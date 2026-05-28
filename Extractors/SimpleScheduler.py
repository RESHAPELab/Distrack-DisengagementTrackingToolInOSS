"""
SimpleScheduler.py — minimal in-memory page scheduler for parallel GitHub extraction.

Design:
- Token 0 does one lightweight pass per stream to enumerate all page cursors.
- Workers claim pages from any stream, process ONE page, and submit rows to the scheduler.
- The scheduler holds out-of-order results in memory and commits them to CSV strictly in order.
- Crash recovery: any non-committed pages (in-progress or in memory) are simply re-done on restart.
  Only 'committed_through' and the CSV files are durable; nothing else needs to be.

State file: scheduler_state.json (in organizationFolder)
"""

import json
import threading
from pathlib import Path
from datetime import datetime, timezone

import portalocker
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import Utilities as util


# ---------------------------------------------------------------------------
# Lightweight enumeration queries — fetch ONLY pageInfo + totalCount, no data
# ---------------------------------------------------------------------------

_ENUM_QUERY_ISSUES = """
query($owner:String!, $name:String!, $first:Int!, $after:String) {
  repository(owner:$owner, name:$name) {
    issues(states:[OPEN, CLOSED], first:$first, after:$after, orderBy:{field:CREATED_AT, direction:ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
    }
  }
}"""

_ENUM_QUERY_PRS = """
query($owner:String!, $name:String!, $first:Int!, $after:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:$first, after:$after, orderBy:{field:CREATED_AT, direction:ASC}) {
      totalCount
      pageInfo { hasNextPage endCursor }
    }
  }
}"""

_ENUM_QUERY_COMMITS = """
query($owner:String!, $name:String!, $first:Int!, $after:String) {
  repository(owner:$owner, name:$name) {
    defaultBranchRef {
      target {
        ... on Commit {
          history(first:$first, after:$after) {
            totalCount
            pageInfo { hasNextPage endCursor }
          }
        }
      }
    }
  }
}"""

_ENUM_QUERIES = {
    "issues_with_timeline": (_ENUM_QUERY_ISSUES,
                             ("data", "repository", "issues")),
    "prs_with_comments":    (_ENUM_QUERY_PRS,
                             ("data", "repository", "pullRequests")),
    "commits":              (_ENUM_QUERY_COMMITS,
                             ("data", "repository", "defaultBranchRef", "target", "history")),
}


class SimpleScheduler:
    """
    Coordinates N parallel workers across three extraction streams.
    Thread-safe via a single reentrant lock.
    """

    STREAM_ORDER = ["issues_with_timeline", "prs_with_comments", "commits"]

    # maps stream → ordered list of CSV table keys (primary first)
    STREAM_TABLES = {
        "issues_with_timeline": ["issues", "issue_activity"],
        "prs_with_comments":    ["prs_repo", "prs_comments"],
        "commits":              ["commits", "per_file_commits"],
    }

    # maps stream → key in pbars dict
    STREAM_PBAR = {
        "issues_with_timeline": "issues",
        "prs_with_comments":    "prs",
        "commits":              "commits",
    }

    def __init__(self, organizationFolder, table_paths, pbars=None):
        """
        organizationFolder : Path-like root for this repo's data
        table_paths        : dict mapping CSV key → Path  (e.g. {"issues": Path(...)})
        pbars              : dict mapping pbar key → tqdm  (attached later via scheduler.pbars)
        """
        self.folder = Path(organizationFolder)
        self.state_path = self.folder / "scheduler_state.json"
        self.table_paths = table_paths
        self.pbars = pbars or {}
        self.lock = threading.Lock()
        self._pending = {s: {} for s in self.STREAM_ORDER}  # {stream: {page_idx: rows_by_type}}
        self.state = self._load_or_init()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _empty_stream(self):
        return {
            "cursors": [],          # [null, "cursor1", "cursor2", …] — one entry per page
            "total_count": None,    # total items in stream (from GraphQL totalCount)
            "committed_through": -1,
            "page_statuses": {},    # {"0": "pending"|"in_progress"|"committed"}
            "complete": False,
        }

    def _load_or_init(self):
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Crash recovery: reset any in_progress pages back to pending
            for stream in state["streams"].values():
                for idx, status in stream["page_statuses"].items():
                    if status == "in_progress":
                        stream["page_statuses"][idx] = "pending"
            self._save_unlocked(state)
            return state
        else:
            state = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "streams": {s: self._empty_stream() for s in self.STREAM_ORDER},
            }
            self._save_unlocked(state)
            return state

    def _save_unlocked(self, state=None):
        """Write state JSON. Caller must hold self.lock (or call during __init__)."""
        data = state if state is not None else self.state
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with portalocker.Lock(str(self.state_path), timeout=15, flags=portalocker.LOCK_EX) as fh:
            fh.seek(0)
            fh.truncate()
            json.dump(data, fh, indent=2)

    def _save(self):
        """Write state JSON under lock."""
        with self.lock:
            self._save_unlocked()

    # ------------------------------------------------------------------
    # Phase 1: cursor enumeration (Token 0, sequential, idempotent)
    # ------------------------------------------------------------------

    def enumerate_stream(self, requester, stream_name, owner, name, per_page=100):
        """
        Lightweight pass: fetch only pageInfo for the given stream.
        Idempotent — resumes from the last recorded cursor if partially done.
        Saves state after each page.
        """
        stream = self.state["streams"][stream_name]

        # Already fully enumerated?  page_statuses is populated after enumeration.
        if stream["page_statuses"]:
            return

        query, conn_path = _ENUM_QUERIES[stream_name]

        cursors = stream["cursors"] if stream["cursors"] else [None]
        after = cursors[-1]  # resume from last known cursor (None = start)

        # If we have cursors stored (partial enumeration), the last entry IS the resume point
        # but we haven't yet confirmed there's a next page, so re-query from it.
        # Reset to whatever we stored and continue.
        if not stream["cursors"]:
            stream["cursors"] = [None]
            cursors = stream["cursors"]

        # Walk from the last cursor we recorded
        after = cursors[-1]

        while True:
            variables = {"owner": owner, "name": name, "first": per_page, "after": after}
            _, data = requester.graphql_query(query=query, variables=variables)

            conn = data
            for key in conn_path:
                conn = conn[key]

            page_info = conn["pageInfo"]
            total_count = conn.get("totalCount")

            if stream["total_count"] is None and total_count is not None:
                stream["total_count"] = total_count

            if page_info["hasNextPage"]:
                after = page_info["endCursor"]
                cursors.append(after)
            else:
                break

            # Save incrementally so a crash during enumeration can resume
            with self.lock:
                self._save_unlocked()

        # Build page_statuses: one entry per page (0-indexed)
        n_pages = len(cursors)
        stream["page_statuses"] = {str(i): "pending" for i in range(n_pages)}
        with self.lock:
            self._save_unlocked()

    # ------------------------------------------------------------------
    # Phase 2: work claiming
    # ------------------------------------------------------------------

    def claim_page(self):
        """
        Thread-safe. Iterate STREAM_ORDER and return the first pending page as
        (stream_name, page_idx, cursor).  Returns None if all pages are claimed/done.
        """
        with self.lock:
            for stream_name in self.STREAM_ORDER:
                stream = self.state["streams"][stream_name]
                if stream["complete"]:
                    continue
                committed = stream["committed_through"]
                for idx_str, status in stream["page_statuses"].items():
                    idx = int(idx_str)
                    if idx <= committed:
                        continue  # already committed
                    if status == "pending":
                        stream["page_statuses"][idx_str] = "in_progress"
                        cursor = stream["cursors"][idx]
                        self._save_unlocked()
                        return (stream_name, idx, cursor)
            return None

    def mark_failed(self, stream_name, page_idx):
        """Reset a failed page back to pending so it gets retried."""
        with self.lock:
            self.state["streams"][stream_name]["page_statuses"][str(page_idx)] = "pending"
            self._save_unlocked()

    # ------------------------------------------------------------------
    # Phase 2: result submission and in-order commit
    # ------------------------------------------------------------------

    def submit_page(self, stream_name, page_idx, rows_by_type):
        """
        Store this page's rows in memory and attempt an in-order commit.
        rows_by_type: e.g. {"issues": [...], "issue_activity": [...]}
        """
        with self.lock:
            self._pending[stream_name][page_idx] = rows_by_type
            self._try_commit(stream_name)

    def _try_commit(self, stream_name):
        """
        Must be called under self.lock.
        Advance committed_through as far as possible by writing consecutive ready pages to CSV.
        """
        stream = self.state["streams"][stream_name]
        next_idx = stream["committed_through"] + 1
        pending = self._pending[stream_name]

        while next_idx in pending:
            rows_by_type = pending.pop(next_idx)

            # Write each table's rows to its CSV file
            for table_key in self.STREAM_TABLES[stream_name]:
                rows = rows_by_type.get(table_key, [])
                if rows:
                    util.append_rows_csv(self.table_paths[table_key], rows)

            # Advance committed_through and persist
            stream["committed_through"] = next_idx
            stream["page_statuses"][str(next_idx)] = "committed"
            self._save_unlocked()

            next_idx += 1

        # Mark stream complete when every page is committed
        total_pages = len(stream["cursors"])
        if total_pages > 0 and stream["committed_through"] >= total_pages - 1:
            stream["complete"] = True
            self._save_unlocked()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def all_complete(self):
        return all(
            self.state["streams"][s]["complete"] for s in self.STREAM_ORDER
        )

    def total_pages(self, stream_name):
        return len(self.state["streams"][stream_name]["cursors"])

    def committed_count(self, stream_name):
        """Number of pages committed so far (committed_through + 1, minimum 0)."""
        return max(0, self.state["streams"][stream_name]["committed_through"] + 1)

    def reset_stream_full(self, stream_name):
        """Hard-reset a stream so collection restarts from page 1."""
        with self.lock:
            self.state["streams"][stream_name] = self._empty_stream()
            self._save_unlocked()
