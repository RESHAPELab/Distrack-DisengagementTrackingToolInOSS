#   conda activate osslab
#   python CommitExtractorV3.py
### IMPORT EXCEPTION MODULES
import uuid
from requests.exceptions import Timeout
from github import GithubException, UnknownObjectException, IncompletableObject

### IMPORT SYSTEM MODULES
from github import Github
import os, logging, pandas, csv, tempfile, shutil, functools
from datetime import datetime, timezone, timedelta
from tqdm import tqdm, tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal, threading
import json
from contextlib import contextmanager
from dateutil import tz as _tz
from collections import Counter


### IMPORT CUSTOM MODULES
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import Settings as cfg
import Utilities as util
from pathlib import Path
import subprocess, tempfile, shutil, logging
from truckfactor.compute import main as compute_tf
import portalocker              # pip install portalocker
import warnings
from SimpleScheduler import SimpleScheduler

from dataclasses import dataclass

warnings.filterwarnings("ignore")
from git import Repo, exc as git_exc

### DEFINE CONSTANTS
COMPLETE = "COMPLETE"
STOP_EVENT = threading.Event()

def updateAlldata(organizationFolder, organization, project, full_extraction):
    repo_full_name = f"{organization}/{project}"
    #output
    commits_csv        = cfg.commit_list_file_name
    prs_csv            = cfg.PR_list_file_name
    prs_comments_csv   = cfg.prs_comments_csv
    issues_csv         = cfg.issue_list_file_name
    issue_activity_csv = cfg.issue_activity_file_name
    per_file_commits_path = cfg.per_file_commits_path
    repo_tree_path = cfg.repo_tree_path

    excluded_path = Path(organizationFolder, cfg.excluded_csv)


    # data collected
    commit_cols = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email",  
                   "sha", "filename_list", "fileschanged_count", "additions_sum", "deletions_sum", ]
    
    pr_cols     = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "PR_id",
                   "state", "merged", "closed_at", "merged_at"]
                
    prcom_cols  = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "PR_id",
                   "comment_id", "event"]

    issue_cols  = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "issue_number", "title",
                   "state", "closed_at", "labels", "assignees", "milestone"]

    issue_activity_cols = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "issue_number", 
                           "activity_id", "item_type", "event", "body"]

    excluded_cols = ["source_file", "error_id", "additional_info"]

    per_file_commit_columns=["repo", "author_id", "author_login","author_name","author_email","sha", "committed_at", "committer_login",
            "file_path","status","additions","deletions","changes","previous_filename"]

    repo_tree_columns = ["repo", "path", "type", "sha", "size", "mode"]

    # check what data needs to be extracted and collected.
    #cursor
    cursor_path = Path(organizationFolder,  cfg.data_cursor)

    if cursor_path.exists():
        with open(cursor_path, 'r') as file:
            states = json.load(file)

        # Check completed streams and reset if older than threshold
        RESET_THRESHOLD = timedelta(weeks=52)
        now = datetime.now(timezone.utc)
        updated_at = datetime.fromisoformat(states["updated_at"])

        reset_streams = []
        for stream_name, stream in states["streams"].items():
            if stream.get("complete") == True:
                age = now - updated_at
                if age > RESET_THRESHOLD:
                    print(f"[cursor] '{stream_name}' completed but is {age.days} days old — resetting.")
                    stream["complete"] = False
                    reset_streams.append(stream_name)

        # save back so the reset is persisted
        util.save_json(states, cursor_path, type="data_cursor")

        # Sync resets into scheduler_state.json so it re-enumerates those streams
        scheduler_path = Path(organizationFolder, "scheduler_state.json")
        if reset_streams and scheduler_path.exists():
            with open(scheduler_path, "r") as f:
                sched_state = json.load(f)
            for sname in reset_streams:
                if sname in sched_state.get("streams", {}):
                    n_old = len(sched_state["streams"][sname].get("cursors", []))
                    sched_state["streams"][sname]["committed_through"] = n_old - 1
                    sched_state["streams"][sname]["page_statuses"] = {}
                    sched_state["streams"][sname]["complete"] = False
            with open(scheduler_path, "w") as f:
                json.dump(sched_state, f, indent=2)


        commits_df      = (pandas.read_csv(Path(organizationFolder, commits_csv), sep=cfg.CSV_separator)
                       if Path(organizationFolder, commits_csv).exists()
                       else pandas.DataFrame(columns=commit_cols))
        
        per_file_commits_df = (pandas.read_csv(Path(organizationFolder, per_file_commits_path), sep=cfg.CSV_separator)
                               if Path(organizationFolder, per_file_commits_path).exists()
                               else pandas.DataFrame(columns=per_file_commit_columns))
        repo_tree_df = (pandas.read_csv(Path(organizationFolder, repo_tree_path), sep=cfg.CSV_separator)
                        if Path(organizationFolder, repo_tree_path).exists()
                        else pandas.DataFrame(columns=repo_tree_columns))

        prs_df          = (pandas.read_csv(Path(organizationFolder, prs_csv),
                                    sep=cfg.CSV_separator)
                        if Path(organizationFolder, prs_csv).exists()
                        else pandas.DataFrame(columns=pr_cols))
        
        prs_comments_df = (pandas.read_csv(Path(organizationFolder, prs_comments_csv),
                                    sep=cfg.CSV_separator)
                        if Path(organizationFolder, prs_comments_csv).exists()
                        else pandas.DataFrame(columns=prcom_cols))
        
        issues_df       = (pandas.read_csv(Path(organizationFolder, issues_csv), 
                                    sep=cfg.CSV_separator)
                        if Path(organizationFolder, issues_csv).exists()
                        else pandas.DataFrame(columns=issue_cols))

        issue_activity_df = (pandas.read_csv(Path(organizationFolder, issue_activity_csv),
                                    sep=cfg.CSV_separator)
                        if Path(organizationFolder, issue_activity_csv).exists()
                        else pandas.DataFrame(columns=issue_activity_cols))


    else:
        # we need to make the json file and set it all to zero
        states = {
          "repo": repo_full_name,
          "updated_at": datetime.now(timezone.utc).isoformat(),
          "streams": {
            "issues_with_timeline": { "after": None, "processed": 0, "total": None, "complete": False },
            "prs_with_comments": { "after": None, "processed": 0, "total": None, "complete": False },
            "commits": { "after": None, "processed": 0, "total": None, "complete": False }
          },
          "meta": {
            "last_error": None,
            "last_rate_limit_reset": None
          }
        }
        # we need to make the csv files too
        util.ensure_csv(Path(organizationFolder, commits_csv), columns=commit_cols, sep=cfg.CSV_separator)
        util.ensure_csv(Path(organizationFolder, prs_csv), columns=pr_cols, sep=cfg.CSV_separator)
        util.ensure_csv(Path(organizationFolder, prs_comments_csv), columns=prcom_cols, sep=cfg.CSV_separator)
        util.ensure_csv(Path(organizationFolder, issues_csv), columns=issue_cols, sep=cfg.CSV_separator)
        util.ensure_csv(Path(organizationFolder, issue_activity_csv), columns=issue_activity_cols, sep=cfg.CSV_separator)
        util.ensure_csv(excluded_path, columns=excluded_cols, sep=cfg.CSV_separator)
        util.ensure_csv(Path(organizationFolder, per_file_commits_path), columns=per_file_commit_columns, sep=cfg.CSV_separator)
        util.ensure_csv(Path(organizationFolder, repo_tree_path), columns=repo_tree_columns, sep=cfg.CSV_separator)
        commits_df      = pandas.DataFrame(columns=commit_cols)
        prs_df          = pandas.DataFrame(columns=pr_cols)
        prs_comments_df = pandas.DataFrame(columns=prcom_cols)
        issues_df       = pandas.DataFrame(columns=issue_cols)
        issue_activity_df = pandas.DataFrame(columns=issue_activity_cols)
        per_file_commits_df = pandas.DataFrame(columns=per_file_commit_columns)
        repo_tree_df = pandas.DataFrame(columns=repo_tree_columns)
        # excluded df
        excluded_df     = pandas.DataFrame(columns=excluded_cols)
        #makes the new blank json file 
        util.save_json(states, cursor_path, type= "data_cursor")


    tables = { "issues": issues_df, "issue_activity": issue_activity_df,
               "prs_repo": prs_df, "prs_comments": prs_comments_df,
               "commits": commits_df }

    _new_updateAlldata_dispatch(
        states, tables, repo_full_name, organizationFolder, full_extraction
    )
    return


def extraction_worker(order, tables, repo_full_name, organizationFolder):
    # This function will be run in a separate thread for each work order
    # It will handle the extraction of data for a specific kind of entity (issue, PR, commit)
    # and write the results to the appropriate CSV files

    token = util.getSpisificToken(order["token_idx"])
    g = Github(token)
    g.per_page = cfg.items_per_page

    if order["kind"] == "Issue":
        return updateIssueListFile(g, token, repo_full_name, organizationFolder, order, tables)
    if order["kind"] == "PR":
        return updatePRListFile(g, token, repo_full_name, organizationFolder, order, tables)
    if order["kind"] == "Commit":
        return updateCommitListFile(g, token, repo_full_name, organizationFolder, order, tables,
                                    collect_per_file=order.get("collect_per_file", False))

    return None


# ---------------------------------------------------------------------------
# Parallel dispatch (called by updateAlldata when 2+ tokens are available)
# ---------------------------------------------------------------------------

def _new_updateAlldata_dispatch(states, tables, repo_full_name, organizationFolder, full_extraction):
    tokens_list, _ = util.getTokensList()
    cursor_path = Path(organizationFolder, cfg.data_cursor)

    # ── Serial fallback: 1 token → reuse existing extraction_worker unchanged ──
    if len(tokens_list) == 1:
        for stream_name, kind in [
            ("issues_with_timeline", "Issue"),
            ("prs_with_comments",    "PR"),
            ("commits",              "Commit"),
        ]:
            if not states["streams"][stream_name]["complete"]:
                order = {
                    "kind": kind,
                    "state": states["streams"][stream_name],
                    "token_idx": 0,
                }
                extraction_worker(order, tables, repo_full_name, organizationFolder)
        return

    # ── Parallel path: 2+ tokens ────────────────────────────────────────────
    table_paths = {
        "issues":           Path(organizationFolder, cfg.issue_list_file_name),
        "issue_activity":   Path(organizationFolder, cfg.issue_activity_file_name),
        "prs_repo":         Path(organizationFolder, cfg.PR_list_file_name),
        "prs_comments":     Path(organizationFolder, cfg.prs_comments_csv),
        "commits":          Path(organizationFolder, cfg.commit_list_file_name),
        "per_file_commits": Path(organizationFolder, cfg.per_file_commits_path),
    }

    scheduler = SimpleScheduler(organizationFolder, table_paths)

    # Phase 1: Token 0 enumerates page cursors for every incomplete stream
    token0 = util.getSpisificToken(0)
    g0 = Github(token0)
    g0.per_page = cfg.items_per_page
    repo0 = g0.get_repo(repo_full_name)
    requester0 = getattr(repo0, "requester", None) or getattr(repo0, "_requester", None)
    owner, name = repo0.owner.login, repo0.name

    print("[scheduler] Phase 1: enumerating page cursors with token 0…")
    for stream_name in SimpleScheduler.STREAM_ORDER:
        if not states["streams"][stream_name]["complete"]:
            scheduler.enumerate_stream(requester0, stream_name, owner, name,
                                       per_page=cfg.items_per_page)
            n = scheduler.total_pages(stream_name)
            print(f"  {stream_name}: {n} page(s) to process")
  
    # Phase 2: Build shared progress bars initialised to already-committed pages
    def _make_pbar(stream_name, desc, position):
        total = scheduler.total_pages(stream_name) * cfg.items_per_page
        pb = tqdm(total=total or None, desc=desc, position=position, leave=True)
        already_done = scheduler.committed_count(stream_name) * cfg.items_per_page
        if already_done:
            pb.update(already_done)
        return pb

    pbars = {
        "issues":  _make_pbar("issues_with_timeline", "Issues", 0),
        "commits": _make_pbar("commits",              "Commit", 2),
        "prs":     _make_pbar("prs_with_comments",   "PRs",    1),
    }
    scheduler.pbars = pbars

    # Phase 3: One worker thread per token
    print(f"[scheduler] Phase 2: dispatching {len(tokens_list)} workers…")
    with ThreadPoolExecutor(max_workers=len(tokens_list)) as ex:
        futures = [
            ex.submit(
                _parallel_worker, idx, tokens_list[idx],
                scheduler, tables, repo_full_name, organizationFolder,
            )
            for idx in range(len(tokens_list))
        ]
        for f in as_completed(futures):
            f.result()   # propagate any unhandled worker exception

    # Sync completion flags back to data_cursor.json so serial mode sees them
    for stream_name in SimpleScheduler.STREAM_ORDER:
        if scheduler.state["streams"][stream_name]["complete"]:
            states["streams"][stream_name]["complete"] = True
    util.save_json(states, cursor_path, type="data_cursor")

    for pb in pbars.values():
        pb.close()
    #os.system('cls' if os.name == 'nt' else 'clear')

def _parallel_worker(token_idx, token, scheduler, tables, repo_full_name, organizationFolder):
    """Worker thread body: claim a page, process it, submit rows, repeat."""
    g = Github(token)
    g.per_page = cfg.items_per_page

    KIND_MAP = {
        "issues_with_timeline": "Issue",
        "prs_with_comments":    "PR",
        "commits":              "Commit",
    }

    while True:
        claim = scheduler.claim_page()
        if claim is None:
            break   # all pages claimed or complete

        stream_name, page_idx, cursor = claim

        order = {
            "kind":        KIND_MAP[stream_name],
            "state":       {"after": cursor, "processed": page_idx * cfg.items_per_page,
                            "total": None, "complete": False},
            "token_idx":   token_idx,
            # Parallel-mode extras — read via order.get() inside each update function
            "page_idx":    page_idx,
            "scheduler":   scheduler,
            "stream_name": stream_name,
        }

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES + 1):
            try:
                if order["kind"] == "Issue":
                    updateIssueListFile(g, token, repo_full_name, organizationFolder, order, tables)
                elif order["kind"] == "PR":
                    updatePRListFile(g, token, repo_full_name, organizationFolder, order, tables)
                elif order["kind"] == "Commit":
                    updateCommitListFile(g, token, repo_full_name, organizationFolder, order, tables,
                                        collect_per_file=order.get("collect_per_file", False))
                break  # success

            except Exception as e:
                err = str(e)

                if "RATE_LIMIT" in err or "rate limit" in err.lower():
                    if attempt < MAX_RETRIES:
                        wait = 60 * (attempt + 1)
                        tqdm.write(
                            f"[worker {token_idx}] rate-limit on {stream_name} "
                            f"page {page_idx} — retrying in {wait}s "
                            f"({attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(wait)
                    else:
                        tqdm.write(
                            f"[worker {token_idx}] {stream_name} page {page_idx} "
                            f"gave up after {MAX_RETRIES} rate-limit retries"
                        )
                        scheduler.mark_failed(stream_name, page_idx)

                elif "Schema mismatch" in err:
                    tqdm.write(
                        f"[worker {token_idx}] SCHEMA ERROR in {stream_name} — "
                        f"delete the CSV and re-run. Details: {err.splitlines()[0]}"
                    )
                    scheduler.mark_failed(stream_name, page_idx)
                    break  # retrying won't help

                else:
                    tqdm.write(
                        f"[worker {token_idx}] {stream_name} page {page_idx} failed: {e}"
                    )
                    scheduler.mark_failed(stream_name, page_idx)
                    break


def updateIssueListFile(g, token, repo_full_name: str, organizationFolder: str, order, tables=None):
    """
    Single-worker durable-cursor extractor:
    - Writes issues metadata to `issues.csv` (1 row/issue)
    - Writes ALL activity traces (events + comments + cross-refs) to `issue_activity.csv` from the Timeline API
      Columns: repo, issue_number, activity_id, item_type, event, body, created_at, actor
    """
    if order["state"]["complete"]:
        return
    
    issues_csv = cfg.issue_list_file_name 
    activity_csv = cfg.issue_activity_file_name

    out_dir = Path(organizationFolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cursor_path = Path(organizationFolder, cfg.data_cursor)

    repo = g.get_repo(repo_full_name)
    owner, name = repo.owner.login, repo.name
    requester = getattr(repo, "requester", None) or getattr(repo, "_requester", None)

    # Ensure CSVs exist with headers
    
    base_variables={"owner": owner, "name": name}
    issue_query = """
    query($owner:String!, $name:String!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        issues(states:[OPEN, CLOSED], first:$first, after:$after, orderBy:{field:CREATED_AT, direction:ASC}) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            title
            state
            createdAt
            closedAt
            author { 
              login
              ... on User { 
                id
                name
                email
              }
            }
            labels(first:20) { nodes { name } }
            assignees(first:10) { 
              nodes { 
                id
                login 
              } 
            }
            milestone { title }
          }
        }
      }
    }"""

    issue_connection_path = ("data","repository","issues")

    issue_timeline_query = """
    query($owner:String!, $name:String!, $number:Int!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        issue(number:$number) {
          timelineItems(first:$first, after:$after) {
            pageInfo { hasNextPage endCursor }
            nodes {
              __typename
              ... on IssueComment {
                id
                createdAt
                author { 
                  login
                  ... on User { 
                    id
                    name
                    email
                  }
                }
                bodyText
              }
              ... on ClosedEvent {
                id
                createdAt
                actor { 
                  login
                  ... on User { 
                    id
                    name
                    email
                  }
                }
              }
              ... on ReopenedEvent {
                id
                createdAt
                actor { 
                  login
                  ... on User { 
                    id
                    name
                    email
                  }
                }
              }
              ... on LabeledEvent {
                id
                createdAt
                actor { 
                  login
                  ... on User { 
                    id
                    name
                    email
                  }
                }
                label { name }
              }
              ... on UnlabeledEvent {
                id
                createdAt
                actor { 
                  login
                  ... on User { 
                    id
                    name
                    email
                  }
                }
                label { name }
              }
              ... on CrossReferencedEvent {
                id
                createdAt
                actor { 
                  login
                  ... on User { 
                    id
                    name
                    email
                  }
                }
              }
            }
          }
        }
      }
    }"""

    per_page= cfg.items_per_page
    # We need to look at the state file which can tell us about
    # the current progress of this extraction
    # we need to check two things
    # 1. if the extraction is already complete
    # 2. if not complete where did we leave off

    after = order["state"]["after"]

    # ── PARALLEL MODE: one page only, collect rows, hand off to scheduler ──
    if order.get("scheduler") is not None:
        _scheduler  = order["scheduler"]
        _page_idx   = order["page_idx"]
        _stream     = order["stream_name"]

        vars_page = {"owner": owner, "name": name, "first": per_page, "after": after}
        _, issue_data = requester.graphql_query(query=issue_query, variables=vars_page)
        issue_conn = issue_data
        for key in issue_connection_path:
            issue_conn = issue_conn[key]
        nodes = issue_conn.get("nodes", []) or []

        issue_rows = []
        for nd in nodes:
            author_id    = (nd.get("author") or {}).get("id")
            author_login = (nd.get("author") or {}).get("login")
            author_name  = (nd.get("author") or {}).get("name")
            author_email = (nd.get("author") or {}).get("email")
            labels    = [x["name"] for x in (nd.get("labels") or {}).get("nodes", [])]
            assignees = [x["login"] for x in (nd.get("assignees") or {}).get("nodes", [])]
            issue_rows.append({
                "repo": repo_full_name,
                "created_at": nd["createdAt"],
                "author_id": author_id,
                "author_name": author_name,
                "author_login": author_login,
                "author_email": author_email,
                "issue_number": nd["number"],
                "title": nd["title"],
                "state": nd["state"],
                "closed_at": nd["closedAt"],
                "labels": "|".join(labels),
                "assignees": "|".join(assignees),
                "milestone": (nd.get("milestone") or {}).get("title"),
            })

        activity_all_rows = []
        for nd in nodes:
            num = nd["number"]
            after_tl = None
            while True:
                vars_tl = {"owner": owner, "name": name, "number": int(num), "first": 100, "after": after_tl}
                _, tl_d = requester.graphql_query(query=issue_timeline_query, variables=vars_tl)
                tl_conn = tl_d["data"]["repository"]["issue"]["timelineItems"]
                tl_nodes = tl_conn.get("nodes", []) or []
                for it in tl_nodes:
                    t = it.get("__typename")
                    aid = it.get("id")
                    created = it.get("createdAt")
                    body = ""
                    event_name = ""
                    if t == "IssueComment":
                        body = it.get("bodyText") or ""
                        event_name = ""
                    elif t in ("ClosedEvent", "ReopenedEvent"):
                        event_name = "closed" if t == "ClosedEvent" else "reopened"
                    elif t in ("LabeledEvent", "UnlabeledEvent"):
                        lbl = (it.get("label") or {}).get("name")
                        event_name = f"{'labeled' if t=='LabeledEvent' else 'unlabeled'}:{lbl}" if lbl else ("labeled" if t=="LabeledEvent" else "unlabeled")
                    elif t == "CrossReferencedEvent":
                        event_name = "cross_referenced"
                    else:
                        event_name = ""
                    _actor = it.get("author") or it.get("actor") or {}
                    author_id    = _actor.get("id")
                    author_login = _actor.get("login")
                    author_name  = _actor.get("name")
                    author_email = _actor.get("email")
                    activity_all_rows.append({
                        "repo": repo_full_name,
                        "created_at": created,
                        "author_id": author_id,
                        "author_name": author_name,
                        "author_login": author_login,
                        "author_email": author_email,
                        "issue_number": num,
                        "activity_id": aid,
                        "item_type": t,
                        "event": event_name,
                        "body": body,
                    })
                if tl_conn["pageInfo"]["hasNextPage"]:
                    after_tl = tl_conn["pageInfo"]["endCursor"]
                else:
                    break
            # one issue fully collected (matches serial pbar_issue.update(1) behaviour)
            _pbar = _scheduler.pbars.get("issues")
            if _pbar:
                _pbar.update(1)

        _scheduler.submit_page(_stream, _page_idx,
                               {"issues": issue_rows, "issue_activity": activity_all_rows})
        return
    # ── END PARALLEL MODE ──────────────────────────────────────────────────

    # ---------- set pbar.total from totalCount (once) ----------
    total_count_query = """
    query($owner:String!, $name:String!) {
      repository(owner:$owner, name:$name) {
        issues(states:[OPEN, CLOSED]) { totalCount }
      }
    }"""
    _, total_d = requester.graphql_query(query=total_count_query, variables={"owner": owner, "name": name})
    total_issues = total_d["data"]["repository"]["issues"]["totalCount"]
    order["state"]["total"] = order["state"]["total"] or int(total_issues)
    
    util.save_json(order["state"], data_cursor_path, type= "issues_with_timeline")
    pbar_issue = tqdm(total=total_issues, desc="Issues", position=0, leave=True)
    pbar_issue.update(order["state"]["processed"])
    #PBAR done
    
    per_page = cfg.items_per_page

    while True:
        vars_page = {"owner": owner, "name": name, "first": per_page, "after": after}
        _, issue_data = requester.graphql_query(query=issue_query, variables=vars_page)

        # walk to connection dict
        issue_conn = issue_data
        for key in issue_connection_path:
            issue_conn = issue_conn[key]

        nodes = issue_conn.get("nodes", []) or []
        page_info = issue_conn["pageInfo"]

        # ---- write issues rows ----
        issue_rows = []
        for nd in nodes:
            
            author_id    = (nd.get("author") or {}).get("id")
            author_login = (nd.get("author") or {}).get("login")
            author_name  = (nd.get("author") or {}).get("name")
            author_email = (nd.get("author") or {}).get("email")

            labels    = [x["name"] for x in (nd.get("labels") or {}).get("nodes", [])]
            assignees = [x["login"] for x in (nd.get("assignees") or {}).get("nodes", [])]
            issue_cols  = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "issue_number", "title",
                   "state", "closed_at", "labels", "assignees", "milestone"]
            issue_rows.append({
                "repo": repo_full_name,
                "created_at": nd["createdAt"],
                "author_id": author_id,
                "author_name": author_name,
                "author_login": author_login,
                "author_email": author_email,
                "issue_number": nd["number"],
                "title": nd["title"],
                "state": nd["state"],
                "closed_at": nd["closedAt"],
                "labels": "|".join(labels),
                "assignees": "|".join(assignees),
                "milestone": (nd.get("milestone") or {}).get("title"),
            })
        if issue_rows:
            util.append_rows_csv(Path(organizationFolder, issues_csv), issue_rows, sep=cfg.CSV_separator)

        # ---- for each issue, fully page timeline and write activity ----
        activity_all_rows = []
        for nd in nodes:
            num = nd["number"]

            after_tl = None
            while True:
                vars_tl = {"owner": owner, "name": name, "number": int(num), "first": 100, "after": after_tl}
                _, tl_d = requester.graphql_query(query=issue_timeline_query, variables=vars_tl)

                tl_conn = tl_d["data"]["repository"]["issue"]["timelineItems"]
                tl_nodes = tl_conn.get("nodes", []) or []

                # map timeline nodes -> activity rows
                for it in tl_nodes:
                    t = it.get("__typename")
                    aid = it.get("id")
                    created = it.get("createdAt")
                    actor = (it.get("author") or it.get("actor") or {}).get("login")
                    body = ""
                    event_name = ""

                    if t == "IssueComment":
                        body = it.get("bodyText") or ""
                        event_name = ""
                    elif t in ("ClosedEvent", "ReopenedEvent"):
                        event_name = "closed" if t == "ClosedEvent" else "reopened"
                    elif t in ("LabeledEvent", "UnlabeledEvent"):
                        lbl = (it.get("label") or {}).get("name")
                        event_name = f"{'labeled' if t=='LabeledEvent' else 'unlabeled'}:{lbl}" if lbl else ("labeled" if t=="LabeledEvent" else "unlabeled")
                    elif t == "CrossReferencedEvent":
                        event_name = "cross_referenced"
                    else:
                        # keep unknowns but still record type+timestamps
                        event_name = ""
                    
                    _actor = it.get("author") or it.get("actor") or {}
                    author_id    = _actor.get("id")
                    author_login = _actor.get("login")
                    author_name  = _actor.get("name")
                    author_email = _actor.get("email")

                    activity_all_rows.append({
                        "repo": repo_full_name,
                        "created_at": created,
                        "author_id": author_id,
                        "author_name": author_name,
                        "author_login": author_login,
                        "author_email": author_email,
                        "issue_number": num,
                        "activity_id": aid,
                        "item_type": t,
                        "event": event_name,
                        "body": body,
                    })

                if tl_conn["pageInfo"]["hasNextPage"]:
                    after_tl = tl_conn["pageInfo"]["endCursor"]
                else:
                    break
            pbar_issue.update(1)

        if activity_all_rows:
            # optional: de-dup by activity_id before writing
            util.append_rows_csv(Path(organizationFolder, activity_csv), activity_all_rows, sep=cfg.CSV_separator)

        # ---- advance durable cursor (only after timelines done) ----
        processed_issues = len(nodes)
        order["state"]["processed"] += processed_issues

        after = page_info.get("endCursor")
        order["state"]["after"] = after
        util.save_json(order["state"], data_cursor_path, type= "issues_with_timeline")

        if not page_info.get("hasNextPage"):
            order["state"]["complete"] = True
            util.save_json(order["state"], data_cursor_path, type= "issues_with_timeline")
            break

    #os.system('cls' if os.name=='nt' else 'clear')

    return

def updateCommitListFile(g, token, repo_full_name: str, organizationFolder: str, order, tables=None, collect_per_file=False):
    # --- paths & setup
    out_dir = Path(organizationFolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cursor_path = Path(organizationFolder, cfg.data_cursor)
    commit_csv_path  = Path(organizationFolder, cfg.commit_list_file_name)

    g, token, search_limit ,   reset_time = util.getSameToken(g, token, order['token_idx'])

    repo      = g.get_repo(repo_full_name)
    owner, name = repo.owner.login, repo.name
    requester = getattr(repo, "requester", None) or getattr(repo, "_requester", None)
    commit_query_safe = """
    query($owner:String!, $name:String!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        defaultBranchRef {
          target {
            ... on Commit {
              history(first:$first, after:$after) {
                totalCount
                pageInfo { hasNextPage endCursor }
                nodes {
                  oid
                  committedDate
                  author {
                    name
                    email
                    user {
                      id
                      login
                      name
                      email
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    after = order["state"]["after"]

    # ── PARALLEL MODE: one page only, collect commit + file rows, submit ───
    if order.get("scheduler") is not None:
        _scheduler = order["scheduler"]
        _page_idx  = order["page_idx"]
        _stream    = order["stream_name"]

        per_page_par = min(cfg.items_per_page, 100)
        # offset by 3 so reset bars go below Issues(0)/PRs(1)/Commits(2) main bars
        g, token, search_limit, reset_time = util.getSameToken(g, token, 3 + order["token_idx"])
        vars_page = {"owner": owner, "name": name, "first": per_page_par, "after": after}
        _, commit_data = requester.graphql_query(query=commit_query_safe, variables=vars_page)
        repo_d = commit_data.get("data", {}).get("repository", {})
        dref   = repo_d.get("defaultBranchRef")
        if not dref or not dref.get("target"):
            _scheduler.mark_failed(_stream, _page_idx)
            return
        hist = dref["target"].get("history")
        if not hist:
            _scheduler.mark_failed(_stream, _page_idx)
            return

        nodes = hist.get("nodes", []) or []
        commit_rows = []
        file_rows   = []
        for nd in nodes:
            sha            = nd.get("oid")
            committed_date = nd.get("committedDate")
            author_block   = nd.get("author") or {}
            auser = (author_block.get("user") or {})
            author_id    = auser.get("id")
            author_login = auser.get("login")
            author_name  = author_block.get("name")
            author_email = author_block.get("email")
            adds    = nd.get("additions") or 0
            dels    = nd.get("deletions") or 0
            changed = nd.get("changedFilesIfAvailable") or 0
            filenames = []
            if collect_per_file:
                try:
                    c = repo.get_commit(sha)
                    files = c.files or []
                    for f in files:
                        file_rows.append({
                            "repo": repo_full_name,
                            "author_id": author_id,
                            "author_login": author_login,
                            "author_name": author_name,
                            "author_email": author_email,
                            "sha": sha,
                            "committed_at": committed_date,
                            "committer_login": (c.committer.login if c.committer else None),
                            "file_path": f.filename,
                            "status": getattr(f, "status", None),
                            "additions": int(getattr(f, "additions", 0) or 0),
                            "deletions": int(getattr(f, "deletions", 0) or 0),
                            "changes": int(getattr(f, "changes", 0) or 0),
                            "previous_filename": getattr(f, "previous_filename", None),
                        })
                except Exception:
                    pass
            if (author_email is None) and (author_name is None) and (author_id is None):
                continue
            commit_rows.append({
                "repo": repo_full_name,
                "created_at": committed_date,
                "author_id": author_id,
                "author_name": author_name,
                "author_login": author_login,
                "author_email": author_email,
                "sha": sha,
                "filename_list": "|".join(filenames),
                "fileschanged_count": int(changed),
                "additions_sum": int(adds),
                "deletions_sum": int(dels),
            })
            # one commit fully collected (matches serial pb_commit.update(1) behaviour)
            _pbar = _scheduler.pbars.get("commits")
            if _pbar:
                _pbar.update(1)

        _scheduler.submit_page(_stream, _page_idx,
                               {"commits": commit_rows, "per_file_commits": file_rows})
        return
    # ── END PARALLEL MODE ──────────────────────────────────────────────────

    # ---------- set pbar.total from totalCount (once) ----------
    commits_pl = repo.get_commits()
    order["state"]["total"] = int(commits_pl.totalCount)
    util.save_json(order["state"], data_cursor_path, type= "commits")
    pb_commit = tqdm(total=commits_pl.totalCount, desc="Commit", position=2, leave=True)
    #PBAR done
    pb_commit.update(order["state"]["processed"])
    per_page = min(cfg.items_per_page, 100)
    if order["state"]["complete"] == True:
        return

    #my code above this part do not touch!!
    #--------------------------------------------------------------------------------------------------
    # per_file_commits for TF
    #--------------------------------------------------------------------------------------------------

    per_file_commits_path = Path(organizationFolder, cfg.per_file_commits_path)
    repo_tree_csv_path = Path(organizationFolder, cfg.repo_tree_path)

    
    if not repo_tree_csv_path.exists():
        util.ensure_csv(repo_tree_csv_path, columns=["repo", "path", "type", "sha", "size", "mode"], sep=cfg.CSV_separator)

    try:
        existing_tree = pandas.read_csv(repo_tree_csv_path, sep=cfg.CSV_separator)
    except Exception as e:
        existing_tree = pandas.DataFrame()

    if existing_tree.empty:
        try:
            default_branch = repo.default_branch
            branch_sha = repo.get_branch(default_branch).commit.sha
            tree = repo.get_git_tree(branch_sha, recursive=True)
            tree_rows = []
            for item in tree.tree or []:
                tree_rows.append({
                    "repo": repo_full_name,
                    "path": getattr(item, "path", None),
                    "type": getattr(item, "type", None),
                    "sha": getattr(item, "sha", None),
                    "size": getattr(item, "size", None),
                    "mode": getattr(item, "mode", None),
                })
            if tree_rows:
                util.append_rows_csv(repo_tree_csv_path, tree_rows, sep=cfg.CSV_separator)
        except Exception:
            pass
        
    #my code below do not touch this part
    while True:

        vars_page = {"owner": owner, "name": name, "first": per_page, "after": after}
        # we are getting stuck on this line is there a way to print out what we are feeding to the query

        g, token, search_limit ,   reset_time = util.getSameToken(g, token, order['token_idx'])

        _, commit_data = requester.graphql_query(query=commit_query_safe, variables=vars_page)
        repo_d = commit_data.get("data", {}).get("repository", {})
        dref   = repo_d.get("defaultBranchRef")
        if not dref or not dref.get("target"):
            print("error at dref", dref, "\n", repo_d)
            return None
        tgt = dref["target"]
        hist = tgt.get("history")
        if not hist:
            print(f"No commit history found for {repo_full_name}. It may be an empty repository.")
            return None

        # ---- write commit rows ----
        nodes     = hist.get("nodes", []) or []
        page_info = hist["pageInfo"]

        commit_rows = []
        file_rows   = []
        for nd in nodes:
          
            sha            = nd.get("oid")
            committed_date = nd.get("committedDate")
            author_block   = nd.get("author") or {}
            auser = (author_block.get("user") or {})

            author_id    = auser.get("id")
            author_login = auser.get("login")
            author_name  = author_block.get("name")   # <- name/email live at author level too
            author_email = author_block.get("email")

            # counts
            adds = nd.get("additions") or 0
            dels = nd.get("deletions") or 0
            changed = nd.get("changedFilesIfAvailable") or 0

            # filename list via REST (GraphQL does not expose per-commit file list)
            filenames = []
            if collect_per_file:
                try:
                    c = repo.get_commit(sha)  # REST — 1 call per commit, skip when not needed
                    files = c.files or []
                    for f in files:
                        file_rows.append({
                            "repo": repo_full_name,
                            "author_id": author_id,
                            "author_login": author_login,
                            "author_name": author_name,
                            "author_email": author_email,
                            "sha": sha,
                            "committed_at": committed_date,
                            "committer_login": (c.committer.login if c.committer else None),
                            "file_path": f.filename,
                            "status": getattr(f, "status", None),
                            "additions": int(getattr(f, "additions", 0) or 0),
                            "deletions": int(getattr(f, "deletions", 0) or 0),
                            "changes": int(getattr(f, "changes", 0) or 0),
                            "previous_filename": getattr(f, "previous_filename", None),
                        })
                except Exception:
                    pass

            # optional: exclude commits with no author info (mirror old behavior)
            if (author_email is None) and (author_name is None) and (author_id is None):
                # If you keep an "excluded" file, add it here; otherwise just skip.
                # util.add(excluded_df, [sha])  # if you maintain an excluded CSV
                continue
            
            commit_rows.append({
                "repo": repo_full_name,
                "created_at": committed_date,
                "author_id": author_id,
                "author_name": author_name,
                "author_login": author_login,
                "author_email": author_email,
                "sha": sha,
                "filename_list": "|".join(filenames),
                "fileschanged_count": int(changed),
                "additions_sum": int(adds),
                "deletions_sum": int(dels),
            })
            pb_commit.update(1)

        if commit_rows:
            util.append_rows_csv(commit_csv_path, commit_rows, sep=cfg.CSV_separator)
        if file_rows:
            util.append_rows_csv(per_file_commits_path, file_rows, sep=cfg.CSV_separator)

        processed = len(nodes)
        order["state"]["processed"] += processed
        
        after = page_info.get("endCursor")
        order["state"]["after"] = after
        util.save_json(order["state"], data_cursor_path, type= "commits")

        if not page_info.get("hasNextPage"):
            order["state"]["complete"] = True
            util.save_json(order["state"], data_cursor_path, type= "commits")
            break
    #os.system('cls' if os.name=='nt' else 'clear')

    return
    
def updatePRListFile(g, token, repo_full_name: str, organizationFolder: str, order, tables=None):
    # --- paths & setup
    out_dir = Path(organizationFolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    prs_csv_path         = Path(organizationFolder, cfg.PR_list_file_name)
    prs_comments_csv_path  = Path(organizationFolder, cfg.prs_comments_csv)

    data_cursor_path = Path(organizationFolder, cfg.data_cursor)

    repo      = g.get_repo(repo_full_name)
    owner, name = repo.owner.login, repo.name
    requester = getattr(repo, "requester", None) or getattr(repo, "_requester", None)

    # Load existing comment ids to avoid duplicates (idempotent writes)
    existing_comments = set()
    try:
        df_existing = pandas.read_csv(prs_comments_csv_path, sep=cfg.CSV_separator, usecols=["comment_id"])
        existing_comments = set(df_existing["comment_id"].astype(str))
    except Exception:
        pass

    pr_query = """
    query($owner:String!, $name:String!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        pullRequests(states:[OPEN, MERGED, CLOSED], orderBy:{field:CREATED_AT, direction:ASC}, first:$first, after:$after) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            createdAt
            closedAt
            mergedAt
            merged
            state
            author { 
              login
              ... on User { 
                id
                name
                email
              }
            }
          }
        }
      }
    }"""

    # Page PR comments for a single PR
    pr_comments_query = """
    query($owner:String!, $name:String!, $number:Int!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        pullRequest(number:$number) {
          comments(first:$first, after:$after) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              createdAt
              author { 
                login
                ... on User { 
                  id
                  name
                  email
                }
              }
            }
          }
        }
      }
    }"""

    # Page PR reviews for a single PR
    pr_reviews_query = """
    query($owner:String!, $name:String!, $number:Int!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        pullRequest(number:$number) {
          reviews(first:$first, after:$after) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              submittedAt
              author { 
                login
                ... on User { 
                  id
                  name
                  email
                }
              }
            }
          }
        }
      }
    }"""
    


    if order["state"]["complete"]:
        return

    after = order["state"]["after"]

    # ── PARALLEL MODE: one page of PRs + their comments/reviews, submit ───
    if order.get("scheduler") is not None:
        _scheduler = order["scheduler"]
        _page_idx  = order["page_idx"]
        _stream    = order["stream_name"]

        per_page_par = cfg.items_per_page
        vars_page = {"owner": owner, "name": name, "first": per_page_par, "after": after}
        _, pr_data = requester.graphql_query(query=pr_query, variables=vars_page)
        pr_conn = pr_data.get("data", {}).get("repository", {}).get("pullRequests", {}) or {}
        nodes   = pr_conn.get("nodes", []) or []

        pr_rows = []
        for nd in nodes:
            author_id    = (nd.get("author") or {}).get("id")
            author_login = (nd.get("author") or {}).get("login")
            author_name  = (nd.get("author") or {}).get("name")
            author_email = (nd.get("author") or {}).get("email")
            pr_rows.append({
                "repo":         repo_full_name,
                "created_at":   nd.get("createdAt"),
                "author_id":    author_id,
                "author_name":  author_name,
                "author_login": author_login,
                "author_email": author_email,
                "PR_id":        nd.get("number"),
                "state":        nd.get("state"),
                "merged":       bool(nd.get("merged")),
                "closed_at":    nd.get("closedAt"),
                "merged_at":    nd.get("mergedAt"),
            })

        comment_rows = []
        for nd in nodes:
            pr_number = int(nd.get("number"))

            # Comments
            after_c = None
            while True:
                vars_c = {"owner": owner, "name": name, "number": pr_number, "first": 100, "after": after_c}
                _, c_d = requester.graphql_query(query=pr_comments_query, variables=vars_c)
                c_conn = c_d["data"]["repository"]["pullRequest"]["comments"]
                c_nodes = c_conn.get("nodes", []) or []
                for c in c_nodes:
                    cid = str(c.get("id"))
                    if cid in existing_comments:
                        continue
                    comment_rows.append({
                        "repo":         repo_full_name,
                        "created_at":   c.get("createdAt"),
                        "author_id":    (c.get("author") or {}).get("id"),
                        "author_name":  (c.get("author") or {}).get("name"),
                        "author_login": (c.get("author") or {}).get("login"),
                        "author_email": (c.get("author") or {}).get("email"),
                        "PR_id":        pr_number,
                        "comment_id":   cid,
                        "event":        "comment",
                    })
                    existing_comments.add(cid)
                if c_conn["pageInfo"]["hasNextPage"]:
                    after_c = c_conn["pageInfo"]["endCursor"]
                else:
                    break

            # Reviews
            after_r = None
            while True:
                vars_r = {"owner": owner, "name": name, "number": pr_number, "first": 100, "after": after_r}
                _, r_d = requester.graphql_query(query=pr_reviews_query, variables=vars_r)
                r_conn = r_d["data"]["repository"]["pullRequest"]["reviews"]
                r_nodes = r_conn.get("nodes", []) or []
                for r in r_nodes:
                    rid = str(r.get("id"))
                    if rid in existing_comments:
                        continue
                    comment_rows.append({
                        "repo":         repo_full_name,
                        "created_at":   r.get("submittedAt"),
                        "author_id":    (r.get("author") or {}).get("id"),
                        "author_name":  (r.get("author") or {}).get("name"),
                        "author_login": (r.get("author") or {}).get("login"),
                        "author_email": (r.get("author") or {}).get("email"),
                        "PR_id":        pr_number,
                        "comment_id":   rid,
                        "event":        "review",
                    })
                    existing_comments.add(rid)
                if r_conn["pageInfo"]["hasNextPage"]:
                    after_r = r_conn["pageInfo"]["endCursor"]
                else:
                    break
            # one PR fully collected (matches serial pb_pr.update(1) behaviour)
            _pbar = _scheduler.pbars.get("prs")
            if _pbar:
                _pbar.update(1)

        _scheduler.submit_page(_stream, _page_idx,
                               {"prs_repo": pr_rows, "prs_comments": comment_rows})
        return
    # ── END PARALLEL MODE ──────────────────────────────────────────────────

    # ---------- set pbar.total from totalCount (once) ----------
    repo     = g.get_repo(repo_full_name)
    prs_pl = repo.get_pulls(state="all")
    order["state"]["total"] = int(prs_pl.totalCount)
    util.save_json(order["state"], data_cursor_path, type= "prs_with_comments")
    total_prs     = repo.get_pulls(state="all").totalCount
    pb_pr     = tqdm(total=total_prs,     desc="PRs    ", position=1, leave=True)
    #PBAR done]
    pb_pr.update(order["state"]["processed"])

    per_page = cfg.items_per_page

    while True:
        vars_page = {"owner": owner, "name": name, "first": per_page, "after": after}
        _, pr_data = requester.graphql_query(query=pr_query, variables=vars_page)

        pr_conn = pr_data.get("data", {}).get("repository", {}).get("pullRequests", {}) or {}
        nodes   = pr_conn.get("nodes", []) or []
        page_info = pr_conn.get("pageInfo", {}) or {}



        # ---- write PR rows ----
        pr_rows = []
        for nd in nodes:
            author_id    = (nd.get("author") or {}).get("id")
            author_login = (nd.get("author") or {}).get("login")
            author_name  = (nd.get("author") or {}).get("name")
            author_email = (nd.get("author") or {}).get("email")

            pr_rows.append({
                "repo":        repo_full_name,
                "created_at":  nd.get("createdAt"),
                "author_id":  author_id,
                "author_name":  author_name,
                "author_login": author_login,
                "author_email": author_email,
                "PR_id":       nd.get("number"),
                "state":       nd.get("state"),
                "merged":      bool(nd.get("merged")),
                "closed_at":   nd.get("closedAt"),
                "merged_at":   nd.get("mergedAt"),
            })
        if pr_rows:
            util.append_rows_csv(prs_csv_path, pr_rows, sep=cfg.CSV_separator)

        # ---- for each PR, fully page comments and reviews ----
        comment_rows = []

        for nd in nodes:
            pr_number = int(nd.get("number"))

            # Comments
            after_c = None
            while True:
                vars_c = {"owner": owner, "name": name, "number": pr_number, "first": 100, "after": after_c}
                _, c_d = requester.graphql_query(query=pr_comments_query, variables=vars_c)
                c_conn = c_d["data"]["repository"]["pullRequest"]["comments"]
                c_nodes = c_conn.get("nodes", []) or []

                for c in c_nodes:
                    cid = str(c.get("id"))
                    if cid in existing_comments:
                        continue
                    author_id    = (c.get("author") or {}).get("id")
                    author_login = (c.get("author") or {}).get("login")
                    author_name  = (c.get("author") or {}).get("name")
                    author_email = (c.get("author") or {}).get("email")
                    comment_rows.append({
                        "repo":        repo_full_name,
                        "created_at":  c.get("createdAt"),
                        "author_id":    author_id,
                        "author_name":  author_name,
                        "author_login": author_login,
                        "author_email": author_email,
                        "PR_id":       pr_number,
                        "comment_id":  cid,
                        "event":       "comment",
                    })
                    existing_comments.add(cid)

                if c_conn["pageInfo"]["hasNextPage"]:
                    after_c = c_conn["pageInfo"]["endCursor"]
                else:
                    break

            # Reviews
            after_r = None
            while True:
                vars_r = {"owner": owner, "name": name, "number": pr_number, "first": 100, "after": after_r}
                _, r_d = requester.graphql_query(query=pr_reviews_query, variables=vars_r)
                r_conn = r_d["data"]["repository"]["pullRequest"]["reviews"]
                r_nodes = r_conn.get("nodes", []) or []

                for r in r_nodes:
                    rid = str(r.get("id"))
                    if rid in existing_comments:
                        continue
                    comment_rows.append({
                        "repo":        repo_full_name,
                        "created_at":  r.get("submittedAt"),
                        "author_id":    (r.get("author") or {}).get("id"),
                        "author_name":  (r.get("author") or {}).get("name"),
                        "author_login": (r.get("author") or {}).get("login"),
                        "author_email": (r.get("author") or {}).get("email"),
                        "PR_id":       pr_number,
                        "comment_id":  rid,
                        "event":       "review",
                    })
                    existing_comments.add(rid)

                if r_conn["pageInfo"]["hasNextPage"]:
                    after_r = r_conn["pageInfo"]["endCursor"]
                else:
                    break
            pb_pr.update(1)

        if comment_rows:
            util.append_rows_csv(prs_comments_csv_path, comment_rows, sep=cfg.CSV_separator)

        # ---- advance durable cursor after writing everything on this page ----
        processed = len(nodes)
        order["state"]["processed"] += processed

        after = page_info.get("endCursor")
        order["state"]["after"] = after
        util.save_json(order["state"], data_cursor_path, type= "prs_with_comments")

        if not page_info.get("hasNextPage"):
            order["state"]["complete"] = True
            util.save_json(order["state"], data_cursor_path, type= "prs_with_comments")
            break
    
    #os.system('cls' if os.name=='nt' else 'clear')
    return

def collect_org_activity(organization: str, org_folder: str):
    """
    Lightweight org-wide commit sweep.

    For every non-archived, non-fork repo in `organization`, collect only
    (repo, author_login, created_at) using pure GraphQL — no REST per-commit calls.

    Output: <org_folder>/_org_activity.csv
    Cursor:  <org_folder>/_org_activity_cursor.json
    """
    ORG_ACTIVITY_CSV    = "_org_activity.csv"
    ORG_ACTIVITY_CURSOR = "_org_activity_cursor.json"
    ORG_COLS = ["repo", "author_login", "created_at"]

    # Minimal commit query — only what we need for "active elsewhere" detection
    ORG_COMMIT_QUERY = """
    query($owner:String!, $name:String!, $first:Int!, $after:String) {
      repository(owner:$owner, name:$name) {
        defaultBranchRef {
          target {
            ... on Commit {
              history(first:$first, after:$after) {
                pageInfo { hasNextPage endCursor }
                nodes {
                  committedDate
                  author {
                    user { login }
                    name
                    email
                  }
                }
              }
            }
          }
        }
      }
    }
    """

    out_dir = Path(org_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path    = out_dir / ORG_ACTIVITY_CSV
    cursor_path = out_dir / ORG_ACTIVITY_CURSOR

    util.ensure_csv(csv_path, columns=ORG_COLS, sep=cfg.CSV_separator)

    # Load or initialise cursor
    if cursor_path.exists():
        with open(cursor_path) as f:
            cursor_state = json.load(f)
    else:
        cursor_state = {}   # repo_full_name -> {"after": None, "complete": False}

    token = util.getSpisificToken(0)
    g = Github(token)
    g.per_page = cfg.items_per_page

    org = g.get_organization(organization)
    repos = [r for r in org.get_repos(type="sources") if not r.archived and not r.fork]

    print(f"[org_activity] {organization}: {len(repos)} repo(s) to sweep")

    for repo in repos:
        repo_full_name = repo.full_name
        state = cursor_state.setdefault(repo_full_name, {"after": None, "complete": False})

        if state.get("complete"):
            print(f"  [skip] {repo_full_name} already complete")
            continue

        owner, name = repo.owner.login, repo.name
        requester   = getattr(repo, "requester", None) or getattr(repo, "_requester", None)
        after       = state["after"]
        per_page    = min(cfg.items_per_page, 100)

        print(f"  [collect] {repo_full_name} …")
        page_count = 0

        while True:
            try:
                _, data = requester.graphql_query(
                    query=ORG_COMMIT_QUERY,
                    variables={"owner": owner, "name": name, "first": per_page, "after": after},
                )
            except Exception as e:
                print(f"    [warn] {repo_full_name} page failed: {e}")
                break

            dref = (data.get("data", {}) or {}).get("repository", {}).get("defaultBranchRef")
            if not dref or not (dref.get("target") or {}).get("history"):
                print(f"    [skip] {repo_full_name} — empty or no default branch")
                state["complete"] = True
                break

            hist      = dref["target"]["history"]
            nodes     = hist.get("nodes", []) or []
            page_info = hist["pageInfo"]

            rows = []
            for nd in nodes:
                author_block = nd.get("author") or {}
                login = (author_block.get("user") or {}).get("login")
                if not login:
                    # Fall back to name/email so we don't silently drop the activity
                    login = author_block.get("name") or author_block.get("email")
                if login:
                    rows.append({
                        "repo":         repo_full_name,
                        "author_login": login,
                        "created_at":   nd.get("committedDate"),
                    })

            if rows:
                util.append_rows_csv(csv_path, rows, sep=cfg.CSV_separator)

            page_count += 1
            after = page_info.get("endCursor")
            state["after"] = after

            if not page_info.get("hasNextPage"):
                state["complete"] = True
                break

        # Persist cursor after each repo so we can resume on interrupt
        with open(cursor_path, "w") as f:
            json.dump(cursor_state, f, indent=2)

        print(f"    done — {page_count} page(s)")

    print(f"[org_activity] {organization} sweep complete → {csv_path}")


_STREAM_CSV_MAP = {
    "issues_with_timeline": [cfg.issue_list_file_name, cfg.issue_activity_file_name],
    "prs_with_comments":    [cfg.PR_list_file_name, cfg.prs_comments_csv],
    "commits":              [cfg.commit_list_file_name, cfg.per_file_commits_path],
}

def _reset_repo_streams(org_folder, stream_names):
    """Full-reset specified streams for one repo: clear cursors, scheduler pages, and CSV files."""
    cursor_path    = Path(org_folder, cfg.data_cursor)
    scheduler_path = Path(org_folder, "scheduler_state.json")

    if cursor_path.exists():
        with open(cursor_path, "r") as f:
            states = json.load(f)
        for sname in stream_names:
            if sname in states.get("streams", {}):
                states["streams"][sname] = {
                    "after": None, "processed": 0, "total": None, "complete": False
                }
        with open(cursor_path, "w") as f:
            json.dump(states, f, indent=2)

    if scheduler_path.exists():
        with open(scheduler_path, "r") as f:
            sched = json.load(f)
        for sname in stream_names:
            if sname in sched.get("streams", {}):
                sched["streams"][sname] = {
                    "cursors": [], "total_count": None,
                    "committed_through": -1, "page_statuses": {}, "complete": False,
                }
        with open(scheduler_path, "w") as f:
            json.dump(sched, f, indent=2)

    for sname in stream_names:
        for csv_name in _STREAM_CSV_MAP.get(sname, []):
            csv_path = Path(org_folder, csv_name)
            if csv_path.exists():
                csv_path.unlink()
                print(f"    [reset] deleted {csv_path.name}")


### MAIN FUNCTION
def main(gitRepoName, collect_org=True):
    ### get the org and repo
    splitRepoName = gitRepoName.split('/')
    organization = splitRepoName[0]
    repo = splitRepoName[1]

    #Make the MAIN organization folder if not exists
    organizationsFolder = cfg.main_folder
    os.makedirs(organizationsFolder, exist_ok=True)

    #Make the new organization folder if not exists
    organizationFolder = os.path.join(organizationsFolder, organization)
    os.makedirs(organizationFolder, exist_ok=True)

    organizationFolder = os.path.join(organizationFolder, repo)
    os.makedirs(organizationFolder, exist_ok=True)

    # collect all user data from this repo
    updateAlldata(organizationFolder, organization, repo, full_extraction=True)

    # Org-wide lightweight sweep: (repo, author_login, created_at) from every repo in the org.
    # Used to detect "developer was active elsewhere" for Mismatch 4 fix.
    if collect_org:
        org_folder = os.path.join(cfg.main_folder, organization)
        collect_org_activity(organization, org_folder)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GitHub data collection")
    parser.add_argument(
        "--reset-stream", action="append", dest="reset_streams",
        choices=list(_STREAM_CSV_MAP.keys()), metavar="STREAM",
        help="Full-reset one stream before collecting (can be repeated). "
             "Choices: issues_with_timeline, prs_with_comments, commits",
    )
    parser.add_argument(
        "--reset-all", action="store_true",
        help="Full-reset all three streams before collecting",
    )
    args = parser.parse_args()

    streams_to_reset = list(_STREAM_CSV_MAP.keys()) if args.reset_all else (args.reset_streams or [])

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_FOLDER)

    with open(cfg.repos_file) as f:
        repo_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    repo_names = [
        l.replace('https://github.com/', '').strip()
        for l in repo_lines
        if '/' in l.replace('https://github.com/', '').strip()
    ]

    # Apply resets before collection starts
    if streams_to_reset:
        print(f"[reset] Resetting streams: {streams_to_reset}")
        for gitRepoName in repo_names:
            parts = gitRepoName.split('/')
            org_folder = os.path.join(cfg.main_folder, parts[0], parts[1])
            if os.path.isdir(org_folder):
                print(f"  [reset] {gitRepoName}")
                _reset_repo_streams(org_folder, streams_to_reset)
        print("[reset] Done — starting collection.\n")

    for gitRepoName in repo_names:
        #os.system('cls' if os.name=='nt' else 'clear')
        tqdm.write('Running Commit Extraction for {}'.format(gitRepoName))
        try:
            main(gitRepoName)
            tqdm.write('Commit Extraction for {} Completed'.format(gitRepoName))
        except Exception as e:
            tqdm.write(f'[ERROR] {gitRepoName} failed: {type(e).__name__}: {e}')
            tqdm.write(f'[ERROR] Skipping {gitRepoName} and continuing to next repo.')
    tqdm.write('Done.')