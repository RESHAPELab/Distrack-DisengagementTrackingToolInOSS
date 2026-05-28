#   conda activate CS485
#   python CommitExtractor.py

### IMPORT EXCEPTION MODULES
import uuid
from requests.exceptions import Timeout
from github import GithubException, UnknownObjectException, IncompletableObject

### IMPORT SYSTEM MODULES
from github import Github
import os, logging, pandas, csv, tempfile, shutil, functools
from datetime import datetime, timezone
from tqdm import tqdm, TqdmWarning, tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal, threading
import json
from typing import Optional
from contextlib import contextmanager
from dateutil import tz as _tz
from collections import Counter

### IMPORT CUSTOM MODULES
import sys
sys.path.append('../')
import Settings as cfg
import Utilities as util
from pathlib import Path
import subprocess, tempfile, shutil, logging
from truckfactor.compute import main as compute_tf
import portalocker              # pip install portalocker
import warnings
warnings.filterwarnings("ignore")
from git import Repo, exc as git_exc

### DEFINE CONSTANTS
COMPLETE = "COMPLETE"
STOP_EVENT = threading.Event()

def _signal_handler(sig, frame):
    print("\n⏹️  Stop requested – finishing current API call …", flush=True)
    STOP_EVENT.set()
    # re-raise the usual KeyboardInterrupt in the main thread
    signal.default_int_handler(sig, frame)

def getExtractionStatus(folder, statusFile):
    status = "NOT-STARTED"
    if(statusFile in os.listdir(folder)):
        with open(os.path.join(folder, statusFile)) as f:
            content = f.readline().strip()
        status = content.split(';')
    return status

def runALLExtractionRoutine(organizationFolder, organization, project, extraction_type= True):
    
    workingFolder = (os.path.join(organizationFolder, project))
    os.makedirs(workingFolder, exist_ok=True)

    #one of the main Repo and one for the side repos.
    #we only collect commtis from non main repo
    if extraction_type == True:
        work_orders = [
            {"kind": "Issue", "token_idx": 0},
            {"kind": "PR"   , "token_idx": 1},
            {"kind": "Commit", "token_idx": 2},
            {"kind": "Commit", "token_idx": 3}
        ]
    else:
        work_orders = [
            {"kind": "Commit", "token_idx": 0},
            {"kind": "Commit", "token_idx": 1},
            {"kind": "Commit", "token_idx": 2},
            {"kind": "Commit", "token_idx": 3},
        ]

    #we need to chekc if all of the extracotrs are done 
    #Change the check to require all three per‑stream files are COMPLETE before skipping.
    #if all three are done then return
    #if one of them is Incompleat then run everything
    count = 0
    for order in work_orders:
        statusFile = f"_{order['kind']}_extractionStatus.tmp"
        if getExtractionStatus(workingFolder, statusFile) == "COMPLETE":
            count += 1
    if count == len(work_orders):
        return

    g0       = Github(util.getSpisificToken(0))
    repo     = g0.get_repo(f"{organization}/{project}")
    project_start_dt = repo.created_at
    collection_day   = datetime.strptime(cfg.data_collection_date, "%Y-%m-%d")

    total_commits = repo.get_commits(since=project_start_dt, until=collection_day).totalCount
    
    total_prs     = repo.get_pulls(state="all").totalCount 
        
    query = """
    query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
        issues(states:[OPEN, CLOSED])   { totalCount }
    }
    }
    """
    vars = {"owner": repo.owner.login, "name": repo.name}
    data = repo.requester.graphql_query(query=query, variables=vars)[1]
    total_issues: int = data["data"]["repository"]["issues"]["totalCount"]

    if Path(workingFolder, cfg.last_page_commits).exists() and Path(workingFolder, cfg.last_page).exists():
        total_commits = total_commits - (util.getLastPageRead(Path(workingFolder, cfg.last_page_commits))*cfg.items_per_page)
        total_prs = total_prs - (util.getLastPageRead(Path(workingFolder, cfg.last_page))*cfg.items_per_page)

    pb_commit = tqdm(total=total_commits, desc="Commits", position=0, leave=True)
    pb_pr     = tqdm(total=total_prs,     desc="PRs    ", position=1, leave=True)
    pb_issue  = tqdm(total=total_issues,  desc="Issues ", position=2, leave=True)


    bars = {"Commit": pb_commit, "Issue": pb_issue,  "PR": pb_pr}


    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
        pool.submit(extraction_worker, order, organizationFolder, organization, project,
                    project_start_dt, collection_day, bars[order["kind"]] ): order
        for order in work_orders
        }
        
        try:
            for fut in as_completed(futures):
                fut.result()
        except KeyboardInterrupt:
            STOP_EVENT.set()
            for f in futures:
                f.cancel()
            raise
        except Exception:
            # non-Ctrl+C failure in any worker → stop everyone
            STOP_EVENT.set()
            for f in futures:
                f.cancel()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    
    return

def extraction_worker(order, org_folder, org, project,
                      project_start_dt, collection_day, pbar):

    token = util.getSpisificToken(order["token_idx"])

    g = Github(token)
    g.per_page = cfg.items_per_page
    repo_name = f"{org}/{project}"
    working   = os.path.join(org_folder, project)

    if order["kind"] == "Issue":
        return updateIssueListFile(
            g, token, repo_full_name=repo_name,
            working_folder=working, pbar=pbar
        )

    if order["kind"] == "PR":
        return updatePRListFile(
            g, token, repo_name, project_start_dt, collection_day,
            working, pbar
        )

    if order["kind"] == "Commit":
        return updateCommitListFile(
            g, token, repo_name, project_start_dt, collection_day,
            working, pbar
        )

    return None

def updateCommitListFile(g, token, repoName, start_date, end_date, workingFolder, pbar, position = 0):
    
    commits_csv        = cfg.commit_list_file_name
    next_page           = cfg.next_page_commits
    last_page_save      = cfg.last_page_commits
    excl_tmp           = "_excludedNoneTypeCommits.tmp"
    status_tmp         = "Commits_extractionStatus.tmp"

    commit_cols = ["repo", "created_at", "created_by","author_name","author_email",  "committer_id",
                   "sha", "filename_list",
                   "fileschanged_count", "additions_sum", "deletions_sum"]


    status = getExtractionStatus(workingFolder, status_tmp)
    if status == COMPLETE:
        return g, token

    os.makedirs(workingFolder, exist_ok=True)
    with open(Path(workingFolder, status_tmp), "w") as fh:
        fh.write(f"INCOMPLETE;{datetime.today():%Y-%m-%d %H:%M:%S}")

    commits_df      = (pandas.read_csv(Path(workingFolder, commits_csv),
                                   sep=cfg.CSV_separator)
                       if Path(workingFolder, commits_csv).exists()
                       else pandas.DataFrame(columns=commit_cols))
    
    excluded        = (pandas.read_csv(Path(workingFolder, excl_tmp),
                                   sep=cfg.CSV_separator)
                       if Path(workingFolder, excl_tmp).exists()
                       else pandas.DataFrame(columns=["sha"]))
    def mark_excluded(sha: str):
        """Add sha to excluded df + set (idempotent)."""
        if sha and sha not in processed_excluded:
            util.add(excluded, [sha])
            processed_excluded.add(sha)

    processed_commits   = set(commits_df.sha)
    processed_excluded  = set(excluded.sha)

    g, token, search_limit ,   reset_time = util.getSameToken(g, token, position)
    repo     = g.get_repo(repoName)
    commits_pl = repo.get_commits(since=start_date, until=end_date)

    last_page   = int(commits_pl.totalCount / cfg.items_per_page)
    last_page   = max(last_page, 0)
    start_page  = (util.getLastPageRead(Path(workingFolder, cfg.last_page_commits))
                   if Path(workingFolder, cfg.last_page_commits).exists()
                   else 0)
    total_pages = last_page - start_page + 1
    current_page = start_page

    # ── main extraction loop with retry wrapper ─────────────────────────
    exception_thrown = True
    count = 0
    commit_count = 0

    while exception_thrown:
        exception_thrown = False

        try:
            #while current page is not larger than the last page
            while current_page <= last_page and not STOP_EVENT.is_set():

                current_page = updatePageFile(workingFolder, next_page)
                # progress bar / rate-limit housekeeping

                if STOP_EVENT.is_set():
                    break
                
                page_commits = commits_pl.get_page(current_page)   # <-- new var

                g, token, search_limit, reset_time = util.getSameToken(g, token, position)
                
                add_sum = del_sum = 0

                # ── iterate PRs on this page ────────
                for commit in page_commits:
                    if STOP_EVENT.is_set():
                        break
                    pbar.update(1)
                    commit_count += 1
                    sha = commit.sha
                    if sha is None:
                        tqdm.write(f"Commit has no sha or name, skipping.")
                        continue
                    if sha in processed_commits:
                        tqdm.write(f"Commit {sha} already processed, skipping.") 
                        continue
                    if sha in processed_excluded:
                        tqdm.write(f"Commit {sha} already excluded, skipping.")
                        continue

                    # handle ghost / deleted users
                    author_id    = (commit.author.login
                                    if commit.author else "NA")
                    author_name = (commit.author.name
                                    if commit.author else "NA")
                    committer_id = (commit.committer.login
                                    if commit.committer else "NA")
                    author_email = commit.commit.author.email


                    if author_email is None and author_name is None and author_id is None:
                        util.add(excluded, [sha])
                        processed_excluded.add(sha)
                        tqdm.write(f"Commit {sha} has no author, skipping.")
                        continue

                    g, token, search_limit ,   reset_time = util.getSameToken(g, token, position)

                    commit_created_at = commit.commit.author.date
                    
                    files             = commit.files
                    filenames         = [f.filename for f in files]
                    fchg_cnt          = len(filenames)
                    adds              = sum(f.additions for f in files)
                    dels              = sum(f.deletions for f in files)

                    add_sum += adds
                    del_sum += dels

                    new_commit = [repo, commit_created_at, author_id, author_name, author_email,
                         committer_id, sha, "|".join(filenames), fchg_cnt, adds, dels]
                    util.add(
                        commits_df, new_commit
                    )

                    #pbar.set_postfix({
                    #    "commit_count": commit_count,
                    #    "page": f"{current_page}/{last_page}",
                    #    "search_limit": search_limit,
                    #    "user": author_id,
                    #    "reset_time": reset_time.strftime("%H:%M:%S"),
                    #    "repo": repoName.split("/")[-1]}, refresh=True)
                    
                    processed_commits.add(sha)

                _flush_data_to_csv(commits_df = commits_df, prs_df = None, prs_comments_df = None, excluded = excluded, workingFolder = workingFolder, page = current_page, save_last_page = last_page_save, save_current_page = next_page, is_page_finished = True)
                
                time.sleep(0.1)  # avoid hitting the API too hard
        except KeyboardInterrupt:
            # ── 1.  user pressed Ctrl-C  ────────────────────────────────
            logging.warning("Extraction interrupted by user – saving progress and exiting.")
            _flush_data_to_csv(commits_df = commits_df, prs_df = None, prs_comments_df = None, excluded = excluded, workingFolder = workingFolder, page = current_page, save_last_page = last_page_save, save_current_page = next_page, is_page_finished = False)
            
            raise  

        except (UnknownObjectException, GithubException, AttributeError) as e:
            count += 1
            logging.warning(f"Extraction interrupted: {e}")

            # ── 2.  rate-limit or abuse-limit  ──────────────────────────
            if isinstance(e, GithubException) and e.status in (403, 429):
                delay = 300
                logging.warning(f"Rate-limited – sleeping {delay}s")
                time.sleep(delay)
                continue

            # ── 3.  forbidden but not back-off related  ─────────────────
            if isinstance(e, GithubException) and e.status == 403:
                mark_excluded(sha)

            # ── 4.  write whatever we have so far  ──────────────────────
            _flush_data_to_csv(commits_df = commits_df, prs_df = None, prs_comments_df = None, excluded = excluded, workingFolder = workingFolder, page = current_page, save_last_page = last_page_save, save_current_page = next_page, is_page_finished = False)
            exception_thrown = True

        except Exception as e:
            logging.warning(f"Unhandled exception: {e}")
            _flush_data_to_csv(commits_df = commits_df, prs_df = None, prs_comments_df = None, excluded = excluded, workingFolder = workingFolder, page = current_page, save_last_page = last_page_save, save_current_page = next_page, is_page_finished = False)
            raise

    # ── final flush / bookkeeping --------------------------------------
    _flush_data_to_csv(commits_df = commits_df, prs_df = None, prs_comments_df = None, excluded = excluded, workingFolder = workingFolder, page = current_page, save_last_page = last_page_save, save_current_page = next_page, is_page_finished = False)


    if len(excluded):
        excluded.to_csv(Path(workingFolder, excl_tmp),
                        sep=cfg.CSV_separator, index=False,
                        lineterminator="\n")

    with open(Path(workingFolder, status_tmp), "w") as fh:
        fh.write(f"COMPLETE;{cfg.data_collection_date}")

    logging.info("Commit + PR extraction COMPLETE for %s", repoName)
    return g, token

def updatePRListFile(g, token, repoName, start_date, end_date, workingFolder, pbar, position = 1):
    prs_csv            = cfg.PR_list_file_name
    prs_comments_csv   = cfg.prs_comments_csv
    next_page           = cfg.next_page
    last_page_save          = cfg.last_page
    excl_tmp           = "_excludedNoneType.tmp"
    status_tmp         = "PR_extractionStatus.tmp"

    pr_cols     = ["repo", "created_at", "created_by", "PR_id",
                   "state", "merged", "closed_at", "merged_at"]
                
    prcom_cols  = ["repo", "created_at", "created_by", "PR_id",
                   "comment_id", "event"]

    # ── early-exit if already complete ──────────────────────────────────
    status = getExtractionStatus(workingFolder, status_tmp)
    if status == COMPLETE:
        return g, token

    os.makedirs(workingFolder, exist_ok=True)
    with open(Path(workingFolder, status_tmp), "w") as fh:
        fh.write(f"INCOMPLETE;{datetime.today():%Y-%m-%d %H:%M:%S}")

    # ── load existing CSVs (or empty frames) ────────────────────────────
    
    prs_df          = (pandas.read_csv(Path(workingFolder, prs_csv),
                                   sep=cfg.CSV_separator)
                       if Path(workingFolder, prs_csv).exists()
                       else pandas.DataFrame(columns=pr_cols))

    prs_comments_df = (pandas.read_csv(Path(workingFolder, prs_comments_csv),
                                   sep=cfg.CSV_separator)
                       if Path(workingFolder, prs_comments_csv).exists()
                       else pandas.DataFrame(columns=prcom_cols))

    excluded        = (pandas.read_csv(Path(workingFolder, excl_tmp),
                                   sep=cfg.CSV_separator)
                       if Path(workingFolder, excl_tmp).exists()
                       else pandas.DataFrame(columns=["sha"]))
    def mark_excluded(sha: str):
        """Add sha to excluded df + set (idempotent)."""
        if sha and sha not in processed_excluded:
            util.add(excluded, [sha])
            processed_excluded.add(sha)
    # ── reusable sets for “skip-if-seen” ────────────────────────────────
    processed_comments  = set(prs_comments_df.comment_id)
    processed_excluded  = set(excluded.sha)

    # ── GitHub paging setup ─────────────────────────────────────────────
    g, token, search_limit ,   reset_time = util.getSameToken(g, token, position)
    repo     = g.get_repo(repoName)
    pulls = repo.get_pulls(state="all", sort="created", direction="desc")
    

    last_page   = int(pulls.totalCount / cfg.items_per_page)
    last_page   = max(last_page, 0)
    start_page  = (util.getLastPageRead(Path(workingFolder, cfg.last_page))
                   if Path(workingFolder, cfg.last_page).exists()
                   else 0)
    current_page = start_page

    # ── main extraction loop with retry wrapper ─────────────────────────
    page = start_page  # make sure it's defined for finally/_flush
    exception_thrown = True
    count = 0
    pr_count =0
    commit_count = 0
    comments_count = 0

    def handle_github_backoff(exc, attempt=0, base_delay=60, cap=900):
            """
            Decide how long to sleep after a GithubException.
            Returns the delay in seconds (may be 0).
            """
            hdrs = getattr(exc, "headers", {}) or {}

            # 1. Retry-After wins
            if "Retry-After" in hdrs:
                return int(hdrs["Retry-After"])

            # 2. Primary rate-limit exhausted
            if hdrs.get("X-RateLimit-Remaining") == "0":
                reset = int(hdrs.get("X-RateLimit-Reset", 0))
                return max(reset - int(time.time()), 1)

            # 3. Secondary limit: exponential back-off
            return min(base_delay * (2 ** attempt), cap)
    
    while exception_thrown:
        exception_thrown = False
        try:
            #while current page is not larger than the last page
            while current_page <= last_page and not STOP_EVENT.is_set():

                current_page = updatePageFile(workingFolder, next_page)
                if STOP_EVENT.is_set():
                    break
                # progress bar / rate-limit housekeeping
                g, token, search_limit, reset_time = util.getSameToken(g, token, position)

                pulls_page = pulls.get_page(current_page)

                # ── iterate PRs on this page ───────────────────────────
                
                for pr in pulls_page:
                    if STOP_EVENT.is_set():
                        break
                    
                    pbar.update(1)
                    g, token, search_limit ,   reset_time = util.getSameToken(g, token, position)
                    pr_count += 1
                    
                    pr_id      = pr.number
                    repo_name  = repo.full_name

                    # PR meta row -------------------------------------------------
                    util.add(
                        prs_df,
                        [repo_name, pr.created_at, pr.user.login
                         if pr.user else None,
                         pr_id, pr.state, bool(pr.merged),
                         pr.closed_at, pr.merged_at]
                    )
                
                    # PR comments -------------------------------------------------
                    for cmt in pr.get_comments():
                        if cmt.id in processed_comments:
                            continue
                        comments_count += 1
                        util.add(
                            prs_comments_df,
                            [repo_name, cmt.created_at,
                             cmt.user.login if cmt.user else None,
                             pr_id, cmt.id, "comment"]
                        )
                        processed_comments.add(cmt.id)

                    # PR reviews --------------------------------------------------
                    for rvw in pr.get_reviews():
                        if rvw.id in processed_comments:
                            continue
                        comments_count += 1
                        util.add(
                            prs_comments_df,
                            [repo_name, rvw.submitted_at,
                             rvw.user.login if rvw.user else None,
                             pr_id, rvw.id, "review"]
                        )
                        processed_comments.add(rvw.id)
                    if STOP_EVENT.is_set():
                        break 
                _flush_data_to_csv( commits_df= None, prs_df = prs_df, prs_comments_df = prs_comments_df,
                            excluded= excluded, workingFolder= workingFolder,  save_last_page = last_page_save , save_current_page = next_page , page= current_page, is_page_finished = True)
                
                time.sleep(0.1)  # avoid hitting the API too hard
        except KeyboardInterrupt:
            # ── 1.  user pressed Ctrl-C  ────────────────────────────────
            logging.warning("Extraction interrupted by user – saving progress and exiting.")
            _flush_data_to_csv( commits_df= None, prs_df = prs_df, prs_comments_df = prs_comments_df,
                            excluded= excluded, workingFolder= workingFolder,  save_last_page = last_page_save , save_current_page = next_page , page= current_page, is_page_finished = False)
            
            raise  

        except (UnknownObjectException, GithubException, AttributeError) as e:
            count += 1
            logging.warning(f"Extraction interrupted: {e}")

            # ── 2.  rate-limit or abuse-limit  ──────────────────────────
            if isinstance(e, GithubException) and e.status in (403, 429):
                delay = handle_github_backoff(e, attempt=count)
                logging.warning(f"Rate-limited – sleeping {delay}s")
                time.sleep(delay)
                continue

            # ── 3.  forbidden but not back-off related  ─────────────────
            if isinstance(e, GithubException) and e.status == 403:
                mark_excluded(pr_id)

            # ── 4.  write whatever we have so far  ──────────────────────
            _flush_data_to_csv( commits_df= None, prs_df = prs_df, prs_comments_df = prs_comments_df,
                            excluded= excluded, workingFolder= workingFolder,  save_last_page = last_page_save , save_current_page = next_page , page= current_page, is_page_finished = False)
            exception_thrown = True

        except Exception as e:
            logging.warning(f"Unhandled exception: {e}")
            _flush_data_to_csv( commits_df= None, prs_df = prs_df, prs_comments_df = prs_comments_df,
                            excluded= excluded, workingFolder= workingFolder,  save_last_page = last_page_save , save_current_page = next_page , page= current_page, is_page_finished = False)            
            raise
        


    # ── final flush / bookkeeping --------------------------------------
    _flush_data_to_csv( commits_df= None, prs_df = prs_df, prs_comments_df = prs_comments_df,
                            excluded= excluded, workingFolder= workingFolder,  save_last_page = last_page_save , save_current_page = next_page , page= current_page, is_page_finished = False)
    if len(excluded):
        excluded.to_csv(Path(workingFolder, excl_tmp),
                        sep=cfg.CSV_separator, index=False,
                        lineterminator="\n")

    with open(Path(workingFolder, status_tmp), "w") as fh:
        fh.write(f"COMPLETE;{cfg.data_collection_date}")

    logging.info("Commit + PR extraction COMPLETE for %s", repoName)
    return g, token

def _load_state(state_path: Path):
    if not state_path.exists():
        return {"next_cursor": None, "complete": False}
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return {"next_cursor": None, "complete": False}
    # sanitize legacy fields
    st.pop("inflight", None)
    st.pop("last_done_cursor", None)
    st.setdefault("next_cursor", None)
    st.setdefault("complete", False)
    return st

def _save_state_atomic(state_path: Path, state: dict):
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(state_path)

def _read_existing_ids(csv_path: Path, colname: str, sep: str):
    if not csv_path.exists():
        return set()
    try:
        import pandas as pd
        s = pd.read_csv(csv_path, usecols=[colname], sep=sep)
        return set(s[colname].dropna().astype(str))
    except Exception:
        seen = set()
        with csv_path.open(newline="", encoding="utf-8") as fh:
            r = csv.DictReader(fh, delimiter=sep)
            for row in r:
                v = str(row.get(colname, "")).strip()
                if v:
                    seen.add(v)
        return seen

def _append_rows(csv_path: Path, rows: list, sep: str):
    if not rows:
        return
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=sep)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)

def updateIssueListFile(
    g, token,                      # token unused but kept for signature compatibility
    repo_full_name: str,
    working_folder: str,
    *,
    issues_csv_name: str = "issues.csv",
    activity_csv_name: str = "issue_activity.csv",
    csv_sep: str = ",",
    state_name: str = "_issues_cursor_state.json",
    pbar=None,
):
    """
    Single-worker durable-cursor extractor:
    - Writes issues metadata to `issues.csv` (1 row/issue)
    - Writes ALL activity traces (events + comments + cross-refs) to `issue_activity.csv` from the Timeline API
      Columns: repo, issue_number, activity_id, item_type, event, body, created_at, actor
    """

    out_dir = Path(working_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / state_name
    issues_csv_path = out_dir / issues_csv_name
    activity_csv_path = out_dir / activity_csv_name

    # load/clean state
    st = _load_state(state_path)
    if st.get("complete") is True:
        return

    # de-dupe sets across re-runs
    seen_issue_numbers = _read_existing_ids(issues_csv_path, "issue_number", csv_sep)
    seen_activity_ids  = _read_existing_ids(activity_csv_path, "activity_id", csv_sep)

    repo = g.get_repo(repo_full_name)

    page_query = """
    query($owner:String!, $name:String!, $after:String) {
      repository(owner:$owner, name:$name) {
        issues(
          first: 100,
          orderBy: { field: CREATED_AT, direction: DESC },
          states: [OPEN, CLOSED],
          after: $after
        ) {
          pageInfo { hasNextPage endCursor }
          nodes {
            number
            createdAt
            title
            state
            closedAt
            author { login }
            labels(first: 20)    { nodes { name } }
            assignees(first: 20) { nodes { login } }
            milestone { title }
          }
        }
      }
    }"""

    # optional: set pbar total
    try:
        totalCount_query = """
        query($owner:String!, $name:String!) {
          repository(owner:$owner, name:$name) {
            issues(states:[OPEN, CLOSED]) { totalCount }
          }
        }"""
        owner, name = repo.owner.login, repo.name
        total_data = repo.requester.graphql_query(query=totalCount_query, variables={"owner": owner, "name": name})[1]
        total_issues = total_data["data"]["repository"]["issues"]["totalCount"]
        if hasattr(pbar, "total") and (pbar.total is None or pbar.total == 0):
            pbar.reset(total=total_issues)
    except Exception:
        pass

    after = st.get("next_cursor")  # None or a cursor string

    while True:
        # fetch a page
        try:
            owner, name = repo.owner.login, repo.name
            payload = {"owner": owner, "name": name, "after": after}
            data = repo.requester.graphql_query(query=page_query, variables=payload)[1]
        except Exception:
            time.sleep(5)
            continue

        issues_block = data["data"]["repository"]["issues"]
        nodes      = issues_block.get("nodes", []) or []
        has_next   = issues_block["pageInfo"]["hasNextPage"]
        end_cursor = issues_block["pageInfo"]["endCursor"]

        # stale cursor -> reset and retry
        if after is not None and len(nodes) == 0:
            after = None
            st["next_cursor"] = None
            _save_state_atomic(state_path, st)
            continue

        # per-page accumulators
        issue_rows = []
        activity_rows = []

        for nd in nodes:
            if pbar is not None:
                pbar.update(1)

            try:
                num = int(nd["number"])
            except Exception:
                continue

            num_s = str(num)

            # ---- issues.csv row (issue metadata; 1 row/issue across all runs)
            if num_s not in seen_issue_numbers:
                labels    = ",".join([x["name"] for x in (nd.get("labels", {}) or {}).get("nodes", [])])
                assignees = ",".join([x["login"] for x in (nd.get("assignees", {}) or {}).get("nodes", [])])
                milestone = (nd.get("milestone") or {}).get("title") or ""

                issue_rows.append({
                    "repo": repo_full_name,
                    "created_at": nd.get("createdAt"),
                    "created_by": (nd.get("author") or {}).get("login") or "",
                    "issue_number": num,
                    "title": nd.get("title") or "",
                    "state": nd.get("state") or "",
                    "closed_at": nd.get("closedAt"),
                    "labels": labels,
                    "assignees": assignees,
                    "milestone": milestone,
                })
                seen_issue_numbers.add(num_s)

            # ---- activity: single stream from Timeline (events + comments + cross-refs)
            try:
                issue_obj = repo.get_issue(number=num)
            except Exception:
                continue

            try:
                for item in issue_obj.get_timeline():  # PyGithub handles pagination internally
                    # robust attribute access
                    aid = getattr(item, "id", None)
                    if not aid:
                        continue
                    aid = str(aid)
                    if aid in seen_activity_ids:
                        continue

                    # classify
                    item_type = item.__class__.__name__
                    # events usually have .event; comments have .body and .user
                    event_name = getattr(item, "event", "") or ""
                    body = getattr(item, "body", None) or ""

                    # actor can be .actor or .user depending on item kind
                    actor_obj = getattr(item, "actor", None) or getattr(item, "user", None)
                    actor = getattr(actor_obj, "login", "") if actor_obj else ""

                    # created time can vary (created_at / submitted_at)
                    created = getattr(item, "created_at", None) or getattr(item, "submitted_at", None)

                    activity_rows.append({
                        "repo": repo_full_name,
                        "issue_number": num,
                        "activity_id": aid,
                        "item_type": item_type,     # e.g., IssueComment, LabeledEvent, ClosedEvent, CrossReferencedEvent
                        "event": event_name,        # empty for comments
                        "body": body,               # empty for non-comment events
                        "created_at": created,
                        "created_by": actor,
                    })
                    seen_activity_ids.add(aid)
            except Exception:
                # If timeline preview isn’t enabled in your PyGithub version or GitHub returns 415,
                # you can fall back to issue_obj.get_comments() and issue_obj.get_events() here.
                pass

        # flush this page
        _append_rows(issues_csv_path, issue_rows, csv_sep)
        _append_rows(activity_csv_path, activity_rows, csv_sep)

        # advance cursor AFTER successful flush
        after = end_cursor
        st["next_cursor"] = after
        st["complete"] = (not has_next)
        _save_state_atomic(state_path, st)

        if not has_next:
            break

def _safe_append(df, path):
    if df is None or df.empty:
        return
    path = Path(path)
    with path.open("a", encoding="utf-8") as fh:
        portalocker.lock(fh, portalocker.LOCK_EX)   # 2 block other writers
        df.to_csv(fh,
                  header=fh.tell() == 0,            # 1 write header only if file was empty
                  sep=cfg.CSV_separator,
                  index=False,
                  lineterminator="\n")
        portalocker.unlock(fh)

def _flush_data_to_csv(commits_df, prs_df, prs_comments_df, excluded, workingFolder, save_last_page, save_current_page, page, is_page_finished = False):
    if commits_df is not None:
        _safe_append(commits_df,  Path(workingFolder, cfg.commit_list_file_name))
    if prs_df is not None:
        _safe_append(prs_df,      Path(workingFolder, cfg.PR_list_file_name))
    if prs_comments_df is not None:
        _safe_append(prs_comments_df,
                 Path(workingFolder, cfg.prs_comments_csv))
    
    _safe_append(excluded,
                 Path(workingFolder, "_excludedNoneType.tmp"))
    
    current_page = (util.getLastPageRead(Path(workingFolder, save_current_page))
                   if Path(workingFolder, save_current_page).exists()
                   else 0)
    
    if is_page_finished:
        with open(os.path.join(workingFolder, save_last_page), "w") as fh:
            fh.write(f"last_page_flushed:{page}")
    else:
        #do nothing if the current page is not larger than the last page
        with open(os.path.join(workingFolder, save_last_page), "w") as fh:
            fh.write(f"last_page_flushed:{current_page}")

def updatePageFile(workingFolder, next_page):

    current_page = (util.getLastPageRead(Path(workingFolder, next_page))
                   if Path(workingFolder, next_page).exists()
                   else 0)
    
    new_page = current_page + 1

    with open(os.path.join(workingFolder, next_page), "w") as fh:
        fh.write(f"Next Page:{new_page}")
    return current_page

def get_commit_based_core_devs(commits, threshold=0.8):
    """
    commits: List[dict] where each dict contains at least the 'author' key.
    Example: [{'author': 'alice'}, {'author': 'bob'}, {'author': 'alice'}, ...]

    Returns: List of core developers (author names) who together authored >= threshold of commits.
    """
    # Count commits per developer
    author_commit_counts = Counter(commit["author_id"] for commit in commits)

    # Sort developers by number of commits (descending)
    sorted_authors = author_commit_counts.most_common()

    total_commits = sum(author_commit_counts.values())
    cumulative = 0
    core_devs = []

    for author, count in sorted_authors:
        cumulative += count
        core_devs.append(author)
        if cumulative / total_commits >= threshold:
            break

    return core_devs

### MAIN FUNCTION
def main(gitRepoName):
    ### get the org and repo
    splitRepoName = gitRepoName.split('/')
    organization = splitRepoName[0]
    project = splitRepoName[1]

    #Make the MAIN organization folder if not exists
    organizationsFolder = cfg.main_folder
    os.makedirs(organizationsFolder, exist_ok=True)

    #Make the new organization folder if not exists
    organizationFolder = os.path.join(organizationsFolder, organization)
    os.makedirs(organizationFolder, exist_ok=True)


    # new need to collect all user data from this repo
    runALLExtractionRoutine(organizationFolder, organization, project, extraction_type= True)
    token = util.getSpisificToken(0)
    g0       = Github(token)
    org = g0.get_organization(organization)

    #get all repos in the organization
    org_repos = [r for r in org.get_repos(type='sources')
             if not r.archived and not r.fork]
    
    #this collects all commit data from other repos in the same organization
    repo_num = 0
    for repo in org_repos:
        project_name = repo.name
        if project_name != project:
            repo_num += 1
            print('Running Commit Extraction for {} ({}/{})'.format(project_name, repo_num, len(org_repos) - 1))
            runALLExtractionRoutine(organizationFolder, organization, project_name, False)

if __name__ == "__main__":
    #add an atribute of id when call the file
    if len(sys.argv) < 1:
        print("Usage: python CommitExtractor.py")
        sys.exit(1)

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_FOLDER)

    os.makedirs(cfg.logs_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
    logfile = cfg.logs_folder+f"/Commit_Extraction-{timestamp}.log"
    logging.basicConfig(filename=logfile, level=logging.INFO)
    
    repoUrls = '../' + cfg.repos_file
    with open(repoUrls) as f:
        repoUrls = f.readlines()
        for repoUrl in repoUrls:
            os.system('cls' if os.name=='nt' else 'clear')

            gitRepoName = repoUrl.replace('https://github.com/', '').strip()
            tqdm.write('Running Commit Extraction for {}'.format(gitRepoName))
            main(gitRepoName)
            tqdm.write('Commit Extraction for {} Completed'.format(gitRepoName))
        tqdm.write('Done.')