#cd C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Extractors
#streamlit run DemoUsersExperience.py
import streamlit as st
import pandas
import numpy as np
import os
import csv
import tempfile
import shutil
import logging
import re
import unicodedata
import sys
import subprocess
import time
import joblib
import matplotlib.pyplot as plt

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Tuple, List, Set, Dict, Optional, Literal
from collections import Counter
from pathlib import Path
from git import Repo, exc as git_exc
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, label_binarize

from dataclasses import dataclass
from github import Github, GithubException, UnknownObjectException, IncompletableObject

sys.path.append('../')
import Settings as cfg
import Utilities as util
from requests.exceptions import Timeout
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import threading
import portalocker
import warnings
from truckfactor.compute import main as compute_tf

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="missing ScriptRunContext!")


# ---------- CONFIG ----------
ARTIFACTS = Path("./artifacts")  # ./artifacts/<owner>/<repo>/
STATE_ORDER = ["ACTIVE", "NON_CODING", "INACTIVE", "GONE"]  #
MODEL_PATH = Path(r"C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\PredictionModel\model.joblib")
BREAK_STATES = {"NON_CODING", "INACTIVE"} 
FEATURE_COLS = [
    # TODO: put the exact columns your model expects (no leakage!)
    # e.g., "active_days_14", "nc_days_14", "streak_active", "days_since_any", ...
]
# ---------- UTIL ----------
ORG_BASE = Path(r"C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY") / "Organizations"
COMPLETE = "COMPLETE"
STOP_EVENT = threading.Event()

@st.cache_data(show_spinner=False)
def list_orgs(base: Path = ORG_BASE) -> list[str]:
    """All org folder names under Organizations/"""
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()], key=str.casefold)

@st.cache_data(show_spinner=True)
def list_repos_for(org: str, base: Path = ORG_BASE) -> list[str]:
    """All repo folder names under Organizations/<org>/"""
    root = base / org
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()], key=str.casefold)

@dataclass
class SplitConfig:
    strategy: str = "holdout_authors"   # "time_by_repo" | "time_global" | "holdout_authors"
    val_months: int = 6              # last N months per repo for validation
    holdout_fraction: float = 0.2    # for holdout_authors

#----------------------------------------
# Data Collection
#----------------------------------------

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
        status, _ = content.split(';')
    return status

def runALLExtractionRoutine(organizationFolder, organization, project,
                            extraction_type: bool = True,
                            since_days: int = 30):
    workingFolder = os.path.join(organizationFolder, project)
    os.makedirs(workingFolder, exist_ok=True)

    # Work orders (we’ll add window_start per order)
    if extraction_type:
        work_orders = [
            {"kind": "Issue",  "token_idx": 0},
            {"kind": "PR",     "token_idx": 1},
            {"kind": "Commit", "token_idx": 2},
            {"kind": "Commit", "token_idx": 3},
        ]
    else:
        work_orders = [
            {"kind": "Commit", "token_idx": 0},
            {"kind": "Commit", "token_idx": 1},
            {"kind": "Commit", "token_idx": 2},
            {"kind": "Commit", "token_idx": 3},
        ]

    # Skip if ALL streams previously marked COMPLETE
    done = 0
    for order in work_orders:
        statusFile = f"{order['kind']}_extractionStatus.tmp"
        if getExtractionStatus(workingFolder, statusFile) == "COMPLETE":
            done += 1
    if done == len(work_orders):
        return

    g0   = Github(util.getSpisificToken(0))
    repo = g0.get_repo(f"{organization}/{project}")

    project_start_dt = datetime.strptime(repo.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"), "%Y-%m-%dT%H:%M:%SZ")
    if getattr(project_start_dt, "tzinfo", None) is not None:
        project_start_dt = project_start_dt.replace(tzinfo=None)

    # I want to collect data from today back till a few months ago
    now = datetime.utcnow().replace(tzinfo=None)

    project_start_dt = repo.created_at
    if getattr(project_start_dt, "tzinfo", None) is not None:
        project_start_dt = project_start_dt.replace(tzinfo=None)

    end_date = now                                    # or parse cfg.data_collection_date and THEN compute start from it
    start_date = max(project_start_dt, end_date - timedelta(days=since_days))

    # Totals (for progress bars). These remain “global” counts; progress bars will just overprovision.
    total_commits = repo.get_commits(since=start_date, until=end_date).totalCount
    total_prs     = repo.get_pulls(state="all").totalCount
    query = """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) { issues(states:[OPEN, CLOSED]) { totalCount } }
    }"""
    vars = {"owner": repo.owner.login, "name": repo.name}
    data = repo.requester.graphql_query(query=query, variables=vars)[1]
    total_issues = data["data"]["repository"]["issues"]["totalCount"]

    pb_commit = tqdm(total=total_commits, desc="Commits", position=0, leave=True)
    pb_pr     = tqdm(total=total_prs,     desc="PRs    ", position=1, leave=True)
    pb_issue  = tqdm(total=total_issues,  desc="Issues ", position=2, leave=True)
    bars = {"Commit": pb_commit, "Issue": pb_issue, "PR": pb_pr}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                extraction_worker,
                {**order, "start_date": start_date, "end_date": end_date},
                organizationFolder, organization, project,
                bars[order["kind"]]
            ): order for order in work_orders
        }
        try:
            for fut in as_completed(futures):
                fut.result()
        except KeyboardInterrupt:
            STOP_EVENT.set()
            for f in futures: f.cancel()
            raise
        except Exception:
            STOP_EVENT.set()
            for f in futures: f.cancel()
            raise

def extraction_worker(order, org_folder, org, project, pbar):
    token = util.getSpisificToken(order["token_idx"])
    g = Github(token); g.per_page = cfg.items_per_page

    repo_name      = f"{org}/{project}"
    working        = os.path.join(org_folder, project)
    start_date   = order["start_date"]
    end_date = order["end_date"]

    if order["kind"] == "Issue":
        return updateIssueListFile(g, token, repo_name, start_date, end_date, working, pbar)
    if order["kind"] == "PR":
        return updatePRListFile(g, token, repo_name, start_date, end_date, working, pbar)
    if order["kind"] == "Commit":
        return updateCommitListFile(g, token, repo_name, start_date, end_date, working, pbar)

def updateCommitListFile(g, token, repoName, start_date, end_date, workingFolder, pbar, position = 0):
    
    commits_csv        = cfg.commit_list_file_name
    next_page           = cfg.next_page_commits
    last_page_save      = cfg.last_page_commits
    excl_tmp           = "_excludedNoneTypeCommits.tmp"
    status_tmp         = "Commits_extractionStatus.tmp"

    commit_cols = ["repo", "created_at", "author_id","author_name","author_email",  "committer_id",
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
                st.write(f"Processing page {current_page} of commits")
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
            st.write(f"Processin pr")

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

                    if pr.created_at.replace(tzinfo=None) < start_date:
                        # finalize current page flush and then break the outer loops
                        _flush_data_to_csv(commits_df=None, prs_df=prs_df, prs_comments_df=prs_comments_df,
                                        excluded=excluded, workingFolder=workingFolder,
                                        save_last_page=last_page_save, save_current_page=next_page,
                                        page=current_page, is_page_finished=True)
                        return g, token


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

def updateIssueListFile(
        g, token,
        repoName, start_date, end_date,
        workingFolder,
        pbar,
        position = 2):
    """Writes the list of the Issues for the given repository"""
    last_issue = 0  

    next_page           = cfg.next_page_issues
    last_page_save          = cfg.last_page_issues
    excl_tmp = "_excludedNoneType_Issues.tmp"
    status_tmp = "Issue_extractionStatus.tmp"

    cursor_tmp       = "_cursor_Issues.tmp"            # NEW: stores GraphQL endCursor


    #commit file names
    issues_csv = cfg.issue_list_file_name
    events_csv = cfg.issue_events_list_file_name
    issues_comments_csv = cfg.issue_comments_list_file_name
    timeline_csv = cfg.issue_timeline_file_name

    save_tmp    = "_saveFile_Issues.tmp"

    issues_cols = [          "repo", "created_at", "created_by", "issue_id", "title", "labels", "state", "body", "assignees", "milestone" ]
    issue_events_cols = [    "repo", "created_at", "created_by", "issue_id", "event_id", "event" ]
    issues_comments_cols = [ "repo", "created_at", "created_by", "issue_id", "comment_id", "body" ]
    timeline_cols = [        "repo", "created_at", "created_by", "issue_id", "event_id", "event" ] 
    

    status = getExtractionStatus(workingFolder, status_tmp)
    if status == COMPLETE:                     # already done
        return g, token

    os.makedirs(workingFolder, exist_ok=True)
    with open(os.path.join(workingFolder, status_tmp), "w") as fh:
        fh.write(f"INCOMPLETE;{datetime.today():%Y-%m-%d %H:%M:%S}")

    
    save_path = os.path.join(workingFolder, save_tmp)
    if os.path.exists(save_path):
        last_issue = int(open(save_path).readline().split(';')[0])

    excluded = pandas.read_csv(os.path.join(workingFolder, excl_tmp), sep=cfg.CSV_separator) if excl_tmp in os.listdir(workingFolder) else pandas.DataFrame(columns=["issue_id"])
    
    
    cursor_path = os.path.join(workingFolder, cursor_tmp)
    after_cursor = None
    if os.path.exists(cursor_path):
        c = open(cursor_path, "r", encoding="utf-8").read().strip()
        after_cursor = c if c else None

    def _ids_if_exists(fname, col):
        path = os.path.join(workingFolder, fname)
        if os.path.exists(path):
            try:
                return set(pandas.read_csv(path, usecols=[col], sep=cfg.CSV_separator)[col].dropna().astype("int64").tolist())
            except Exception:
                return set()
        return set()

    processed_issues   = _ids_if_exists(issues_csv, "issue_id")
    processed_events   = _ids_if_exists(events_csv, "event_id")
    processed_comments = _ids_if_exists(issues_comments_csv, "comment_id")
    processed_timeline = _ids_if_exists(timeline_csv, "event_id")

    # commits total-count (for page math)
    issues_df   = pandas.DataFrame(columns=issues_cols)
    events_df   = pandas.DataFrame(columns=issue_events_cols)
    comments_df = pandas.DataFrame(columns=issues_comments_cols)
    timeline_df = pandas.DataFrame(columns=timeline_cols)
    excluded_new = pandas.DataFrame(columns=["issue_id"])
    

    g, token, search_limit, reset_time = util.getSameToken(g, token, position)
    repo     = g.get_repo(repoName)
    
    query = """
    query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
        issues(states:[OPEN, CLOSED])   { totalCount }
    }
    }
    """
    vars = {"owner": repo.owner.login, "name": repo.name}
    data = repo.requester.graphql_query(query=query, variables=vars)[1]
    num_items: int = data["data"]["repository"]["issues"]["totalCount"]
        
    last_page   = int(num_items / cfg.items_per_page)
    last_page   = max(last_page, 0)
    start_page  = (util.getLastPageRead(Path(workingFolder, last_page_save))
                   if Path(workingFolder, last_page_save).exists()
                   else 0)
    current_page = start_page

    page_query = """
    query($owner:String!, $name:String!, $after:String) {
      repository(owner:$owner, name:$name) {
        issues(first:100, orderBy:{field:CREATED_AT, direction:DESC}, states:[OPEN, CLOSED], after:$after) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id
            number
            createdAt
            title
            state
            author { login }
            labels(first:20){ nodes{ name } }
            assignees(first:20){ nodes{ login } }
            milestone { title }
          }
        }
      }
    }
    """
    chunk_idx = 0
   
    exception_thrown = True

    while exception_thrown:
        exception_thrown = False
        try:
            
            while current_page <= last_page and not STOP_EVENT.is_set():

                g, token, search_limit, reset_time = util.getSameToken(g, token, position)
                payload = {"owner": repo.owner.login, "name": repo.name, "after": after_cursor}
                data = repo.requester.graphql_query(query=page_query, variables=payload)[1]


                issues_block = data["data"]["repository"]["issues"]
                nodes = issues_block["nodes"]

                stop_stream = False
                for node in nodes:
                    st.write(f"Processing issue node: {node['id']}")
                    g, token, search_limit, reset_time = util.getSameToken(g, token, position)
                    if STOP_EVENT.is_set():
                        break
                    created_at_iso = node["createdAt"]
                    node_dt = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
                    if node_dt.tzinfo is not None:
                        node_dt = node_dt.replace(tzinfo=None)

                    if node_dt < start_date:
                        stop_stream = True
                        break
                    pbar.update(1)
                    created_by = node["author"]["login"] if node["author"] else None
                    issue = repo.get_issue(number=node["number"])

                    if issue.user is None:
                        excluded.loc[len(excluded)] = {"issue_id": issue.id}
                        continue
                    if issue.id < last_issue:
                        continue
                    if issue.id in processed_issues:
                        continue
                    if issue.id in excluded.issue_id.values:
                        continue
                    #pdar.set_postfix({
                    #        "search_limit": search_limit,
                    #        "reset_time": reset_time,
                    #        "repo": repoName.split("/")[-1]
                    #    })
                    
                    processed_issues.add(issue.id)
                    issues_df.loc[len(issues_df)] = {
                        "repo": repoName,
                        "created_at": issue.created_at,
                        "created_by": issue.user.login if issue.user else created_by or "",
                        "issue_id": issue.id,
                        "title": issue.title,
                        "labels": ",".join([label.name for label in issue.labels]),
                        "state": issue.state,
                        "body": issue.body or "",
                        "assignees": ",".join([assignee.login for assignee in issue.assignees]),
                        "milestone": issue.milestone.title if issue.milestone else "",
                    }

                    for comment in issue.get_comments():
                        if comment.id in processed_comments:
                            continue
                        processed_comments.add(comment.id)
                        comments_df.loc[len(comments_df)] = {
                            "repo": repoName,
                            "created_at": comment.created_at,
                            "created_by": comment.user.login if comment.user else "",
                            "issue_id": issue.id,
                            "comment_id": comment.id,
                            "body": comment.body or ""
                        }
                    for event in issue.get_events():
                        if event.id in processed_events:
                            continue
                        processed_events.add(event.id)
                        events_df.loc[len(events_df)] = {
                            "repo": repoName,
                            "created_at": event.created_at,
                            "created_by": event.actor.login if event.actor else "",
                            "issue_id": issue.id,
                            "event_id": event.id,
                            "event": event.event
                        }
                    for interaction in issue.get_timeline():
                        if interaction.id in processed_timeline:
                            continue
                        processed_timeline.add(interaction.id)

                        timeline_df.loc[len(timeline_df)] = {
                            "repo": repoName,
                            "created_at": interaction.created_at,
                            "created_by": interaction.actor.login if interaction.actor else "",
                            "issue_id": issue.id,
                            "event_id": interaction.id,
                            "event": interaction.event
                        }

                    last_issue = issue.id

                # ----- exception handlers unchanged ----------------------
                _flush_issues_data_to_csv(
                    issues_df, events_df, comments_df, timeline_df, excluded,
                    workingFolder, save_tmp, last_issue=last_issue,
                    page=chunk_idx,
                    save_last_page=cfg.last_page_issues,
                    save_current_page=cfg.next_page_issues,
                    is_page_finished=True
                )
                end_cursor = issues_block["pageInfo"]["endCursor"]
                with open(cursor_path, "w", encoding="utf-8") as fh:
                    fh.write(end_cursor or "")

                # clear frames
                issues_df   = issues_df.iloc[0:0]
                events_df   = events_df.iloc[0:0]
                comments_df = comments_df.iloc[0:0]
                timeline_df = timeline_df.iloc[0:0]
                chunk_idx  += 1

                if stop_stream or not issues_block["pageInfo"]["hasNextPage"]:
                    break
                after_cursor = end_cursor

            exception_thrown = False
            break  # Exit the while loop if no exception was thrown
        except KeyboardInterrupt as e:
            # ----- 1. user pressed Ctrl-C --------------------------------
            _flush_issues_data_to_csv(issues_df, events_df, comments_df, timeline_df, excluded, 
                                      workingFolder, save_tmp, last_issue, page=current_page,
                                      save_last_page=last_page_save, save_current_page=next_page, is_page_finished=False)            
            raise e
        # ----- exception handlers unchanged ------------------------
        except (GithubException, AttributeError, Exception) as e:
            _flush_issues_data_to_csv(issues_df, events_df, comments_df, timeline_df, excluded, 
                                      workingFolder, save_tmp, last_issue, page=current_page,
                                      save_last_page=last_page_save, save_current_page=next_page, is_page_finished=False)
            exception_thrown = True
            raise

        finally:
            _flush_issues_data_to_csv(issues_df, events_df, comments_df, timeline_df, excluded, 
                                      workingFolder, save_tmp, last_issue, page=current_page,
                                      save_last_page=last_page_save, save_current_page=next_page, is_page_finished=False)

    with open(os.path.join(workingFolder, status_tmp), "w") as fh:
        fh.write(f"COMPLETE;{cfg.data_collection_date}")

    return g, token

def _flush_issues_data_to_csv(
        issues_df, events_df, comments_df,
        timeline_df, excluded_df,
        workingFolder, save_tmp, last_issue, page,
        save_last_page, save_current_page, is_page_finished=False):

    _safe_append(issues_df, Path(workingFolder, cfg.issue_list_file_name))
    _safe_append(events_df,  Path(workingFolder, cfg.issue_events_list_file_name))
    _safe_append(comments_df,Path(workingFolder, cfg.issue_comments_list_file_name))
    _safe_append(timeline_df, Path(workingFolder, cfg.issue_timeline_file_name))
    _safe_append(excluded_df, Path(workingFolder, "_excludedNoneType_Issues.tmp"))

    # remember progress
    with open(Path(workingFolder, save_tmp), "w") as fh:
        fh.write(str(last_issue))

    # mirror the commit logic for finished / unfinished pages
    current_page = (
        util.getLastPageRead(Path(workingFolder, save_current_page))
        if Path(workingFolder, save_current_page).exists()
        else 0
    )
    with open(Path(workingFolder, save_last_page), "w") as fh:
        fh.write(
            f"last_page_flushed:{page}" if is_page_finished
            else f"last_page_flushed:{current_page}"
        )

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

def main_commit_extraction(repo, since_days):

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_FOLDER)
    
    os.system('cls' if os.name=='nt' else 'clear')

    gitRepoName = repo.replace('https://github.com/', '').strip()
    tqdm.write('Running Commit Extraction for {}'.format(gitRepoName))


    ### SET THE PROJECT
    splitRepoName = gitRepoName.split('/')
    organization = splitRepoName[0]
    project = splitRepoName[1]

    organizationsFolder = cfg.temp_data_folder
    os.makedirs(organizationsFolder, exist_ok=True)

    organizationFolder = os.path.join(organizationsFolder, organization)
    os.makedirs(organizationsFolder, exist_ok=True)
    
    full_extraction= True

    #first check if we have data if so return data
    if os.path.exists(os.path.join(organizationFolder, project, cfg.commit_list_file_name)):
        tqdm.write('Commit data already exists. Skipping extraction.')
        return pandas.read_csv(os.path.join(organizationFolder, project, cfg.commit_list_file_name), sep=cfg.CSV_separator)

    runALLExtractionRoutine(organizationFolder, organization, project, full_extraction, since_days)
    token = util.getSpisificToken(0)
    g0       = Github(token)
    org = g0.get_organization(organization)

    org_repos = [r for r in org.get_repos(type='sources')
             if not r.archived and not r.fork]

    try: ### Only for Log (Block)
        num_repos = org_repos.totalCount - 1
    except:
        num_repos = 'Unknown'

    full_extraction = False
    repo_num = 0 ### Only for Log
    for repo in org_repos:
        project_name = repo.name
        if project_name != project:
            repo_num += 1 ### Only for Log
            print('Running Commit Extraction for {} ({}/{})'.format(project_name, repo_num, num_repos))
            runALLExtractionRoutine( organizationFolder, organization, project_name, False, since_days )
    
    tqdm.write('Commit Extraction for {} Completed'.format(gitRepoName))
    tqdm.write('Done.')

    out_path = os.path.join(organizationFolder, project, cfg.commit_list_file_name)
    if not os.path.exists(out_path):
        # return an empty DataFrame with the expected schema
        print(f"No commit data found at {out_path}. Returning empty DataFrame.")
        return pandas.DataFrame(columns=[
            "repo","created_at","author_id","author_name","author_email",
            "committer_id","sha","filename_list","fileschanged_count",
            "additions_sum","deletions_sum"
        ])
    return pandas.read_csv(out_path, sep=cfg.CSV_separator)

def write_pauses_table(
        df: pandas.DataFrame,
        out_path: os.PathLike,
        authors: list[str] | None = None,
        *,
        user_col: str = "author_id",
        date_col: str = "created_at",
        tail_to_today: bool = False
    ) -> pandas.DataFrame:

    df[date_col] = pandas.to_datetime(df[date_col]).dt.normalize()

    if authors is None:
        authors = df[user_col].unique()

    rows = []
    
    count =0
    for dev in authors:
        user_df = df[df[user_col] == dev]
        pause_len_1 = len(user_df[date_col].dt.date.unique())
        pause_len_2 =len(user_df)
        if user_df.empty:
            continue

        active_days = sorted(user_df[date_col].dt.date.unique())
        current_row = [dev]

        for i in range(len(active_days) - 1):
            prev_day = active_days[i]
            next_day = active_days[i + 1]
            gap = (next_day - prev_day).days
            if gap > 1:
                # Inactivity starts the day after prev_day
                current_row.append(f"{(prev_day + pandas.Timedelta(days=1)).strftime('%Y-%m-%d')}/{next_day.strftime('%Y-%m-%d')}")
            else:
                count += 1


        if tail_to_today and active_days:
            today = _date.today()
            gap = (today - active_days[-1]).days
            if gap > 1:
                current_row.append(f"{active_days[-1]}/{today}")

        if len(current_row) > 1:
            rows.append(current_row)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="",encoding="utf-8" ) as f:
        csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE).writerows(rows)

    return pandas.DataFrame(rows)

def get_NONCODING(folder: str, dev_login: str) -> pandas.DataFrame:

    """
    Build the developer's DAILY 'other-actions' table.
    Returns a dataframe whose index is the *action*
    ('issues/pull_requests', 'issues_comments', …) and whose
    columns are day-strings.
    """
    files = {
    "prs": (
        "prs_repo.csv",
        {"PR_id": "id", "created_at": "date", "created_by": "creator_login"},
    ),
    "prs_comments": (
        "prs_comments.csv",
        {"comment_id": "id", "created_at": "date", "created_by": "creator_login"},
    ),
    "issues": (
        "issues_repo.csv",
        {"issue_id": "id", "created_at": "date", "created_by": "creator_login"},
    ),
    "issues_comments": (
        "issues_comments_repo.csv",
        {"comment_id": "id", "created_at": "date", "created_by": "creator_login"},
    ),
    "issues_events": (
        "issues_events_repo.csv",
        {"event_id": "id", "created_at": "date", "created_by": "creator_login"},
    ),
    "issues_timeline": (
        "issues_timeline_repo.csv",
        {"event_id": "id", "created_at": "date", "created_by": "creator_login"},
    )
    }

    # ---------- read / filter every file ----------
    dfs = {}
    for key, (fname, rename_map) in files.items():
        dfs[key] = _load_activity_csv(folder, fname, rename_map, dev_login)

    # ---------- split issues vs PRs -------------
    # Old logic: issues endpoint also returns PRs; remove rows whose id
    # matches a PR id so we don’t double-count.
    if not dfs["issues"].empty and not dfs["prs"].empty:
        dfs["issues"] = dfs["issues"][~dfs["issues"].id.isin(dfs["prs"].id)]

    # ---------- build the day range -------------
    # Derive it from the *actual* activity we just read.
    #
    # 1) gather every non-empty dataframe
    non_empty = [df for df in dfs.values() if not df.empty]

    if non_empty:
        # 2) earliest / latest date across *all* action types
        min_date = min(df["date"].min() for df in non_empty)
        max_date = max(df["date"].max() for df in non_empty)
    else:
        # Developer has no activity at all → default to one-day range
        min_date = max_date = pandas.Timestamp.today()

    # 3) full, dense list of day strings
    day_cols = (
        pandas.date_range(
            start=pandas.to_datetime(min_date).normalize(),
            end=pandas.to_datetime(max_date).normalize(),
            freq="D",
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )

    # ---------- helper to create one timeline row ----------
    def _timeline_row(action_name, df_raw):
        row = [action_name]
        if df_raw.empty:
            row += [0] * len(day_cols)
            return row
        counts = (
            pandas.to_datetime(df_raw["date"])
            .dt.date
            .value_counts()
            .to_dict()
        )
        for d in day_cols:
            row.append(counts.get(pandas.to_datetime(d).date(), 0))
        return row

    # ---------- compile all action rows ----------
    rows = []
    if not dfs["issues"].empty:
        rows.append(_timeline_row("issues", dfs["issues"]))
    if not dfs["issues_comments"].empty:
        rows.append(_timeline_row("issues_comments", dfs["issues_comments"]))
    if not dfs["issues_events"].empty:
        rows.append(_timeline_row("issues_events", dfs["issues_events"]))
    if not dfs["prs"].empty:
        rows.append(_timeline_row("pull_requests", dfs["prs"]))
    if not dfs["prs_comments"].empty:
        rows.append(_timeline_row("pull_requests_comments", dfs["prs_comments"]))

    # (commits are already encoded in coding_history_table, so we skip them here)

    actions = pandas.DataFrame(rows, columns=["action"] + day_cols).set_index("action")

    return actions

def _load_activity_csv(folder: str,
                       filename: str,
                       rename_map: Dict[str, str],
                       dev_login,
                       usecols: list[str] = None,
                       ) -> pandas.DataFrame:
    """
    Read *filename* in *folder*, rename to the canonical columns
    ('id','date','creator_login'), keep ONLY the specified dev, and
    return three columns.  On any problem → empty df.
    """
    path = os.path.join(folder, filename)
    try:
        df = pandas.read_csv(path, sep=cfg.CSV_separator, usecols=usecols)
    except FileNotFoundError:
        logging.info("File %s not found – skipping", path)
        return pandas.DataFrame(columns=["id", "date", "creator_login"])
    except Exception as e:
        logging.warning("Could not read %s: %s", path, e)
        return pandas.DataFrame(columns=["id", "date", "creator_login"])

    df = df.rename(columns=rename_map)
    # keep only the columns we need, ignore anything extra
    df = df[["id", "date", "creator_login"]]
    df = df[df.creator_login == dev_login]
    # allow str OR list[str]
    if isinstance(dev_login, list):
        df = df[df.creator_login.isin(dev_login)]
    else:
        df = df[df.creator_login == dev_login]
    return df.reset_index(drop=True)

def get_ACTIVITY(folder: str, dev: str) -> pandas.DataFrame:

    path = os.path.join(folder, "commit_list.csv")

    # ─── load & clean ──────────────────────────────────────────────────
    df = pandas.read_csv(path, sep=cfg.CSV_separator, parse_dates=["created_at"])

    df = df[df["author_id"] == dev]           # keep only this dev
    if df.empty:                                    # no commits at all
        raise ValueError(f"No commits found for {dev}")

    dates = df["created_at"].dt.normalize()
    if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
    # ─── per-day aggregation ──────────────────────────────────────────
    daily_counts = (
        df.groupby(df["created_at"].dt.normalize())       # strip and remove hh:mm:ss 
          .size()
          .rename("commits")
    )

    daily_counts = (
        dates.value_counts()
             .sort_index()
             .rename("commits")
             .astype("int64")
    )
    daily_counts.index.name = "date"
    return daily_counts.to_frame()

def get_timeline(folder: str, dev: str) -> pandas.DataFrame:
    """ActivitiesExtractor.py
    This function will take all the activity and make a daily aggregated df
    """
    # actions: rows = action names, cols = day strings "YYYY-MM-DD"
    actions = get_NONCODING(folder, dev)
    actions = actions.transpose()  # make the index a DatetimeIndex


    # make the index a DatetimeIndex (tz-naive, normalized)
    actions.index = pandas.to_datetime(actions.index, errors="coerce").tz_localize(None)
    actions.index.name = "date"

    # commits: DataFrame with index=date (DatetimeIndex), col 'commits'
    commits = get_ACTIVITY(folder, dev)

    # union the date ranges and align
    # build a FULL daily index from min→max, then align
    if not actions.empty and not commits.empty:
        start = min(actions.index.min(), commits.index.min())
        end   = max(actions.index.max(), commits.index.max())
    elif not actions.empty:
        start, end = actions.index.min(), actions.index.max()
    elif not commits.empty:
        start, end = commits.index.min(), commits.index.max()
    else:
        # no activity at all → return an empty, well-typed frame
        cols = ["commits","pull_requests","issues","issues_comments",
                "issues_events","pull_requests_comments","coding_day","nc_day"]
        return pandas.DataFrame(columns=cols).astype({
            "commits":"int64","pull_requests":"int64","issues":"int64",
            "issues_comments":"int64","issues_events":"int64",
            "pull_requests_comments":"int64","coding_day":"bool","nc_day":"bool"
        })

    full_idx = pandas.date_range(start=start, end=end, freq="D")
    actions = actions.reindex(full_idx, fill_value=0)
    commits = commits.reindex(full_idx, fill_value=0)


    # merge
    df = actions.join(commits, how="outer")
    df = df.fillna(0)

    # ensure expected columns exist even if that action never occurred
    for col in ["pull_requests", "issues", "issues_comments", "issues_events", "pull_requests_comments"]:
        if col not in df.columns:
            df[col] = 0

    # booleans
    df["coding_day"] = (df["commits"] > 0)


    df["nc_day"] = (df[["issues", "issues_comments", "issues_events", "pull_requests_comments"]].sum(axis=1) > 0)

    # nice ordering
    cols = ["commits", "pull_requests", "issues", "issues_comments", "issues_events", "pull_requests_comments",
            "coding_day", "nc_day"]
    # keep any extra columns too (if you later add more)
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    return df.sort_index()

def write_pauses_table(
        df: pandas.DataFrame,
        out_path: os.PathLike,
        authors: list[str] | None = None,
        *,
        user_col: str = "author_id",
        date_col: str = "created_at",
        tail_to_today: bool = False
    ) -> pandas.DataFrame:

    df[date_col] = pandas.to_datetime(df[date_col]).dt.normalize()

    if authors is None:
        authors = df[user_col].unique()

    rows = []
    
    count =0
    for dev in authors:
        user_df = df[df[user_col] == dev]
        pause_len_1 = len(user_df[date_col].dt.date.unique())
        pause_len_2 =len(user_df)
        if user_df.empty:
            continue

        active_days = sorted(user_df[date_col].dt.date.unique())
        current_row = [dev]

        for i in range(len(active_days) - 1):
            prev_day = active_days[i]
            next_day = active_days[i + 1]
            gap = (next_day - prev_day).days
            if gap > 1:
                # Inactivity starts the day after prev_day
                current_row.append(f"{(prev_day + pandas.Timedelta(days=1)).strftime('%Y-%m-%d')}/{next_day.strftime('%Y-%m-%d')}")
            else:
                count += 1


        if tail_to_today and active_days:
            today = _date.today()
            gap = (today - active_days[-1]).days
            if gap > 1:
                current_row.append(f"{active_days[-1]}/{today}")

        if len(current_row) > 1:
            rows.append(current_row)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="",encoding="utf-8" ) as f:
        csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE).writerows(rows)

    return pandas.DataFrame(rows)

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

#-----------------------------
def getFarOutThreshold(values, dev): ### If it is satisfying, move the function into UTILITIES
    import numpy
    th = 0
    q_3rd = numpy.percentile(values,75)
    q_1st = numpy.percentile(values,25)
    iqr = q_3rd-q_1st
    if iqr > 1:
        th = q_3rd + 3*iqr
    return th

def addToBreaksList(pauses, currentBreaks, th):
    for _, p in pauses.iterrows():
        if (p['len'] > th) and (p['dates'] not in currentBreaks.dates.tolist()):
            util.add(currentBreaks, [p['len'], p['dates'], th])
    return currentBreaks

def cleanClearBreaks(clearBreaks, breaks):
    for _, b in breaks.iterrows():
        clearBreaks = clearBreaks[clearBreaks.dates != b['dates']] # If it was in the long_breaks list, remove ot from there
    return clearBreaks

def identifyBreaks(pauses_dates_list, dev, window, shift,
                   debug_folder=None):           # NEW ARG
    '''
    Removes SURE BREAKS from windows to calculate Tfov
    and — with debug_folder — writes a per-window diagnostics CSV.
    '''
    breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])
    diagnostics = []                             # NEW
    count = 0
    for row in pauses_dates_list:
        if row[0] != dev:              # ⬅️  ignore other developers
            continue
        
        count += 1
        if count % 50 == 0:  # Print progress every 100 developers
            print(count)
        intervals_list = [ x for x in row[1:]
                          if isinstance(x, str) and '/' in x and x.strip()]
        
        intervals_list.sort(key=lambda s: s.split('/')[0])

        if not all(a.split('/')[0] <= b.split('/')[0]
                for a, b in zip(intervals_list, intervals_list[1:])):
            print("⚠️  intervals_list UNSORTED for", dev)

        if not intervals_list:
            print(dev, 'has NO valid pauses')
            continue                      # <- don’t bail out; just skip

        clear_breaks = pandas.DataFrame(columns=['len', 'dates'])

        FPS_dt = datetime.strptime(intervals_list[0].split('/')[0], '%Y-%m-%d')
        LPE_dt = datetime.strptime(intervals_list[-1].split('/')[1], '%Y-%m-%d')

        win_start, win_end = FPS_dt, FPS_dt + timedelta(days=window)
        last_th = 0
        while win_end < LPE_dt:
            win_pauses_list = pandas.DataFrame(columns=['len', 'dates'])
            partially_included_pauses_list = pandas.DataFrame(columns=['len', 'dates'])

            for interval in intervals_list:
                int_start_str, int_end_str = interval.split('/')          # keep strings
                int_start_dt  = datetime.strptime(int_start_str, '%Y-%m-%d')
                int_end_dt    = datetime.strptime(int_end_str,   '%Y-%m-%d')
                pause_len = util.daysBetween(int_start_str, int_end_str)
                # fully inside
                if int_start_dt >= win_start and int_end_dt <= win_end:
                    util.add(win_pauses_list, [pause_len, interval])
                # touches boundary
                if ((int_start_dt <= win_end and int_end_dt > win_end) or
                    (int_end_dt >= win_start and int_start_dt < win_start)):
                    util.add(partially_included_pauses_list, [pause_len, interval])

            win_pauses = len(win_pauses_list)
            pauses = pandas.concat([win_pauses_list,
                                    partially_included_pauses_list],
                                    ignore_index=True)

            # --- decision logic (unchanged) ---------------------------------
            win_th = None
            added_flag = False
            if win_pauses >= 4:
                win_th = getFarOutThreshold(win_pauses_list['len'], dev)
                if win_th > 0:
                    before = len(breaks_df)
                    breaks_df = addToBreaksList(pauses, breaks_df, win_th)
                    added_flag = len(breaks_df) > before
                    last_th = win_th
                elif last_th > 0:
                    before = len(breaks_df)
                    breaks_df = addToBreaksList(pauses, breaks_df, last_th)
                    added_flag = len(breaks_df) > before
            else:
                if last_th > 0:
                    before = len(breaks_df)
                    breaks_df = addToBreaksList(pauses, breaks_df, last_th)
                    added_flag = len(breaks_df) > before

                clear_breaks = cleanClearBreaks(clear_breaks, breaks_df)
                for _, p in pauses.iterrows():
                    if (p['len'] >= window and
                        p['dates'] not in clear_breaks.dates.tolist() and
                        p['dates'] not in breaks_df.dates.tolist()):
                        util.add(clear_breaks, p)

            # ----------- NEW: record diagnostics for this window -------------
            diagnostics.append({
                'win_start': win_start.date(),
                'win_end':   win_end.date(),
                'win_pauses': win_pauses,
                'pause_lengths': ';'.join(map(str, win_pauses_list['len'].tolist())),
                'partial_lengths': ';'.join(map(str, partially_included_pauses_list['len'].tolist())),
                'win_th': win_th,
                'last_th': last_th,
                'added_as_break': 'yes' if added_flag else 'no'
            })
            # -----------------------------------------------------------------

            win_start += timedelta(days=shift)
            win_end   = win_start + timedelta(days=window)


    return breaks_df
#--------------------------

def label_timeline(user_timeline, breaks_df):
    """
    Make a labled timeline of devlopers breaks

    given a user_timeline and breaks_df
    user_timeline
    commits,pull_requests,issues,issues_comments,issues_events,pull_requests_comments,coding_day,nc_day
    0,0,2,1,0,0,False,True
    3,0,0,0,0,0,True,False
    0,0,0,0,0,0,False,False

    breaks_df
    len,dates,th
    72,2015-05-05/2015-07-16,59.25
    """
    df = user_timeline.copy()

    df["break_day"] = None

    #this marks the break days from breaks_df onto user_timeline
    for breaks in breaks_df.itertuples():
        start = breaks.dates.split('/')[0]
        end = breaks.dates.split('/')[1]

        if start in df.index and bool(df.at[start, "coding_day"]):
            start = start + pandas.Timedelta(days=1)

        end = pandas.to_datetime(end) - pandas.Timedelta(days=1)

        break_range = pandas.date_range(start=pandas.to_datetime(start)+pandas.Timedelta(days=1),
                                end=pandas.to_datetime(end)-pandas.Timedelta(days=1))

        if pandas.to_datetime(start) <= end:

           for date in break_range:
                date = date.strftime("%Y-%m-%d")
                df.at[date, "break_day"] = True
                df.at[date, "th"] = breaks.th
                df.at[date, "len"] = breaks.Index

    #turns all the NA into false
    for col in ["coding_day", "nc_day", "break_day"]:
        df[col] = df[col].fillna(False).astype("bool")


    df.index = pandas.to_datetime(df.index)
    df = df.sort_index()

    # Optional: unmark the break *end* day (commit day) so it’s not counted as break
    # break end - 1 day = end non coding
    for breaks in breaks_df.itertuples():
        end = pandas.to_datetime(breaks.dates.split('/')[1])
        if end in df.index:
            df.at[end, "break_day"] = False

    gone_days = 365

    df["event_day"] = df["coding_day"] | df["nc_day"]

    df["state"] = "ACTIVE"

    # Identify contiguous break windows (groups of consecutive True in break_day)
    bd = df["break_day"]
    group_id = (bd != bd.shift(1)).cumsum()

    # Precompute last event BEFORE a given date (global, across timeline)
    all_events_idx = df.index[df["event_day"]]

    for gid, block in df.groupby(group_id):
        if not block["break_day"].iloc[0]:
            continue  # not a break chunk

        # This is one contiguous break [start .. end] (inclusive)
        start_ts = block.index[0]
        end_ts   = block.index[-1]

        # Lookahead info: is there any non-coding event in this break?
        has_nc = bool((df.loc[start_ts:end_ts, "nc_day"]).any())

        th_vals = df.loc[start_ts:end_ts, "th"].dropna()
        Tfov = int(round(th_vals.iloc[0])) if not th_vals.empty else 14

        # Anchor silence to the last event (coding or non-coding) before the break starts
        prev_nc_idx = df.index[df["nc_day"] & (df.index < start_ts)]
        last_nc_before = prev_nc_idx.max() if len(prev_nc_idx) else None

        last_nc = None  # most recent non-coding event inside this break
        if last_nc_before is not None and (start_ts - last_nc_before).days <= Tfov:
            # Seed the NON_CODING hold across the start of the break
            last_nc = last_nc_before

        # Walk day by day inside the break
        for d in block.index:

            # Non-coding event day => NON_CODING and update last_nc
            if bool(df.at[d, "nc_day"]):
                df.at[d, "state"] = "NON_CODING"
                last_nc = d
                continue

            # Silent day inside a break -> decide via Tfov and gone
            # Compute silence since the most relevant last event:
            # - Prefer last NC inside break; else use last event before break; else start-of-break as approximate anchor.
            ref_nc = last_nc if last_nc is not None else None
            if (ref_nc is not None) and ((d - ref_nc).days <= Tfov):
                df.at[d, "state"] = "NON_CODING"
                continue

            # No recent NC: INACTIVE vs GONE (since last ANY event)
            last_any = ref_nc if ref_nc is not None else (all_events_idx[all_events_idx < start_ts].max() if len(all_events_idx[all_events_idx < start_ts]) else None)

            if last_any is not None:
                silent_days = (d - last_any).days
                if silent_days > gone_days:
                    df.at[d, "state"] = "GONE"
                else:
                    df.at[d, "state"] = "INACTIVE"
            else:
                df.at[d, "state"] = "INACTIVE"


    return df

#--------------------------
#Prediction helper functions
#--------------------------
def users_activity_concat(authors, repo):
    frames = []

    repo = repo.rstrip('\n')
    base = cfg.temp_data_folder + "/" + repo + "/" + cfg.results_folder
    print(base)
    for dev in authors:
        file_name = Path(base) / f"{dev}_labeled_timeline.csv"
        if not file_name.exists():
            # Optional: log this instead of printing
            continue
        #parse date is the index column
        df = pandas.read_csv(
            file_name, parse_dates=["date"]
        )

        df["author"] = dev
        df["repo"] = repo

        df["date"] = df["date"].dt.normalize()

        df = df.sort_values("date")

        frames.append(df)


    out = pandas.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["author", "repo", "date"]).reset_index(drop=True)
    st.write("out", out)

    return out

def segmentize_timeline(
    daily: pandas.DataFrame,
    state_order: List[str] = ("ACTIVE", "NON_CODING", "INACTIVE", "GONE"),
) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    Convert a per-day timeline into consecutive same-state segments per (author, repo).

    Inputs
    ------
    daily : DataFrame with at least columns:
        ['author','repo','date','state',
         'commits','pull_requests','issues','issues_comments',
         'issues_events','pull_requests_comments','coding_day','nc_day','break_day','th','len','event_day']
        (extras are okay)

    state_order : list of states in desired categorical order.

    Returns
    -------
    daily_with_segments : original daily rows +:
        'segment_id'  (int, 0-based within each (author,repo))
        'segment_pos' (1-based position within the segment)
        'segment_len' (length of the segment in days)

    segments : one row per segment with keys & aggregates:
        ['author','repo','segment_id','state_curr','start_date','end_date','seg_len',
         'commits_sum','pr_sum','issues_sum','issues_comments_sum','issues_events_sum','pr_comments_sum',
         'coding_days','nc_days','break_days','prev_state','prev_seg_len','next_state']
    """
    df = daily.copy()

    # --- Normalize key types & sort deterministically ---
    df["author"] = df["author"].astype(str).str.strip()
    df["repo"]   = df["repo"].astype(str).str.strip()
    df["date"]   = pandas.to_datetime(df["date"], utc=True, errors="coerce")
    df["state"]  = pandas.Categorical(df["state"].astype(str),
                                  categories=list(state_order), ordered=True)

    df = df.sort_values(["author","repo","date"], kind="stable").reset_index(drop=True)

    # --- Compute per-(author,repo) segment boundaries ---
    # A new segment starts whenever state != previous state's value (or first row)
    grp = df.groupby(["author","repo"], sort=False, observed=True)
    first_in_group = grp.cumcount().eq(0)
    state_changed  = df["state"].ne(grp["state"].shift(1))
    prev_state = grp["state"].shift(1)
    next_state = grp["state"].shift(-1)
    one_day_flip = prev_state.eq(next_state) & df["state"].ne(prev_state)
    df.loc[one_day_flip, "state"] = prev_state[one_day_flip]
    is_seg_start   = first_in_group | state_changed

    # segment_id: 0,1,2,... within each (author,repo)
    df["segment_id"] = grp.apply(lambda g: is_seg_start.loc[g.index].cumsum() - 1).reset_index(level=[0,1], drop=True)

    # Position & length inside segment (handy during analysis or later expansions)
    df["segment_pos"] = df.groupby(["author","repo","segment_id"], observed=True).cumcount() + 1
    df["segment_len"] = df.groupby(["author","repo","segment_id"], observed=True)["date"].transform("size")

    # --- Build segment-level table with aggregates ---
    agg_map = {
        "date": ["min","max","size"],
        "commits": "sum",
        "pull_requests": "sum",
        "issues": "sum",
        "issues_comments": "sum",
        "issues_events": "sum",
        "pull_requests_comments": "sum",
        "coding_day": "sum",
        "nc_day": "sum",
        "break_day": "sum",
    }

    seg = (df
           .groupby(["author","repo","segment_id","state"], observed=True, as_index=False)
           .agg(start_date=("date","min"),
                end_date=("date","max"),
                seg_len=("date","size"),
                commits_sum=("commits","sum"),
                pr_sum=("pull_requests","sum"),
                issues_sum=("issues","sum"),
                issues_comments_sum=("issues_comments","sum"),
                issues_events_sum=("issues_events","sum"),
                pr_comments_sum=("pull_requests_comments","sum"),
                coding_days=("coding_day","sum"),
                nc_days=("nc_day","sum"),
                break_days=("break_day","sum"))
          )
    seg.rename(columns={"state": "state_curr"}, inplace=True)

    # Previous/next segment context (per author,repo)
    seg = seg.sort_values(["author","repo","segment_id"], kind="stable")
    seg["prev_state"]   = seg.groupby(["author","repo"], observed=True)["state_curr"].shift(1)
    seg["prev_seg_len"] = seg.groupby(["author","repo"], observed=True)["seg_len"].shift(1)
    seg["next_state"]   = seg.groupby(["author","repo"], observed=True)["state_curr"].shift(-1)

    # Keep date columns as date (drop TZ & normalize to date if you prefer)
    # seg["start_date"] = seg["start_date"].dt.tz_localize(None).dt.date
    # seg["end_date"]   = seg["end_date"].dt.tz_localize(None).dt.date

    return df, seg

def add_predictors(seg: pandas.DataFrame) -> pandas.DataFrame:
    """
    Segment-level feature builder for next-segment-state prediction.

    Input  : daily timeline with columns >=
             ['author','repo','date','state','commits','pull_requests','issues',
              'issues_comments','issues_events','pull_requests_comments',
              'coding_day','nc_day','break_day','th','len','event_day']

    Output : segment table with features and targets:
             - keys:   author, repo, segment_id, start_date, end_date
             - labels: state_t, target_next_state
             - features: seg_len, densities, proportions, prev_* and rolling stats
             - a 'date' column (== start_date) kept for downstream compatibility
    """
    # 1) segmentize

    if seg.empty:
        return pandas.DataFrame(columns=[
            "author","repo","segment_id","date","start_date","end_date",
            "state_t","target_next_state"
        ])

    # 2) core, leakage-safe features (predicting at segment END)
    out = seg.copy()

    # densities / proportions within the segment
    eps = 1e-9
    out["avg_commits_per_day"]   = out["commits_sum"] / (out["seg_len"] + eps)
    out["avg_prs_per_day"]       = out["pr_sum"] / (out["seg_len"] + eps)
    out["avg_issues_per_day"]    = out["issues_sum"] / (out["seg_len"] + eps)
    out["avg_comments_per_day"]  = out["issues_comments_sum"] / (out["seg_len"] + eps)
    out["avg_events_per_day"]    = out["issues_events_sum"] / (out["seg_len"] + eps)
    out["avg_pr_comments_per_day"] = out["pr_comments_sum"] / (out["seg_len"] + eps)

    out["pct_coding_days"]    = out["coding_days"] / (out["seg_len"] + eps)
    out["pct_noncoding_days"] = out["nc_days"]     / (out["seg_len"] + eps)
    out["pct_break_days"]     = out["break_days"]  / (out["seg_len"] + eps)

    # calendar at segment start
    out["start_date"] = pandas.to_datetime(out["start_date"], utc=True, errors="coerce")
    out["start_dow"]  = out["start_date"].dt.dayofweek  # 0=Mon
    out["start_woy"]  = out["start_date"].dt.isocalendar().week.astype(int)
    out["start_month"]= out["start_date"].dt.month
    out["start_qtr"]  = ((out["start_month"] - 1) // 3 + 1).astype(int)

    # prior segment context (already present): prev_state, prev_seg_len
    # simple bigram feature
    out["prev_curr_pair"] = (out["prev_state"].astype(str) + "→" + out["state_curr"].astype(str))

    # author/repo-level rolling stats on previous segments (k = 3, 5)
    by = ["author","repo"]
    for k in (3, 5):
        out[f"roll{k}_seg_len_mean"]   = _roll_feat(out, by, "seg_len", k, "mean")
        out[f"roll{k}_seg_len_std"]    = _roll_feat(out, by, "seg_len", k, "std")
        out[f"roll{k}_commits_mean"]   = _roll_feat(out, by, "commits_sum", k, "mean")
        out[f"roll{k}_nc_days_mean"]   = _roll_feat(out, by, "nc_days", k, "mean")
        out[f"roll{k}_break_days_mean"]= _roll_feat(out, by, "break_days", k, "mean")

    # 3) labels and presentation
    out.rename(columns={"state_curr": "state_t"}, inplace=True)
    out["state_t"] = out["state_t"].astype(str)
    out["target_next_state"] = out["next_state"].astype(str)

    # keep only rows with a known next segment (drop last segment per series)
    out = out[out["target_next_state"].notna()].copy()

    # compatibility: expose a 'date' column (use segment start date)
    out["date"] = out["start_date"]

    # keys to keep for later joins/inspection
    keep_first = ["author","repo","segment_id","date","start_date","end_date","state_t","target_next_state"]
    # rest are features
    feat_cols = [c for c in out.columns if c not in keep_first + ["next_state"]]

    # reorder nicely: keys/labels first, then features
    out = out[keep_first + feat_cols].sort_values(["author","repo","segment_id"]).reset_index(drop=True)
    return out

def _roll_feat(seg: pandas.DataFrame, by_cols: List[str], col: str, k: int, fn: str) -> pandas.Series:
    """Groupwise rolling (on previous segments only)."""
    s = (seg
         .groupby(by_cols, observed=True)[col]
         .apply(lambda x: getattr(x.shift(1).rolling(k, min_periods=1), fn)()))
    # the groupby/apply preserves a hierarchical index; align back:
    return s.reset_index(level=by_cols, drop=True)

def tail_features(daily_with_segments, ks=(3,7,14)):
    d = daily_with_segments.sort_values(["author","repo","segment_id","date"]).copy()
    d["any_event"] = (
        d[["commits","pull_requests","issues","issues_comments","issues_events","pull_requests_comments"]]
        .sum(axis=1) > 0
    ).astype(int)

    out = d[["author","repo","segment_id"]].drop_duplicates().copy()
    g = d.groupby(["author","repo","segment_id"], observed=True)

    for k in ks:
        tail_ev = g["any_event"].apply(lambda s: s.tail(k).mean()).reset_index(level=[0,1], drop=True)
        tail_cd = g["coding_day"].apply(lambda s: s.tail(k).mean()).reset_index(level=[0,1], drop=True)
        out[f"tail{k}_event_rate"] = tail_ev.values
        out[f"tail{k}_coding_rate"] = tail_cd.values
    return out

def make_splits(df: pandas.DataFrame, cfg: SplitConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns boolean masks (train_mask, val_mask) aligned to df rows.
    """
    idx = np.arange(len(df))
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask   = np.zeros(len(df), dtype=bool)

    if cfg.strategy == "time_by_repo":
        for (a, r), g in df.groupby(["author","repo"], sort=False):
            if g.empty: continue
            last_date = g["date"].max()
            cutoff = last_date - pandas.Timedelta(days=cfg.val_months*30)
            sel_train = g["date"] <= cutoff
            sel_val   = g["date"] >  cutoff
            train_mask[g.index] = sel_train.values
            val_mask[g.index]   = sel_val.values

    elif cfg.strategy == "time_global":
        cutoff = df["date"].max() - pandas.Timedelta(days=cfg.val_months*30)
        train_mask = (df["date"] <= cutoff).values
        val_mask   = ~train_mask

    elif cfg.strategy == "holdout_authors":
        val_mask   = df["author"].isin(["jangorecki"]).values
        train_mask = ~val_mask
    else:
        raise ValueError(f"Unknown split strategy: {cfg.strategy}")

    # Ensure both non-empty
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("Empty train or val split; adjust SplitConfig.")

    return train_mask, val_mask

def _normalize_state_series(s: pandas.Series) -> pandas.Series:
    # make robust to stray variants like "Non-Coding", "gone ", etc.
    norm = (s.astype(str)
              .str.strip()
              .str.upper()
              .str.replace("-", "_"))
    # map anything unexpected to NaN so we drop it later
    norm = norm.where(norm.isin(STATE_ORDER), other=np.nan)
    return norm

def add_predictors_segments(segments: pandas.DataFrame) -> pandas.DataFrame:
    """
    Build segment-level features for next-segment-state prediction.
    Input: segments from segmentize_timeline()
    Output: 1 row per segment (excluding final segments without a next_state).
    """
    seg = segments.copy()

    # normalize states
    seg["state_curr"] = _normalize_state_series(seg["state_curr"])
    if "next_state" in seg.columns:
        seg["next_state"] = _normalize_state_series(seg["next_state"])

    # drop rows without a defined next segment (last segment in a series)
    if "next_state" in seg.columns:
        seg = seg[seg["next_state"].notna()].copy()
    else:
        raise ValueError("segments must include 'next_state' to define the target.")

    # densities/proportions
    eps = 1e-9
    seg["avg_commits_per_day"]     = seg["commits_sum"] / (seg["seg_len"] + eps)
    seg["avg_prs_per_day"]         = seg["pr_sum"] / (seg["seg_len"] + eps)
    seg["avg_issues_per_day"]      = seg["issues_sum"] / (seg["seg_len"] + eps)
    seg["avg_comments_per_day"]    = seg["issues_comments_sum"] / (seg["seg_len"] + eps)
    seg["avg_events_per_day"]      = seg["issues_events_sum"] / (seg["seg_len"] + eps)
    seg["avg_pr_comments_per_day"] = seg["pr_comments_sum"] / (seg["seg_len"] + eps)

    seg["pct_coding_days"]    = seg["coding_days"] / (seg["seg_len"] + eps)
    seg["pct_noncoding_days"] = seg["nc_days"]     / (seg["seg_len"] + eps)
    seg["pct_break_days"]     = seg["break_days"]  / (seg["seg_len"] + eps)

    # calendar at segment start
    seg["start_date"] = pandas.to_datetime(seg["start_date"], utc=True, errors="coerce")
    seg["start_dow"]   = seg["start_date"].dt.dayofweek
    seg["start_month"] = seg["start_date"].dt.month
    seg["start_qtr"]   = ((seg["start_month"] - 1) // 3 + 1).astype(int)

    # previous context
    if "prev_state" not in seg.columns:
        seg["prev_state"] = np.nan
    if "prev_seg_len" not in seg.columns:
        seg["prev_seg_len"] = np.nan

    seg["prev_curr_pair"] = (seg["prev_state"].astype(str)
                             + "→" + seg["state_curr"].astype(str))

    # labels + compatibility
    seg.rename(columns={"state_curr": "state_t"}, inplace=True)
    seg["target_next_state"] = seg["next_state"].astype(str)
    seg["date"] = seg["start_date"]  # keep a 'date' column for split code

    # ensure keys exist
    needed = ["author","repo","segment_id","state_t","target_next_state","date"]
    missing = [c for c in needed if c not in seg.columns]
    if missing:
        raise ValueError(f"segments missing required columns: {missing}")

    # sort and return
    seg = seg.sort_values(["author","repo","segment_id"]).reset_index(drop=True)
    return seg

    """
    Build X (features) and y (labels) for NEXT-SEGMENT-STATE classification.
    Returns (X_df, y_np, feature_cols).
    """
    df = df.copy()

    # categorical features to one-hot
    cat_cols = ["state_t", "prev_state", "prev_curr_pair"]  # optional: "author","repo"
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "∅"

    X_cat = pandas.get_dummies(df[cat_cols].astype("category"), prefix=cat_cols, dummy_na=False)

    # numeric features (ensure present; fill missing with 0)
    num_cols = [
        "seg_len",
        "avg_commits_per_day","avg_prs_per_day","avg_issues_per_day",
        "avg_comments_per_day","avg_events_per_day","avg_pr_comments_per_day",
        "pct_coding_days","pct_noncoding_days","pct_break_days",
        "start_dow","start_month","start_qtr",
        "prev_seg_len", "tail"
        # rolling segment stats if you added them (safe if absent):
        "roll3_seg_len_mean","roll3_seg_len_std",
        "roll3_commits_mean","roll3_nc_days_mean","roll3_break_days_mean",
        "roll5_seg_len_mean","roll5_seg_len_std",
        "roll5_commits_mean","roll5_nc_days_mean","roll5_break_days_mean",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
    X_num = df[num_cols].astype(float)

    # concat features
    X = pandas.concat([X_num, X_cat], axis=1)
    feature_cols = list(X.columns)

    # labels — map to 0..K-1 and DROP any -1 rows
    cat = pandas.Categorical(df["target_next_state"], categories=STATE_ORDER, ordered=True)
    y = cat.codes.astype(np.int64)

    valid = y >= 0
    if not valid.all():
        X = X.loc[valid].reset_index(drop=True)
        y = y[valid]
    return X, y, feature_cols, valid

    K = len(STATE_ORDER)
    y_tr = np.asarray(y_tr, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    class_counts = np.bincount(y_tr, minlength=K)
    total = class_counts.sum()
    weights = {i: (total / (K * max(1, int(c)))) for i, c in enumerate(class_counts)}
    sample_w = np.vectorize(weights.get)(y_tr)

    model = XGBClassifier(
        n_estimators=2000,              # allow ES to pick best
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=K,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,                      # use all cores
        eval_metric=["mlogloss","merror"]
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_w,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def attach_predictions_to_segments(
    segments_raw: pandas.DataFrame,      # output of segmentize_timeline(...)[1]
    features_df: pandas.DataFrame,       # output of add_predictors(segments or daily)
    X,                                # features matrix aligned to features_df
    model,                            # trained classifier
    val_mask,                         # boolean mask over features_df for held-out rows
    state_order: List[str],           # ["ACTIVE","NON_CODING","INACTIVE","GONE"]
    out_csv_path: str,
    *,
    filter_to_val_authors: bool = True,
    only_author: Optional[str] = None # e.g., "jangorecki"
) -> pandas.DataFrame:
    """
    Merge next-state predictions back onto the *segment table* (not daily).
    Keeps one row per (author,repo,segment_id). Filters to held-out author(s) by default.
    """

    # --- sanity: must have segment keys ---
    required_keys = {"author","repo","segment_id"}
    if not required_keys.issubset(segments_raw.columns):
        raise ValueError(f"segments_raw missing keys: {required_keys - set(segments_raw.columns)}")
    if not required_keys.issubset(features_df.columns):
        raise ValueError(f"features_df missing keys: {required_keys - set(features_df.columns)}")

    # --- who is held out? ---
    heldout_authors = (
        features_df.loc[val_mask, "author"]
        .astype(str).str.strip().dropna().unique().tolist()
    )
    if only_author:
        heldout_authors = [only_author]

    # --- predict on held-out rows in *features_df* ---
    X_val = X[val_mask]
    yhat = model.predict(X_val)
    try:
        proba = model.predict_proba(X_val)
    except Exception:
        proba = None

    preds = features_df.loc[val_mask, ["author","repo","segment_id","state_t","target_next_state"]].copy()
    preds.rename(columns={"target_next_state": "actual_next_state"}, inplace=True)

    X_val = X[val_mask]
    yhat = model.predict(X_val)
    proba = model.predict_proba(X_val)

    # blend with prior from TRAIN rows only
    P_prior = transition_prior(features_df, train_mask=~val_mask, states=state_order, alpha=1.0)
    lam = 0.35   # small prior weight (tune 0.2–0.5)
    rows = []
    for st, p in zip(preds["state_t"].to_numpy(), proba):
        prior = P_prior.loc[st].to_numpy()
        rows.append((1-lam)*p + lam*prior)
    proba = np.vstack(rows)

    # write calibrated columns as before
    for i, name in enumerate(state_order):
        preds[f"p_{name.lower()}"] = proba[:, i]
    preds["pred_next_state"] = pandas.Categorical.from_codes(
        yhat, categories=state_order, ordered=True
    ).astype(str)
    preds["pred_confidence"] = proba.max(axis=1)

    # --- normalize join keys & filter left table to held-out authors if requested ---
    def _norm(df: pandas.DataFrame) -> pandas.DataFrame:
        out = df.copy()
        out["author"] = out["author"].astype(str).str.strip()
        out["repo"]   = out["repo"].astype(str).str.strip()
        out["segment_id"] = pandas.to_numeric(out["segment_id"], errors="coerce").astype("Int64")
        return out

    left  = _norm(segments_raw)
    right = _norm(preds)

    if filter_to_val_authors:
        left = left[left["author"].isin(heldout_authors)].copy()

    # --- merge segment-level predictions back onto the raw segments ---
    keep_cols = ["author","repo","segment_id","pred_next_state","actual_next_state"]
    if proba is not None:
        keep_cols += [f"p_{n.lower()}" for n in state_order] + ["pred_confidence"]

    right = right[keep_cols].drop_duplicates(subset=["author","repo","segment_id"])

    seg_out = left.merge(
        right, on=["author","repo","segment_id"], how="left", validate="one_to_one"
    )

    # --- correctness (NaN for last segments that naturally have no next state) ---
    seg_out["correct_next"] = (
        seg_out["pred_next_state"].notna() &
        (seg_out["pred_next_state"] == seg_out["actual_next_state"])
    )

    # --- nice column order: keys, timing, current-state, then predictions ---
    front = [
        "author","repo","segment_id",
        "state_curr","start_date","end_date","seg_len",
        "commits_sum","pr_sum","issues_sum","issues_comments_sum","issues_events_sum","pr_comments_sum",
        "coding_days","nc_days","break_days",
        # prev context if present
        *([c for c in ["prev_state","prev_seg_len"] if c in seg_out.columns]),
        # predictions
        "pred_next_state","actual_next_state","correct_next"
    ]
    prob_cols = [c for c in seg_out.columns if c.startswith("p_")] + (["pred_confidence"] if "pred_confidence" in seg_out.columns else [])
    # keep any other columns at the end
    remaining = [c for c in seg_out.columns if c not in set(front + prob_cols)]
    ordered_cols = [c for c in front if c in seg_out.columns] + prob_cols + remaining

    seg_out = seg_out[ordered_cols].sort_values(["author","repo","segment_id"]).reset_index(drop=True)

    seg_out.to_csv(out_csv_path, index=False)
    view_df(seg_out, "seg_out")                 # optional

    return seg_out

def transition_prior(df, train_mask, states=STATE_ORDER, alpha=1.0):
    # counts of state_t -> target_next_state over TRAIN rows only
    t = (df.loc[train_mask, ["state_t","target_next_state"]]
           .dropna()
           .value_counts()
           .rename("n")
           .reset_index())
    M = pandas.DataFrame(alpha, index=states, columns=states, dtype=float)
    for _, r in t.iterrows():
        M.loc[str(r["state_t"]), str(r["target_next_state"])] += r["n"]
    P = M.div(M.sum(axis=1), axis=0)   # row-normalize
    return P
    """
    proba: shape (n_samples, n_classes) from model.predict_proba
    model_classes: labels in the same order as proba columns (e.g., ["ACTIVE", "GONE", ...])
    state_order: desired order for probability columns
    """
    # map class -> column index
    idx = {cls: i for i, cls in enumerate(model_classes)}
    cols = {}
    for s in state_order:
        if s in idx:
            cols[f"p_{s.lower()}"] = proba[:, idx[s]]
        else:
            # class missing in the model (unlikely) -> zeros
            cols[f"p_{s.lower()}"] = np.zeros((proba.shape[0],), dtype=float)
    return pandas.DataFrame(cols)

def _predict_labels_from_proba(proba: np.ndarray, model_classes: List[str]) -> np.ndarray:
    """Return predicted class labels using argmax over proba."""
    best = proba.argmax(axis=1)
    return np.array([model_classes[i] for i in best])

def _ensure_feature_matrix(df: pandas.DataFrame, required_cols: list[str]) -> Tuple[pandas.DataFrame, list[str]]:
    """
    Make df have exactly the columns the model expects:
      - Add any missing columns filled with 0
      - Drop any extra columns not expected
      - Preserve the order of required_cols
    """
    X = df.copy()
    missing = [c for c in required_cols if c not in X.columns]
    for c in missing:
        X[c] = 0
    # drop extras
    X = X[required_cols]
    return X, required_cols


#--------------------------
# Driver functions
#--------------------------

def predict_state(authors: list[str], repo, split_cfg: SplitConfig = SplitConfig()):
    STATE_ORDER = ["ACTIVE", "NON_CODING", "INACTIVE", "GONE"]

    # 1) LOAD
    raw = users_activity_concat(authors, repo)

    # 1a) SEGMENTIZE
    daily_with_segments, segments = segmentize_timeline(raw, state_order=STATE_ORDER)

    tf = tail_features(daily_with_segments)
    segments = segments.merge(tf, on=["author","repo","segment_id"], how="left")

    # 2) FEATURES (segment-level)
    data = add_predictors_segments(segments)  # << updated

    
    #load pretrained model
    if not Path(cfg.model_path).exists():
        raise FileNotFoundError(f"Pretrained model not found at: {cfg.model_path}")
    else:
        model = joblib.load(cfg.model_path)

    model_classes = list(getattr(model, "classes_", STATE_ORDER))


    X = _ensure_feature_matrix(data, FEATURE_COLS)


    # ---------- 4) PREDICT ----------
    # predict_proba should exist for XGBClassifier / sklearn-compatible classifiers
    proba = model.predict_proba(X.values)
    pred_labels = _predict_labels_from_proba(proba, model_classes)


    # 5) ATTACH TO SEGMENTS (held-out author only)
    out_csv = r"C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\PredictionModel\predictions_segments.csv"
    out = attach_predictions_to_segments(
        segments_raw=segments,
        features_df=data,
        X=X.values,                     # XGB accepts numpy arrays; you can also pass DataFrame
        model=model,
        val_mask=val_mask,
        state_order=STATE_ORDER,
        out_csv_path=out_csv,
        filter_to_val_authors=True,
        only_author="jangorecki"
    )
    out["correct_next"].value_counts()
    #count wrong state predictions and which state was predicted wrong
    wrong_predictions = out[out["correct_next"] == False]
    wrong_counts = wrong_predictions["pred_next_state"].value_counts()
    correct_predictions = out[out["correct_next"] == True]
    correct_counts = correct_predictions["pred_next_state"].value_counts()
    print("Correct state predictions:", correct_counts)
    print("Wrong state predictions:", wrong_counts)


    y_true = pandas.Categorical(out["actual_next_state"], categories=STATE_ORDER, ordered=True).codes
    y_pred = pandas.Categorical(out["pred_next_state"],  categories=STATE_ORDER, ordered=True).codes

    print(classification_report(y_true, y_pred, target_names=STATE_ORDER, zero_division=0))
    print(pandas.DataFrame(confusion_matrix(y_true, y_pred),
                        index=[f"true_{s}" for s in STATE_ORDER],
                        columns=[f"pred_{s}" for s in STATE_ORDER]))

    # top-2 accuracy (uses your p_* columns)
    P = out[[f"p_{s.lower()}" for s in STATE_ORDER]].to_numpy()
    top2 = np.argsort(P, axis=1)[:, -2:]
    top2_hit = np.array([y_true[i] in top2[i] for i in range(len(y_true))])
    print("Top-1 acc:", (y_true==y_pred).mean(), " | Top-2 acc:", top2_hit.mean())

    #rearrange rows
    #i want the order to be segment_id, author, state_curr, pred_next_state,	actual_next_state,	correct_next
    out = out[["segment_id", "author", "state_curr", "pred_next_state", "actual_next_state", "correct_next", "repo", "start_date", "end_date", "seg_len", "commits_sum", "pr_sum", "issues_sum", "issues_comments_sum", "issues_events_sum", "pr_comments_sum", "coding_days", "nc_days", "break_days", "prev_state", "prev_seg_len", "p_active", "p_non_coding", "p_inactive", "p_gone", "pred_confidence", "next_state"]]

    view_df(out, "seg_out")
    out.to_csv(r"C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\PredictionModel\predictions_segments.csv", index=False)


    return out

def collect_repo(repo, since_days):

    organizationFolder = cfg.main_folder + "/" + repo

    # collect data from the past few months
    commits = main_commit_extraction(repo, since_days=since_days)

    # find key developers
    commit_authors = get_commit_based_core_devs(commits.to_dict(orient='records'))

    pauses = write_pauses_table(commits, organizationFolder + "/pauses_commits.csv", commit_authors, user_col = "author_id", date_col="created_at")
    return commit_authors, pauses

def clean_data(repo, authors, pauses):

    pauses_list = pauses.values.tolist()
    print(f"{len(authors)} Developers inactivity periods identified")

    input_folder = cfg.temp_data_folder + '/' + repo
    output_folder = cfg.temp_data_folder + '/' + repo + "/Results"
    os.makedirs(output_folder, exist_ok=True)
    tf_devs= []

    for dev in authors:
        tf_devs.append(dev)
        timeline_folder = output_folder + '/' + cfg.timeline_folder_name
        os.makedirs(timeline_folder, exist_ok=True)
    
        timeline_path = Path(timeline_folder) / f"{dev}_timeline.csv"

        if timeline_path.is_file():
            user_timeline = pandas.read_csv(timeline_path, sep=cfg.CSV_separator, index_col=0)
        else:
            user_timeline = get_timeline(input_folder, dev)

            #transpose the user_timeline making the frist row the first column
            user_timeline.to_csv(timeline_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n')

        print(dev)
    
        #make a break folder 
        breaks_folder = cfg.main_folder + '/' + repo + "/Breaks"
        os.makedirs(breaks_folder, exist_ok=True)

        breaks_path =  Path(breaks_folder)/  f"{dev}_breaks.csv"              

        if breaks_path.is_file():
            breaks_df = pandas.read_csv(breaks_path, sep=cfg.CSV_separator, index_col=0)  

        else:
            breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])
            breaks_df = identifyBreaks(pauses_list, dev=dev, window=cfg.sliding_window_size, shift=cfg.shift, debug_folder=output_folder)
            breaks_df.to_csv(breaks_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, index=False, lineterminator="\n")
                    
        #add label_timeline
        user_timeline = label_timeline(user_timeline, breaks_df)

        out_csv = Path(output_folder) / f"{dev}_labeled_timeline.csv"
        user_timeline.to_csv(out_csv,
                                sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n', index_label='date')

        tf_devs_df = pandas.DataFrame(tf_devs, columns=["developer"])
        tf_devs_df.to_csv(Path(output_folder) / "tf_devs.csv", sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n')

    return tf_devs, user_timeline

#----------------------------------------
#app
#----------------------------------------
st.set_page_config(page_title="User Side Devloper Inactivity Predictions Demo", layout="wide")
st.title("User Side Devloper Inactivity Predictions Demo")


if st.button("Rescan folders"):
    st.cache_data.clear()

# make a  area to type out hwat repo you want


orgs = list_orgs()
if not orgs:
    st.warning("No organizations found under your Organizations folder.")
    st.stop()
user_org = st.selectbox("Select an organization", orgs)

user_org_input = st.text_input("Or enter an organization name (priority)")


# 2) Repository dropdown (dependent on org)
repos = list_repos_for(user_org)
if not repos:
    st.warning(f"No repositories found in organization: {user_org}.")
    st.stop()

user_repo = st.selectbox("Select a repository from organization ", repos)


# Combined key and resolved paths for downstream steps

user_repo_input = st.text_input("Or enter a repository name (priority)")

if user_org_input:
    if user_repo_input:
        # If user enters 'org/repo', use it directly
        repo_key = f"{user_org_input}/{user_repo_input}"
    else:
        # If only repo name is entered, combine with selected org
        repo_key = f"{user_org_input}/{user_repo}"
else:
    repo_key = f"{user_org}/{user_repo}"


st.caption(f"Selected repo: **{repo_key}**")
# (Later buttons can use `repo_key` and `paths`, e.g., Update, Label, Predict)

st.divider()

if st.button("Collect Data Demo"):
        commit_authors, pauses = collect_repo(repo_key, since_days=90)
        st.write("pauses collected:")
        st.write(pauses)
        tf_devs, user_timeline = clean_data(repo_key, commit_authors, pauses)
        st.session_state["data"] = user_timeline            # <-- persist
        st.session_state["authors"] = tf_devs               # <-- persist
        st.write(f"clean_data from {tf_devs[-1]}:")
        st.write(user_timeline)


st.divider()
#this part can only show up after labeling

st.subheader("Predict")

if st.button("Predict next states"):
    authors = st.session_state.get("authors")
    # 3) Predict

    out = predict_state(authors, repo_key, split_cfg=SplitConfig())