#   conda activate osslab
#   python ProjectHealthMetrics.py



### IMPORT EXCEPTION MODULES
import uuid
from requests.exceptions import Timeout

### IMPORT SYSTEM MODULES
from github import Github
import os, logging, pandas, csv, tempfile, shutil, functools
from datetime import datetime, timezone
from tqdm import tqdm, tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal, threading
import json
from contextlib import contextmanager
from dateutil import tz as _tz
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

from dataclasses import dataclass
import KnowledgeDistribution as kd
warnings.filterwarnings("ignore")
from git import Repo, exc as git_exc

def project_health_analysis(tables, dev_leaving, as_of_date, window_days):

    #Time to First Response (Issues & PRs)
    #Median + 90th percentile
    #Count of people doing reviews in last N days
    #Open issues
    #Open PRs
    #PR merge time
    #Issue close time
    #Median or P90 (don’t need both)
    
    #Active contributors
    #Commit activity
    #Time-to-first response
    #Time-to-close
    #Contributor churn
    #Commit concentration
    #Open PR age
    #Review backlog
    #Review latency
    #Patch acceptance
    #Maintainer overload
    #File ownership gaps
    #Bus factor
    #Review enforcement
    #Activity recency

    return None


def user_health_analysis(tables, dev_leaving, as_of_date, window_days):
    user_labeled_timeline = tables["user_labeled_timeline"].copy()
    commits        = tables["commits"].copy()
    issues         = tables["issues"].copy()
    issue_activity = tables["issue_activity"].copy()
    prs_repo       = tables["prs_repo"].copy()
    prs_comments   = tables["prs_comments"].copy()

    # -----------------------------
    # 0) Resolve dev identifier
    # dev_leaving can be "author_login|foo" etc, or a raw author_id
    # -----------------------------
    if isinstance(dev_leaving, str) and "|" in dev_leaving:
        left, right = dev_leaving.split("|", 1)
        if left in {"author_id", "author_login", "author_name", "author_email"}:
            dev_column, dev_value = left, right
        else:
            dev_column, dev_value = "author_id", dev_leaving
    else:
        dev_column, dev_value = "author_id", dev_leaving

    # -----------------------------
    # 1) Datetime normalization (FIXES tz-naive vs tz-aware crashes)
    # -----------------------------
    def to_utc(s):
        return pandas.to_datetime(s, errors="coerce", utc=True)

    if "date" in user_labeled_timeline.columns:
        user_labeled_timeline["date"] = to_utc(user_labeled_timeline["date"])

    for df, cols in [
        (commits,        ["created_at"]),
        (issues,         ["created_at", "closed_at"]),
        (issue_activity, ["created_at"]),
        (prs_repo,       ["created_at", "closed_at", "merged_at"]),
        (prs_comments,   ["created_at"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = to_utc(df[c])

    as_of = to_utc(as_of_date)
    if pandas.isna(as_of):
        as_of = user_labeled_timeline["date"].max()

    # -----------------------------
    # 2) Filter timeline for this user
    # (support both dev_leaving and parsed dev_value)
    # -----------------------------
    if "dev" in user_labeled_timeline.columns:
        tl_user = user_labeled_timeline[
            (user_labeled_timeline["dev"] == dev_leaving) |
            (user_labeled_timeline["dev"] == dev_value)
        ].copy()
    else:
        tl_user = user_labeled_timeline.copy()

    tl_user = tl_user.sort_values("date")

    # If caller passes a future as_of_date, clip to what we actually have
    if not tl_user.empty and not pandas.isna(tl_user["date"].max()) and as_of > tl_user["date"].max():
        as_of = tl_user["date"].max()

    window_start = as_of - pandas.Timedelta(days=int(window_days))

    tl_win = tl_user[(tl_user["date"] >= window_start) & (tl_user["date"] <= as_of)]

    user_commits = tl_win["commits"].sum() if "commits" in tl_win.columns else 0
    user_prs     = tl_win["prs"].sum()     if "prs"     in tl_win.columns else 0
    user_issues  = tl_win["issues"].sum()  if "issues"  in tl_win.columns else 0

    # -----------------------------
    # 3) Resolve a login for response-time work (issue_activity uses created_by=login)
    # -----------------------------
    def most_common_login(df, id_col, id_val):
        if id_col in df.columns and "author_login" in df.columns:
            s = df.loc[df[id_col] == id_val, "author_login"].dropna().astype(str)
            if len(s) > 0:
                return s.value_counts().idxmax()
        return None

    if dev_column == "author_login":
        dev_login = str(dev_value)
    else:
        dev_login = (
            most_common_login(commits, dev_column, dev_value) or
            most_common_login(issues, dev_column, dev_value) or
            most_common_login(prs_repo, dev_column, dev_value)
        )

    response_times_hours = []
    first_responder_count = 0

    # -----------------------------
    # 4) Issue response times (first IssueComment by this user vs issue created_at)
    # -----------------------------
    if dev_login and "created_by" in issue_activity.columns and "issue_number" in issue_activity.columns:
        ia = issue_activity.copy()

        # treat only IssueComment as a "response"
        if "item_type" in ia.columns:
            ia = ia[ia["item_type"] == "IssueComment"]

        # user comments within window
        ia_user = ia[
            (ia["created_by"] == dev_login) &
            (ia["created_at"] >= window_start) &
            (ia["created_at"] <= as_of)
        ]

        # build a quick lookup for issues created_at / author_login
        if "issue_number" in issues.columns and "created_at" in issues.columns:
            issues_lookup = issues.set_index("issue_number", drop=False)

            for issue_num, grp in ia_user.groupby("issue_number"):
                if issue_num not in issues_lookup.index:
                    continue

                issue_created = issues_lookup.loc[issue_num, "created_at"]
                if pandas.isna(issue_created):
                    continue

                first_user_comment = grp["created_at"].min()
                if pandas.isna(first_user_comment) or first_user_comment < issue_created:
                    continue

                response_times_hours.append((first_user_comment - issue_created).total_seconds() / 3600)

                # first responder (first non-author IssueComment)
                global_comments = ia[ia["issue_number"] == issue_num]
                issue_author = issues_lookup.loc[issue_num, "author_login"] if "author_login" in issues_lookup.columns else None
                if isinstance(issue_author, str) and len(issue_author) > 0:
                    global_comments = global_comments[global_comments["created_by"] != issue_author]

                global_first = global_comments["created_at"].min()
                if not pandas.isna(global_first) and abs((first_user_comment - global_first).total_seconds()) <= 1:
                    first_responder_count += 1

    # -----------------------------
    # 5) PR response times (first comment/review by this user vs PR created_at)
    # -----------------------------
    if dev_login and "PR_id" in prs_comments.columns and "created_at" in prs_comments.columns:
        pc = prs_comments.copy()

        # actor column differs for comment vs review rows in your extractor
        if "created_by" in pc.columns:
            actor = pc["created_by"].copy()
            if "author_login" in pc.columns:
                actor = actor.fillna(pc["author_login"])
        elif "author_login" in pc.columns:
            actor = pc["author_login"].copy()
        else:
            actor = pandas.Series([None] * len(pc), index=pc.index)

        pc["_actor"] = actor.astype("string")

        pc_user = pc[
            (pc["_actor"] == dev_login) &
            (pc["created_at"] >= window_start) &
            (pc["created_at"] <= as_of)
        ]

        if "PR_id" in prs_repo.columns and "created_at" in prs_repo.columns:
            prs_lookup = prs_repo.set_index("PR_id", drop=False)

            for pr_id, grp in pc_user.groupby("PR_id"):
                if pr_id not in prs_lookup.index:
                    continue

                pr_created = prs_lookup.loc[pr_id, "created_at"]
                if pandas.isna(pr_created):
                    continue

                first_user_action = grp["created_at"].min()
                if pandas.isna(first_user_action) or first_user_action < pr_created:
                    continue

                response_times_hours.append((first_user_action - pr_created).total_seconds() / 3600)

                # first responder (first non-author action)
                global_actions = pc[pc["PR_id"] == pr_id]
                pr_author = prs_lookup.loc[pr_id, "author_login"] if "author_login" in prs_lookup.columns else None
                if isinstance(pr_author, str) and len(pr_author) > 0:
                    global_actions = global_actions[global_actions["_actor"] != pr_author]

                global_first = global_actions["created_at"].min()
                if not pandas.isna(global_first) and abs((first_user_action - global_first).total_seconds()) <= 1:
                    first_responder_count += 1

    out = pandas.DataFrame({
        "user_commits": [user_commits],
        "user_prs": [user_prs],
        "user_issues": [user_issues],
        "median_response_time_hours": [np.median(response_times_hours) if len(response_times_hours) else None],
        "first_responder_count": [first_responder_count],
    })

    print("\n\nuser_health_analysis output\n")
    print(out)

    return out

def knowledge_distribution_analysis(tables, dev_leaving):
    """
    Given precomputed DOE and author map, summarize how much knowledge
    a developer 'dev_leaving' holds in this repo.

    Returns:
        dev_summary_df: 1-row DataFrame with overall stats for this dev
        folder_summary_df: DataFrame with one row per folder the dev owns files in
    """

    df_DOE = tables['df_DOE']          # DOE metrics for each file/developer
    author_map = tables['author_map']  # dict: dev -> set(file_paths)

    # ------------------------------------------------------------
    # Collect all developers listed in author_map
    # ------------------------------------------------------------
    devs_in_author_map = set(author_map.keys())


    # Files the developer OWNS (max DOE)  — comes directly from author_map
    owned_files = set(author_map.get(dev_leaving, set()))


    single_contributor_files = set()
    single_expert_files = set()
    multi_expert_files = set()

    alpha = 0.8  # threshold for "co-expert" DOE closeness

    for f in owned_files:
        doe_rows = df_DOE[df_DOE["file_path"] == f][["developer", "DOE"]]
        if doe_rows.empty:
            continue

        authors = doe_rows['developer'].unique()

        # Case 1: only contributor to this file
        if len(authors) == 1:
            single_contributor_files.add(f)
            # Treat "only contributor" as a special case of single-expert
            single_expert_files.add(f)
            continue

        # Case 2/3: multiple contributors. Check if dev_leaving has co-experts.
        dev_row = doe_rows[doe_rows['developer'] == dev_leaving]

        dev_doe = dev_row['DOE'].iloc[0]

        co_experts = doe_rows[
            (doe_rows['developer'] != dev_leaving) &
            (doe_rows['DOE'] >= alpha * dev_doe)
        ]

        if len(co_experts) > 0:
            multi_expert_files.add(f)
        else:
            single_expert_files.add(f)


    high_doe_files = set(
        df_DOE[
            (df_DOE['developer'] == dev_leaving) &
            (df_DOE['DOE'] >= 3)
        ]['file_path'].tolist()
    )

    # 4. Build folder list from owned files
    #    (we only consider folders where the dev owns at least one file)
    folders = set()
    for f in owned_files:
        p = Path(f).parent
        # Stop when we reach '.' or root-equivalent
        while str(p) not in ('.', ''):
            folders.add(str(p))
            p = p.parent

    list_of_folders = list(folders)

    # 5. Folder-level stats + Truck Factor per folder
    folder_stats = {}

    for folder in list_of_folders:
        # Files in this folder (any author)
        files_in_folder = set(
            df_DOE[df_DOE['file_path'].str.startswith(folder)]['file_path'].unique()
        )
        total_files_in_folder = len(files_in_folder)
        if total_files_in_folder <= 1:
            continue  # nothing to do

        # Files in this folder owned by dev_leaving
        owned_in_folder = files_in_folder & owned_files
        single_contributor_in_folder = owned_in_folder & single_contributor_files
        single_expert_in_folder = owned_in_folder & single_expert_files
        multi_expert_in_folder = owned_in_folder & multi_expert_files
        high_doe_in_folder = files_in_folder & high_doe_files

        # Truck factor for the folder (treat as mini-project)
        df_DOE_folder = df_DOE[df_DOE['file_path'].str.startswith(folder)]
        folder_authors_map = kd.build_authors_map_from_doe(df_DOE_folder)
        tf, tf_devs = kd.runTruckFactor(df_DOE_folder, folder_authors_map)

        n_only_contributor = len(single_contributor_in_folder)
        n_single_expert = len(single_expert_in_folder)
        n_multi_expert = len(multi_expert_in_folder)

        # Simple risk score as you proposed
        risk_score = 3 * n_only_contributor + 2 * n_single_expert + 1 * n_multi_expert

        folder_stats[folder] = {
            'total_files': total_files_in_folder,
            'n_owned': len(owned_in_folder),
            'n_only_contributor': n_only_contributor,
            'n_single_expert': n_single_expert,
            'n_multi_expert': n_multi_expert,
            'n_high_doe': len(high_doe_in_folder),
            'tf': tf,
            'tf_devs': tf_devs,
            'is_truck_factor_dev': dev_leaving in tf_devs,
            'risk_score': risk_score,
        }

    # 6. Dev-level summary DataFrame
    n_owned_files = len(owned_files)
    n_only_contributor_files = len(single_contributor_files)
    n_single_expert_files = len(single_expert_files)
    n_multi_expert_files = len(multi_expert_files)
    n_high_doe_files = len(high_doe_files)

    dev_summary = [{
        "dev": dev_leaving,
        "n_owned_files": n_owned_files,
        "n_only_contributor_files": n_only_contributor_files,
        "n_single_expert_files": n_single_expert_files,
        "n_multi_expert_files": n_multi_expert_files,
        "n_high_doe_files": n_high_doe_files,
        # simple shares; you can drop these if you truly only want counts
        "share_only_contributor_owned": (
            n_only_contributor_files / n_owned_files if n_owned_files else 0.0
        ),
        "share_single_expert_owned": (
            n_single_expert_files / n_owned_files if n_owned_files else 0.0
        ),
        "share_multi_expert_owned": (
            n_multi_expert_files / n_owned_files if n_owned_files else 0.0
        ),
    }]

    dev_summary_df = pandas.DataFrame(dev_summary)

    # This is where you get:
    # "D is top DOE expert for {n_owned_files} files in this repo,
    #  including {n_single_expert_files} single-expert files."

    # 7. Folder-level summary DataFrame
    folder_records = []
    for folder, stats in folder_stats.items():
        total = stats['total_files'] or 1  # avoid division by zero

        folder_records.append({
            "dev": dev_leaving,
            "folder": folder,
            "total_files": stats['total_files'],
            "n_owned": stats['n_owned'],
            "n_only_contributor": stats['n_only_contributor'],
            "n_single_expert": stats['n_single_expert'],
            "n_multi_expert": stats['n_multi_expert'],
            "n_high_doe": stats['n_high_doe'],
            "risk_score": stats['risk_score'],
            "tf": stats['tf'],
            "is_truck_factor_dev": stats['is_truck_factor_dev'],
            "tf_devs": stats['tf_devs'],
            # simple ratios for "18 of 25 files" style statements
            "owned_share": stats['n_owned'] / total,
            "only_contributor_share": stats['n_only_contributor'] / total,
            "single_expert_share": stats['n_single_expert'] / total,
            "multi_expert_share": stats['n_multi_expert'] / total,
            "high_doe_share": stats['n_high_doe'] / total,
        })

    folder_summary_df = pandas.DataFrame(folder_records)

    # This is where you get sentences like:
    # "In src/core/, D is the only expert on 18 of 25 files (TF ≈ 1)."

    # You can sort folders by risk_score when you display:
    # folder_summary_df.sort_values("risk_score", ascending=False)

    # fore each devloper they own n files, they are single expert on 50%
    #D is top DOE expert for 137 files in this repo, including 54 single-expert files.
    #In src/core/, D is the only expert on 18 of 25 files (TF≈1)
    # for each folder we have N-total files, our user owns n of them, of thoes he owns s single-expert, m multi-expert, c only-contributor, he has contributed to h high-doe files
    # lastly he is/is not a truck factor dev for the folder with TF = t and TF_devs = [..]. these are the people to replace him/ reach out to if he does leave
    # we show a list of folders ordered by risk score = 3*s + 2*m + 1*c

    return dev_summary_df, folder_summary_df

def main(repo_full_name=None, dev_leaving=None, tables = None ):

    # Initialize
    org, repo = repo_full_name.split('/')
    organization_folder = Path(cfg.main_folder, org, repo)
    project_health_metrics_folder = Path(organization_folder, cfg.project_health_metrics_folder)

    os.makedirs(project_health_metrics_folder, exist_ok=True)

    # covert the time into 
    #nvalid comparison between dtype=datetime64[ns] and Timestamp
    as_of_date=pandas.Timestamp.now(tz=timezone.utc) - pandas.Timedelta(days=900)
    as_of_date = pandas.to_datetime(as_of_date).tz_convert(None)
    #
    out = user_health_analysis(tables, dev_leaving, as_of_date=as_of_date, window_days=360)

    # Step 1:
    # we need to
    dev_summary_df, folder_summary_df = knowledge_distribution_analysis(tables, dev_leaving)

    return dev_summary_df, folder_summary_df, out

if __name__ == "__main__":
    # let the user spesify the repo to process
    repo_full_name = "Rdatatable/data.table"  # Example: "organization/repo"

    main(repo_full_name)
