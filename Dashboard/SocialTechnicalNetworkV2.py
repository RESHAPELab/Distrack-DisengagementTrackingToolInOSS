#   conda activate CS485
#   python SocialTechnicalNetwork.py



### IMPORT EXCEPTION MODULES
import pandas as pd
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
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

### IMPORT CUSTOM MODULES
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import Settings as cfg
import Utilities as util
from pathlib import Path
import subprocess, tempfile, shutil, logging
from truckfactor.compute import main as compute_tf
import portalocker              # pip install portalocker
import warnings

from dataclasses import dataclass
import tempfile, webbrowser, json

warnings.filterwarnings("ignore")
from git import Repo, exc as git_exc
#_________________________________________________________
#
#Social Technical Networks
#
#_________________________________________________________

def _norm_time(s):
    return pandas.to_datetime(s, utc=True, errors="coerce")

def _resolve_login(row) -> str | None:
    """Return the best human-readable identifier for a row.
    Priority: author_login > author_id > author_name > author_email.
    Returns None if nothing usable is found.
    """
    for field in ("author_login", "author_id", "author_name", "author_email"):
        val = row.get(field) if hasattr(row, "get") else getattr(row, field, None)
        if val and str(val).strip() and str(val).strip().lower() not in ("nan", "na", "none", ""):
            return str(val).strip()
    return None

def _is_bot(login: str) -> bool:
    """Return True if the login looks like a bot account."""
    login_lower = login.lower()
    if login_lower.endswith("[bot]"):
        return True
    if any(p in login_lower for p in ("-bot", "_bot", "bot-", "bot_")):
        return True
    if login_lower == "bot":
        return True
    known_bots = {
        "dependabot", "renovate", "renovate-bot", "greenkeeper",
        "coveralls", "codecov", "snyk-bot", "stale",
        "imgbot", "allcontributors", "netlify", "vercel",
        "github-actions", "pull", "restyled-io",
    }
    return login_lower in known_bots

def view_df(df, name="DataFrame"):
    ''' Simple HTML table viewer for DataFrames with CSV download button '''
    import tempfile, webbrowser, json

    # Serialize CSV data to embed in the page
    csv_data = df.to_csv(index=False)
    csv_json = json.dumps(csv_data)          # safely escape for JS string
    safe_name = name.replace('"', '').replace("'", "")

    html = "\n".join([
        "<meta charset='utf-8'>",
        "<style>",
        "  body { font-family: system-ui, 'Segoe UI', Arial; padding: 16px; }",
        "  table { border-collapse: collapse; }",
        "  th, td { border: 1px solid #ddd; padding: 6px; }",
        "  th { position: sticky; top: 0; background: #fafafa; }",
        "  #dl-btn {",
        "    display: inline-flex; align-items: center; gap: 6px;",
        "    margin-bottom: 12px; padding: 7px 14px;",
        "    background: #2563eb; color: #fff; border: none;",
        "    border-radius: 6px; font-size: 14px; cursor: pointer;",
        "    text-decoration: none;",
        "  }",
        "  #dl-btn:hover { background: #1d4ed8; }",
        "</style>",
        f"<h3>{name}</h3>",
        f"<button id='dl-btn' onclick=\"downloadCSV()\">&#8681; Download CSV</button>",
        df.to_html(index=False, escape=False),
        "<script>",
        f"  const CSV_DATA = {csv_json};",
        f"  const FILE_NAME = '{safe_name}.csv';",
        "  function downloadCSV() {",
        "    const blob = new Blob([CSV_DATA], { type: 'text/csv;charset=utf-8;' });",
        "    const url  = URL.createObjectURL(blob);",
        "    const a    = document.createElement('a');",
        "    a.href     = url;",
        "    a.download = FILE_NAME;",
        "    a.click();",
        "    URL.revokeObjectURL(url);",
        "  }",
        "</script>",
    ])

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        f.write(html)
        webbrowser.open("file://" + f.name)


#_________________________
# Step 0: load data
#_________________________

_PR_REPO_OLD_COLS = ["repo", "created_at", "created_by", "PR_id", "state", "merged", "closed_at", "merged_at"]
_PR_REPO_NEW_COLS = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "PR_id", "state", "merged", "closed_at", "merged_at"]
_PR_COM_OLD_COLS  = ["repo", "created_at", "created_by", "PR_id", "comment_id", "event"]
_PR_COM_NEW_COLS  = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email", "PR_id", "comment_id", "event"]

# commit_list: old used 'created_by' (login) + author_name/email; new adds author_id + author_login
_COMMIT_OLD_COLS  = ["repo", "created_at", "created_by", "author_name", "author_email",
                     "sha", "filename_list", "fileschanged_count", "additions_sum", "deletions_sum"]
_COMMIT_NEW_COLS  = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email",
                     "sha", "filename_list", "fileschanged_count", "additions_sum", "deletions_sum"]


# issues: old had only created_by; new has full author breakdown
_ISSUE_OLD_COLS   = ["repo", "created_at", "created_by", "issue_number", "title",
                     "state", "closed_at", "labels", "assignees", "milestone"]
_ISSUE_NEW_COLS   = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email",
                     "issue_number", "title", "state", "closed_at", "labels", "assignees", "milestone"]

# issue_activity: old had created_at/created_by at the END; new has full author fields near the front
_ISSUE_ACT_OLD_COLS = ["repo", "issue_number", "activity_id", "item_type", "event", "body", "created_at", "created_by"]
_ISSUE_ACT_NEW_COLS = ["repo", "created_at", "author_id", "author_name", "author_login", "author_email",
                       "issue_number", "activity_id", "item_type", "event", "body"]

def load_users_activity(repo_full_name, organization_folder):

    # BUG 1 WAS: orgs_dir = organization_folder / "Organizations"
    # organization_folder is already .../Organizations/Rdatatable/data.table
    # so this was building .../Organizations/Rdatatable/data.table/Organizations
    # FIX: the files live directly inside organization_folder, no need to go up and back down

    target_files = {
        "issues":         cfg.issue_list_file_name,
        "issue_activity": cfg.issue_activity_file_name,
        "prs_repo":       cfg.PR_list_file_name,
        "prs_comments":   cfg.prs_comments_csv,
        "commit_list":    cfg.commit_list_file_name,
        "perfile_commit": cfg.per_file_commits_path
    }

    issues         = pandas.DataFrame()
    issue_activity = pandas.DataFrame()
    prs_repo       = pandas.DataFrame()
    prs_comments   = pandas.DataFrame()
    commits        = pandas.DataFrame()
    perfile_commits = pandas.DataFrame()

    # BUG 2 WAS: looping over organization_path.iterdir() (all repos)
    # then ignoring the repo_full_name you passed in
    # FIX: organization_folder IS the specific repo folder — just use it directly

    if not organization_folder.exists():
        print(f"Repo folder not found: {organization_folder}")
        return {}   # return empty dict, not 0 — so timeline() fails gracefully

    for file_key, file_name in target_files.items():
        file_path = organization_folder / file_name
        if file_path.exists():
            print(f"Found file: {file_path}")
            try:
                if file_key == "prs_repo":
                    df = _read_pr_csv_compat(file_path, _PR_REPO_OLD_COLS, _PR_REPO_NEW_COLS)
                    prs_repo = pandas.concat([prs_repo, df], ignore_index=True)
                elif file_key == "prs_comments":
                    df = _read_pr_csv_compat(file_path, _PR_COM_OLD_COLS, _PR_COM_NEW_COLS)
                    prs_comments = pandas.concat([prs_comments, df], ignore_index=True)
                elif file_key == "commit_list":
                    df = _read_pr_csv_compat(file_path, _COMMIT_OLD_COLS, _COMMIT_NEW_COLS)
                    commits = pandas.concat([commits, df], ignore_index=True)
                elif file_key == "issues":
                    df = _read_pr_csv_compat(file_path, _ISSUE_OLD_COLS, _ISSUE_NEW_COLS)
                    issues = pandas.concat([issues, df], ignore_index=True)
                elif file_key == "issue_activity":
                    df = _read_pr_csv_compat(file_path, _ISSUE_ACT_OLD_COLS, _ISSUE_ACT_NEW_COLS)
                    issue_activity = pandas.concat([issue_activity, df], ignore_index=True)
                else:
                    df = pandas.read_csv(file_path)
                    if file_key == "perfile_commit":
                        perfile_commits = pandas.concat([perfile_commits, df], ignore_index=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    return {
        "issues":         issues,
        "issue_activity": issue_activity,
        "prs_repo":       prs_repo,
        "prs_comments":   prs_comments,
        "commits":        commits,
        "perfile_commits": perfile_commits
    }

def _read_pr_csv_compat(file_path, old_cols, new_cols):
    """
    Read a PR CSV that may contain rows written under two different schemas.
    Old rows use 'created_by' for the author login; new rows use the full
    author_id / author_name / author_login / author_email breakdown.
    Returns a DataFrame with the new-schema columns.
    """
    # Some fields (issue body, filename_list) can exceed the default 131072-byte limit.
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2 ** 31 - 1)

    records = []
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header line (may be either schema)
            for row in reader:
                if len(row) == len(old_cols):
                    d = dict(zip(old_cols, row))
                    d['author_login'] = d.pop('created_by')
                    d.setdefault('author_id', None)
                    d.setdefault('author_name', None)
                    d.setdefault('author_email', None)
                    records.append({c: d.get(c) for c in new_cols})
                elif len(row) == len(new_cols):
                    records.append(dict(zip(new_cols, row)))
                # skip malformed rows silently
    except Exception as e:
        print(f"Warning: could not read {file_path}: {e}")
    return pandas.DataFrame(records, columns=new_cols) if records else pandas.DataFrame(columns=new_cols)

#_________________________
# Step 0: timeline
#_________________________

def timeline(tables):
    '''
    Create issue and PR timelines by merging main tables with their activity tables.
    We need to be ure about the author.
    '''

    issues = tables["issues"]
    issue_activity = tables["issue_activity"]
    prs_repo = tables["prs_repo"]
    prs_comments = tables["prs_comments"]

    issues['created_at_issue'] = issues['created_at']

    issues["created_by_issue"]           = issues.apply(_resolve_login, axis=1)
    issue_activity["created_at_activity"] = issue_activity["created_at"]
    issue_activity["created_by_activity"] = issue_activity.apply(_resolve_login, axis=1)
    # issue_activity: `author_*` columns hold the issue OPENER's identity, not the
    # commenter's. The actual performer of each activity lives in `created_by`.
    if "created_by" in issue_activity.columns:
        cb = issue_activity["created_by"].astype(str).str.strip()
        valid = cb.str.len() > 0
        valid &= ~cb.str.lower().isin(["nan", "na", "none", ""])
        issue_activity.loc[valid, "created_by_activity"] = cb[valid]
    prs_repo["created_at_pr"]             = prs_repo["created_at"]
    prs_repo["created_by_pr"]             = prs_repo.apply(_resolve_login, axis=1)
    prs_comments["created_at_comment"]    = prs_comments["created_at"]
    prs_comments["created_by_comment"]    = prs_comments.apply(_resolve_login, axis=1)

    issue_timeline = (
        issues.merge(issue_activity, on="issue_number", how="left")
              .sort_values(["issue_number", "created_at_activity"])
              .reset_index(drop=True)
    )

    pr_timeline = (
        prs_repo.merge(prs_comments, on="PR_id", how="left")
                .sort_values(["PR_id", "created_at_comment"])
                .reset_index(drop=True)
    )

    return issue_timeline, pr_timeline

#_________________________
#  Step 2: interaction network
#_________________________

def interaction_network(timeline, type, folder_path=None):
    # we need to create the interaction network from the issue and pr timelines
    # we have:
    # repo_pr,created_at_pr,created_by_pr,PR_id,state,merged,closed_at,merged_at,repo_comment,created_at_comment,created_by_comment,comment_id,event
    # Rdatatable/data.table,2020-01-24T12:11:47Z,sritchie73,4196,CLOSED,False,2025-06-27T17:25:04Z,,Rdatatable/data.table,2020-05-19T11:08:35Z,sritchie73,MDEyOklzc3VlQ29tbWVudDYzMDc1MDEyOQ==,comment
    #repo_issue,created_at_issue,created_by_issue,issue_number,title,state,closed_at,labels,assignees,milestone,repo_activity,activity_id,item_type,event,body,created_at_activity,created_by_activity
    #Rdatatable/data.table,2017-12-01T14:14:47Z,MichaelChirico,2505,fread support for parquet,CLOSED,2018-08-16T11:08:24Z,,,,Rdatatable/data.table,MDEyOklzc3VlQ29tbWVudDM0OTcyODc1Mw==,IssueComment,,"Yeah, it would be nice reading parquet into R without using Spark",2017-12-06T18:18:51Z,DavidArenburg
    # we need to group by PR_id and then create an interaction of a user with all other users who commented on the PR before them and are not themself

    interactions_df = []

    created_at = "created_at_comment" if type == "PR" else "created_at_activity"
    created_at_2 = "created_at_pr" if type == "PR" else "created_at_issue"
    created_by = "created_by_comment" if type == "PR" else "created_by_activity"
    id_name = "PR_id" if type == "PR" else "issue_number"

    timeline[created_at] = _norm_time(timeline[created_at])

    # Loop 1: connect each activity/comment author back to the item creator.
    # This is the primary source of interactions for issues (where issue_activity
    # only stores one user's activities per issue, so Loop 2 produces nothing).
    created_by_creator = "created_by_pr" if type == "PR" else "created_by_issue"
    for i, row in timeline.iterrows():
        user_1 = row[created_by]
        user_2 = row[created_by_creator]
        if pandas.isna(user_1) or pandas.isna(user_2) or user_1 == user_2:
            continue
        if _is_bot(str(user_1)) or _is_bot(str(user_2)):
            continue
        interactions_df.append({
            "from_user": user_1,
            "to_user": user_2,
            "event_1": row["event"],
            "event_2": "created",
            id_name: row[id_name],
            "event_1_timestamp": row[created_at],
            "event_2_timestamp": row[created_at_2],
            "distance": 0,
            'author_id_x': row['author_id_x'],
            'author_name_x': row['author_name_x'],
            'author_login_x': row['author_login_x'],
            'author_email_x': row['author_email_x'],
            'author_id_y': row['author_id_y'],
            'author_name_y': row['author_name_y'],
            'author_login_y': row['author_login_y'],
            'author_email_y': row['author_email_y'],
        })

    # Loop 2: connect user j to user i when j acted before i on the same item.
    # This captures commenter-to-commenter interactions (works for PRs where
    # prs_comments contains multiple users per PR).
    for pr_id, group in timeline.groupby(id_name):
        users = group[created_by].tolist()
        timestamps = group[created_at].tolist()
        event = group["event"].tolist()
        for i in range(len(users)):
            user_i = users[i]
            time_i = timestamps[i]
            for j in range(len(users)):
                user_j = users[j]
                time_j = timestamps[j]
                if user_i != user_j and time_j <= time_i and not _is_bot(str(user_i)) and not _is_bot(str(user_j)):
                    row_j = group.iloc[j]
                    interactions_df.append({
                        "from_user": user_j,
                        "to_user": user_i,
                        "event_1": event[j],
                        "event_2": event[i],
                        id_name: pr_id,
                        "event_1_timestamp": time_i,
                        "event_2_timestamp": time_j,
                        "distance": i - j,
                        'author_id_x': row_j['author_id_x'],
                        'author_name_x': row_j['author_name_x'],
                        'author_login_x': row_j['author_login_x'],
                        'author_email_x': row_j['author_email_x'],
                        'author_id_y': row_j['author_id_y'],
                        'author_name_y': row_j['author_name_y'],
                        'author_login_y': row_j['author_login_y'],
                        'author_email_y': row_j['author_email_y']

                    })
    _cols = ["from_user","to_user","event_1","event_2",id_name,
             "event_1_timestamp","event_2_timestamp","distance",
             "author_id_x","author_name_x","author_login_x","author_email_x",
             "author_id_y","author_name_y","author_login_y","author_email_y"]
    if interactions_df:
        interactions_df = pandas.DataFrame(interactions_df)
    else:
        interactions_df = pandas.DataFrame(columns=_cols)

    interactions_df = interactions_df.sort_values(by=[id_name,"event_1_timestamp"])
    if folder_path is not None:
        interactions_df.to_csv(Path(folder_path) / f"{type}_interactions.csv", index=False)
    else:
        interactions_df.to_csv(f"{type}_interactions.csv", index=False)

    return interactions_df

#_________________________
#  Step 3: build_graph_tables
#_________________________

def build_graph_tables(combined_interactions: pandas.DataFrame,
                       window_days: int = 30,
                       as_of_date=None):
    """
    Parameters
    ----------
    combined_interactions : DataFrame
        Output of Step 2. Must have columns:
            from_user, to_user, event_1_timestamp (datetime, UTC-aware)
    window_days : int
        How many days back to look. Default 30.

    Returns
    -------
    nodes_df : DataFrame   columns: id, out, inc
    links_df : DataFrame   columns: s, t, w
    """

    df = combined_interactions.copy()

    # ── Step 0: normalise the timestamp and filter to the window ──────────────
    df["event_1_timestamp"] = pandas.to_datetime(
        df["event_1_timestamp"], utc=True, errors="coerce"
    )
    if as_of_date is None:
        cutoff_end = datetime.now(tz=timezone.utc)
    elif hasattr(as_of_date, 'tzinfo') and as_of_date.tzinfo is not None:
        cutoff_end = as_of_date
    else:
        cutoff_end = datetime(as_of_date.year, as_of_date.month, as_of_date.day,
                              23, 59, 59, tzinfo=timezone.utc)
    cutoff = cutoff_end - timedelta(days=window_days)
    df = df[(df["event_1_timestamp"] >= cutoff) & (df["event_1_timestamp"] <= cutoff_end)].copy()

    if df.empty:
        print(f"Warning: no interactions found in the last {window_days} days.")
        nodes_df = pandas.DataFrame(columns=["id", "out", "inc"])
        links_df = pandas.DataFrame(columns=["s", "t", "w"])
        return nodes_df, links_df

    # ── Step 1: nodes ─────────────────────────────────────────────────────────
    # out = how many times this user appears as the FROM side (they acted)
    # inc = how many times this user appears as the TO side (someone acted toward them)

    out_counts = (
        df.groupby("from_user")
        .size()
        .reset_index(name="out")
        .rename(columns={"from_user": "id"})
    )

    inc_counts = (
        df.groupby("to_user")
        .size()
        .reset_index(name="inc")
        .rename(columns={"to_user": "id"})
    )

    # outer join so users who only send OR only receive still appear
    nodes_df = pandas.merge(out_counts, inc_counts, on="id", how="outer").fillna(0)
    nodes_df["out"] = nodes_df["out"].astype(int)
    nodes_df["inc"] = nodes_df["inc"].astype(int)

    # ── Step 2: links ─────────────────────────────────────────────────────────
    # We want ONE undirected edge per pair (A, B) regardless of who went first.
    # Weight = total number of interaction rows between A and B.
    #
    # sorted() on the pair means (A, B) and (B, A) both map to the same key
    # so we count them together.

    df = df.dropna(subset=["from_user", "to_user"])
    df = df[df["from_user"].apply(lambda x: isinstance(x, str)) &
            df["to_user"].apply(lambda x: isinstance(x, str))]
    df["pair"] = df.apply(
        lambda r: tuple(sorted([r["from_user"], r["to_user"]])), axis=1
    )

    links_df = (
        df.groupby("pair")
        .size()
        .reset_index(name="w")
    )

    # split the pair tuple back into s and t columns
    links_df[["s", "t"]] = pandas.DataFrame(
        links_df["pair"].tolist(), index=links_df.index
    )
    links_df = links_df[["s", "t", "w"]].sort_values("w", ascending=False).reset_index(drop=True)

    return nodes_df, links_df

#_________________________
#  Step 4: generate_and_open
#_________________________

def generate_and_open(
    nodes_df: pandas.DataFrame,
    links_df: pandas.DataFrame,
    folder_path: Path,
    nodes_filename: str = "stn_nodes.csv",
    links_filename: str = "stn_links.csv",
    html_filename:  str = "stn_network.html",
    repo_full_name: str = "",
    open_browser:   bool = True,
    as_of_date=None,
    window_days: int = 30,) -> Path:
    """
    1. Saves nodes_df and links_df as CSVs into folder_path.
    2. Generates a standalone HTML file with the data embedded.
    3. Opens the HTML in the default browser (optional).

    Returns the path to the generated HTML file.
    """

    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Save CSVs ──────────────────────────────────────────────────────────
    nodes_path = folder_path / nodes_filename
    links_path = folder_path / links_filename
    nodes_df.to_csv(nodes_path, index=False)
    links_df.to_csv(links_path, index=False)
    print(f"  Saved nodes → {nodes_path}")
    print(f"  Saved links → {links_path}")

    # ── 2. Convert DataFrames to JavaScript arrays ────────────────────────────
    # nodes: [{id, out, inc}, ...]
    # links: [{s, t, w}, ...]
    nodes_js = json.dumps(
        nodes_df.rename(columns={"id": "id", "out": "out", "inc": "inc"})
                .to_dict(orient="records"),
        indent=2
    )
    links_js = json.dumps(
        links_df.rename(columns={"s": "s", "t": "t", "w": "w"})
                .to_dict(orient="records"),
        indent=2
    )

    # ── 3. Build full HTML ────────────────────────────────────────────────────
    html = _build_html(nodes_js, links_js, repo_full_name,
                       window_days=window_days, as_of_date=as_of_date)

    html_path = folder_path / html_filename
    html_path.write_text(html, encoding="utf-8")
    print(f"  Saved HTML  → {html_path}")

    # ── 4. Open in browser ────────────────────────────────────────────────────
    if open_browser:
        webbrowser.open(html_path.as_uri())
        print(f"  Opened in browser.")

    return html_path

def _build_html(nodes_js: str, links_js: str, repo_full_name: str,
                window_days: int = 30, as_of_date=None, initial_node: str = None) -> str:
    """Returns standalone HTML with before/after departure simulation networks."""
    from datetime import date as _date
    if as_of_date is None:
        subtitle_date = _date.today().strftime('%b %d, %Y')
    elif hasattr(as_of_date, 'strftime'):
        subtitle_date = as_of_date.strftime('%b %d, %Y')
    else:
        subtitle_date = str(as_of_date)
    _subtitle = f"{window_days}-day window ending {subtitle_date}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Departure Simulation — {repo_full_name}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  :root {{
    --bg: #ffffff; --bg2: #f5f4f0;
    --text: #1a1a18; --text2: #73726c; --text3: #9c9a92;
    --b1: rgba(0,0,0,0.10); --b2: rgba(0,0,0,0.18); --b3: rgba(0,0,0,0.30);
    --red: #E24B4A; --green: #1D9E75; --accent: #378ADD; --orange: #F09050;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1c1c1a; --bg2: #252522;
      --text: #e8e6de; --text2: #9c9a92; --text3: #6b6964;
      --b1: rgba(255,255,255,0.10); --b2: rgba(255,255,255,0.18); --b3: rgba(255,255,255,0.30);
    }}
  }}
  body {{ margin:0; padding:16px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; font-size:13px; background:var(--bg); color:var(--text); }}
  h1 {{ font-size:15px; font-weight:500; margin:0 0 3px; }}
  .sub {{ font-size:11px; color:var(--text2); margin:0 0 14px; }}
  .networks {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }}
  .net-wrap {{ border:1px solid var(--b1); border-radius:8px; overflow:hidden; background:var(--bg); }}
  .net-label {{ padding:7px 12px; font-size:11px; font-weight:600; border-bottom:1px solid var(--b1); display:flex; align-items:center; gap:6px; color:var(--text); }}
  .dot {{ width:8px; height:8px; border-radius:50%; flex-shrink:0; }}
  svg {{ display:block; width:100%; }}
  .mc {{ background:var(--bg2); border-radius:8px; padding:8px 10px; }}
  .ml {{ font-size:10px; color:var(--text2); margin:0 0 2px; }}
  .mv {{ font-family:'Courier New',monospace; font-size:17px; font-weight:500; margin:0; line-height:1; color:var(--text); }}
  .msub {{ font-size:9px; color:var(--text2); margin:3px 0 0; }}
  .stats-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:10px; }}
  .insight {{ padding:8px 10px; border-radius:8px; border:0.5px solid var(--b1); font-size:11px; color:var(--text2); line-height:1.5; }}
  .hint {{ font-size:10px; color:var(--text3); margin-top:8px; text-align:center; }}
  .legend {{ display:flex; gap:14px; padding:5px 12px 8px; font-size:10px; color:var(--text2); flex-wrap:wrap; }}
  .leg-item {{ display:flex; align-items:center; gap:5px; }}
  #tooltip {{
    position:fixed; pointer-events:none; opacity:0; z-index:99;
    background:var(--bg2); border:1px solid var(--b2); border-radius:6px;
    padding:7px 11px; font-size:11px; color:var(--text); line-height:1.6;
    box-shadow:0 2px 8px rgba(0,0,0,0.15); transition:opacity 0.08s;
  }}
</style>
</head>
<body>
<h1>Departure Simulation — {repo_full_name}</h1>
<p class="sub">{_subtitle} &nbsp;·&nbsp; <span id="removed-label">click a node to simulate removal</span></p>

<div class="networks">
  <div class="net-wrap">
    <div class="net-label"><div class="dot" style="background:var(--accent)"></div>Before — full network</div>
    <svg id="svg-before" style="height:400px;"></svg>
    <div class="legend">
      <span class="leg-item"><div class="dot" style="background:var(--red)"></div>departing dev</span>
      <span class="leg-item"><div class="dot" style="background:var(--accent)"></div>other devs</span>
    </div>
  </div>
  <div class="net-wrap">
    <div class="net-label"><div class="dot" style="background:var(--red)"></div><span id="after-label-text">After — select a node</span></div>
    <svg id="svg-after" style="height:400px;"></svg>
    <div class="legend">
      <span class="leg-item"><div class="dot" style="background:var(--orange)"></div>lost direct connection</span>
      <span class="leg-item"><div class="dot" style="background:var(--accent);opacity:0.3"></div>unaffected</span>
    </div>
  </div>
</div>

<div id="impact-panel" style="display:none;">
  <div class="stats-row">
    <div class="mc"><p class="ml">Developer</p><p class="mv" id="stat-dev" style="font-size:13px;">—</p></div>
    <div class="mc"><p class="ml">Edges cut</p><p class="mv" id="stat-edges">—</p><p class="msub">direct connections lost</p></div>
    <div class="mc"><p class="ml">Volume lost</p><p class="mv" id="stat-vol">—</p><p class="msub">% of total interactions</p></div>
    <div class="mc"><p class="ml">Fragments</p><p class="mv" id="stat-frags">—</p><p class="msub">disconnected groups</p></div>
  </div>
  <div class="insight" id="stat-insight"></div>
</div>
<p class="hint">Click any node in the "Before" network to simulate their departure.</p>
<div id="tooltip"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const RAW_NODES = {nodes_js};
const RAW_LINKS = {links_js};
const INITIAL_NODE = {json.dumps(initial_node)};

// All edges between kept nodes are displayed — k-core guarantees every node
// has connections, so no edge filter is needed or safe here.
const DISPLAY_LINKS = RAW_LINKS;
const NODE_COLOR = '#378ADD';

function buildAdj(nodes, rawLinks) {{
  const adj = {{}};
  nodes.forEach(n => {{ adj[n.id] = []; }});
  rawLinks.forEach(l => {{
    if (adj[l.s] !== undefined && !adj[l.s].includes(l.t)) adj[l.s].push(l.t);
    if (adj[l.t] !== undefined && !adj[l.t].includes(l.s)) adj[l.t].push(l.s);
  }});
  return adj;
}}

function findComponents(nodes, adj) {{
  const vis = new Set(), comps = [];
  for (const n of nodes) {{
    if (vis.has(n.id)) continue;
    const comp = [], q = [n.id];
    while (q.length) {{
      const cur = q.pop();
      if (vis.has(cur)) continue;
      vis.add(cur); comp.push(cur);
      for (const nb of (adj[cur] || [])) if (!vis.has(nb)) q.push(nb);
    }}
    comps.push(comp);
  }}
  return comps;
}}

function computeBetweenness(nodes, adj) {{
  const ids = nodes.map(n => n.id), n = ids.length;
  const idx = Object.fromEntries(ids.map((id, i) => [id, i]));
  const btw = new Float64Array(n);
  for (let si = 0; si < n; si++) {{
    const stack = [], pred = ids.map(() => []);
    const sigma = new Float64Array(n), dist = new Int32Array(n).fill(-1);
    sigma[si] = 1; dist[si] = 0;
    const q = [si]; let qi = 0;
    while (qi < q.length) {{
      const v = q[qi++]; stack.push(v);
      for (const nb of (adj[ids[v]] || [])) {{
        const w = idx[nb]; if (w === undefined) continue;
        if (dist[w] < 0) {{ q.push(w); dist[w] = dist[v] + 1; }}
        if (dist[w] === dist[v] + 1) {{ sigma[w] += sigma[v]; pred[w].push(v); }}
      }}
    }}
    const delta = new Float64Array(n);
    while (stack.length) {{
      const w = stack.pop();
      for (const v of pred[w]) delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]);
      if (w !== si) btw[w] += delta[w];
    }}
  }}
  const norm = (n > 2) ? (n - 1) * (n - 2) : 1;
  return Object.fromEntries(ids.map((id, i) => [id, btw[i] / norm]));
}}

// Pre-compute full-network metrics
const adj0  = buildAdj(RAW_NODES, RAW_LINKS);
const btw0  = computeBetweenness(RAW_NODES, adj0);
const deg0  = {{}}, wdeg0 = {{}};
RAW_NODES.forEach(n => {{ deg0[n.id] = 0; wdeg0[n.id] = 0; }});
RAW_LINKS.forEach(l => {{ deg0[l.s]++; deg0[l.t]++; wdeg0[l.s] += l.w; wdeg0[l.t] += l.w; }});
const NODES = RAW_NODES.map(n => ({{
  ...n,
  degree:      deg0[n.id]  || 0,
  wdegree:     wdeg0[n.id] || 0,
  betweenness: btw0[n.id]  || 0,
}}));

const maxW   = Math.max(...RAW_LINKS.map(l => l.w), 1);
const wScale = d3.scaleLinear().domain([1, maxW]).range([1, 5]).clamp(true);

// Rank-based node sizing: sort nodes by total interactions, map rank → radius.
// This replaces log/sqrt compression — the top developer always gets the largest
// circle, the least active always gets the smallest, and everyone is evenly spread
// between them regardless of the raw magnitude gap between values.
const _byActivity = [...NODES].sort((a, b) =>
  ((a.out||0)+(a.inc||0)) - ((b.out||0)+(b.inc||0))
);
const _rankMap = Object.fromEntries(_byActivity.map((n, i) => [n.id, i]));
const _nNodes  = Math.max(NODES.length - 1, 1);
const nR = n => 6 + (_rankMap[n.id] / _nNodes) * 22;

let simBefore = null, simAfter = null;

function makeNetwork(svgId, removeId, highlightId, clickable) {{
  const svgEl = document.getElementById(svgId);
  if (!svgEl) return null;
  const W = svgEl.getBoundingClientRect().width || 500, H = 400;
  const svg = d3.select('#' + svgId);
  svg.selectAll('*').remove();
  const gL = svg.append('g'), gN = svg.append('g');

  const nodeData    = NODES.filter(n => n.id !== removeId).map(n => ({{...n}}));
  const rawFiltered = DISPLAY_LINKS.filter(l => l.s !== removeId && l.t !== removeId);
  const linkData    = rawFiltered.map(l => ({{source: l.s, target: l.t, w: l.w}}));

  // Nodes that had a direct edge to the removed developer
  const affectedSet = new Set();
  if (removeId) {{
    RAW_LINKS.forEach(l => {{
      if (l.s === removeId) affectedSet.add(l.t);
      if (l.t === removeId) affectedSet.add(l.s);
    }});
  }}

  const sim = d3.forceSimulation(nodeData)
    .force('link',      d3.forceLink(linkData).id(d => d.id)
                          .distance(d => 110 + (30 - Math.min(d.w, 30)) * 0.8))
    .force('charge',    d3.forceManyBody().strength(-350))
    .force('center',    d3.forceCenter(W / 2, H / 2))
    .force('collision', d3.forceCollide().radius(d => nR(d) + 10))
    .force('x',         d3.forceX(W / 2).strength(0.04))
    .force('y',         d3.forceY(H / 2).strength(0.04));

  const linkSel = gL.selectAll('line').data(linkData).enter().append('line')
    .attr('stroke', 'rgba(0,0,0,0.15)')
    .attr('stroke-width', d => wScale(d.w))
    .attr('opacity', 0.55);

  const nodeSel = gN.selectAll('circle').data(nodeData).enter().append('circle')
    .attr('r',    d => nR(d))
    .attr('fill', d => {{
      if (d.id === highlightId)              return '#E24B4A';
      if (removeId && affectedSet.has(d.id)) return '#F09050';
      return NODE_COLOR;
    }})
    .attr('stroke', d => {{
      if (d.id === highlightId)              return '#991B1B';
      if (removeId && affectedSet.has(d.id)) return '#C06020';
      return 'none';
    }})
    .attr('stroke-width', d =>
      (d.id === highlightId || (removeId && affectedSet.has(d.id))) ? 1.5 : 0)
    .attr('opacity', d => removeId ? (affectedSet.has(d.id) ? 1 : 0.3) : 1)
    .attr('cursor', clickable ? 'pointer' : 'default')
    .on('click', clickable ? (e, d) => simulateRemoval(d.id) : null)
    .on('mouseover', (e, d) => {{
      const total = (d.out||0) + (d.inc||0);
      const tip = document.getElementById('tooltip');
      tip.innerHTML = '<strong>' + d.id + '</strong><br>'
        + 'Total interactions: ' + total.toLocaleString() + '<br>'
        + 'Sent: ' + (d.out||0).toLocaleString()
        + ' &nbsp; Received: ' + (d.inc||0).toLocaleString();
      tip.style.opacity = 1;
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY + 14) + 'px';
    }})
    .on('mousemove', e => {{
      const tip = document.getElementById('tooltip');
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top  = (e.clientY + 14) + 'px';
    }})
    .on('mouseout', () => {{
      document.getElementById('tooltip').style.opacity = 0;
    }});

  const labelSel = gN.selectAll('text').data(nodeData).enter().append('text')
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
    .attr('fill', '#1a1a18').attr('font-size', 9).attr('font-weight', 500)
    .attr('pointer-events', 'none')
    .text(d => d.id.length > 8 ? d.id.slice(0, 7) + '…' : d.id);

  sim.on('tick', () => {{
    linkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
           .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeSel.attr('cx', d => d.x = Math.max(nR(d), Math.min(W - nR(d), d.x)))
           .attr('cy', d => d.y = Math.max(nR(d), Math.min(H - nR(d), d.y)));
    labelSel.attr('x', d => d.x).attr('y', d => d.y);
  }});

  return sim;
}}

function renderImpact(nodeId) {{
  const node = NODES.find(n => n.id === nodeId);
  if (!node) return;
  const removedLinks = RAW_LINKS.filter(l => l.s === nodeId || l.t === nodeId);
  const vol      = removedLinks.reduce((s, l) => s + l.w, 0);
  const exNodes  = NODES.filter(n => n.id !== nodeId);
  const exLinks  = RAW_LINKS.filter(l => l.s !== nodeId && l.t !== nodeId);
  const exAdj    = buildAdj(exNodes, exLinks);
  const comps    = findComponents(exNodes, exAdj);
  const totalVol = RAW_LINKS.reduce((s, l) => s + l.w, 0);
  const pct      = totalVol > 0 ? Math.round((vol / totalVol) * 100) : 0;

  document.getElementById('stat-dev').textContent   = nodeId;
  document.getElementById('stat-edges').textContent = removedLinks.length;
  document.getElementById('stat-vol').textContent   = vol + ' (' + pct + '%)';
  const fragsEl = document.getElementById('stat-frags');
  fragsEl.textContent = comps.length;
  fragsEl.style.color = comps.length > 1 ? 'var(--red)' : 'var(--green)';

  let insight = comps.length > 1
    ? `<strong style="color:var(--red)">Network splits into ${{comps.length}} disconnected groups.</strong> `
      + `${{nodeId}} was bridging separate communities — reassigning volume alone cannot restore connectivity.`
    : `Network stays connected after removing ${{nodeId}}. `
      + `The ${{vol}} interactions (${{pct}}% of total) need to be redistributed among neighbors.`;
  if (node.betweenness > 0.15)
    insight += ` <strong>High betweenness (${{node.betweenness.toFixed(3)}})</strong> confirms a bridging role.`;
  document.getElementById('stat-insight').innerHTML = insight;
  document.getElementById('impact-panel').style.display = 'block';
}}

function simulateRemoval(nodeId) {{
  document.getElementById('removed-label').textContent    = 'Simulating removal of ' + nodeId;
  document.getElementById('after-label-text').textContent = 'After — ' + nodeId + ' removed';
  if (simBefore) simBefore.stop();
  if (simAfter)  simAfter.stop();
  simBefore = makeNetwork('svg-before', null,   nodeId, true);
  simAfter  = makeNetwork('svg-after',  nodeId, null,   false);
  renderImpact(nodeId);
}}

function init() {{
  if (typeof d3 === 'undefined') {{ setTimeout(init, 50); return; }}
  if (INITIAL_NODE) {{
    simulateRemoval(INITIAL_NODE);
  }} else {{
    simBefore = makeNetwork('svg-before', null, null, true);
    simAfter  = makeNetwork('svg-after',  null, null, false);
  }}
}}
init();
</script>
</body>
</html>"""


#_________________________
#  Streamlit entry point
#_________________________

def _prune_nodes_kcore(nodes_df, links_df, initial_node=None, tf_devs=None):
    """K-core filtering: keep only the densely-connected core.

    Uses k-core decomposition (Seidman 1983). Searches upward from k=2 to find
    the smallest k where the k-core fits within 25 nodes (but keeps ≥ 5).
    Every node in the resulting k-core has ≥ k connections to OTHER nodes in
    the same set — so no node can be isolated after pruning.

    Pinned nodes (focal developer, their direct neighbours, truck-factor devs)
    are always included.
    """
    _MAX = 25

    if nodes_df.empty:
        return nodes_df, links_df

    G = nx.Graph()
    for nid in nodes_df["id"]:
        G.add_node(nid)
    for _, row in links_df.iterrows():
        G.add_edge(row["s"], row["t"], weight=int(row["w"]))

    # Nodes that must always appear
    pinned = set()
    if initial_node and initial_node in G:
        pinned.add(initial_node)
        pinned.update(G.neighbors(initial_node))
    if tf_devs:
        valid_ids = set(nodes_df["id"])
        pinned.update(d for d in tf_devs if d in valid_ids)

    # Walk k upward from 2 until the core fits within _MAX nodes (but stays ≥ 5).
    # The first k that satisfies both bounds is used — it shows the largest
    # densely-connected subgraph that is still readable.
    core_nums = nx.core_number(G)
    max_k = max(core_nums.values(), default=0)
    chosen_k = 1   # k=1 fallback: show everything
    for k in range(2, max_k + 1):
        candidate = {n for n, c in core_nums.items() if c >= k} | pinned
        if len(candidate) < 5:
            break   # further increases only shrink further — stop here
        if len(candidate) <= _MAX:
            chosen_k = k
            break   # fits: use this k

    keep_set = {n for n, c in core_nums.items() if c >= chosen_k} | pinned

    nodes_pruned = nodes_df[nodes_df["id"].isin(keep_set)].reset_index(drop=True)
    links_pruned = links_df[
        links_df["s"].isin(keep_set) & links_df["t"].isin(keep_set)
    ].reset_index(drop=True)
    return nodes_pruned, links_pruned


def get_html_for_streamlit(repo_full_name: str, as_of_date=None, window_days: int = 30,
                           initial_node: str = None, tf_devs=None) -> str:
    """
    Build the D3 interaction graph HTML for embedding in Streamlit via
    st.components.v1.html().

    Loads pre-saved interaction CSVs from SocialTechnicalNetwork/ if available
    (fast — avoids re-processing all issues/PRs on every date change).
    Falls back to re-computing from raw data only when those CSVs are absent.

    Parameters
    ----------
    repo_full_name : str   e.g. "Rdatatable/data.table"
    as_of_date     : date | datetime | None
        Snapshot date for the window. None → today.
    window_days    : int
        Width of the rolling window. Default 30.

    Returns
    -------
    str  — complete standalone HTML ready for components.html()
    """
    org, repo = repo_full_name.split('/')
    organization_folder = Path(cfg.main_folder, org, repo)
    folder_path = organization_folder / cfg.social_technical_metrics_folder

    issue_cache = folder_path / "issue_interactions.csv"
    pr_cache    = folder_path / "PR_interactions.csv"

    if issue_cache.exists() and pr_cache.exists():
        issue_interactions = pandas.read_csv(issue_cache)
        pr_interactions    = pandas.read_csv(pr_cache)
    else:
        raw_data_tables = load_users_activity(
            repo_full_name=repo_full_name,
            organization_folder=organization_folder,
        )
        if not raw_data_tables:
            return "<p>No raw data found for this repository.</p>"
        issue_timeline, pr_timeline = timeline(raw_data_tables)
        folder_path.mkdir(parents=True, exist_ok=True)
        issue_interactions = interaction_network(issue_timeline, "issue",
                                                 folder_path=folder_path)
        pr_interactions    = interaction_network(pr_timeline,   "PR",
                                                 folder_path=folder_path)

    issue_interactions["source"] = "issue"
    pr_interactions["source"]    = "PR"
    combined = pandas.concat([issue_interactions, pr_interactions], ignore_index=True)

    nodes_df, links_df = build_graph_tables(combined,
                                            window_days=window_days,
                                            as_of_date=as_of_date)

    if nodes_df.empty:
        end_str = (as_of_date.strftime('%b %d, %Y')
                   if hasattr(as_of_date, 'strftime') else str(as_of_date))
        return (f"<p style='font-family:sans-serif;color:#888;padding:16px;'>"
                f"No interactions found in the {window_days}-day window ending {end_str}.</p>")

    nodes_df, links_df = _prune_nodes_kcore(nodes_df, links_df,
                                            initial_node=initial_node,
                                            tf_devs=tf_devs)

    nodes_js = json.dumps(nodes_df.to_dict(orient="records"), indent=2)
    links_js = json.dumps(links_df.to_dict(orient="records"), indent=2)
    return _build_html(nodes_js, links_js, repo_full_name,
                       window_days=window_days, as_of_date=as_of_date, initial_node=initial_node)


#_________________________
#  Main
#_________________________

def main(repo_full_name=None, tf_devs=None, tables = None):

    # Initialize
    org, repo = repo_full_name.split('/')
    organization_folder = Path(cfg.main_folder, org, repo)
    social_technical_metrics_folder = Path(organization_folder, cfg.social_technical_metrics_folder)

    os.makedirs(social_technical_metrics_folder, exist_ok=True)
    # Save output
    out_file = Path(social_technical_metrics_folder ,cfg.social_technical_metrics_file)

    print(organization_folder)

    raw_data_tables = load_users_activity(repo_full_name=repo_full_name, organization_folder = organization_folder)

    # Step 1:
    # we need to create the issue timeline
    issue_timeline, pr_timeline = timeline(raw_data_tables)
    print("Timelines Created.")

    # Step 2:
    # now we need to take this joined data and create the interaction network
    issue_interactions = interaction_network(issue_timeline, "issue",
                                             folder_path=social_technical_metrics_folder)
    print("Issue Interactions Created.")
    pr_interactions = interaction_network(pr_timeline, "PR",
                                          folder_path=social_technical_metrics_folder)
    print("PR Interactions Created.")
    #we need to combine these interaction networks into a single one
    # can you add a column to tell which file it came from
    issue_interactions["source"] = "issue"
    pr_interactions["source"] = "PR"

    #Step 3:
    combined_interactions = pandas.concat(
        [issue_interactions, pr_interactions], ignore_index=True
    )

    nodes_df, links_df = build_graph_tables(combined_interactions, window_days=30)
    print(nodes_df.head())
    print(links_df.head())

    #Step 4:
    # in step 4 we will anayise the tables issue and issue_activity to make daily metrics.
    folder_path = Path(cfg.main_folder, org, repo, cfg.social_technical_metrics_folder)

    generate_and_open(
        nodes_df       = nodes_df,
        links_df       = links_df,
        folder_path    = folder_path,
        nodes_filename = cfg.social_technical_nodes_file,   # "stn_nodes.csv"
        links_filename = cfg.social_technical_links_file,   # "stn_links.csv"
        html_filename  = cfg.social_technical_html_file,    # "stn_network.html"
        repo_full_name = repo_full_name,
        open_browser   = True,
    )

    return


if __name__ == "__main__":
    # let the user spesify the repo to process
    repo_full_name = "Rdatatable/data.table"  # Example: "organization/repo"

    main(repo_full_name)
