#   conda activate CS485
#   cd C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Extractors
#   streamlit run DemoApp.py

import json
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
import altair as alt

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from dataclasses import dataclass
from github import Github, GithubException, UnknownObjectException, IncompletableObject

from ProjectHealthAnaysis import build_repo_health as project_health_main
from CommitExtractor import main as extract_repo_main


sys.path.append('../')
import Settings as cfg
import Utilities as util


#from truckfactor.compute import main as compute_tf

#-----------------------------------------
# Program Start
#-----------------------------------------


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

def mock_update_repo(repo: str):
    # TODO: plug in your fast incremental updater if available.
    # For now, simulate progress so the flow is obvious.
    total = np.random.randint(20, 30)
    pbar  = tqdm(total=total,     desc="Issues ", position=2, leave=True)

    progress = st.progress(0)
    for i in range(total):
        pbar.update(1)
        time.sleep(np.random.randint(1, 2) / 1000)
        progress.progress((i+1)/total)

    path = ORG_BASE / repo / "commit_list.csv"
    st.write("Reading commit list from:", path)
    df = pandas.read_csv(path)

    st.success("Data is up to date (demo).")

    return df

#-----------------------------------------
# Label Developers Activity
#-----------------------------------------

def label_developers_activity(repo, process_all: bool = False) -> pandas.DataFrame:
    """
    main function for labeling developers

    sets up varables to call label timeline

    """

    # "../Organizations"
    organizationFolder = cfg.main_folder

    win = cfg.sliding_window_size
    shift = cfg.shift

    repos_txt = '../' + cfg.repos_file
    repos_to_process = []

    if process_all or repo is None:
        if not os.path.isfile(repos_txt):
            st.error(f"Repositories file not found: {repos_txt}")
            return []
        
    if process_all is True:
        with open(repos_txt, 'r') as f:
            repos = f.readlines()
            repos_to_process = [r.strip() for r in repos if r.strip()]
            st.write(f"Found {len(repos)} repositories in {repos_txt}")
    else:
        repos_to_process = [repo]


    tf_devs= []

    for repo in repos_to_process:


        organization, project = repo.split('/')
        if Path(organizationFolder, organization).exists() == False:
            st.write(f"Organization folder not found: {Path(organizationFolder, organization)}")
            continue

        print(f"Start Identifying inactivity periods for {organization}/{project}...")
        st.write(f"Start Identifying inactivity periods for {organization}/{project}...")

        authors, pauses = identifyInactivityPeriods( organizationFolder, organization, project)

        #make pauses to a csv file at this location C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Organizations\Rdatatable\data.table\Results

        pauses_list = pauses.values.tolist()
        print(f"{len(authors)} Developers inactivity periods identified")
        st.write(f"{len(authors)} Developers inactivity periods identified")

        output_folder = organizationFolder + '/' + repo + "/Results"
        os.makedirs(output_folder, exist_ok=True)
        progress = st.progress(0)

        for i, dev in enumerate(authors):
            progress.progress((i+1)/len(authors))

            tf_devs.append(dev)
            timeline_folder = organizationFolder + '/' + repo + '/' + cfg.timeline_folder_name
            os.makedirs(timeline_folder, exist_ok=True)
        
            timeline_path = Path(timeline_folder) / f"{dev}_timeline.csv"

            if timeline_path.is_file():
                user_timeline = pandas.read_csv(timeline_path, sep=cfg.CSV_separator, index_col=0)
            else:
                folder = organizationFolder + '/' + repo
                user_timeline = get_timeline(folder, dev)

                #transpose the user_timeline making the frist row the first column
                user_timeline.to_csv(timeline_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n')

            print(dev)
            #make a break folder 
            breaks_folder = organizationFolder + '/' + repo + "/Breaks"
            os.makedirs(breaks_folder, exist_ok=True)

            breaks_path =  Path(breaks_folder)/  f"{dev}_breaks.csv"              

            if breaks_path.is_file():
                breaks_df = pandas.read_csv(breaks_path, sep=cfg.CSV_separator, index_col=0)  

            else:
                breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])
                breaks_df = identifyBreaks(pauses_list, dev=dev, window=win, shift=shift, debug_folder=output_folder )
                breaks_df.to_csv(breaks_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, index=False, lineterminator="\n")
                        
            #add label timeline
            user_timeline = label_timeline(user_timeline, breaks_df)

            out_csv = Path(output_folder) / f"{dev}_labeled_timeline.csv"
            user_timeline.to_csv(out_csv,
                                sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n', index_label='date')

            tf_devs_df = pandas.DataFrame(authors, columns=["developer"])
            tf_devs_df.to_csv(Path(output_folder) / "tf_devs.csv", sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n')

    return tf_devs

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
    df["break_day"] = pandas.Series(False, index=df.index, dtype="boolean")
    df["th"] = pandas.Series(pandas.NA, index=df.index, dtype="Float64")
    df["len"] = pandas.Series(pandas.NA, index=df.index, dtype="Int64") 

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
        df[col] = df[col].astype("boolean").fillna(False)


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
            silent_days = (d - last_any).days
            if silent_days > gone_days:
                df.at[d, "state"] = "GONE"
            else:
                df.at[d, "state"] = "INACTIVE"


    return df

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

def get_commit_based_core_devs(
    commits,
    threshold=0.7):
    """
    commits: iterable of dicts with some author identifier in one of `author_keys`
    Returns a list of core developers (author strings) whose commits cover >= threshold.
    """

    # Common NA tokens and GitHub bot patterns; override via args if needed.
    na_tokens = {"", "na", "n/a", "none", "null", "nan", "unknown", "ghost"}

    bot_patterns = [
        r"\[bot\]$",           # e.g., dependabot[bot]
        r"(^|[-_.])bot($|[-_.])",
        r"\bbot\b",
        r"^dependabot",
        r"^renovate",
        r"^github-actions",
    ]
    bot_res = [re.compile(p, re.IGNORECASE) for p in bot_patterns]

    # Count commits per (filtered) author
    author_commit_counts = Counter()
    for c in commits:
    
        v = c.get("author_id")
        if isinstance(v, dict):
            v = v.get("login") or v.get("name") or v.get("email")
        elif v is not None:
            a = str(v).strip()
        else:
            return None       
        if not a:
            continue
        al = a.lower()
        if al in na_tokens:
            continue
        if any(rx.search(al) for rx in bot_res):
            continue
        author_commit_counts[a] += 1

    if not author_commit_counts:
        return []

    # Sort by count desc, then name asc for stable ties
    sorted_authors = sorted(author_commit_counts.items(), key=lambda t: (-t[1], t[0].lower()))
    total_commits = sum(author_commit_counts.values())

    cumulative = 0
    core_devs = []
    for author, count in sorted_authors:
        cumulative += count
        core_devs.append(author)
        if cumulative / total_commits >= threshold:
            break
    return core_devs

def identifyInactivityPeriods(organizationFolder, organization, project):
    """Identifies the inactivity periods of the developers in the organization"""
    #url = "https://github.com/" + organization + "/" + project + ".git"
    #authors, emails = findCoreDevelopers(url, name=project)
    
    organizationFolder = organizationFolder + "/" + organization + "/" + project

    commits =  pandas.read_csv(organizationFolder + "/commit_list.csv", parse_dates=["created_at"], encoding="utf-8", header=0, sep=cfg.CSV_separator)

    commit_authors = get_commit_based_core_devs(commits.to_dict(orient='records'))

    pauses = write_pauses_table(commits, organizationFolder + "/pauses_commits.csv", commit_authors, user_col = "author_id", date_col="created_at")
    return commit_authors, pauses

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

def get_NONCODING(folder: str, dev_login: str) -> pandas.DataFrame:
    """
    Build the developer's DAILY 'other-actions' table from the new extractor:
      - issues:              counts from issues.csv (issue creations)
      - issues_comments:     counts from issue_activity.csv where item_type == "IssueComment"
      - issues_events:       counts from issue_activity.csv where item_type != "IssueComment"
      - pull_requests / pull_requests_comments: unchanged (existing PR CSVs)
    Output index = day ("YYYY-MM-DD"); rows = action names.
    """

    # ---- filenames (prefer cfg names, fall back to defaults) ----
    issues_csv = getattr(cfg, "issue_list_file_name", "issues.csv")
    activity_csv = getattr(cfg, "issue_activity_file_name", "issue_activity.csv")

    # ---- read PRs / PR comments with the existing helper ----
    dfs = {}
    dfs["prs"] = _load_activity_csv(
        folder, getattr(cfg, "PR_list_file_name", "prs_repo.csv"),
        {"PR_id": "id", "created_at": "date", "created_by": "creator_login"},
        dev_login
    )
    dfs["prs_comments"] = _load_activity_csv(
        folder, getattr(cfg, "prs_comments_csv", "prs_comments.csv"),
        {"comment_id": "id", "created_at": "date", "created_by": "creator_login"},
        dev_login
    )

    # ---- issues (created) from issues.csv ----
    issues_path = os.path.join(folder, issues_csv)
    if os.path.exists(issues_path):
        iss = pandas.read_csv(issues_path, sep=cfg.CSV_separator)
        # normalize a few possible column names from your extractor
        if "issue_number" in iss.columns and "issue_id" not in iss.columns:
            iss = iss.rename(columns={"issue_number": "issue_id"})
        if "created_by" not in iss.columns:
            iss["created_by"] = pandas.NA
        if "created_at" in iss.columns:
            iss["date"] = pandas.to_datetime(iss["created_at"], errors="coerce").dt.normalize()
        else:
            iss["date"] = pandas.NaT

        iss["creator_login"] = iss["created_by"].astype(str)
        # keep only this dev
        iss = iss[iss["creator_login"] == dev_login][["issue_id", "date", "creator_login"]]
        iss = iss.rename(columns={"issue_id": "id"}).dropna(subset=["date"])
        dfs["issues"] = iss.reset_index(drop=True)
    else:
        dfs["issues"] = pandas.DataFrame(columns=["id", "date", "creator_login"])

    # ---- issue timeline (comments + non-comment events) from issue_activity.csv ----
    def _load_issue_activity_counts(folder: str, dev: str) -> tuple[pandas.DataFrame, pandas.DataFrame]:
        path = os.path.join(folder, activity_csv)
        if not os.path.exists(path):
            empty = pandas.DataFrame(columns=["id", "date", "creator_login"])
            return empty, empty
        a = pandas.read_csv(path, sep=cfg.CSV_separator)

        if "created_by" not in a.columns and "actor" in a.columns:
            a = a.rename(columns={"actor": "created_by"})

        a["date"] = pandas.to_datetime(a.get("created_at"), errors="coerce").dt.normalize()
        a["creator_login"] = a.get("created_by", pandas.NA).astype(str)

        item_type = a["item_type"].astype(str) if "item_type" in a.columns else pandas.Series("", index=a.index)
        is_comment = item_type.eq("IssueComment")

        cm = a.loc[is_comment, ["date", "creator_login"]].copy()
        ev = a.loc[~is_comment, ["date", "creator_login"]].copy()
        cm["id"] = range(len(cm))
        ev["id"] = range(len(ev))

        cm = cm[cm["creator_login"] == dev].dropna(subset=["date"]).reset_index(drop=True)
        ev = ev[ev["creator_login"] == dev].dropna(subset=["date"]).reset_index(drop=True)
        return cm[["id","date","creator_login"]], ev[["id","date","creator_login"]]

    dfs["issues_comments"], dfs["issues_events"] = _load_issue_activity_counts(folder, dev_login)
    for k, d in list(dfs.items()):
        if d is None or d.empty:
            # ensure expected columns exist for downstream code
            dfs[k] = pandas.DataFrame(columns=["id", "date", "creator_login"])
            continue
        # make sure a 'date' column exists
        if "date" not in d.columns:
            d["date"] = pandas.NaT
        # coerce to datetime (midnight) and drop rows with invalid dates
        d["date"] = pandas.to_datetime(d["date"], errors="coerce").dt.normalize()
        d = d.dropna(subset=["date"]).reset_index(drop=True)
        dfs[k] = d

    non_empty = [d for d in dfs.values() if not d.empty]

    if non_empty:
        min_date = min(d["date"].min() for d in non_empty)
        max_date = max(d["date"].max() for d in non_empty)
    else:
        min_date = max_date = pandas.Timestamp.today().normalize()
    # ---- derive full day range from whatever we have ----

    day_cols = pandas.date_range(start=pandas.to_datetime(min_date).normalize(),
                                 end=pandas.to_datetime(max_date).normalize(),
                                 freq="D").strftime("%Y-%m-%d").tolist()

    # ---- helper: counts → one action row ----
    def _timeline_row(action_name, df_raw):
        row = [action_name]
        if df_raw.empty:
            row += [0] * len(day_cols)
            return row
        counts = df_raw["date"].dt.date.value_counts().to_dict()
        for d in day_cols:
            row.append(counts.get(pandas.to_datetime(d).date(), 0))
        return row

    # ---- compile rows ----
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
    # build a FULL daily index from min→today, then align
    if not actions.empty and not commits.empty:
        start = min(actions.index.min(), commits.index.min())
        today = pandas.Timestamp.utcnow().normalize()
    elif not actions.empty:
        start, today = actions.index.min(), pandas.Timestamp.utcnow().normalize()
    elif not commits.empty:
        start, today = commits.index.min(), pandas.Timestamp.utcnow().normalize()
    else:
        # no activity at all → return an empty, well-typed frame
        cols = ["commits","pull_requests","issues","issues_comments",
                "issues_events","pull_requests_comments","coding_day","nc_day"]
        return pandas.DataFrame(columns=cols).astype({
            "commits":"int64","pull_requests":"int64","issues":"int64",
            "issues_comments":"int64","issues_events":"int64",
            "pull_requests_comments":"int64","coding_day":"bool","nc_day":"bool"
        })
    #Start and end cannot both be tz-aware with different timezones
    if (start.tz is not None) != (today.tz is not None):
        start = start.tz_localize(None)
        today = today.tz_localize(None)

    full_idx = pandas.date_range(start=start, end=today, freq="D")
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

#-----------------------------------------
# Predict Developers Activity
#-----------------------------------------
def users_activity_for_repo(repo_key: str, authors: List[str]) -> pandas.DataFrame:
    """
    Load the labeled daily timeline ONLY for (org/repo) and ONLY for the provided authors.
    Expects one file per author: <Organizations>/<org>/<repo>/<results_folder>/<author>_labeled_timeline.csv
    """

    org, repo = repo_key.split("/")
    base = ORG_BASE / org / repo
    # If you keep results under a subfolder, adjust here:
    # e.g., base = base / cfg.results_folder
    # For now, look directly under the repo folder for *_labeled_timeline.csv.
    
    frames = []

    for dev in authors:
        # Try common locations
        candidates = [
            base / f"{dev}_labeled_timeline.csv",
            base / "Results" / f"{dev}_labeled_timeline.csv",
        ]
        file = next((p for p in candidates if p.exists()), None)
        if file is None:
            # Skip missing authors quietly in demo
            continue

        df = pandas.read_csv(file, parse_dates=["date"])
        df["author"] = str(dev)
        df["repo"]   = f"{org}/{repo}"
        df["date"]   = df["date"].dt.normalize()
        frames.append(df)

    if not frames:
        return pandas.DataFrame(columns=[
            "author","repo","date","state","commits","pull_requests","issues",
            "issues_comments","issues_events","pull_requests_comments",
            "coding_day","nc_day","break_day","th","len","event_day"
        ])

    out = pandas.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["author","repo","date"]).reset_index(drop=True)
    return out

def users_activity_all_repos(
    authors: List[str] | None = None,
    repos_txt: str | Path = '../' + cfg.repos_file,
    base_dir: Path = ORG_BASE,
) -> pandas.DataFrame:
    """
    Load *all* labeled daily timelines you’ve produced across every repo in Resources/repositories.txt.
    If `authors` is None -> include everyone we find.
    If `authors` is a list -> include those authors when present (others are still included if authors=None).

    Returns a single concatenated daily table with columns like:
    ['author','repo','date','state','commits','pull_requests','issues',
     'issues_comments','issues_events','pull_requests_comments',
     'coding_day','nc_day','break_day','th','len','event_day']
    """
    def _read_one(author: str, org: str, repo: str) -> pandas.DataFrame | None:
        base = base_dir / org / repo
        candidates = [
            base / f"{author}_labeled_timeline.csv",
            base / "Results" / f"{author}_labeled_timeline.csv",
        ]
        file = next((p for p in candidates if p.exists()), None)
        if file is None:
            print(f"Missing file for {author} in {org}/{repo}")
            return None
        df = pandas.read_csv(file, parse_dates=["date"])
        df["author"] = str(author)
        df["repo"]   = f"{org}/{repo}"
        df["date"]   = df["date"].dt.normalize()
        return df

    frames: list[pandas.DataFrame] = []

    # read the repos list (organization/repo per line)
    with open(repos_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    repos = [ln for ln in lines if ln and not ln.startswith("#")]

    for key in repos:
        try:
            org, repo = key.split("/")
        except ValueError:
            continue  # skip malformed lines

        repo_dir = base_dir / org / repo / "Results"
        if authors is None:
            # include everyone we have labeled files for in this repo
            if repo_dir.exists():
                for f in repo_dir.glob("*_labeled_timeline.csv"):
                    dev = f.stem.replace("_labeled_timeline", "")
                    df = _read_one(dev, org, repo)
                    if df is not None:
                        frames.append(df)
            else:
                # Fallback: scan repo root for labeled files
                root = base_dir / org / repo
                for f in root.glob("*_labeled_timeline.csv"):
                    dev = f.stem.replace("_labeled_timeline", "")
                    df = _read_one(dev, org, repo)
                    if df is not None:
                        frames.append(df)
        else:
            # only for requested authors (if the file exists here)
            for dev in authors:
                df = _read_one(dev, org, repo)
                if df is not None:
                    frames.append(df)

    if not frames:
        return pandas.DataFrame(columns=[
            "author","repo","date","state","commits","pull_requests","issues",
            "issues_comments","issues_events","pull_requests_comments",
            "coding_day","nc_day","break_day","th","len","event_day"
        ])

    out = pandas.concat(frames, ignore_index=True, sort=False)
    # normalize / dedupe
    out["author"] = out["author"].astype(str).str.strip()
    out["repo"]   = out["repo"].astype(str).str.strip()
    out["date"]   = pandas.to_datetime(out["date"]).dt.normalize()
    out = (out
           .sort_values(["author","repo","date"])
           .drop_duplicates(subset=["author","repo","date"], keep="last")
           .reset_index(drop=True))
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
    df["segment_id"] = (
        is_seg_start.groupby([df["author"], df["repo"]]).cumsum() - 1
    ).astype("int64")
    
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

def add_targets(segments: pandas.DataFrame) -> pandas.DataFrame:
    """
    Adds three target columns to segments:
      - current_break_length: duration of the current segment
      - next_break_length: duration of the *next* segment (per author/repo)
    """
    segments["start_date"] = pandas.to_datetime(segments["start_date"])
    segments["end_date"] = pandas.to_datetime(segments["end_date"])

    # 1. Current segment length
    segments["current_break_length"] = (segments["end_date"] - segments["start_date"]).dt.days + 1

    # 2. Next segment length: shift within each author+repo group
    segments = segments.sort_values(["author", "repo", "start_date"])
    segments["next_break_length"] = (
        segments.groupby(["author", "repo"])["current_break_length"]
        .shift(-1)
    )

    

    return segments

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

def make_splits(df: pandas.DataFrame, time_col: str = "date", train_frac: float = 0.80) -> Tuple[np.ndarray, np.ndarray]:
    # Always work on a fresh, 0..N-1 view to avoid label/position confusion
    dates = pandas.to_datetime(df[time_col], errors="coerce")
    if dates.isna().all():
        raise ValueError(f"No valid datetimes in column '{time_col}'")

    start_date = dates.min()
    end_date   = dates.max()
    if pandas.isna(start_date) or pandas.isna(end_date):
        raise ValueError("Cannot compute split; start or end date is NaT")

    span = end_date - start_date
    cutoff_date = start_date + pandas.Timedelta(seconds=span.total_seconds() * train_frac)

    mask_train = (dates <= cutoff_date)
    mask_test  = ~mask_train

    # Return **positions** (0..N-1), not index labels
    tr_idx = np.nonzero(mask_train.to_numpy())[0]
    te_idx = np.nonzero(mask_test.to_numpy())[0]
    return tr_idx, te_idx

def build_xy(df: pandas.DataFrame, feature_cols: list[str], target_col: str) -> Tuple[pandas.DataFrame, pandas.Series]:
    X = df[feature_cols].copy()
    y = df[target_col].astype("Int64").fillna(0).astype(int)
    return X, y

def _normalize_state_series(s: pandas.Series) -> pandas.Series:
    norm = (s.astype(str).str.strip().str.upper().str.replace("-", "_"))
    return norm.where(norm.isin(STATE_ORDER), other=np.nan)

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

def prepare_ml_tables_segments(seg_df: pandas.DataFrame):
    df = seg_df.copy()

    # normalize categorical state columns
    if "state_curr" in df.columns:
        df["state_curr"] = _normalize_state_series(df["state_curr"])
    if "next_state" in df.columns:
        df["next_state"] = _normalize_state_series(df["next_state"])

    # drop rows with unknown next_state (final segments)
    df = df[df["next_state"].notna()].copy()

    # feature engineering (you already did most; ensure presence)
    eps = 1e-9
    need_nums = [
        "seg_len",
        "commits_sum","pr_sum","issues_sum","issues_comments_sum","issues_events_sum","pr_comments_sum",
        "coding_days","nc_days","break_days",
        "avg_commits_per_day","avg_prs_per_day","avg_issues_per_day","avg_comments_per_day","avg_events_per_day","avg_pr_comments_per_day",
        "pct_coding_days","pct_noncoding_days","pct_break_days",
        "start_dow","start_month","start_qtr",
        "prev_seg_len",
        "roll3_seg_len_mean","roll3_seg_len_std",
        "roll3_commits_mean","roll3_nc_days_mean","roll3_break_days_mean",
        "roll5_seg_len_mean","roll5_seg_len_std",
        "roll5_commits_mean","roll5_nc_days_mean","roll5_break_days_mean",
    ]
    for c in need_nums:
        if c not in df.columns:
            df[c] = 0.0

    # categorical features
    df["prev_state"] = df.get("prev_state", pandas.Series(pandas.NA, index=df.index)).astype("string").fillna("∅").astype(str)
    df["state_t"]    = df.get("state_curr", pandas.Series(pandas.NA, index=df.index)).astype("string").fillna("∅").astype(str)
    df["prev_curr_pair"] = (df["prev_state"].astype(str) + "→" + df["state_t"].astype(str))

    # target
    cat = pandas.Categorical(df["next_state"], categories=STATE_ORDER, ordered=True)
    y = cat.codes.astype(np.int64)
    valid = y >= 0
    df = df.loc[valid].copy()
    y  = y[valid]
    df.reset_index(drop=True, inplace=True)


    # design matrix
    X_num = df[need_nums].astype(float)
    X_cat = pandas.get_dummies(
        df[["state_t","prev_state","prev_curr_pair"]].astype("category"),
        prefix=["state_t","prev_state","pair"],
        dummy_na=False
    )
    X = pandas.concat([X_num, X_cat], axis=1)
    feature_cols = list(X.columns)

    return df, X, y, feature_cols

def prepare_break_regression_table(
    features_df: pandas.DataFrame,
    segments_with_targets: pandas.DataFrame,
    *,
    which: str,  # "current" or "next"
    break_states: set[str] = BREAK_STATES,
    use_log_target: bool = True,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, np.ndarray, str]:
    """
    Returns:
      dfR : rows aligned to features_df (subset) with keys & target
      X   : numeric/categorical design matrix
      y   : target vector (log1p if use_log_target)
      target_col : the name of the (non-logged) target column
    """
    if which not in {"current","next"}:
        raise ValueError("which must be 'current' or 'next'")

    # Join targets onto features_df by keys
    KEYS = ["author","repo","segment_id"]
    base = features_df.copy()
    # Ensure required targets exist on base; if not, merge only those missing:
    needed = {"current_break_length","next_break_length","next_state","start_date","state_curr"}
    missing = [c for c in needed if c not in base.columns]
    if missing:
        base = base.merge(segments_with_targets[["author","repo","segment_id"] + missing],
                        on=["author","repo","segment_id"], how="left", validate="one_to_one")

    if which == "current":
        target_col = "current_break_length"
        row_mask = base["state_curr"].isin(break_states) & base[target_col].notna()
    else:
        target_col = "next_break_length"
        row_mask = base["next_state"].isin(break_states) & base[target_col].notna()

    dfR = base.loc[row_mask].copy()
    if dfR.empty:
        raise ValueError(f"No rows available for {which}-break regression (check labels or break_states).")

    # === Design matrix: reuse your classifier feature columns as much as possible ===
    # Numeric set (same as you had in prepare_ml_tables_segments, plus safe fallbacks)
    eps = 1e-9
    need_nums = [
        "seg_len",
        "commits_sum","pr_sum","issues_sum","issues_comments_sum","issues_events_sum","pr_comments_sum",
        "coding_days","nc_days","break_days",
        "avg_commits_per_day","avg_prs_per_day","avg_issues_per_day","avg_comments_per_day","avg_events_per_day","avg_pr_comments_per_day",
        "pct_coding_days","pct_noncoding_days","pct_break_days",
        "start_dow","start_month","start_qtr",
        "prev_seg_len",
        "roll3_seg_len_mean","roll3_seg_len_std",
        "roll3_commits_mean","roll3_nc_days_mean","roll3_break_days_mean",
        "roll5_seg_len_mean","roll5_seg_len_std",
        "roll5_commits_mean","roll5_nc_days_mean","roll5_break_days_mean",
        # tails you added
        "tail3_event_rate","tail7_event_rate","tail14_event_rate",
        "tail3_coding_rate","tail7_coding_rate","tail14_coding_rate",
    ]
    for c in need_nums:
        if c not in dfR.columns:
            dfR[c] = 0.0

    # Categorical set reused
    dfR["prev_state"] = dfR.get("prev_state", pandas.Series(pandas.NA, index=dfR.index)).astype("string").fillna("∅").astype(str)
    dfR["state_t"]    = dfR.get("state_t",    pandas.Series(pandas.NA, index=dfR.index)).astype("string").fillna("∅").astype(str)
    dfR["prev_curr_pair"] = (dfR["prev_state"].astype(str) + "→" + dfR["state_t"].astype(str))

    X_num = dfR[need_nums].astype(float)
    X_cat = pandas.get_dummies(
        dfR[["state_t","prev_state","prev_curr_pair"]].astype("category"),
        prefix=["state_t","prev_state","pair"],
        dummy_na=False
    )
    X = pandas.concat([X_num, X_cat], axis=1)

    y_raw = dfR[target_col].astype(float).values
    y = np.log1p(y_raw) if use_log_target else y_raw

    return dfR, X, y, target_col

def train_classifier(X_tr, y_tr, X_val, y_val):
    K = len(STATE_ORDER)
    binc = np.bincount(y_tr, minlength=K)
    tot = binc.sum()
    wts = {i: (tot/(K*max(1,int(c)))) for i,c in enumerate(binc)}
    sw = np.vectorize(wts.get)(y_tr)

    model = XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=K,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        eval_metric=["mlogloss","merror"]  # keep both
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=sw,
        eval_set=[(X_val, y_val)],
        # scikit wrapper accepts list of (name, X, y) since xgboost>=2.0
        verbose=False
    )
    return model

def train_regressor(X_tr, y_tr, X_val=None, y_val=None):
    model = XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    if X_val is not None and y_val is not None:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)
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
    return seg_out

def attach_break_length_predictions(
    segments_raw: pandas.DataFrame,   # from segmentize_timeline(...)[1] with add_targets()
    reg_df: pandas.DataFrame,         # dfR returned by prepare_break_regression_table
    X_reg,                            # aligned design matrix for reg_df
    model,                            # trained regressor
    val_mask: np.ndarray,             # boolean mask over reg_df rows (NOT over full features_df)
    *,
    which: str,                       # "current" or "next"
    used_log_target: bool = True
) -> pandas.DataFrame:

    if which not in {"current","next"}:
        raise ValueError("which must be 'current' or 'next'")

    # Predict on held-out portion of *this regression view*
    X_te = X_reg[val_mask]
    yhat = model.predict(X_te)
    if used_log_target:
        yhat = np.expm1(yhat)

    preds = reg_df.loc[val_mask, ["author","repo","segment_id"]].copy()
    if which == "current":
        preds["pred_current_break_len"]  = yhat
        preds["actual_current_break_len"]= reg_df.loc[val_mask, "current_break_length"].astype(float).values
        preds["ae_current_break_len"]    = (preds["pred_current_break_len"] - preds["actual_current_break_len"]).abs()
    else:
        preds["pred_next_break_len"]   = yhat
        preds["actual_next_break_len"] = reg_df.loc[val_mask, "next_break_length"].astype(float).values
        preds["ae_next_break_len"]     = (preds["pred_next_break_len"] - preds["actual_next_break_len"]).abs()

    # Merge onto segments_raw
    KEYS = ["author","repo","segment_id"]
    out = segments_raw.merge(preds.drop_duplicates(subset=KEYS), on=KEYS, how="left", validate="one_to_one")

    return out

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

def predict_state(repo , tf_devs, use_all=True):
    #we get the repo we are working on and the labeled data for all users in that repo

    if use_all:
        # train on the whole corpus
        raw = users_activity_all_repos(authors=None)
    else:
        # train on this one repo only
        raw = users_activity_for_repo(repo, tf_devs)

    print("\n=== RAW DAILY ===")
    print("shape:", raw.shape)
    print("cols:", raw.columns.tolist())
    print("date range:", raw["date"].min(), "→", raw["date"].max())
    print(raw.head(5))

    # 1a) SEGMENTIZE
    daily_with_segments, segments = segmentize_timeline(raw, state_order=STATE_ORDER)

    segments = add_targets(segments)

    tf = tail_features(daily_with_segments)
    segments = segments.merge(tf, on=["author","repo","segment_id"], how="left")

    print("\n=== SEGMENTS (with targets & tails) ===")
    print("segments shape:", segments.shape)
    print("cols:", segments.columns.tolist())
    print("segment dates:", segments["start_date"].min(), "→", segments["start_date"].max())
    print(segments.head(5))
    # 2) FEATURES (segment-level)
    data = add_predictors_segments(segments)  

    print("\n=== FEATURES (segment-level) ===")
    print("data shape:", data.shape)
    new_cols = [c for c in data.columns if c not in segments.columns]
    print("feature cols added:", new_cols)
    print(data[["author","repo","segment_id","date","state_t","target_next_state"] + new_cols[:8]].head(5))
    # 3b) TABLES (segment-level)
    seg_df, X, y, feature_cols = prepare_ml_tables_segments(data)


    tr_idx, te_idx = make_splits(seg_df, time_col="start_date")
    X_tr, y_tr = X.iloc[tr_idx].to_numpy(), y[tr_idx]
    X_te, y_te = X.iloc[te_idx].to_numpy(), y[te_idx]

    print("\n=== ML TABLE (next-state) ===")
    print("seg_df shape:", seg_df.shape, "| X:", X.shape, "| y:", y.shape)
    print("feature_cols (n={}):".format(len(feature_cols)))
    print(feature_cols[:15], "...")

    cutoff_guess = seg_df.loc[tr_idx, "start_date"].max()
    print("split cutoff ~", cutoff_guess)
    print("train rows:", len(tr_idx), "test rows:", len(te_idx))
    print("train date range:", seg_df.loc[tr_idx, "start_date"].min(), "→", seg_df.loc[tr_idx, "start_date"].max())
    print("test  date range:", seg_df.loc[te_idx, "start_date"].min(), "→", seg_df.loc[te_idx, "start_date"].max())

    # quick sanity: ensure no target columns accidentally in features
    target_like = {"next_state","current_break_length","next_break_length"}
    leak_feats = [c for c in feature_cols if c in target_like]
    print("!! target-like features present in X:", leak_feats)

    # 4) TRAIN
    model = train_classifier(X_tr, y_tr, X_te, y_te)

    val_mask = np.zeros(len(seg_df), dtype=bool)
    val_mask[te_idx] = True

    # (optional) filter to selected repo + TF devs in the output table
    only_author = None   # e.g., set from a Streamlit selectbox if you want
    out_csv = r"C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\PredictionModel\predictions_segments.csv"

    out = attach_predictions_to_segments(
        segments_raw=segments,
        features_df=data,
        X=X.values,                     # XGB accepts numpy arrays; you can also pass DataFrame
        model=model,
        val_mask=val_mask,
        state_order=STATE_ORDER,
        out_csv_path=out_csv,
        filter_to_val_authors=True
    )
    # ===== New: REGRESSION — current break length =====
    dfR_cur, X_cur, y_cur, tgt_cur = prepare_break_regression_table(
        features_df=data,
        segments_with_targets=segments,
        which="current",
        break_states=BREAK_STATES,
        use_log_target=True
    )
    # Build a time split for this regression view using its own dates
    tr_idx_cur, te_idx_cur = make_splits(dfR_cur, time_col="start_date")

    print("\n=== REGRESSION (CURRENT break length) ===")
    print("dfR_cur shape:", dfR_cur.shape, "| X_cur:", X_cur.shape, "| y_cur:", y_cur.shape)
    print("date range:", dfR_cur["start_date"].min(), "→", dfR_cur["start_date"].max())
    print("sample rows:")
    print(dfR_cur[["author","repo","segment_id","state_curr","current_break_length","seg_len"]].head(5))

    # suspicious features check: these are aggregates over the WHOLE segment
    suspicious = [c for c in ["seg_len","commits_sum","pr_sum","issues_sum",
                            "issues_comments_sum","issues_events_sum","pr_comments_sum",
                            "avg_*","pct_*","roll*","tail*"] if any(fnmatch.fnmatchcase(x, c) for x in X_cur.columns)]
    print("!! POSSIBLE leakage features in CURRENT task (computed over the same break): found", len(suspicious), "patterns")
    print("current split: train", len(tr_idx_cur), "test", len(te_idx_cur))
    print("current train date range:", dfR_cur.loc[tr_idx_cur, "start_date"].min(), "→", dfR_cur.loc[tr_idx_cur, "start_date"].max())
    print("current test  date range:", dfR_cur.loc[te_idx_cur, "start_date"].min(), "→", dfR_cur.loc[te_idx_cur, "start_date"].max())

    model_cur = train_regressor(X_cur.iloc[tr_idx_cur].to_numpy(), y_cur[tr_idx_cur],
                                X_cur.iloc[te_idx_cur].to_numpy(),  y_cur[te_idx_cur])

    val_mask_cur = np.zeros(len(dfR_cur), dtype=bool); val_mask_cur[te_idx_cur] = True
    seg_with_cur_pred = attach_break_length_predictions(
        segments_raw=out,
        reg_df=dfR_cur,
        X_reg=X_cur.values,
        model=model_cur,
        val_mask=val_mask_cur,
        which="current",
        used_log_target=True
    )

    # ===== New: REGRESSION — next break length =====
    dfR_next, X_next, y_next, tgt_next = prepare_break_regression_table(
        features_df=data,
        segments_with_targets=segments,
        which="next",
        break_states=BREAK_STATES,
        use_log_target=True
    )
    tr_idx_next, te_idx_next = make_splits(dfR_next, time_col="start_date")
    model_next = train_regressor(X_next.iloc[tr_idx_next].to_numpy(), y_next[tr_idx_next],
                                 X_next.iloc[te_idx_next].to_numpy(),  y_next[te_idx_next])

    val_mask_next = np.zeros(len(dfR_next), dtype=bool); val_mask_next[te_idx_next] = True
    out = attach_break_length_predictions(
        segments_raw=seg_with_cur_pred,
        reg_df=dfR_next,
        X_reg=X_next.values,
        model=model_next,
        val_mask=val_mask_next,
        which="next",
        used_log_target=True
    )
    
    # ===== EARLY WARNING (binary) =====
    EW_HORIZON = 7
    EW_MIN_BREAK = 3

    ew_base, X_ew, y_ew, _ = build_early_warning_table(
        features_df=data,
        segments_with_targets=segments,
        break_states=BREAK_STATES,
        horizon_days=EW_HORIZON,
        min_break_len=EW_MIN_BREAK,
    )
    print("\n=== EARLY WARNING TABLE ===")
    print("ew_base shape:", ew_base.shape, "| X_ew:", X_ew.shape, "| y_ew:", y_ew.shape)
    print("date range:", ew_base["start_date"].min(), "→", ew_base["start_date"].max())
    print("positive rate:", float(y_ew.mean()) if len(y_ew) else np.nan)
    print(ew_base[["author","repo","segment_id","seg_len","next_state","next_break_length"]].head(5))

    # use the same time-based split helper
    tr_idx_ew, te_idx_ew = make_splits(ew_base, time_col="start_date")
    model_ew = train_binary_classifier(
        X_ew.iloc[tr_idx_ew].to_numpy(), y_ew[tr_idx_ew],
        X_ew.iloc[te_idx_ew].to_numpy(), y_ew[te_idx_ew]
    )
    print("EW split: train", len(tr_idx_ew), "test", len(te_idx_ew))
    print("EW train date range:", ew_base.loc[tr_idx_ew, "start_date"].min(), "→", ew_base.loc[tr_idx_ew, "start_date"].max())
    print("EW test  date range:", ew_base.loc[te_idx_ew, "start_date"].min(), "→", ew_base.loc[te_idx_ew, "start_date"].max())

    val_mask_ew = np.zeros(len(ew_base), dtype=bool); val_mask_ew[te_idx_ew] = True
    out = attach_early_warning_predictions(
        segments_raw=out,            # <- the table that already has multi-class outputs
        ew_df=ew_base,
        X_ew=X_ew.values,
        model=model_ew,
        val_mask=val_mask_ew,
        out_col_prob="p_break_soon",
        out_col_pred="ew_alert"
    )

    with st.expander("Per-segment predictions (held-out)"):
    # CHANGE: use the *function parameters* instead of outer-scope vars
        sel = out.copy()
        # nice columns to surface
        show_cols = [
            "author","repo","segment_id","start_date","end_date",
            "state_curr","next_state","pred_next_state",
            "current_break_length","pred_current_break_len","ae_current_break_len",
            "next_break_length","pred_next_break_len","ae_next_break_len", "p_break_soon", "ew_alert"
        ]
        show_cols = [c for c in show_cols if c in sel.columns]

        # CHANGE: don't drop everything on accidental NaNs; do it later if you must
        sel = sel.sort_values(["author","start_date","segment_id"], na_position="last").reset_index(drop=True)

        # CHANGE: coerce to datetime before .dt.strftime (or it will crash)
        for col in ("start_date","end_date"):
            if col in sel.columns:
                if not pandas.api.types.is_datetime64_any_dtype(sel[col]):
                    sel[col] = pandas.to_datetime(sel[col], errors="coerce", utc=True)
                sel[col] = sel[col].dt.strftime("%Y-%m-%d")
        st.write(sel[show_cols])
    return out

#----------------
#test 
#----------------

def build_early_warning_table(
    features_df: pandas.DataFrame,
    segments_with_targets: pandas.DataFrame,
    *,
    break_states: set[str] = BREAK_STATES,
    horizon_days: int = 7,
    min_break_len: int = 3,
) -> tuple[pandas.DataFrame, pandas.DataFrame, np.ndarray, list[str]]:
    """
    Returns a binary-early-warning training view aligned to your segment features.

    Positive (1): next state is a BREAK state AND the transition happens within `horizon_days`
                  (i.e., current segment length <= horizon)
                  AND (optionally) the next break lasts at least `min_break_len` days.
    """
    KEYS = ["author","repo","segment_id"]

    base = features_df.copy()
    needed = {"next_state","seg_len","next_break_length"}
    missing = [c for c in needed if c not in base.columns]
    if missing:
        base = base.merge(segments_with_targets[KEYS + list(needed)],
                          on=KEYS, how="left", validate="one_to_one")

    is_break_next = base["next_state"].isin(break_states)
    within_horizon = base["seg_len"].astype(float) <= float(horizon_days)
    long_enough = base["next_break_length"].fillna(0).astype(float) >= float(min_break_len)

    y = (is_break_next & within_horizon & long_enough).astype(int)

    # Reuse your segment design matrix just like prepare_ml_tables_segments
    # (categoricals + numerics → X)
    df_for_X = base.copy()
    # Ensure the same numeric set as in prepare_ml_tables_segments
    _, X, _, feature_cols = prepare_ml_tables_segments(df_for_X)
    # But we replace y with the binary EW target and keep all rows that have y defined
    # (prepare_ml_tables_segments() filters out rows with unknown next_state already)

    return base.reset_index(drop=True), X, y.to_numpy(), feature_cols

def train_binary_classifier(X_tr, y_tr, X_val=None, y_val=None):
    # Simple XGBoost binary; consistent with your other trainers
    model = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        objective="binary:logistic",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        eval_metric=["logloss","auc"]
    )
    if X_val is not None and y_val is not None:
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)
    return model

def attach_early_warning_predictions(
    segments_raw: pandas.DataFrame,     # original segments with keys
    ew_df: pandas.DataFrame,            # base returned by build_early_warning_table (aligned to features)
    X_ew,                               
    model,                              
    val_mask: np.ndarray,               # boolean over ew_df rows (time-based split)
    *,
    out_col_prob: str = "p_break_soon",
    out_col_pred: str = "ew_alert"
) -> pandas.DataFrame:
    # Predict probabilities on held-out rows
    X_val = X_ew[val_mask]
    p = model.predict_proba(X_val)[:, 1]  # probability of positive
    yhat = (p >= 0.5).astype(int)         # default threshold; adjust in UI

    preds = ew_df.loc[val_mask, ["author","repo","segment_id"]].copy()
    preds[out_col_prob] = p
    preds[out_col_pred] = yhat

    KEYS = ["author","repo","segment_id"]
    out = segments_raw.merge(
        preds.drop_duplicates(subset=KEYS), on=KEYS, how="left", validate="one_to_one"
    )
    return out

def attach_seg_predictions_to_daily(daily_df, seg_preds):
    """
    daily_df: one row per day per dev; must include ['dev','date','state'].
              must include seg_id that groups consecutive equal states (no leakage).
    seg_preds: one row per (dev, seg_id) with your CURRENT model outputs at boundaries, e.g.:
               ['dev','seg_id','pred_next_label','p_ACTIVE','p_NON_CODING','p_INACTIVE','p_GONE',
                'pred_next_start_date' (optional)]
    Returns daily_df with the segment-level predictions replicated to all days in that segment.
    """
    # Ensure seg_id exists on daily
    daily_df = daily_df.sort_values(['dev','date']).copy()
    if 'seg_id' not in daily_df.columns:
        daily_df['seg_id'] = (
            (daily_df['state'] != daily_df.groupby('dev')['state'].shift())
            .groupby(daily_df['dev']).cumsum()
        )

    keep_cols = ['dev','seg_id','pred_next_label','p_ACTIVE','p_NON_CODING','p_INACTIVE','p_GONE']
    if 'pred_next_start_date' in seg_preds.columns:
        keep_cols.append('pred_next_start_date')

    out = daily_df.merge(seg_preds[keep_cols], on=['dev','seg_id'], how='left')
    return out

def _attach_daily_targets_from_segments(daily_with_segments: pandas.DataFrame,
                                        segments: pandas.DataFrame,
                                        horizon_days: int = 14) -> pandas.DataFrame:
    """
    Adds next segment start/label to EACH daily row, then builds:
      - lead_time_days: days until next segment starts
      - inactive_in_H: 1 if next label is INACTIVE or NON_CODING within horizon_days
    NOTE: uses only segment_id grouping (no look-ahead leakage in features).
    """
    d = daily_with_segments.copy().sort_values(["author","repo","date"])
    s = segments.copy().sort_values(["author","repo","segment_id"])
    # compute next segment's start and label at the segment table
    s["next_label"] = s.groupby(["author","repo"])["state_curr"].shift(-1)
    s["next_start"] = s.groupby(["author","repo"])["start_date"].shift(-1)

    # bring next_* onto each daily row via (author,repo,segment_id)
    keep = ["author","repo","segment_id","next_label","next_start"]
    d = d.merge(s[keep], on=["author","repo","segment_id"], how="left")

    # lead time / target
    d["lead_time_days"] = (pandas.to_datetime(d["next_start"]) - pandas.to_datetime(d["date"])).dt.days
    d["inactive_in_H"] = ((d["next_label"] == "INACTIVE") | (d["next_label"] == "NON_CODING")) & (d["lead_time_days"].le(horizon_days))
    return d

def _make_daily_features_simple(d: pandas.DataFrame) -> pandas.DataFrame:
    """
    Minimal, prefix-safe daily features that only use info up to 'date'.
    Requires: ['author','repo','date','state','coding_day','event_day','nc_day','segment_id','segment_pos']
    """
    df = d.sort_values(["author","repo","date"]).copy()

    # Helper counts
    df["any_event"] = df["event_day"].astype(int)

    # Days-since features (per author,repo)
    def _days_since(series):
        grp = (series > 0)
        return ((~grp).astype(int).groupby(grp.cumsum()).cumsum()).astype(int)

    df["days_since_code"] = (df.groupby(["author","repo"])["coding_day"]
                               .apply(_days_since).reset_index(level=[0,1], drop=True))
    df["days_since_any"] = (df.groupby(["author","repo"])["any_event"]
                               .apply(_days_since).reset_index(level=[0,1], drop=True))

    # Rolling sums (include current day; prefix-safe)
    for k in (7, 14, 30):
        df[f"code_days_{k}"] = (df.groupby(["author","repo"])["coding_day"]
                                  .rolling(k, min_periods=1).sum().reset_index(level=[0,1], drop=True))
        df[f"nc_days_{k}"] = (df.groupby(["author","repo"])["nc_day"]
                                  .rolling(k, min_periods=1).sum().reset_index(level=[0,1], drop=True))
        df[f"any_days_{k}"] = (df.groupby(["author","repo"])["any_event"]
                                  .rolling(k, min_periods=1).sum().reset_index(level=[0,1], drop=True))

    # Calendar
    dt = pandas.to_datetime(df["date"])
    df["dow"] = dt.dt.dayofweek
    df["month_sin"] = np.sin(2*np.pi*(dt.dt.month/12.0))
    df["month_cos"] = np.cos(2*np.pi*(dt.dt.month/12.0))

    # Segment position if missing
    if "segment_pos" not in df.columns:
        df["segment_pos"] = df.groupby(["author","repo","segment_id"]).cumcount() + 1

    # One-hot current state, then attach ONLY the OHE columns (avoid duplicating numerics)
    state_ohe = pandas.get_dummies(df["state"].astype(str), prefix="state")
    df = pandas.concat([df, state_ohe], axis=1)

    # Numeric/base feature columns already live in df (don’t concat them again)
    num_cols = [
        "segment_pos", "days_since_code", "days_since_any",
        "code_days_7","code_days_14","code_days_30",
        "nc_days_7","nc_days_14","nc_days_30",
        "any_days_7","any_days_14","any_days_30",
        "dow","month_sin","month_cos",
    ]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = df[c].astype(float, copy=False)

    # FINAL: set attrs AFTER all ops that might replace df
    feat_cols = num_cols + list(state_ohe.columns)
    df.attrs["daily_feature_cols"] = feat_cols
    return df

def prediction_analysis(daily_preds, repo_key):
    """
    Adds:
      1) Per-Author accuracy (N>=100) bar chart
      2) Per-Repo accuracy bar chart
      3) Drilldowns: repo picker and author picker, each with ALERT vs WATCH accuracy + counts

    Expects daily_preds to have at least:
      ['author','repo','date','state','p_inactive_14','alert_level_pred','alert_level_exp']
    Where:
      - alert_level_pred in {'OK','WATCH','ALERT'}
      - alert_level_exp is boolean (True = an alert-worthy event actually happened)
    """

    # Optional: Altair for nice bars; fall back to st.bar_chart if unavailable
           
    

    # ---------- Clean / normalize ----------
    df = daily_preds.copy()
    df["date"] = pandas.to_datetime(df["date"]).dt.date
    df["alert_level_pred"] = df["alert_level_pred"].astype(str).str.upper().str.strip()
    if df["alert_level_exp"].dtype != bool:
        df["alert_level_exp"] = df["alert_level_exp"].astype(bool)
    df["p_inactive_14"] = pandas.to_numeric(df["p_inactive_14"], errors="coerce")

    # ---------- Today's view (keep your original table) ----------
    today_view = (df.sort_values(["date","p_inactive_14"], ascending=[False, False])
                    .loc[lambda x: x["date"] == x["date"].max(), 
                         ["author","date","state","p_inactive_14","alert_level_pred"]])
    st.write(f"### Daily Early Warning (7d) for {repo_key}")
    st.dataframe(today_view, use_container_width=True)

    preds_mask = df["alert_level_pred"].isin(["ALERT","WATCH"])
    preds = df.loc[preds_mask].copy()
    preds["correct"] = preds["alert_level_exp"]  # True == correct guess

    # ---------- Metrics helpers ----------
    # Overall accuracy by group (among guesses)
    def _overall_acc(g):
        return pandas.Series({
            "n": len(g),
            "accuracy": float(g["alert_level_exp"].mean()) if len(g) else np.nan
        })

    # Accuracy by category (ALERT/WATCH) within group
    def _by_cat(df_, group_cols):
        tmp = (df_.groupby(group_cols + ["alert_level_pred"])["alert_level_exp"]
                 .agg(n="size", accuracy="mean")
                 .reset_index())
        acc_pivot = tmp.pivot(index=group_cols, columns="alert_level_pred", values="accuracy").rename_axis(None, axis=1)
        n_pivot   = tmp.pivot(index=group_cols, columns="alert_level_pred", values="n").rename_axis(None, axis=1)
        acc_pivot = acc_pivot.add_suffix("")  # keep columns 'ALERT','WATCH'
        n_pivot   = n_pivot.add_prefix("n_")  # columns 'n_ALERT','n_WATCH'
        return acc_pivot, n_pivot

    # ---------- Per-author ----------
    per_author_all = preds.groupby("author").apply(_overall_acc).reset_index()
    accA, nA = _by_cat(preds, ["author"])
    author_metrics = (per_author_all
                      .merge(accA, on="author", how="left")
                      .merge(nA, on="author", how="left")
                      .fillna({"ALERT":0.0, "WATCH":0.0, "n_ALERT":0, "n_WATCH":0})
                     )
    author_metrics_100 = author_metrics[author_metrics["n"] >= 100] \
                            .sort_values(["accuracy","n"], ascending=[False, False])

    # ---------- Per-repo ----------
    per_repo_all = preds.groupby("repo").apply(_overall_acc).reset_index()
    accR, nR = _by_cat(preds, ["repo"])
    repo_metrics = (per_repo_all
                    .merge(accR, on="repo", how="left")
                    .merge(nR, on="repo", how="left")
                    .fillna({"ALERT":0.0, "WATCH":0.0, "n_ALERT":0, "n_WATCH":0})
                   ) \
                   .sort_values(["accuracy","n"], ascending=[False, False])

    # ---------- Graph 1: Per-Author Accuracy (N ≥ 100) ----------
    st.subheader("Per-Author Metrics (N≥100) — Accuracy (PPV on WATCH+ALERT)")
    top_auth = author_metrics_100[["author","accuracy","n","ALERT","WATCH","n_ALERT","n_WATCH"]]
    if not top_auth.empty:
        chart_auth = (alt.Chart(top_auth)
                        .mark_bar()
                        .encode(
                            x=alt.X("accuracy:Q", title="Accuracy (PPV)"),
                            y=alt.Y("author:N", sort="-x", title="Author"),
                            tooltip=[
                                alt.Tooltip("author:N"),
                                alt.Tooltip("n:Q", title="Total guesses"),
                                alt.Tooltip("accuracy:Q", format=".3f", title="Overall PPV"),
                                alt.Tooltip("ALERT:Q", format=".3f", title="ALERT PPV"),
                                alt.Tooltip("WATCH:Q", format=".3f", title="WATCH PPV"),
                                alt.Tooltip("n_ALERT:Q", title="n ALERT"),
                                alt.Tooltip("n_WATCH:Q", title="n WATCH"),
                            ],
                        ))
        st.altair_chart(chart_auth, use_container_width=True)
    else:
        st.bar_chart(top_auth.set_index("author")["accuracy"])

    # ---------- Graph 2: Per-Repo Accuracy ----------
    st.subheader("Overall & Per-Repo Summary — Accuracy (PPV on WATCH+ALERT)")
    rep_plot_df = repo_metrics[["repo","accuracy","n","ALERT","WATCH","n_ALERT","n_WATCH"]]
    if not rep_plot_df.empty:
        chart_repo = (alt.Chart(rep_plot_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("accuracy:Q", title="Accuracy (PPV)"),
                            y=alt.Y("repo:N", sort="-x", title="Repository"),
                            tooltip=[
                                alt.Tooltip("repo:N"),
                                alt.Tooltip("n:Q", title="Total guesses"),
                                alt.Tooltip("accuracy:Q", format=".3f", title="Overall PPV"),
                                alt.Tooltip("ALERT:Q", format=".3f", title="ALERT PPV"),
                                alt.Tooltip("WATCH:Q", format=".3f", title="WATCH PPV"),
                                alt.Tooltip("n_ALERT:Q", title="n ALERT"),
                                alt.Tooltip("n_WATCH:Q", title="n WATCH"),
                            ],
                        ))
        st.altair_chart(chart_repo, use_container_width=True)
    else:
        st.bar_chart(rep_plot_df.set_index("repo")["accuracy"])

    # ---------- Drilldowns ----------
    st.subheader("Inspect by Repo or Author")

    # Repo drilldown
    repos_available = sorted(preds["repo"].dropna().unique().tolist())
    default_repo_ix = repos_available.index(repo_key) if (repo_key in repos_available) else 0 if repos_available else 0
    repo_sel = st.selectbox("Select a repo", repos_available, index=default_repo_ix if repos_available else 0)
    r = preds[preds["repo"] == repo_sel]
    if not r.empty:
        repo_summary = (r.groupby("alert_level_pred")["alert_level_exp"]
                          .agg(n="size", accuracy="mean").reset_index())
        st.write(f"Repo **{repo_sel}** — n={len(r)}, overall PPV={r['alert_level_exp'].mean():.3f}")
        st.dataframe(repo_summary, use_container_width=True)
        
        st.altair_chart(
            alt.Chart(repo_summary).mark_bar().encode(
                x=alt.X("accuracy:Q", title="Accuracy (PPV)"),
                y=alt.Y("alert_level_pred:N", sort="-x", title="Category"),
                tooltip=[alt.Tooltip("alert_level_pred:N"), alt.Tooltip("n:Q"), alt.Tooltip("accuracy:Q", format=".3f")]
            ),
            use_container_width=True
        )
    

    # Author drilldown
    authors_available = sorted(preds["author"].dropna().unique().tolist())
    author_sel = st.selectbox("Select an author", authors_available, index=0 if authors_available else 0)
    a = preds[preds["author"] == author_sel]
    if not a.empty:
        author_summary = (a.groupby("alert_level_pred")["alert_level_exp"]
                            .agg(n="size", accuracy="mean").reset_index())
        st.write(f"Author **{author_sel}** — n={len(a)}, overall PPV={a['alert_level_exp'].mean():.3f}")
        st.dataframe(author_summary, use_container_width=True)
        
        st.altair_chart(
            alt.Chart(author_summary).mark_bar().encode(
                x=alt.X("accuracy:Q", title="Accuracy (PPV)"),
                y=alt.Y("alert_level_pred:N", sort="-x", title="Category"),
                tooltip=[alt.Tooltip("alert_level_pred:N"), alt.Tooltip("n:Q"), alt.Tooltip("accuracy:Q", format=".3f")]
            ),
            use_container_width=True
        )
    # ---------- “Correct vs incorrect guesses” table (preds only) ----------
    st.write("Correct vs incorrect guesses (WATCH/ALERT only)")
    st.dataframe(preds[["author","repo","date","alert_level_pred","alert_level_exp","correct"]],
                 use_container_width=True)

    # ---------- Keep your end tables ----------
    st.write("Daily Predictions (raw)")
    st.dataframe(df, use_container_width=True)


    # Return the summary frames in case you want to use them elsewhere
    return author_metrics_100, repo_metrics

def predict_daily_early_warning(repo_key: str,
                                authors: list[str] | None,
                                *,
                                use_all: bool = True,
                                horizon_days: int = 14,
                                p_watch: float = 0.75,
                                p_alert: float = 0.85) -> pandas.DataFrame:
    """
    End-to-end: loads daily data, builds targets+features, trains a tiny XGB binary model,
    and returns per-day risk p_inactive_14 with alert labels.
    """
    dbg = _mk_debug_dir(prefix=f"dailyEW-{repo_key.replace('/','_')}")
    
    # 1) Load data the same way your segment model does
    if use_all:
        raw = users_activity_all_repos(authors=None)
    else:
        raw = users_activity_for_repo(repo_key, authors)
        

    if raw.empty:
        return pandas.DataFrame()
    _dump_df(raw, dbg, "raw_empty")

    # 2) Segmentize (you already have this in your pipeline)  :contentReference[oaicite:0]{index=0}
    daily_with_segments, segments = segmentize_timeline(raw, state_order=STATE_ORDER)

    daily_tgt = _attach_daily_targets_from_segments(daily_with_segments, segments, horizon_days=horizon_days)
    daily = _make_daily_features_simple(daily_tgt)
    _dump_df(daily, dbg, "daily")
    _dump_df(segments, dbg, "segments")

    daily = daily.loc[:, ~daily.columns.duplicated()].copy()

    # Capture features from attrs, or rebuild robustly by pattern
    _feat_cols = daily.attrs.get("daily_feature_cols")
    if not _feat_cols:
        base = [
            "segment_pos","days_since_code","days_since_any",
            "code_days_7","code_days_14","code_days_30",
            "nc_days_7","nc_days_14","nc_days_30",
            "any_days_7","any_days_14","any_days_30",
            "dow","month_sin","month_cos",
        ]
        state_cols = [c for c in daily.columns if c.startswith("state_")]
        _feat_cols = [c for c in base + state_cols if c in daily.columns]
        daily.attrs["daily_feature_cols"] = _feat_cols  # persist for later
        

    # NEW: capture before pandas ops drop attrs
    _feat_cols = daily.attrs.get("daily_feature_cols", None)

    # Keep rows that actually have a known next segment (drop last segments)
    m = daily["inactive_in_H"].notna()
    daily = daily.loc[m].reset_index(drop=True)

    # Re-attach (reset_index can drop attrs in some pandas versions)
    if _feat_cols is not None:
        daily.attrs["daily_feature_cols"] = _feat_cols

    feat_cols = daily.attrs["daily_feature_cols"]

    _dump_json({"feature_cols": feat_cols,
                "all_cols": list(daily.columns)}, dbg, "06_feature_columns")
    _dump_df(daily[["author","repo","date","segment_id","state","inactive_in_H"] + feat_cols], dbg, "06a_daily_for_model")

    # 4) Train/test split by time
    tr_idx, te_idx = make_splits(daily, time_col="date")
    #print the columns and split info

    split_meta = {
            "train_n": int(len(tr_idx)),
            "test_n": int(len(te_idx)),
            "train_date_min": str(daily.loc[tr_idx, "date"].min()),
            "train_date_max": str(daily.loc[tr_idx, "date"].max()),
            "test_date_min":  str(daily.loc[te_idx, "date"].min()),
            "test_date_max":  str(daily.loc[te_idx, "date"].max()),
        }
    _dump_json(split_meta, dbg, "07_split_meta")

    X_all = daily[feat_cols].fillna(0.0).to_numpy()
    y_all = daily["inactive_in_H"].astype(int).to_numpy()

    meta_cols = ["author","repo","date","segment_id","state","inactive_in_H"]

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te, y_te = X_all[te_idx], y_all[te_idx]


    # 5) Train a tiny, robust XGB binary model
    pos_weight = max(1.0, float((y_tr == 0).sum()) / max(1, (y_tr == 1).sum()))
    m_bin = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        eval_metric=["logloss","auc"]
    )
    m_bin.fit(X_tr, y_tr)

    # 6) Predict for ALL days (so you can visualize continuously)
    p = m_bin.predict_proba(X_all)[:, 1]
    out = daily[["author","repo","date","segment_id","state"]].copy()
    out["p_inactive_14"] = p
    # Smarter Alert Levels
    out["alert_level_pred"] = np.where(p >= p_alert, "ALERT",
                           np.where(p >= p_watch, "WATCH", "OK"))
    # (Optional) include ground truth lead time for inspection
    out["alert_level_exp"] = daily_tgt["inactive_in_H"].astype(bool)  #make into boolean
    out["lead_time_days_truth"] = daily["lead_time_days"]

    _dump_df(out, dbg, "11_predictions")
    
    if out.empty:
        st.warning("No data available to compute daily predictions.")
    else:
        prediction_analysis(out, repo_key=repo_key)

    return out

#DELEATE LATER
def _dump_df(df: pandas.DataFrame, outdir: Path, name: str, *, index: bool=False):
    p = outdir / f"{name}.csv"
    df.to_csv(p, index=index)
    print(f"[dump] {name}: {df.shape} -> {p}")

def _dump_json(obj, outdir: Path, name: str):
    p = outdir / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[dump] {name} -> {p}")

def _dump_xy(X: np.ndarray, y: np.ndarray, cols: list[str], meta_df: pandas.DataFrame,
             outdir: Path, prefix: str):
    # Save as a human-friendly CSV with column names + meta columns
    dfX = pandas.DataFrame(X, columns=cols)
    df = pandas.concat([meta_df.reset_index(drop=True), dfX], axis=1)
    _dump_df(df, outdir, f"{prefix}_design")
    # Also save a compact npz if you want to reload in Python
    np.savez(outdir / f"{prefix}.npz", X=X, y=y, cols=np.array(cols, dtype=object))

def _mk_debug_dir(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path("debug_runs") / f"{prefix}-{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d

#DELEATE LATER

#---------------------
# APP
#---------------------
st.set_page_config(page_title="Dev Inactivity Demo", layout="wide")
st.title("Developer Inactivity – Demo")

col_refresh, _ = st.columns([1, 3])

orgs = list_orgs()
if not orgs:
    st.warning("No organizations found under your Organizations folder.")
    st.stop()
user_org = st.selectbox("Select an organization", orgs)


# 2) Repository dropdown (dependent on org)
repos = list_repos_for(user_org)
if not repos:
    st.warning(f"No repositories found in organization: {user_org}.")
    st.stop()

user_repo = st.selectbox("Select a repository from organization ", repos)

# Combined key and resolved paths for downstream steps
repo_key = f"{user_org}/{user_repo}" 

st.caption(f"Selected repo: **{repo_key}**")
# (Later buttons can use `repo_key` and `paths`, e.g., Update, Label, Predict)

with col_refresh:
    if st.button("Rescan folders"):
        st.cache_data.clear()

if st.button("Collect Data Demo"):
        extract_repo_main(repo_key)



if st.button("Update Data Demo", type="secondary"):
        data = mock_update_repo(repo_key)
        st.session_state["data"] = data            # <-- persist
        st.write(data)


st.divider()
st.subheader("Label Developers' Activity")

label_all = st.checkbox("Label ALL repos", value=False)


if st.button("Run labeling"):
    tf_devs = label_developers_activity(repo_key, process_all=label_all)
    st.session_state["tf_devs"] = tf_devs            # <-- persist
    st.session_state["labeled_repo"] = repo_key

    output_folder = ORG_BASE / repo_key / "Results"
    file_path = output_folder / f"{tf_devs[0]}_labeled_timeline.csv"
    for tf in tf_devs:
        if not file_path.exists():
            continue
        else:
            user_timeline = pandas.read_csv(file_path)
            st.write(f"User timeline for {tf}", user_timeline)
            st.success("Labeling complete.")
            break

st.divider()
st.subheader("Analyze Project Health")

if st.button("Find Project Health"):
    project_health_main(repo_key)


st.divider()
#this part can only show up after labeling

st.subheader("Predict")

use_all = st.checkbox("Train on ALL labeled repos", value=True)

if st.button("Predict next states"):
    authors = st.session_state.get("tf_devs")
    # 3) Predict
    out = predict_state(repo_key, authors, use_all=use_all)


if st.button("Daily Early Warning (14d)"):
    authors = st.session_state.get("tf_devs")
    daily_preds = predict_daily_early_warning(repo_key, authors, use_all=use_all, horizon_days=7)
