#   conda activate osslab
#   streamlit run DemoAppV2.2.py


#   cd D:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Extractors

from asyncio import Event
import json
from operator import index
from msilib import Table
import streamlit as st
import pandas
import pandas as pd
import numpy as np
import os
import csv
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tempfile, webbrowser, os, pathlib

import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from typing import Iterable, Tuple, List, Set, Dict, Optional, Literal
from collections import Counter
from pathlib import Path
from git import Repo, exc as git_exc
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import random

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from dataclasses import dataclass
from github import Github, GithubException, UnknownObjectException, IncompletableObject
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import Settings as cfg
import Utilities as util
import KnowledgeDistribution as kd
import SocialTechnicalNetwork as stn
import ProjectHealthMetrics as phm
import distrac_writer as dw

LAST_CALLED = [None]

ORG_BASE = PROJECT_ROOT / "Organizations"
IMAGES_BASE = PROJECT_ROOT / cfg.photo_folder
IMAGES_BASE.mkdir(parents=True, exist_ok=True)

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


def _compute_project_health_data(raw_data_tables, repo_full_name, output_path, n_weeks=16):
    """
    Aggregate weekly commit / PR / issue counts per developer and save as
    project_health.json.  Called from the Predictors step.
    """
    commits = raw_data_tables.get("commits",  pandas.DataFrame())
    prs     = raw_data_tables.get("prs_repo", pandas.DataFrame())
    issues  = raw_data_tables.get("issues",   pandas.DataFrame())

    # ── find reference date (most recent event across all tables) ────────────
    ref_date = None
    for df in [commits, prs, issues]:
        if df.empty or "created_at" not in df.columns:
            continue
        ts = pandas.to_datetime(df["created_at"], utc=True, errors="coerce").max()
        if pandas.notna(ts) and (ref_date is None or ts > ref_date):
            ref_date = ts

    if ref_date is None:
        print(f"  [ProjectHealth] No data found for {repo_full_name}, skipping.")
        return

    # ── build n_weeks week-start timestamps ending at ref_date ───────────────
    week_starts = [ref_date - pandas.Timedelta(weeks=(n_weeks - i)) for i in range(n_weeks)]
    week_labels = [d.strftime("%b %d") for d in week_starts]

    def _dev_key(row):
        aid = str(row.get("author_id", "") or "").strip()
        if aid:
            return aid
        login = str(row.get("author_login", "") or row.get("created_by", "") or "").strip()
        return f"author_login|{login}" if login else None

    def _agg_weekly(df):
        if df.empty or "created_at" not in df.columns:
            return {}
        df = df.copy()
        df["_ts"] = pandas.to_datetime(df["created_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["_ts"])

        # Build dev key column
        if "author_id" in df.columns:
            df["_dev"] = df["author_id"].where(
                df["author_id"].notna() & (df["author_id"].astype(str).str.strip() != ""), other=None
            )
            fallback = "author_login" if "author_login" in df.columns else (
                       "created_by"   if "created_by"   in df.columns else None)
            if fallback:
                df["_dev"] = df["_dev"].fillna(
                    df[fallback].apply(lambda x: f"author_login|{x}" if pandas.notna(x) and str(x).strip() else None)
                )
        elif "created_by" in df.columns:
            df["_dev"] = df["created_by"].apply(
                lambda x: f"author_login|{x}" if pandas.notna(x) and str(x).strip() else None
            )
        else:
            return {}

        df = df.dropna(subset=["_dev"])[["_ts", "_dev"]]
        out = {}
        for i, ws in enumerate(week_starts):
            we = ws + pandas.Timedelta(weeks=1)
            for dev, cnt in df.loc[(df["_ts"] >= ws) & (df["_ts"] < we), "_dev"].value_counts().items():
                if dev not in out:
                    out[dev] = [0] * n_weeks
                out[dev][i] = int(cnt)
        return out

    c_agg = _agg_weekly(commits)
    p_agg = _agg_weekly(prs)
    i_agg = _agg_weekly(issues)

    all_devs = set(c_agg) | set(p_agg) | set(i_agg)

    repo_c = [sum(c_agg.get(d, [0]*n_weeks)[w] for d in all_devs) for w in range(n_weeks)]
    repo_p = [sum(p_agg.get(d, [0]*n_weeks)[w] for d in all_devs) for w in range(n_weeks)]
    repo_i = [sum(i_agg.get(d, [0]*n_weeks)[w] for d in all_devs) for w in range(n_weeks)]

    developers = {}
    for dev in all_devs:
        wc = c_agg.get(dev, [0]*n_weeks)
        wp = p_agg.get(dev, [0]*n_weeks)
        wi = i_agg.get(dev, [0]*n_weeks)
        bc = round(sum(wc) / n_weeks, 1)
        bp = round(sum(wp) / n_weeks, 1)
        bi = round(sum(wi) / n_weeks, 1)
        if bc + bp + bi == 0:
            continue
        developers[str(dev)] = {
            "weekly_commits": wc,
            "weekly_prs":     wp,
            "weekly_issues":  wi,
            "baseline_commits": bc,
            "baseline_prs":     bp,
            "baseline_issues":  bi,
        }

    payload = {
        "repo":         repo_full_name,
        "generated_at": ref_date.isoformat(),
        "weeks":        week_labels,
        "repo_totals": {
            "commits_per_week": round(sum(repo_c) / n_weeks, 1),
            "prs_per_week":     round(sum(repo_p) / n_weeks, 1),
            "issues_per_week":  round(sum(repo_i) / n_weeks, 1),
        },
        "developers": developers,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  [ProjectHealth] Saved {len(developers)} developers → {output_path}")


def compute_daily_ph_features(raw_data_tables: dict) -> pandas.DataFrame:
    """
    Build daily rolling-7-day repo-wide totals (commits, PRs, issues, active devs).
    These are repo-level signals broadcast to every developer by a date-only join.

    Returns columns: date, repo_commits_7d, repo_prs_7d, repo_issues_7d, repo_active_devs_7d
    """
    commits = raw_data_tables.get("commits",  pandas.DataFrame())
    prs     = raw_data_tables.get("prs_repo", pandas.DataFrame())
    issues  = raw_data_tables.get("issues",   pandas.DataFrame())

    def _daily_counts(df, date_col):
        if df.empty or date_col not in df.columns:
            return pandas.Series(dtype=float)
        dates = (
            pandas.to_datetime(df[date_col], utc=True, errors="coerce")
            .dt.tz_localize(None).dt.normalize()
        )
        return dates.value_counts().sort_index()

    def _daily_devs(df, date_col):
        dev_col = next((c for c in ("author_login", "author_id") if c in df.columns), None)
        if df.empty or date_col not in df.columns or dev_col is None:
            return pandas.Series(dtype=float)
        df2 = df[[date_col, dev_col]].copy()
        df2["_d"] = (
            pandas.to_datetime(df2[date_col], utc=True, errors="coerce")
            .dt.tz_localize(None).dt.normalize()
        )
        return df2.groupby("_d")[dev_col].nunique()

    commit_s = _daily_counts(commits, "created_at")
    pr_s     = _daily_counts(prs, "created_at")
    issue_s  = _daily_counts(issues, "created_at")
    devs_s   = _daily_devs(commits, "created_at")

    all_dates = sorted(set(commit_s.index) | set(pr_s.index) | set(issue_s.index))
    if not all_dates:
        return pandas.DataFrame()

    idx = pandas.DatetimeIndex(all_dates)
    ph = pandas.DataFrame(index=idx)
    ph["_c"] = commit_s.reindex(idx, fill_value=0)
    ph["_p"] = pr_s.reindex(idx, fill_value=0)
    ph["_i"] = issue_s.reindex(idx, fill_value=0)
    ph["_d"] = devs_s.reindex(idx, fill_value=0)

    ph["repo_commits_7d"]     = ph["_c"].rolling(7, min_periods=1).sum().astype(int)
    ph["repo_prs_7d"]         = ph["_p"].rolling(7, min_periods=1).sum().astype(int)
    ph["repo_issues_7d"]      = ph["_i"].rolling(7, min_periods=1).sum().astype(int)
    ph["repo_active_devs_7d"] = ph["_d"].rolling(7, min_periods=1).sum().astype(int)

    return (
        ph[["repo_commits_7d", "repo_prs_7d", "repo_issues_7d", "repo_active_devs_7d"]]
        .reset_index()
        .rename(columns={"index": "date"})
    )


def load_users_activity(repo_full_name):
    """
        We need to FIND and load all of our raw data
    we have to get the file paths correct
    then we have to load the csv and then put them in a table

    #TODO: this is a long todo but we need to add our daily data collectiong here too
    #insted of just finding the data
    # we need to have a contiuous data collection system were every day it checks for new data and adds it to the data set
    # so we will have two parts of data collection
    # one algorithm that collects every day
    # one algorithm that gets all of that data
    """
    # in one line find the folder we are in
    orgs_dir = PROJECT_ROOT / "Organizations"

    target_files = {
        "issues": cfg.issue_list_file_name,
        "issue_activity": cfg.issue_activity_file_name,
        "prs_repo": cfg.PR_list_file_name,
        "prs_comments": cfg.prs_comments_csv,
        "commit_list": cfg.commit_list_file_name,
        "perfile_commit": cfg.per_file_commits_path
    }
    #we need to make df for each of the target files
    issues = pandas.DataFrame()
    issue_activity = pandas.DataFrame()
    prs_repo = pandas.DataFrame()
    prs_comments = pandas.DataFrame()
    commits = pandas.DataFrame()
    perfile_commits = pandas.DataFrame()

    if not orgs_dir.exists():
        st.write(f"Organizations folder not found: {orgs_dir}")
        return 0

    org, repo_name = repo_full_name.split('/')
    repo_dir = orgs_dir / org / repo_name

    if not repo_dir.is_dir():
        st.warning(f"Repo folder not found: {repo_dir}")
        return {}

    for file_key, file_name in target_files.items():
        file_path = repo_dir / file_name
        if file_path.exists():
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
                st.write(f"Error loading {file_path}: {e}")

    raw_data_tables = { "issues": issues, "issue_activity": issue_activity, 
                       "prs_repo": prs_repo, "prs_comments": prs_comments, 
                       "commits": commits , "perfile_commits": perfile_commits}

    #save the data to a temp file
    
    return raw_data_tables

def _is_repo_ready(org: str, repo: str) -> bool:
    """Return True iff the repo folder exists and all 3 cursor streams are complete."""
    cursor_path = ORG_BASE / org / repo / cfg.data_cursor
    if not cursor_path.exists():
        return False
    try:
        cursor = json.loads(cursor_path.read_text())
    except Exception:
        return False
    streams = cursor.get("streams", {})
    return bool(streams) and all(s.get("complete", False) for s in streams.values())


@st.cache_data(show_spinner=False)
def list_orgs(base: Path = ORG_BASE) -> list[str]:
    """Orgs that have at least one repo in Resources/repo_split.csv (any split)."""
    pairs = cfg.load_repo_split()          # all rows, no filter
    orgs = sorted({org for org, _ in pairs}, key=str.casefold)
    return [o for o in orgs if (base / o).is_dir()]

@st.cache_data(show_spinner=True)
def list_repos_for(org: str, base: Path = ORG_BASE) -> list[str]:
    """Repos for this org that appear in Resources/repo_split.csv (any split)."""
    pairs = cfg.load_repo_split()
    repos = sorted([repo for o, repo in pairs if o == org], key=str.casefold)
    return [r for r in repos if (base / org / r).is_dir()]
                 
def plot_state_comparison_html(dev_id, dev_timeline_raw, breaks_original, breaks_causal, out_path):
    """
    Save an interactive plotly HTML comparing identified breaks (original bidirectional
    algorithm vs causal backward-only algorithm) for a single developer, with activity bars.
    Takes breaks DataFrames directly — no need to run label_timeline.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = dev_timeline_raw.copy()
    df["date"] = pandas.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for _c in ("commits", "prs", "issues"):
        if _c in df.columns:
            df[_c] = pandas.to_numeric(df[_c], errors="coerce").fillna(0)
        else:
            df[_c] = 0

    if df.empty:
        return

    date_min = df["date"].min()
    date_max = df["date"].max()

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Breaks — original (bidirectional window, accurate state)",
                        "Breaks — causal (backward-only, no look-ahead)",
                        "Daily activity (commits / PRs / issues)"],
        row_heights=[0.15, 0.15, 0.70],
        vertical_spacing=0.07,
    )

    # Dense date spine used for all state fills — guarantees datetime axis type
    _spine = pandas.date_range(date_min, date_max + pandas.Timedelta(days=1), freq="D")

    def _shade_breaks(breaks_df, row):
        # Green baseline (ACTIVE): fill from y=0 up to y=1 across the full timeline
        fig.add_trace(go.Scatter(
            x=_spine, y=[1.0] * len(_spine),
            fill="tozeroy", fillcolor="rgba(46,204,113,0.55)",
            mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=1)

        # Red overlay for each break period
        if breaks_df is not None and not breaks_df.empty and "dates" in breaks_df.columns:
            for _, b in breaks_df.iterrows():
                try:
                    start_s, end_s = str(b["dates"]).split("/")
                    brk_spine = pandas.date_range(start_s, end_s, freq="D")
                    fig.add_trace(go.Scatter(
                        x=brk_spine, y=[1.0] * len(brk_spine),
                        fill="tozeroy", fillcolor="rgba(231,76,60,0.80)",
                        mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip",
                    ), row=row, col=1)
                except Exception:
                    pass

        fig.update_yaxes(
            tickvals=[0.25, 0.75], ticktext=["BREAK", "ACTIVE"],
            showticklabels=True, tickfont=dict(size=9),
            showgrid=False, range=[0, 1],
            row=row, col=1,
        )

    _shade_breaks(breaks_original, row=1)
    _shade_breaks(breaks_causal,   row=2)

    fig.add_trace(go.Bar(x=df["date"], y=df["commits"], name="commits",
                         marker_color="#2ecc71", opacity=0.9), row=3, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["prs"],     name="PRs",
                         marker_color="#f1c40f", opacity=0.9), row=3, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["issues"],  name="issues",
                         marker_color="#3498db", opacity=0.9), row=3, col=1)

    fig.update_layout(
        title=f"Break comparison — {dev_id}",
        barmode="stack",
        plot_bgcolor="#1c1c2e", paper_bgcolor="#1c1c2e",
        font=dict(color="white"),
        height=680,
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_xaxes(showgrid=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))

    webbrowser.open("file://" + str(out_path))


#-----------------------
# inactivity labeling
#------------------------
def label_developers_activity(repo, over_write = False, tf_devs = None) -> pandas.DataFrame:
    """
    main function for labeling developers
    sets up varables to call label timeline
    """
    
    # "../Organizations"
    organization, project = repo.split('/')
    organizationFolder = ORG_BASE / organization / project

    win = cfg.sliding_window_size

    repos_txt = '../' + cfg.repos_file
    repos_to_process = []
        
    
    all_timelines = []
    all_diagnostics = []

    # i want to check if the end file has already been made and if it has then we can skip the whole process and just load the file
    output_folder = organizationFolder /  "Results"
    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / "all_users_labeled_timeline.csv"
    
    if out_path.is_file() and over_write == False:
        print("we are loading the file from ", out_path)
        df = pandas.read_csv(out_path, sep=cfg.CSV_separator)
        return df
    elif over_write == False: 
        print(f"we did not find the file at {out_path}, we are generating the file now...")
    
    organization, project = repo.split('/')
    if Path(organizationFolder).exists() == False:
        st.write(f"Organization folder not found: {Path(organizationFolder)}")

    print(f"Start Identifying inactivity periods for {organization}/{project}...")

    commits = pd.read_csv( organizationFolder / "commit_list.csv" , sep=cfg.CSV_separator, parse_dates=["created_at"])

    if commits.empty:
        st.write(f"No commits found for {organization}/{project} at {organizationFolder / 'commit_list.csv'}")
        return pandas.DataFrame()  # Return empty DataFrame if no commits

    commits["created_at"] = pandas.to_datetime(commits["created_at"], utc=True)

    # TruckFactor.json is written by KnowledgeDistribution.py into the KnowledgeDistribution/ subfolder.
    if tf_devs is None:
        tf_devs = _tf_data["tf_list"]
    else:
        with open(organizationFolder / "KnowledgeDistribution" / cfg.truck_factor_file, "r") as f:
            _tf_data = json.load(f)

    tf = _tf_data["tf"]

    timeline_folder = organizationFolder / cfg.timeline_folder.lstrip("/")
    os.makedirs(timeline_folder, exist_ok=True)
        
    timeline_path = Path(timeline_folder, cfg.timeline_file)

    if timeline_path.is_file():
        user_timeline = pandas.read_csv(timeline_path, sep=cfg.CSV_separator)
    else:
        print(f"Timeline not found at {timeline_path}, generating timeline...")

    count = 0


    pauses = write_pauses_table(commits, user_timeline, organizationFolder / "pauses_commits.csv", tf_devs, date_col="created_at")
    #make pauses to a csv file at this location C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Organizations\Rdatatable\data.table\Results

    
    for dev in tf_devs:
        print(f"{tf_devs.index(dev) + 1} / {len(tf_devs)}")
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            column, dev = dev.split('|', maxsplit=1)
        count= count+1

        breaks_folder = organizationFolder /  "Breaks"
        os.makedirs(breaks_folder, exist_ok=True)
        breaks_path =  Path(breaks_folder)/  f"{dev}_breaks.csv"

        # Run original (forward-sliding, bidirectional) algorithm — accurate state for response variable
        if breaks_path.is_file() and not over_write:
            breaks_original = pandas.read_csv(breaks_path, sep=cfg.CSV_separator)
        else:
            breaks_original = identifyBreaks_original(pauses, dev=dev, window=win, shift=cfg.shift)
            breaks_original.to_csv(breaks_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, lineterminator="\n")        # Run causal (backward-only) algorithm — no look-ahead, safe as model predictor
        
        breaks_causal, diagnostics_df = identifyBreaks_causal(pauses, dev=dev, window=win, debug_folder=output_folder)

        # filter to this developer before labeling
        # NOTE: use a local variable so the full user_timeline stays intact for the next dev
        dev_timeline_raw = user_timeline[user_timeline["dev"] == dev]

        labeled_original = label_timeline(dev_timeline_raw.copy(), breaks_original)
        labeled_causal   = label_timeline(dev_timeline_raw.copy(), breaks_causal)

        # Combine: state (accurate, from original algorithm) + state_causal (no look-ahead)
        dev_timeline = labeled_causal.copy()
        dev_timeline["state_causal"] = labeled_causal["state"]

        # Date-aligned assignment — safe even if labeled_original has phantom rows from break ranges
        _orig_dstr = pandas.to_datetime(labeled_original["date"]).dt.strftime("%Y-%m-%d")
        _orig_map  = dict(zip(_orig_dstr, labeled_original["state"]))
        _dt_dstr   = pandas.to_datetime(dev_timeline["date"]).dt.strftime("%Y-%m-%d")
        dev_timeline["state"] = _dt_dstr.map(_orig_map).fillna("ACTIVE")

        _state_order = {"ACTIVE": 0, "NON_CODING": 1, "INACTIVE": 2, "GONE": 3}
        dev_timeline["state_causal_enc"] = dev_timeline["state_causal"].map(_state_order)

        # Save per-dev break comparison HTML (original vs causal, using break date ranges directly)
        _comp_path = Path(output_folder) / "state_comparisons" / f"{dev}_state_comparison.html"

        dev_timeline = dev_timeline.reset_index(drop=True)
        all_timelines.append(dev_timeline)

        all_diagnostics.append(diagnostics_df)

        out_csv = Path(output_folder) / f"{dev}_labeled_timeline.csv"
        dev_timeline.to_csv(out_csv, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, lineterminator='\n', index=False)

        tf_devs_df = pandas.DataFrame(tf_devs, columns=["developer"])
        tf_devs_df.to_csv(Path(output_folder) / "tf_devs.csv", sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n')

        # WE NEED TO MAKE A MASTER USER TIMELINE that we use as the return value


    if tf == 1:
        master_user_timeline = all_timelines[0]
        master_diagnostics = all_diagnostics
    else:
        master_user_timeline = pandas.concat(all_timelines, ignore_index=True)
        master_diagnostics = pandas.concat(all_diagnostics, ignore_index=True)


    # Extract DataFrames from lists if needed
    if isinstance(master_user_timeline, list) and len(master_user_timeline) > 0:
        if isinstance(master_user_timeline[0], pandas.DataFrame):
            master_user_timeline = master_user_timeline[0]

    if isinstance(master_diagnostics, list) and len(master_diagnostics) > 0:
        if isinstance(master_diagnostics[0], pandas.DataFrame):
            master_diagnostics = master_diagnostics[0]

    # Normalize date columns
    master_user_timeline["date"] = pandas.to_datetime(master_user_timeline["date"]).dt.normalize()
    master_diagnostics["date"] = pandas.to_datetime(master_diagnostics["win_end"]).dt.normalize()



    # Merge on dev and date
    master_user_timeline = master_user_timeline.merge(
        master_diagnostics,
        left_on=["dev", "date"],
        right_on=["dev", "date"],
        how="left"
    )

    master_user_timeline.to_csv(out_path, index=False)

    # visualize the breaks
    # for each developer in tf_devs we need to make a plot of their timeline with breaks marked
    devs = sorted(master_user_timeline["dev"].unique())

    return master_user_timeline

def build_dev_names(repo_full_name: str) -> pandas.DataFrame:
    """
    Build a dev_id → readable name lookup table for one repo and save it to
    Organizations/<org>/<repo>/Results/dev_names.csv.

    Reads commit_list.csv, issues.csv, and prs_repo.csv, computes each row's
    dev_id using create_developer_id() from KnowledgeDistribution, then
    aggregates the best available name/login/email/raw_id per dev_id.

    Returns the resulting DataFrame.
    """
    org, repo = repo_full_name.split("/", 1)
    repo_path = ORG_BASE / org / repo

    id_cols = ["author_id", "author_name", "author_login", "author_email"]

    frames = []
    for fname in ("commit_list.csv", "issues.csv", "prs_repo.csv"):
        fpath = repo_path / fname
        if not fpath.exists():
            continue
        df = pandas.read_csv(fpath, usecols=lambda c: c in id_cols, low_memory=False)
        for col in id_cols:
            if col not in df.columns:
                df[col] = None
        frames.append(df[id_cols])

    if not frames:
        print(f"[build_dev_names] No source files found for {repo_full_name}")
        return pandas.DataFrame(columns=["dev_id", "name", "login", "email", "raw_id"])

    combined = pandas.concat(frames, ignore_index=True).drop_duplicates()
    combined["dev_id"] = combined.apply(kd.create_developer_id, axis=1)
    combined = combined.dropna(subset=["dev_id"])

    def first_valid(series):
        vals = series.dropna()
        return vals.iloc[0] if not vals.empty else None

    result = (
        combined.groupby("dev_id", sort=False)
        .agg(
            name=("author_name",  first_valid),
            login=("author_login", first_valid),
            email=("author_email", first_valid),
            raw_id=("author_id",  first_valid),
        )
        .reset_index()
    )

    out_path = repo_path / "Results" / "dev_names.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[build_dev_names] Saved {len(result)} entries -> {out_path}")
    return result


_DEBUG_DEV = "Corey Johnson"  # change to target a different dev in diagnostics

def write_pauses_table(
        df: pandas.DataFrame,
        user_timeline: pandas.DataFrame,
        out_path: os.PathLike,
        tf_devs: list[str] | None = None,
        *,
        date_col: str = "created_at",
        tail_to_today: bool = True) -> pandas.DataFrame:

    df[date_col] = pandas.to_datetime(df[date_col]).dt.normalize()

    rows = []

    count =0
    for dev in tf_devs:
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            column, dev = dev.split('|', maxsplit=1)
            user_df = df[df[column] == dev]
        else:
            user_df = df[df["author_id"] == dev]

        # Use the full timeline (commits + PRs) to compute active days when available.
        # This prevents PR-only days from being swallowed inside a "break" that was
        # detected from commits alone.
        if user_timeline is not None and not user_timeline.empty:
            dev_tl = user_timeline[user_timeline["dev"] == dev].copy()
            dev_tl["date"] = pandas.to_datetime(dev_tl["date"]).dt.normalize()
            coding_mask = (dev_tl["commits"].fillna(0) > 0) | (dev_tl["prs"].fillna(0) > 0)
            active_days = sorted(dev_tl[coding_mask]["date"].dt.date.unique())
            source = "timeline (commits+PRs)"
        else:
            active_days = sorted(user_df[date_col].dt.date.unique())
            source = "commit_list only"

        # DIAGNOSTIC: show what days are treated as active and which gaps become pauses
        if str(dev) == _DEBUG_DEV:
            for _i in range(1, len(active_days)):
                _gap = (active_days[_i] - active_days[_i - 1]).days
                if _gap > 1:
                    _ps = active_days[_i - 1] + timedelta(days=1)
                    _pe = active_days[_i]     - timedelta(days=1)

        current_row = [dev]

        for i in range(len(active_days) - 1):
            #if we are at the first day then there is no prev day and no period
            # so we skip it
            if i == 0:
                continue
            prev_active_day = active_days[i - 1]
            #FOUND ITTTTTTTT
            current_active_day = active_days[i]
            gap = (current_active_day - prev_active_day).days
            if gap > 1:
                # Inactivity starts the day after prev_active_day
                current_row.append(f"{(prev_active_day + pandas.Timedelta(days=1)).strftime('%Y-%m-%d')}/{(current_active_day - pandas.Timedelta(days=1)).strftime('%Y-%m-%d')}")
            else:
                count += 1

        if tail_to_today:
            today = datetime.now().date()
            gap = (today - active_days[-1]).days
            if gap > 1:
                tail_start = active_days[-1] + timedelta(days=1)
                current_row.append(f"{tail_start}/{today}")
        if len(current_row) > 1:
            rows.append(current_row)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="",encoding="utf-8" ) as f:
        csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE).writerows(rows)
    out = pandas.DataFrame(rows)

    return out


def _apply_segment_label(df, seg_start, seg_end, label):
    """Apply label to all days in [seg_start, seg_end) using vectorized assignment."""
    if seg_start >= seg_end:
        return
    mask = (df.index >= seg_start) & (df.index < seg_end)
    df.loc[mask, "state"] = label


def _label_break_ncut(df, block, start_ts, end_ts, Tfov, gone_days=365):
    """
    Labels break days using the splitBreak NCUT state machine from Calefato et al. (2021).

    Sub-intervals are defined by NC event days inside the break.  The break
    boundaries bookend the list exactly as break_range[0/1] do in the original
    splitBreak function. GONE is detected per-segment (gap > gone_days) so NC
    events inside the break naturally reset the GONE clock without needing a
    separate last_event_before tracker.

    States (mirrors original splitBreak):
      ACTIVE  → opening state; first gap > Tfov → INACTIVE/GONE, else → NCUT
      NCUT    → accumulating silence from period_start; total > Tfov → NON_CODING segment
      NON_CODING → extending if next gap ≤ Tfov; else → INACTIVE/GONE
      INACTIVE/GONE → NC event arrives; gap < Tfov → NCUT; else → NON_CODING(+INACTIVE/GONE)
    """
    nc_days_inside = sorted(d for d in block.index if bool(df.at[d, "nc_day"]))

    # End sentinel is exclusive so (sentinel - p0).days matches daysBetween semantics
    sentinel = end_ts + pandas.Timedelta(days=1)
    action_points = [start_ts] + nc_days_inside + [sentinel]

    status = "ACTIVE"
    previously = "ACTIVE"
    period_start = None

    for i in range(len(action_points) - 1):
        p0 = action_points[i]
        p1 = action_points[i + 1]
        gap = (p1 - p0).days

        if status == "ACTIVE":
            if gap > Tfov:
                label = "GONE" if gap > gone_days else "INACTIVE"
                _apply_segment_label(df, p0, p1, label)
                status = label
                previously = label
            else:
                previously = status
                status = "NCUT"
                period_start = p0

        elif status in ("INACTIVE", "GONE"):
            if gap < Tfov:
                previously = status
                status = "NCUT"
                period_start = p0
            else:
                residual = gap - (Tfov + 1)
                if residual > Tfov:
                    nc_boundary = p0 + pandas.Timedelta(days=Tfov + 1)
                    _apply_segment_label(df, p0, nc_boundary, "NON_CODING")
                    label = "GONE" if residual > gone_days else "INACTIVE"
                    _apply_segment_label(df, nc_boundary, p1, label)
                    status = label
                    previously = label
                else:
                    _apply_segment_label(df, p0, p1, "NON_CODING")
                    status = "NON_CODING"
                    previously = "NON_CODING"

        elif status == "NON_CODING":
            if gap > Tfov:
                label = "GONE" if gap > gone_days else "INACTIVE"
                _apply_segment_label(df, p0, p1, label)
                status = label
                previously = label
            else:
                # Sub-interval ≤ Tfov: extend the NON_CODING window
                _apply_segment_label(df, p0, p1, "NON_CODING")

        else:  # NCUT — accumulating from period_start
            size = (p1 - period_start).days
            if size > Tfov:
                residual = size - (Tfov + 1)
                nc_boundary = period_start + pandas.Timedelta(days=Tfov + 1)
                if residual > Tfov:
                    _apply_segment_label(df, period_start, nc_boundary, "NON_CODING")
                    label = "GONE" if residual > gone_days else "INACTIVE"
                    _apply_segment_label(df, nc_boundary, p1, label)
                    status = label
                    previously = label
                else:
                    _apply_segment_label(df, period_start, p1, "NON_CODING")
                    status = "NON_CODING"
                    previously = "NON_CODING"
            # else: still accumulating — no label change yet

    # NCUT tail: inherit the pre-NCUT label (original extends its first segment to cover tail).
    # If previously == "ACTIVE" the days stay ACTIVE (within-threshold natural rhythm).
    if status == "NCUT" and previously != "ACTIVE":
        _apply_segment_label(df, period_start, sentinel, previously)

    # NC event days always become NON_CODING regardless of which segment claimed them.
    for d in nc_days_inside:
        if d in df.index:
            df.at[d, "state"] = "NON_CODING"


def label_timeline(user_timeline, breaks_df):
    """
    Make a labled timeline of devlopers breaks

    given a user_timeline and breaks_df
    user_timeline
    dev,date,commits,issues,prs,files_changed,lines_added,lines_removed,prs_review,prs_comment,issues_commented,issues_activity,labeled,closed,commented,mentioned,subscribed,referenced,renamed,issue_type_added,unsubscribed,pinned,locked,reopened,assigned,unlabeled,connected,milestoned,comment_deleted,unassigned,unpinned,demilestoned,marked_as_duplicate,transferred,unmarked_as_duplicate,unlocked,parent_issue_added,parent_issue_removed,sub_issue_added,sub_issue_removed,disconnected
    jekyllbot,2014-06-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

    breaks_df
    len,dates,th
    72,2015-05-05/2015-07-16,59.25
    """
    df = user_timeline.copy()
    if df.empty:
        return df

    # Ensure the index is a DatetimeIndex BEFORE any df.at[date, ...] calls.
    # If user_timeline has a plain integer index, df.at["2015-05-05", col] would
    # add phantom rows with NaN instead of setting existing ones.
    if not isinstance(df.index, pandas.DatetimeIndex):
        if "date" in df.columns:
            df.index = pandas.to_datetime(df["date"])
        else:
            df.index = pandas.to_datetime(df.index)
    df = df.sort_index()

    # coding activity: commit OR PR
    df["coding_day"] = ((df["commits"] > 0) | (df["prs"] > 0)).astype(int)

    # non-coding activity: any other event > 0 AND no coding
    noncoding_cols = [c for c in df.columns if c in ["issues", "issue_activity", "pr_activity"]]
    df["nc_day"] = ((df[noncoding_cols].sum(axis=1) > 0) & (df["coding_day"] == 0)).astype(int)

    df["break_day"] = pandas.Series(False, index=df.index, dtype="boolean")
    df["th"]        = pandas.Series(pandas.NA, index=df.index, dtype="Float64")
    df["len"]       = pandas.Series(pandas.NA, index=df.index, dtype="Int64")
    df["index"]     = pandas.Series(pandas.NA, index=df.index, dtype="Int64")

    #this marks the break days from breaks_df onto user_timeline
    for breaks in breaks_df.itertuples():
        start = breaks.dates.split('/')[0]
        end = breaks.dates.split('/')[1]

        start = pandas.to_datetime(start)
        end = pandas.to_datetime(end)

        break_range = pandas.date_range(start=pandas.to_datetime(start),
                                end=pandas.to_datetime(end))

        if pandas.to_datetime(start) <= end:

           for date in break_range:
                date = date.strftime("%Y-%m-%d")
                df.at[date, "break_day"] = True
                df.at[date, "th"] = breaks.th
                df.at[date, "len"] = breaks.len
                df.at[date, "index"] = breaks.Index

    df.index = pandas.to_datetime(df.index)
    df = df.sort_index()

    df["state"] = "ACTIVE"

    # Identify contiguous break windows (groups of consecutive True in break_day)
    bd = df["break_day"]
    group_id = (bd != bd.shift(1)).cumsum()

    df["event_day"] = ((df["coding_day"] >= 1) | (df["nc_day"] >= 1)).astype(int)

    for gid, block in df.groupby(group_id):

        if not block["break_day"].iloc[0]:
            continue  # not a break chunk

        start_ts = block.index[0]
        end_ts   = block.index[-1]

        Tfov = int(df.loc[start_ts:end_ts, "th"].iloc[0])

        # Label this break block using the NCUT state machine (Calefato et al. 2021)
        _label_break_ncut(df, block, start_ts, end_ts, Tfov)

    return df

def getFarOutThreshold(values): ### If it is satisfying, move the function into UTILITIES
    th = 0
    q_3rd = np.percentile(values,75)
    q_1st = np.percentile(values,25)
    iqr = q_3rd-q_1st
    if iqr > 1:
        th = q_3rd + 3*iqr
    return th

def addToBreaksList(current_dt, pauses, intervals_list, currentBreaks, th):
    # we need to find the current pause. this is the current date and the last active day before today 
    # we need to find the previous length of pause
    # we can do this by finding the pause that has the current date as the end date
    # then we can check if the length of that pause is greater than the threshold

    #find the row where current_dt = intervals_list["date"].split('/')[1] or the end date
    for interval in intervals_list:
        int_start_str, int_end_str = interval.split('/')
        if int_end_str != current_dt.strftime('%Y-%m-%d'):
            continue
        else:           
            pause_len = util.daysBetween(int_start_str, int_end_str) + 1
            if int_end_str == current_dt.strftime('%Y-%m-%d') and pause_len >= th:
                # check if this break is already in the list
                if not ((currentBreaks['dates'] == interval).any()):
                    util.add(currentBreaks, [pause_len, interval, th])

    return currentBreaks

def cleanClearBreaks(clearBreaks, breaks):
    for _, b in breaks.iterrows():
        clearBreaks = clearBreaks[clearBreaks.dates != b['dates']] # If it was in the long_breaks list, remove ot from there
    return clearBreaks

def identifyBreaks_causal(pauses_dates_list, dev, window, debug_folder=None):
    '''
    Causal (backward-only) break identifier — no look-ahead bias.
    Window: [current_dt - window, current_dt].  Safe to use as a model predictor.
    '''
    pauses_dates_list = pauses_dates_list.values.tolist()

    breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])
    diagnostics = []                             # NEW
    count = 0

    for row in pauses_dates_list:
        # print the first few characters of the row

        if str(row[0]).strip() != str(dev).strip():            # ⬅️  ignore other developers
            continue




        intervals_list = [ x for x in row[1:]
                          if isinstance(x, str) and '/' in x and x.strip()]

        intervals_list.sort(key=lambda s: s.split('/')[0])

        if not all(a.split('/')[0] <= b.split('/')[0]
                for a, b in zip(intervals_list, intervals_list[1:])):
            print("⚠️  intervals_list UNSORTED for", dev[1])

        if not intervals_list:
            print(dev[1], 'has NO valid pauses')
            continue                      # <- don’t bail out; just skip

        clear_breaks = pandas.DataFrame(columns=['len', 'dates'])

        last_th = 0
        if intervals_list:
            FPS_dt = datetime.strptime(intervals_list[0].split('/')[0], '%Y-%m-%d')
            LPE_dt = datetime.strptime(intervals_list[-1].split('/')[1], '%Y-%m-%d')
            print(f"Dev {dev} Start date = {FPS_dt.date()}  End date ={LPE_dt.date()}")
        else:
            print("  (no intervals after filtering)")

        current_dt = FPS_dt
        current_index = 0

        while current_dt < LPE_dt:
            win_start, win_end = current_dt - timedelta(days=window), current_dt
            past_pauses_list = pandas.DataFrame(columns=['len', 'dates'])
            partially_included_pauses_list = pandas.DataFrame(columns=['len', 'dates'])

            for interval in intervals_list:
                int_start_str, int_end_str = interval.split('/')          # keep strings
                int_start_dt  = datetime.strptime(int_start_str, '%Y-%m-%d')
                int_end_dt    = datetime.strptime(int_end_str,   '%Y-%m-%d')
                pause_len = util.daysBetween(int_start_str, int_end_str) + 1
                # fully inside
                if int_start_dt >= win_start and int_end_dt <= win_end:
                    util.add(past_pauses_list, [pause_len, interval])
                # touches boundary but need to still be less than current date
                if ((int_start_dt <= win_end and int_end_dt > win_end and int_end_dt < current_dt) or
                    (int_end_dt >= win_start and int_start_dt < win_start and int_start_dt < current_dt)):
                    util.add(partially_included_pauses_list, [pause_len, interval])
                # we need to add the current pause length to the list of pauses

            
            win_pauses = len(past_pauses_list)
            pauses = pandas.concat([past_pauses_list,
                                    partially_included_pauses_list],
                                    ignore_index=True)

            # --- decision logic (unchanged) ------------------------------
            win_th = None
            added_flag = False
            if win_pauses >= 4:
                # To check if we have look head bias we can look at the
                # data that we are using to calculate the threshold
                #print(pauses)
                win_th = getFarOutThreshold(pauses['len'])
                
                if win_th < 3:
                    #print("we are seeing a very low threshold of ", win_th, "for dev ", dev, "at date ", current_dt.date(), "with window ", window)
                    #print("the pauses['len'] values are ", pauses['len'].tolist())
                    th = 0
                    q_3rd = np.percentile(pauses['len'],75)
                    q_1st = np.percentile(pauses['len'],25)
                    iqr = q_3rd-q_1st
                    if iqr > 1:
                        th = q_3rd + 3*iqr
                    
                    #print("meaning when we run through the calculation")
                    #print("q_3rd is ", q_3rd)
                    #print("q_1st is ", q_1st)
                    #print("iqr is ", iqr)
                    #print("th is ", th)
                    

                #print(win_th)
                if win_th > 0:
                    before = len(breaks_df)
                    breaks_df = addToBreaksList( current_dt, pauses, intervals_list, breaks_df, win_th)
                    added_flag = len(breaks_df) > before
                    last_th = win_th
                elif last_th > 0:
                    win_th = last_th
                    before = len(breaks_df)
                    breaks_df = addToBreaksList( current_dt, pauses, intervals_list, breaks_df, last_th)
                    added_flag = len(breaks_df) > before
                #what if the window threshold is 0?
                
            else:
            

                if last_th > 0:
                    win_th = last_th
                    before = len(breaks_df)
                    breaks_df = addToBreaksList(current_dt, pauses, intervals_list, breaks_df, last_th)
                    added_flag = len(breaks_df) > before
                else:
                    # If a user is new and doesnt have more than 4 breaks
                    # and they cannot rely on the past breaks to set a threshold
                    # we need to set a very basic threshold
                    # we set it to window size.
                    # meaning if they paused for a length of time equal to 
                    # the entire window we count it as a break
                    win_th = window
                    last_th = win_th
                    before = len(breaks_df)
                    breaks_df = addToBreaksList(current_dt, pauses, intervals_list, breaks_df, win_th)
                    added_flag = len(breaks_df) > before 



            # We need to move to the next acctive day
            # We can find this inside of the pauses list
            current_index += 1
            if current_index >= len(intervals_list):
                break
            current_dt = datetime.strptime(intervals_list[current_index].split('/')[1], '%Y-%m-%d')

            # this is very useful for debugging
            # you can see how each window is decied as a break or not
            diagnostics.append({
                'dev': dev,
                'win_start': win_start.date(),
                'win_end':   win_end.date(),
                'win_pauses': win_pauses,
                'pause_lengths': ';'.join(map(str, pauses['len'].tolist())),
                'partial_lengths': ';'.join(map(str, partially_included_pauses_list['len'].tolist())),
                'win_th': win_th,
                'last_th': last_th,
                'added_as_break': 'yes' if added_flag else 'no'
            })
            # -----------------------------------------------------------------

    diagnostics_df = pandas.DataFrame(diagnostics)

    return breaks_df, diagnostics_df

def identifyBreaks_original(pauses_dates_list, dev, window, shift=7):
    '''
    Original forward-sliding window break identifier (Calefato et al. 2021).
    Window: [win_start, win_start + window] advances by `shift` days from FPS to LPE.
    Uses bidirectional context — do NOT use as a model predictor (look-ahead bias).
    Returns breaks_df only (no diagnostics).
    '''
    def _add_breaks(pauses_df, current_breaks, th):
        for _, p in pauses_df.iterrows():
            if (p['len'] > th) and (p['dates'] not in current_breaks.dates.tolist()):
                util.add(current_breaks, [p['len'], p['dates'], th])
        return current_breaks

    pauses_dates_list = pauses_dates_list.values.tolist()
    breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])

    for row in pauses_dates_list:
        if str(row[0]).strip() != str(dev).strip():
            continue

        intervals_list = [x for x in row[1:] if isinstance(x, str) and '/' in x and x.strip()]
        intervals_list.sort(key=lambda s: s.split('/')[0])

        if not intervals_list:
            print(dev, 'has NO valid pauses')
            continue

        FPS_dt = datetime.strptime(intervals_list[0].split('/')[0], '%Y-%m-%d')
        LPE_dt = datetime.strptime(intervals_list[-1].split('/')[1], '%Y-%m-%d')
        print(f"[original] Dev {dev}  FPS={FPS_dt.date()}  LPE={LPE_dt.date()}")

        win_start = FPS_dt
        win_end = FPS_dt + timedelta(days=window)
        clear_breaks = pandas.DataFrame(columns=['len', 'dates'])
        last_th = 0

        while win_end < LPE_dt:
            win_pauses_list = pandas.DataFrame(columns=['len', 'dates'])
            partial_list = pandas.DataFrame(columns=['len', 'dates'])

            for interval in intervals_list:
                int_start_str, int_end_str = interval.split('/')
                int_start_dt = datetime.strptime(int_start_str, '%Y-%m-%d')
                int_end_dt = datetime.strptime(int_end_str, '%Y-%m-%d')
                pause_len = util.daysBetween(int_start_str, int_end_str) + 1

                if int_start_dt >= win_start and int_end_dt <= win_end:
                    util.add(win_pauses_list, [pause_len, interval])
                if ((int_start_dt <= win_end and int_end_dt > win_end) or
                        (int_end_dt >= win_start and int_start_dt < win_start)):
                    util.add(partial_list, [pause_len, interval])

            win_pauses = len(win_pauses_list)
            all_pauses = pandas.concat([win_pauses_list, partial_list], ignore_index=True)

            if win_pauses >= 4:
                win_th = getFarOutThreshold(win_pauses_list['len'])
                if win_th > 0:
                    breaks_df = _add_breaks(all_pauses, breaks_df, win_th)
                    last_th = win_th
                elif last_th > 0:
                    breaks_df = _add_breaks(all_pauses, breaks_df, last_th)
            else:
                if last_th > 0:
                    breaks_df = _add_breaks(all_pauses, breaks_df, last_th)
                clear_breaks = cleanClearBreaks(clear_breaks, breaks_df)
                for _, p in all_pauses.iterrows():
                    if (p['len'] >= window and
                            p['dates'] not in clear_breaks.dates.tolist() and
                            p['dates'] not in breaks_df.dates.tolist()):
                        util.add(clear_breaks, p)

            win_start += timedelta(days=shift)
            win_end = win_start + timedelta(days=window)

        # Flush remaining clear_breaks using average threshold
        if len(clear_breaks) > 0:
            mean_th = breaks_df.th.mean() if len(breaks_df) > 0 else 0
            try:
                avg_th = int(round(mean_th, 0))
            except Exception:
                avg_th = 7
            for _, p in clear_breaks.iterrows():
                if p['dates'] not in breaks_df.dates.tolist() and p['len'] > avg_th:
                    util.add(breaks_df, [p['len'], p['dates'], avg_th])

        print(dev, '[original] Done')

    return breaks_df

def timeline(tables, tf_devs, repo_full_name=None, overwrite=False) -> pandas.DataFrame:
    # given 3 files I need you to count the rows per day per user
    # you are given commits.csv issues.csv prs.csv
    # dev (string; developer id/handle)
    # date (date at daily granularity, e.g., YYYY-MM-DD)
    # commits (non-negative integer)
    # prs (non-negative integer)
    # issues (non-negative integer)

    # Fast path: return cached file if it exists and overwrite is not requested
    if repo_full_name and not overwrite:
        _tl_path = Path(ORG_BASE) / repo_full_name / cfg.timeline_folder.lstrip("/") / cfg.timeline_file
        if _tl_path.is_file():
            print("Loading cached timeline from", _tl_path)
            return pandas.read_csv(_tl_path, sep=cfg.CSV_separator)

    issues = tables['issues']
    commits = tables['commits']
    prs = tables['prs_repo']
    issue_activity = tables['issue_activity']
    pr_activity = tables['prs_comments']

    # Convert created_at columns to datetime
    commits['created_at'] = pandas.to_datetime(commits['created_at'])
    issues['created_at'] = pandas.to_datetime(issues['created_at'])
    prs['created_at'] = pandas.to_datetime(prs['created_at'])
    issue_activity['created_at'] = pandas.to_datetime(issue_activity['created_at'])
    pr_activity['created_at'] = pandas.to_datetime(pr_activity['created_at'])

    # We sometimes do not have a author_id column
    # we add a unique identifier ate the start for the column that we used
    # if it has the string "author_login_" at the start use author_login
    # if it has the strung "author_name_" at the start use author_name
    # if it has the string "author_email_" at the start use author_email
    # if there is nothing at the start we use id

    # Count rows per day per user
    user_activity = []
    for dev in tf_devs:
        print(f"{tf_devs.index(dev) + 1} / {len(tf_devs)}") 
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            column, dev = dev.split('|', maxsplit=1)
            dev_commits = commits[commits[column] == dev]
            dev_issues = issues[issues[column] == dev]
            dev_prs = prs[prs[column] == dev]
        else:
            column = "author_id"
            dev_commits = commits[commits[column] == dev]
            dev_issues = issues[issues[column] == dev]
            dev_prs = prs[prs[column] == dev]
        
        dev_issue_activity = issue_activity[issue_activity[column] == dev]
        dev_pr_activity = pr_activity[pr_activity[column] == dev]

        # Set created_at as index, then resample to daily frequency and count rows
        daily_commits = dev_commits.set_index('created_at').resample('D').size()
        daily_issues = dev_issues.set_index('created_at').resample('D').size()
        daily_prs = dev_prs.set_index('created_at').resample('D').size()
        daily_issue_activity = dev_issue_activity.set_index('created_at').resample('D').size()
        daily_pr_activity = dev_pr_activity.set_index('created_at').resample('D').size()

        # Combine all dates. Include NC-only dates so that break days where the
        # developer only reviews PRs or comments on issues are not silently dropped.
        all_dates = (daily_commits.index
                     .union(daily_issues.index)
                     .union(daily_prs.index)
                     .union(daily_issue_activity.index)
                     .union(daily_pr_activity.index))
        
        for date in all_dates:
            user_activity.append({
                'dev': dev,
                'date': date.strftime('%Y-%m-%d'),
                'commits': daily_commits.get(date, 0),
                'prs': daily_prs.get(date, 0),
                'issues': daily_issues.get(date, 0),
                'issue_activity': daily_issue_activity.get(date, 0),
                'pr_activity': daily_pr_activity.get(date, 0)
            })

    user_activity = pandas.DataFrame(user_activity)

    organizationFolder = Path(ORG_BASE) / repo_full_name

    timeline_folder = organizationFolder / cfg.timeline_folder.lstrip("/")
    os.makedirs(timeline_folder, exist_ok=True)
        
    timeline_path = Path(timeline_folder, cfg.timeline_file)

    user_activity.to_csv(timeline_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, lineterminator="\n", index=False)

    return user_activity

#-----------------------
# Developer timeline prediction
#------------------------
def build_response(df_in, N, label_col = "state"):

    d = df_in.sort_values(["dev", "date"]).copy()

    y = d[["dev", "date", label_col]].copy()
    
    y["active_to_inactive"]      = False
    y["non_coding_to_inactive"]  = False
    y["active_to_non_coding"]    = False

    for dev, g in d.groupby("dev", sort=False):
        g = g.sort_values("date").copy()
        next_state = g[label_col].shift(-N)  # look-ahead N days

        # current-state masks
        m_active     = (g[label_col] == "ACTIVE")
        m_noncoding  = (g[label_col] == "NON_CODING")

        # next-state masks (t+N)
        m_next_inact = (next_state == "INACTIVE") | (next_state == "GONE")
        m_next_nc    = (next_state == "NON_CODING")

        # assign ONLY within this group's rows
        y.loc[g.index, "active_to_inactive"]     = (m_active & m_next_inact).values
        y.loc[g.index, "non_coding_to_inactive"] = (m_noncoding & m_next_inact).values
        y.loc[g.index, "active_to_non_coding"]   = (m_active & m_next_nc).values

    y["transition_to_inactive"] = (
        y["active_to_inactive"] | y["non_coding_to_inactive"]
    )

    # final numeric response (avoid NaN->int errors)
    y["transition_to_inactive"] = y["transition_to_inactive"].astype("int8")
    y["active_to_inactive"]    = y["active_to_inactive"].astype("int8")
    y["non_coding_to_inactive"] = y["non_coding_to_inactive"].astype("int8")
    y["active_to_non_coding"]  = y["active_to_non_coding"].astype("int8")
    
    return y

def make_confusion_mats(pred_df, thr_rf=0.6, per_dev=False, date_col="date"):
    """
    Reports:
      (A) Row-level metrics: TP, FP, TN, FN, Precision, Recall, FPR, Specificity
      (B) True-window hit rate: contiguous runs where y_true==1; hit if any pred==1 in run
      (C) Predicted-episode precision: contiguous runs where pred==1; TP episode if overlaps any y_true==1
    """
    df = pred_df.copy()
    df = df.dropna(subset=["y_true", "rf_proba"])
    df["y_true"] = df["y_true"].astype(int)
    df["rf_pred"] = (df["rf_proba"] >= thr_rf).astype(int)

    group_cols = ["dev"] if per_dev and "dev" in df.columns else []
    sort_cols = group_cols + ([date_col] if date_col in df.columns else [])
    if sort_cols:
        df = df.sort_values(sort_cols)

    def _contiguous_run_id(mask: pandas.Series) -> pandas.Series:
        # run id increments at each False->True transition
        starts = mask & (~mask.shift(fill_value=False))
        return starts.cumsum()

    def _metrics_one_group(g: pandas.DataFrame) -> pandas.Series:
        y = g["y_true"].values
        p = g["rf_pred"].values

        # --- (A) Row-level confusion
        TP = int(((y == 1) & (p == 1)).sum())
        FP = int(((y == 0) & (p == 1)).sum())
        TN = int(((y == 0) & (p == 0)).sum())
        FN = int(((y == 1) & (p == 0)).sum())

        precision = TP / (TP + FP) if (TP + FP) else np.nan
        recall    = TP / (TP + FN) if (TP + FN) else np.nan
        fpr       = FP / (FP + TN) if (FP + TN) else np.nan
        spec      = TN / (TN + FP) if (TN + FP) else np.nan

        # --- (B) True-window hit rate (your "window accuracy" is actually window recall)
        true_mask = g["y_true"].eq(1)
        if true_mask.any():
            true_run = _contiguous_run_id(true_mask)
            g2 = g.copy()
            g2["true_run_id"] = np.where(true_mask, true_run, np.nan)

            # each true run is "hit" if any pred==1 inside it
            hits = (g2.loc[true_mask]
                      .groupby("true_run_id")["rf_pred"]
                      .apply(lambda s: int((s == 1).any())))
            true_windows_total = int(hits.size)
            true_windows_hit   = int(hits.sum())
            true_windows_missed = true_windows_total - true_windows_hit
            window_recall = true_windows_hit / true_windows_total if true_windows_total else np.nan
        else:
            true_windows_total = 0
            true_windows_hit = 0
            true_windows_missed = 0
            window_recall = np.nan

        # --- (C) Predicted-episode precision (contiguous runs of pred==1)
        pred_mask = g["rf_pred"].eq(1)
        if pred_mask.any():
            pred_run = _contiguous_run_id(pred_mask)
            g3 = g.copy()
            g3["pred_run_id"] = np.where(pred_mask, pred_run, np.nan)

            # episode is TP if overlaps any y_true==1 within that predicted run
            ep = (g3.loc[pred_mask]
                    .groupby("pred_run_id")["y_true"]
                    .apply(lambda s: int((s == 1).any())))
            pred_episodes_total = int(ep.size)
            pred_episodes_tp = int(ep.sum())
            pred_episodes_fp = pred_episodes_total - pred_episodes_tp
            episode_precision = pred_episodes_tp / pred_episodes_total if pred_episodes_total else np.nan
        else:
            pred_episodes_total = 0
            pred_episodes_tp = 0
            pred_episodes_fp = 0
            episode_precision = np.nan

        return pandas.Series({
            # Row-level
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "specificity": spec,

            # True-window (event recall)
            "true_windows_total": true_windows_total,
            "true_windows_hit": true_windows_hit,
            "true_windows_missed": true_windows_missed,
            "window_recall": window_recall,

            # Predicted episodes (event precision)
            "pred_episodes_total": pred_episodes_total,
            "pred_episodes_tp": pred_episodes_tp,
            "pred_episodes_fp": pred_episodes_fp,
            "episode_precision": episode_precision,
        })

    if group_cols:
        per = df.groupby(group_cols, dropna=False).apply(_metrics_one_group).reset_index()

        # overall row: sum confusion counts, recompute ratios from sums
        sums = per[["TP","FP","TN","FN",
                    "true_windows_total","true_windows_hit","true_windows_missed",
                    "pred_episodes_total","pred_episodes_tp","pred_episodes_fp"]].sum()

        TP, FP, TN, FN = [int(sums[k]) for k in ["TP","FP","TN","FN"]]
        precision = TP/(TP+FP) if (TP+FP) else np.nan
        recall = TP/(TP+FN) if (TP+FN) else np.nan
        fpr = FP/(FP+TN) if (FP+TN) else np.nan
        spec = TN/(TN+FP) if (TN+FP) else np.nan

        twt = int(sums["true_windows_total"])
        twh = int(sums["true_windows_hit"])
        window_recall = twh/twt if twt else np.nan

        pet = int(sums["pred_episodes_total"])
        petp = int(sums["pred_episodes_tp"])
        episode_precision = petp/pet if pet else np.nan

        overall = pandas.DataFrame([{
            group_cols[0]: "__ALL__",
            **{k:int(v) for k,v in sums.items()},
            "precision": precision, "recall": recall, "fpr": fpr, "specificity": spec,
            "window_recall": window_recall,
            "episode_precision": episode_precision,
        }])

        return {"rf": pandas.concat([per, overall], ignore_index=True)}
    else:
        return {"rf": _metrics_one_group(df).to_frame().T}
    
def run_prediction_pipeline(
    df,
    repo_key,
    tf_devs,
    response_col,
    predictor_cols,
    window_size=90,
    epochs=100):

    # ---- SPLIT ----
    train_df, test_df = test_train_split_method(
        df=df,
        pred_cols=predictor_cols,
        response_col=response_col,
        method="Per Dev",
        tf_devs=tf_devs
    )

    # ---- BUILD SEQUENCES ----
    Xtr_tensor, ytr_tensor = build_rolling_sequences(
        train_df,
        feature_cols=predictor_cols,
        label_col=response_col,
        window_size=window_size
    )

    Xte_tensor, yte_tensor = build_rolling_sequences(
        test_df,
        feature_cols=predictor_cols,
        label_col=response_col,
        window_size=window_size
    )

    # ---- MODEL ----
    model = DeveloperLSTM(
        input_size=Xtr_tensor.shape[2],
        hidden_size=126,
        num_layers=3,
        output_size=1
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- TRAIN LOOP ----
    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(Xtr_tensor)
        loss = criterion(outputs, ytr_tensor)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # ---- EVALUATE ----
    model.eval()
    with torch.no_grad():
        if len(Xte_tensor) > 0:
            test_outputs = model(Xte_tensor)

    evaluate_and_plot_developer(
        model,
        test_df,
        Xte_tensor,
        yte_tensor,
        feature_cols=predictor_cols,
        window_size=window_size
    )

    return model

def test_train_split_method(
    df,
    pred_cols,
    response_col,
    method="Per Dev",
    tf_devs=None,
    repos=None,
    split_ratio=0.3,
    random_seed=42):

    random.seed(random_seed)

    if method == "Per Dev":

        n = len(tf_devs)
        size_test = int(n * split_ratio)

        test_devs = random.choices(tf_devs, k=size_test)
        train_devs = random.choices(tf_devs, k=n - size_test)

        test_df = df[df["dev"].isin(test_devs)].copy()
        train_df = df[df["dev"].isin(train_devs)].copy()

        print("Test devs:", test_devs)
        print("Train devs:", train_devs)

    elif method == "Per Repo":

        n = len(repos)
        size_test = int(n * split_ratio)

        test_repos = random.choices(repos, k=size_test)
        train_repos = random.choices(repos, k=n - size_test)

        test_df = df[df["repo"].isin(test_repos)].copy()
        train_df = df[df["repo"].isin(train_repos)].copy()

        print("Test repos:", test_repos)
        print("Train repos:", train_repos)

    else:
        raise ValueError("Invalid method selected")

    # ---- SCALE AFTER SPLIT ----
    scaler = StandardScaler()
    train_df[pred_cols] = scaler.fit_transform(train_df[pred_cols])
    test_df[pred_cols] = scaler.transform(test_df[pred_cols])

    return train_df, test_df

class DeveloperLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(DeveloperLSTM, self).__init__()
        
        # PyTorch only applies inter-layer dropout (dropout > 0 requires num_layers > 1).
        # Passing dropout > 0 with num_layers == 1 raises a UserWarning and has no effect.
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Take last time step
        last_output = lstm_out[:, -1, :]
        
        out = self.fc(last_output)
        return out 

def build_rolling_sequences_with_meta(
    df, feature_cols, label_col,
    dev_col="dev", date_col="date", window_size=90):
    """
    Like build_rolling_sequences, but also returns the (dev, date) of the
    TARGET row (index i) so predictions can be merged back to the dataframe.
    """
    all_sequences, all_labels, all_devs, all_dates = [], [], [], []

    df = df.sort_values([dev_col, date_col]).copy()

    for dev, group in df.groupby(dev_col):
        group  = group.sort_values(date_col)
        feature_cols = [c for c in feature_cols if c in group.columns]
        X_vals = group[feature_cols].values
        y_vals = group[label_col].values
        dates  = group[date_col].values

        if len(group) < window_size:
            continue

        for i in range(window_size, len(group)):
            all_sequences.append(X_vals[i - window_size:i])
            all_labels.append(y_vals[i])
            all_devs.append(dev)
            all_dates.append(dates[i])

    X_tensor = torch.tensor(all_sequences, dtype=torch.float32)
    y_tensor = torch.tensor(all_labels, dtype=torch.long)

    return X_tensor, y_tensor, all_devs, all_dates

def evaluate_model(
    test_df,
    model,
    label_encoder,
    predictor_cols,
    encoded_col,
    window_size=90,
    shift_days=14):

    import torch
    import torch.nn.functional as F
    import pandas as pd
    import numpy as np

    model.eval()
    test_df = test_df.copy()

    class_names    = list(label_encoder.classes_)
    prob_col_names = [f"prob_{c}" for c in class_names]

    # ── fill any predictor columns missing in this df (e.g. dev_ one-hot cols
    #    from training that test-repo developers don't have) ──────────────────
    for col in predictor_cols:
        if col not in test_df.columns:
            test_df[col] = 0

    # ── ensure the encoded label column exists ────────────────────────────────
    if encoded_col not in test_df.columns:
        known_classes = set(label_encoder.classes_)
        state_col = encoded_col.replace("_encoded", "")
        if state_col in test_df.columns:
            test_df[encoded_col] = test_df[state_col].astype(str).apply(
                lambda x: int(label_encoder.transform([x])[0]) if x in known_classes else 0
            )
        else:
            test_df[encoded_col] = 0

    # ── build sequences the same way training did ────────────────────────────
    # build_rolling_sequences_with_meta returns (X_tensor, y_tensor, devs, dates)
    X_tensor, _, all_devs, all_dates = build_rolling_sequences_with_meta(
        test_df,
        feature_cols=predictor_cols,
        label_col=encoded_col,
        window_size=window_size,
    )

    # ── run inference in batches to stay memory safe ─────────────────────────
    BATCH = 256
    all_probs = []
    _infer_device = next(model.parameters()).device   # use whatever device the model is on

    with torch.no_grad():
        for start in range(0, len(X_tensor), BATCH):
            batch   = X_tensor[start : start + BATCH].to(_infer_device)
            logits  = model(batch)                          # (B, num_classes)
            probs   = F.softmax(logits, dim=1)              # (B, num_classes)
            # .tolist() avoids the numpy-not-available error entirely
            all_probs.extend(probs.tolist())                # list of [p0,p1,...] per row

    # ── build a predictions lookup: (dev, date) → prob vector ────────────────
    pred_lookup = {}
    for dev, date, prob_vec in zip(all_devs, all_dates, all_probs):
        pred_lookup[(dev, pd.Timestamp(date))] = prob_vec

    # ── attach probabilities back onto test_df ────────────────────────────────
    result_df = test_df.copy()
    result_df["date"] = pd.to_datetime(result_df["date"])

    for col in prob_col_names:
        result_df[col] = np.nan

    for col in prob_col_names:
        result_df[col] = result_df[col].astype(float)

    for i, row in result_df.iterrows():
        key = (row["dev"], row["date"])
        if key in pred_lookup:
            for col_name, prob_val in zip(prob_col_names, pred_lookup[key]):
                result_df.at[i, col_name] = prob_val

    # we need to shift the probabilities back by shift_days to align with the original response variable
    for col_name in prob_col_names:
        result_df[col_name] = result_df.groupby("dev")[col_name].shift(shift_days)

    # ── compute TDR metrics on the test set ──────────────────────────────────
    compute_tdr_metrics(result_df, shift_days=shift_days)

    return result_df

def compute_tdr_metrics(result_df, shift_days=14, prob_threshold=0.3,
                        output_file="tdr_report.txt"):
    """
    Transition Detection Rate with Lead Time (TDR@shift_days).

    The model predicts state `shift_days` days ahead.  After the shift applied
    in evaluate_model, prob_INACTIVE at row date=D holds the prediction that
    was made at day D-shift_days.  This means the first `shift_days` rows of
    every INACTIVE period contain predictions that were generated *before* the
    period began — they are the genuine early-warning window.

    For each contiguous INACTIVE period (per developer, on the test set):
      - "detected"  = any day in the early-warning window has
                      prob_INACTIVE >= prob_threshold
      - "lead time" = shift_days - (offset of first detection from period start)
                      (max = shift_days days, min = 1 day)

    Prints a summary to stdout and writes it + a per-period table to
    `output_file` (set to None to skip file output).

    Returns a dict of summary stats plus a 'period_records' DataFrame.
    """
    import pandas as pd
    import numpy as np

    df = result_df.dropna(subset=["state"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["dev", "date"]).reset_index(drop=True)

    # Prefer named prob_INACTIVE (future_state / shifted_state modes);
    # fall back to prob_1 (legacy survival_binary binary mode)
    if "prob_INACTIVE" in df.columns:
        prob_col = "prob_INACTIVE"
    elif "prob_1" in df.columns:
        prob_col = "prob_1"
    else:
        available = [c for c in df.columns if c.startswith("prob_")]
        raise ValueError(
            f"Need 'prob_INACTIVE' or 'prob_1' in result_df. "
            f"Available prob columns: {available}"
        )

    total_periods    = 0
    detected_periods = 0
    lead_times       = []
    period_records   = []

    for dev, group in df.groupby("dev"):
        group  = group.sort_values("date").reset_index(drop=True)
        states = group["state"].values
        probs  = group[prob_col].astype(float).values
        dates  = group["date"].values

        i = 0
        while i < len(states):
            if states[i] != "INACTIVE":
                i += 1
                continue

            # Walk to the end of this contiguous INACTIVE run
            j = i
            while j < len(states) and states[j] == "INACTIVE":
                j += 1

            # Early-warning window = first min(shift_days, period_length) days
            # of the period.  Each prob_INACTIVE[i+k] was predicted at
            # date[i+k] - shift_days, i.e. before the period started.
            win_len      = min(shift_days, j - i)
            window_probs = probs[i : i + win_len]

            all_nan  = np.all(np.isnan(window_probs))
            detected = (not all_nan) and (np.nanmax(window_probs) >= prob_threshold)

            lead_time = None
            if detected:
                # np.argmax on a bool array → index of first True
                first_offset = int(np.argmax(window_probs >= prob_threshold))
                lead_time    = shift_days - first_offset
                lead_times.append(lead_time)
                detected_periods += 1

            period_records.append({
                "dev":                dev,
                "period_start":       pd.Timestamp(dates[i]),
                "period_end":         pd.Timestamp(dates[j - 1]),
                "period_length_days": j - i,
                "detected":           detected,
                "lead_time_days":     lead_time,
                "max_prob_in_window": float(np.nanmax(window_probs)) if not all_nan else float("nan"),
            })
            total_periods += 1
            i = j

    tdr         = detected_periods / total_periods if total_periods > 0 else 0.0
    avg_lead    = float(np.mean(lead_times))    if lead_times else 0.0
    median_lead = float(np.median(lead_times))  if lead_times else 0.0

    lines = [
        "=" * 60,
        "  TRANSITION DETECTION RATE (TDR) — TEST SET",
        "=" * 60,
        f"  Probability threshold      : {prob_threshold}",
        f"  Early-warning window       : {shift_days} days",
        f"  Total INACTIVE periods     : {total_periods}",
        f"  Detected periods           : {detected_periods}",
        f"  Transition Detection Rate  : {tdr:.1%}",
        f"  Mean lead time (detected)  : {avg_lead:.1f} days",
        f"  Median lead time           : {median_lead:.1f} days",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print(report)

    if output_file:
        import os
        out_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(report + "\n\n")
            fh.write("Per-period breakdown:\n")
            fh.write(pd.DataFrame(period_records).to_string(index=False))
            fh.write("\n")
        print(f"\n  [TDR] Results written to: {out_path}")

    return {
        "tdr":              tdr,
        "total_periods":    total_periods,
        "detected_periods": detected_periods,
        "mean_lead_time":   avg_lead,
        "median_lead_time": median_lead,
        "period_records":   pd.DataFrame(period_records),
    }

# ---------------------------------------------------------------------------
# Response-variable engineering
# ---------------------------------------------------------------------------

_LEAKY_FEATURES = {"break_day", "th", "len", "win_th"}

# ---------------------------------------------------------------------------
# Feature group constants — used by the evaluation suite and Streamlit UI.
# Keep these in sync with _BASE_COLS below.
# ---------------------------------------------------------------------------
_ACTIVITY_COLS = [
    'commits', 'prs', 'issues', 'issue_activity', 'pr_activity',
    'coding_day', 'nc_day', 'break_day', 'th', 'len', 'event_day',
    'win_pauses', 'win_th', 'last_th', 'added_as_break',
    'commits_7d_mean', 'commits_30d_mean', 'commits_90d_mean',
    'commits_7d_std', 'commits_30d_std',
    'coding_days_30d', 'pr_activity_30d_mean', 'issue_activity_30d_mean',
    'total_commits_lifetime',
    'days_since_last_break', 'length_of_last_break_days',
    'n_breaks_past_90d', 'n_breaks_past_365d',
    'lines_added_today', 'lines_deleted_today',
    'churn_today', 'churn_ratio_today', 'churn_30d_mean',
    'tenure_days',
    'org_commits_today', 'this_repo_share_today',
    'org_active_elsewhere_today', 'org_active_elsewhere_7d',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day_of_year',
    'state_causal_enc', 'response_col_noise',
]
_STN_COLS = [
    'issue_interactions_today', 'issue_unique_partners_today',
    'issue_new_partners_today', 'issue_threads_today',
    'pr_interactions_today', 'pr_unique_partners_today',
    'pr_new_partners_today', 'pr_threads_today',
    'total_unique_partners_today',
    'mention_out_today', 'mention_in_today', 'solo_commit_day',
    'all_new_partners_today', 'new_to_community_today', 'regulars_today',
]
_KD_COLS = [
    'files_worked_today', 'owned_files_today',
    'collab_files_today', 'collab_commit_ratio',
]
_PH_COLS = [
    'repo_commits_7d', 'repo_prs_7d',
    'repo_issues_7d', 'repo_active_devs_7d',
]
_BASE_COLS = _ACTIVITY_COLS + _STN_COLS + _KD_COLS + _PH_COLS

def build_break_level_dataset(list_of_repos, pre_break_window=30):
    """
    Walks all Organizations/{org}/{repo}/Breaks/ directories, reads each
    developer's breaks CSV and their matching labeled timeline CSV, then sums
    the developer's activity in the `pre_break_window` days BEFORE each break
    starts.

    Collecting pre-break activity (not in-break activity) gives non-zero,
    meaningful features: during INACTIVE breaks the developer has no commits
    or PRs by definition, so in-break sums are always 0.

    Developer discovery: derived from Breaks/{dev}_breaks.csv file names.
    Timeline source:     Results/{dev}_labeled_timeline.csv

    Breaks CSV format:   index, len, dates ("YYYY-MM-DD/YYYY-MM-DD"), th
    Timeline CSV format: dev, date, commits, prs, issues,
                         issue_activity, pr_activity, ...

    Returns a DataFrame with one row per break event and columns:
        dev, org, repo, len, dates, th,
        commits, prs, issues, issue_activity, pr_activity
        (activity columns = sums over the pre_break_window days before break start)
    """
    all_rows = []

    for org_path in ORG_BASE.iterdir():
        if not org_path.is_dir():
            continue

        for repo_path in org_path.iterdir():
            if not repo_path.is_dir():
                continue
            print(repo_path)

            breaks_folder  = repo_path / "Breaks"
            results_folder = repo_path / "Results"

            if not breaks_folder.exists():
                continue

            # Discover developers from break file names
            for break_file in breaks_folder.glob("*_breaks.csv"):
                dev = break_file.stem.removesuffix("_breaks")

                timeline_file = results_folder / f"{dev}_labeled_timeline.csv"
                if not timeline_file.exists():
                    continue

                breaks_df = pandas.read_csv(
                    break_file, sep=cfg.CSV_separator,
                    na_values=[cfg.CSV_missing], index_col=0
                )
                if breaks_df.empty:
                    continue

                timeline_df = pandas.read_csv(
                    timeline_file, sep=cfg.CSV_separator,
                    na_values=[cfg.CSV_missing]
                )
                # Old files were written with index_label='date', which creates a
                # duplicate date column that pandas renames to date.1 on read.
                # Drop the spurious numeric index column and restore the real date.
                if "date.1" in timeline_df.columns:
                    timeline_df = timeline_df.drop(columns=["date"]).rename(columns={"date.1": "date"})
                timeline_df["date"] = pandas.to_datetime(timeline_df["date"])
                
                for _, break_row in breaks_df.iterrows():
                    # dates column is "YYYY-MM-DD/YYYY-MM-DD"
                    date_parts = str(break_row["dates"]).split("/")
                    if len(date_parts) != 2:
                        continue

                    break_start = pandas.to_datetime(date_parts[0].strip())

                    # Collect activity in the pre_break_window days BEFORE the break,
                    # not during it — in-break activity is 0 for INACTIVE breaks.
                    window_start = break_start - pandas.Timedelta(days=pre_break_window)
                    mask     = (timeline_df["date"] >= window_start) & (timeline_df["date"] < break_start)
                    activity = timeline_df[mask]

                    all_rows.append({
                        "dev":            dev,
                        "org":            org_path.name,
                        "repo":           repo_path.name,
                        "len":            break_row["len"],
                        "dates":          break_row["dates"],
                        "th":             break_row["th"],
                        "commits":        activity["commits"].sum(),
                        "prs":            activity["prs"].sum(),
                        "issues":         activity["issues"].sum(),
                        "issue_activity": activity["issue_activity"].sum(),
                        "pr_activity":    activity["pr_activity"].sum(),
                    })

    all_rows = pandas.DataFrame(all_rows)
    return all_rows

def genarate_responce_column(
    prediction_df,
    tf_devs,
    response_col,
    predictor_cols,
    window_size=90,
    mode="survival_binary",   # "shifted_state" | "break_length" | "survival_binary"
    shift_days=14,):
    """
    Engineers new response-variable columns on prediction_df.

    Always adds all three engineered columns to the df.
    Returns the (df, response_col, predictor_cols) appropriate for `mode`.

    Modes:
      shifted_state   — classify state shift_days ahead; drops leaky features
      break_length    — regression target (engineers column only; model unchanged)
      survival_binary — binary: will dev be ACTIVE in shift_days days?
    """
    df = prediction_df.sort_values(["dev", "date"]).copy()

    # ── always engineer all three columns ──────────────────────────────────

    # 1. Shifted state
    shifted_col = f"state_shifted_{shift_days}"
    df[shifted_col] = df.groupby("dev")["state"].shift(-shift_days)

    # 2. Break length regression target (first day of each break only)
    df["_break_int"] = df["break_day"].astype(int)
    df["_break_start"] = df.groupby("dev")["_break_int"].transform(
        lambda x: (x.diff().fillna(0) > 0).astype(int)
    )
    df["break_length_target"] = df["len"].where(df["_break_start"] == 1)
    df = df.drop(columns=["_break_int", "_break_start"])

    # 3. Survival binary
    #
    df = df.sort_values(["dev", "date"]).copy()
    
    prev_state = df.groupby("dev")["state"].shift(1)
    df["is_onset"] = (
        (df["state"] == "INACTIVE") & 
        (prev_state != "INACTIVE")
    ).astype(float)

    # for each row, is there an onset in the next `window` rows?
    df[f"break_starts_in_{shift_days}d"] = (
        df.groupby("dev")["is_onset"]
          .transform(lambda s: s[::-1].rolling(shift_days, min_periods=1).max()[::-1])
    ).shift(-1)  # shift 1 so current day isn't included

    df.drop(columns=["is_onset"], inplace=True)


    # ── select active target based on mode ─────────────────────────────────

    if mode == "shifted_state":
        df = df.dropna(subset=[shifted_col])
        active_response_col = [shifted_col]
        active_predictor_cols = [c for c in predictor_cols if c not in _LEAKY_FEATURES]

    elif mode == "break_length":
        # Return unchanged — regression not yet supported by run prediction pipeline_2
        active_response_col = response_col
        active_predictor_cols = predictor_cols

    elif mode == "survival_binary":
        df = df.dropna(subset=[f"break_starts_in_{shift_days}d"])
        df[f"break_starts_in_{shift_days}d"] = df[f"break_starts_in_{shift_days}d"].astype(int).astype(str)
        active_response_col = [f"break_starts_in_{shift_days}d"]   # f"active_in_{shift_days}"
        active_predictor_cols = [c for c in predictor_cols if c not in _LEAKY_FEATURES]

    elif mode == "future_state":
        # Direct N-day-ahead state prediction: "what state will this developer be in
        # exactly shift_days from now?"  Label is one of ACTIVE / NON_CODING / INACTIVE / GONE.
        # This is the cleanest early-warning framing — the model is explicitly trained to
        # forecast forward at a fixed horizon, so evaluation is intuitive and defensible.
        # `state_shifted_N` is already computed above; we just drop the tail rows where
        # the shifted target is NaN (end of each developer's timeline).
        df = df.dropna(subset=[shifted_col])
        active_response_col = [shifted_col]
        # state_causal_enc captures current state without look-ahead — valid predictor.
        # We still remove _LEAKY_FEATURES (break_day, th, win_th) which encode current
        # break membership redundantly, and `len` which requires knowing break end date.
        active_predictor_cols = [c for c in predictor_cols if c not in _LEAKY_FEATURES]

    elif mode == "inactivity_window":
        # Binary at-risk label that covers BOTH:
        #   (a) the pre-break warning window (shift_days before onset), AND
        #   (b) the break period itself (while developer is INACTIVE or GONE).
        #
        # This is more actionable than future_state for a project manager: the model
        # outputs a single risk signal that stays high throughout the entire at-risk
        # window, not just in the days immediately before the break starts.
        #
        # Label = "1"  →  currently INACTIVE/GONE  OR  onset arrives within shift_days
        # Label = "0"  →  otherwise safe

        _at_risk_states = {"INACTIVE", "GONE"}
        _currently_at_risk = df["state"].isin(_at_risk_states)

        # Detect first day of any INACTIVE/GONE period (onset)
        _prev_state_iw = df.groupby("dev")["state"].shift(1)
        df["_is_onset_iw"] = (
            _currently_at_risk &
            ~_prev_state_iw.isin(_at_risk_states)
        ).astype(float)

        # For each row, is there an onset in the NEXT shift_days rows?
        # (reverse-rolling max then shift -1 so current day is not self-counted)
        df["_onset_ahead_iw"] = (
            df.groupby("dev")["_is_onset_iw"]
              .transform(lambda s: s[::-1].rolling(shift_days, min_periods=1).max()[::-1])
        ).shift(-1)

        _risk_col = f"inactivity_window_{shift_days}d"
        df[_risk_col] = (
            _currently_at_risk | (df["_onset_ahead_iw"] == 1.0)
        ).astype(int).astype(str)


        df.drop(columns=["_is_onset_iw", "_onset_ahead_iw"], inplace=True)
        df = df.dropna(subset=[_risk_col])
        active_response_col = [_risk_col]
        active_predictor_cols = [c for c in predictor_cols if c not in _LEAKY_FEATURES]

    else:
        raise ValueError(f"Unknown mode: {mode!r}. "
                         "Choose 'shifted_state', 'break_length', 'survival_binary', "
                         "'future_state', or 'inactivity_window'.")

    return df, active_response_col, active_predictor_cols

def _preprocess_df_for_model(df, label_encoder, response_col, encoded_col):
    """
    Apply the same type coercions that run_prediction_pipeline_2 does,
    so inference on a freshly-built df works the same as during training.
    """
    df = df.copy()
    bool_map = {"yes": 1, "no": 0, "True": 1, "False": 0, True: 1, False: 0}

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "win_start" in df.columns:
        df["win_start"] = pd.to_datetime(df["win_start"])
    if "win_end" in df.columns:
        df["win_end"] = pd.to_datetime(df["win_end"])

    if "break_day" in df.columns:
        df["break_day"] = df["break_day"].astype(int)
    if "th" in df.columns:
        df["th"] = df["th"].astype(float)
    if "len" in df.columns:
        df["len"] = df["len"].astype(float)
    if "added_as_break" in df.columns and df["added_as_break"].dtype == object:
        df["added_as_break"] = df["added_as_break"].map(bool_map)

    # encode the response column if missing
    if encoded_col not in df.columns:
        rc = response_col[0] if isinstance(response_col, list) else response_col
        if rc in df.columns:
            known_classes = set(label_encoder.classes_)
            df[encoded_col] = df[rc].astype(str).apply(
                lambda x: int(label_encoder.transform([x])[0]) if x in known_classes else 0
            )
        else:
            df[encoded_col] = 0

    df = df.fillna(0)
    return df


def run_prediction_pipeline_2(train_df, test_df,
    tf_devs,
    response_col,
    predictor_cols,
    window_size=90,
    epochs=100,
    hidden_size=126,
    num_layers=3,
    lr=0.001,
    dropout=0.2,
    patience=25):
    """
    Train LSTM on train_df, validate on test_df.
    train_df and test_df are already split by repo (repo_split.csv).
    No internal developer split is performed here.
    """

    if isinstance(response_col, list):
        response_col = response_col[0]

    train_df = train_df.copy()
    test_df  = test_df.copy() if test_df is not None and not test_df.empty else train_df.copy()

    # ---- Encode state labels → integers (fit on train only) ----
    label_encoder = LabelEncoder()
    encoded_col   = response_col + "_encoded"
    train_df[encoded_col] = label_encoder.fit_transform(train_df[response_col].astype(str))

    # Transform test labels — unknown labels (states not seen in train) default to 0
    known_classes = set(label_encoder.classes_)
    test_df[encoded_col] = test_df[response_col].astype(str).apply(
        lambda x: int(label_encoder.transform([x])[0]) if x in known_classes else 0
    )

    print(f"State classes: {list(enumerate(label_encoder.classes_))}")

    # dev_activity_tier is already in predictor_cols (computed before calling this function).
    # No one-hot encoding — 100 sparse dev columns add no signal and bloat input size.

    for _df in (train_df, test_df):
        _df["date"] = pd.to_datetime(_df["date"])
        if "win_start" in _df.columns:
            _df["win_start"] = pd.to_datetime(_df["win_start"])
        if "win_end" in _df.columns:
            _df["win_end"] = pd.to_datetime(_df["win_end"])

    bool_map = {"yes": 1, "no": 0, "True": 1, "False": 0, True: 1, False: 0}
    for _df in (train_df, test_df):
        _df["break_day"] = _df["break_day"].astype(int)
        _df["th"]        = _df["th"].astype(float)
        _df["len"]       = _df["len"].astype(float)
        if "added_as_break" in _df.columns and _df["added_as_break"].dtype == object:
            _df["added_as_break"] = _df["added_as_break"].map(bool_map)

    train_df = train_df.fillna(0)
    test_df  = test_df.fillna(0)

    # Intersect predictor_cols with both DataFrames so missing columns in either
    # don't crash build_rolling_sequences_with_meta.
    predictor_cols = [c for c in predictor_cols
                      if c in train_df.columns and c in test_df.columns]
    print(f"[Predictor cols after intersection] {len(predictor_cols)} columns: {predictor_cols}")

    # ---- BUILD SEQUENCES ----
    Xtr_tensor, ytr_tensor = build_rolling_sequences_with_meta(
        train_df, feature_cols=predictor_cols,
        label_col=encoded_col, window_size=window_size
    )[:2]

    Xte_tensor, yte_tensor = build_rolling_sequences_with_meta(
        test_df, feature_cols=predictor_cols,
        label_col=encoded_col, window_size=window_size
    )[:2]

    num_classes = len(label_encoder.classes_)

    device = torch.device('cuda')
    print(f"[GPU] Using device: {device}  ({torch.cuda.get_device_name(0)})")

    # ---- MODEL ----
    model = DeveloperLSTM(
        input_size=Xtr_tensor.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=num_classes,
        dropout=dropout,
    ).to(device)

    # ---- CLASS IMBALANCE: inverse-frequency weights (works for binary and multiclass) ----
    class_names_list = list(label_encoder.classes_)
    inactive_idx = class_names_list.index("INACTIVE") if "INACTIVE" in class_names_list else None

    class_counts = torch.bincount(ytr_tensor, minlength=num_classes).float()
    # Inverse frequency, normalised so the mean weight is 1.0
    class_weights_raw = 1.0 / class_counts.clamp(min=1.0)
    class_weights = (class_weights_raw / class_weights_raw.mean()).to(device)

    count_str = {n: int(c) for n, c in zip(class_names_list, class_counts.tolist())}
    weight_str = {n: f"{w:.2f}" for n, w in zip(class_names_list, class_weights.cpu().tolist())}
    print(f"Class counts (train sequences): {count_str}")
    print(f"Class weights (inverse-freq):   {weight_str}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- CLASS IMBALANCE: weighted sampler (balances each mini-batch) ----
    # Each sample draws its class weight, so minority classes appear ~equally often
    sample_weights = class_weights.cpu()[ytr_tensor].float()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- TRAIN (mini-batched to avoid OOM on large datasets) ----
    from torch.utils.data import DataLoader, TensorDataset

    BATCH_SIZE = 128   # lower to 64 or 32 if still OOM

    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(
        TensorDataset(Xtr_tensor, ytr_tensor),
        batch_size=BATCH_SIZE,
        sampler=sampler,   # replaces shuffle=True; balances classes per batch
    )
    test_loader = DataLoader(
        TensorDataset(Xte_tensor, yte_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.train()
    epoch_results = []   # accumulates per-epoch metrics for HP search

    # ── early stopping state ──────────────────────────────────────────────────
    best_val_f1       = -1.0
    patience_counter  = 0
    best_state_dict   = None   # weights of the best-so-far checkpoint

    for epoch in range(epochs):
        # ── training pass ─────────────────────────────────────────────────
        epoch_loss_tr = 0.0
        total_tr      = 0
        all_preds_tr  = []
        all_labels_tr = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(Xb)
            loss    = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss_tr += loss.item() * len(yb)
            total_tr      += len(yb)
            all_preds_tr.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels_tr.extend(yb.cpu().numpy())
        loss_tr = epoch_loss_tr / total_tr
        from sklearn.metrics import precision_score
        f1_tr        = f1_score(all_labels_tr, all_preds_tr, average='macro', zero_division=0)
        recall_tr    = recall_score(all_labels_tr, all_preds_tr, average='macro', zero_division=0)
        precision_tr = precision_score(all_labels_tr, all_preds_tr, average='macro', zero_division=0)
        # Track INACTIVE recall specifically — this is our TDR proxy
        recall_inactive_tr = (
            recall_score(all_labels_tr, all_preds_tr, labels=[inactive_idx], average='macro', zero_division=0)
            if inactive_idx is not None else recall_tr
        )

        # ── evaluation pass (no gradients stored) ─────────────────────────
        model.eval()
        epoch_loss_te = 0.0
        total_te      = 0
        all_preds_te  = []
        all_labels_te = []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                outputs       = model(Xb)
                epoch_loss_te += criterion(outputs, yb).item() * len(yb)
                total_te      += len(yb)
                all_preds_te.extend(outputs.argmax(dim=1).cpu().numpy())
                all_labels_te.extend(yb.cpu().numpy())
        loss_te      = epoch_loss_te / total_te
        f1_te        = f1_score(all_labels_te, all_preds_te, average='macro', zero_division=0)
        recall_te    = recall_score(all_labels_te, all_preds_te, average='macro', zero_division=0)
        precision_te = precision_score(all_labels_te, all_preds_te, average='macro', zero_division=0)
        recall_inactive_te = (
            recall_score(all_labels_te, all_preds_te, labels=[inactive_idx], average='macro', zero_division=0)
            if inactive_idx is not None else recall_te
        )
        model.train()

        epoch_results.append({
            'epoch':                    epoch + 1,
            'train_loss':               loss_tr,
            'train_f1_macro':           f1_tr,
            'train_recall_macro':       recall_tr,
            'train_precision_macro':    precision_tr,
            'train_recall_INACTIVE':    recall_inactive_tr,
            'test_loss':                loss_te,
            'test_f1_macro':            f1_te,
            'test_recall_macro':        recall_te,
            'test_precision_macro':     precision_te,
            'test_recall_INACTIVE':     recall_inactive_te,
        })

        timer()
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Training: Loss: {loss_tr:.4f}, F1(macro): {f1_tr:.4f}, "
              f"Recall(macro): {recall_tr:.4f}, Recall(INACTIVE): {recall_inactive_tr:.4f}, "
              f"Precision(macro): {precision_tr:.4f}")
        print(f"  Testing:  Loss: {loss_te:.4f}, F1(macro): {f1_te:.4f}, "
              f"Recall(macro): {recall_te:.4f}, Recall(INACTIVE): {recall_inactive_te:.4f}, "
              f"Precision(macro): {precision_te:.4f}")
        print(f"  --> TDR proxy: test recall on INACTIVE class = {recall_inactive_te:.1%}")

        # ── early stopping check ───────────────────────────────────────────────
        if f1_te > best_val_f1:
            best_val_f1      = f1_te
            patience_counter = 0
            best_state_dict  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [Early Stop] New best val F1={best_val_f1:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            print(f"  [Early Stop] No improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"  [Early Stop] Stopping at epoch {epoch + 1}.")
                break

    # restore best weights before returning
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[Early Stop] Best weights restored (val F1={best_val_f1:.4f}).")

    # ── return encoded test_df so the caller can slice it per repo ───────────
    return model, label_encoder, test_df, predictor_cols, encoded_col, epoch_results

def plot_hp_epoch_curves(epoch_curves_path):
    """
    Read hp_epoch_curves.csv and render 4 Streamlit charts:
    train vs test  loss / f1_macro / recall_macro / precision_macro.
    Each config gets its own line, labelled by its hyperparameters.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    curves_path = Path(epoch_curves_path)
    if not curves_path.exists():
        st.info("No epoch curves file found yet — run the HP search first.")
        return

    df = pd.read_csv(curves_path)
    if df.empty:
        st.info("Epoch curves file is empty.")
        return

    metrics = [
        ("loss",              "Loss",               "train_loss",            "test_loss"),
        ("f1_macro",          "F1 Macro",           "train_f1_macro",        "test_f1_macro"),
        ("recall_macro",      "Recall Macro",       "train_recall_macro",    "test_recall_macro"),
        ("precision_macro",   "Precision Macro",    "train_precision_macro", "test_precision_macro"),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m[1] for m in metrics],
        shared_xaxes=False,
    )

    config_cols = ['config_id', 'hidden_size', 'num_layers', 'lr', 'dropout']
    for group_keys, grp in df.groupby(config_cols):
        cid, hs, nl, lr_val, do = group_keys
        label = f"cfg{cid} h{hs} L{nl} lr{lr_val} d{do}"
        grp = grp.sort_values('epoch')

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        for (row, col), (_, _, train_col, test_col) in zip(positions, metrics):
            show_legend = (row == 1 and col == 1)

            if train_col in grp.columns:
                fig.add_trace(go.Scatter(
                    x=grp['epoch'], y=grp[train_col],
                    mode='lines', name=f"{label} train",
                    line=dict(dash='solid'),
                    legendgroup=label,
                    showlegend=show_legend,
                ), row=row, col=col)

            if test_col in grp.columns:
                fig.add_trace(go.Scatter(
                    x=grp['epoch'], y=grp[test_col],
                    mode='lines', name=f"{label} test",
                    line=dict(dash='dot'),
                    legendgroup=label,
                    showlegend=show_legend,
                ), row=row, col=col)

    fig.update_layout(
        height=700,
        title_text="HP Search — Training Curves (solid=train, dotted=test)",
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Baseline models and evaluation metrics for the paper
# ---------------------------------------------------------------------------

def naive_baseline_scores(X_np, col_names):
    """
    Score = 1.0 if commits_7d_mean (or 'commits') is 0 on the last window day.
    Mirrors the "no commit in 7 days → flag as at-risk" heuristic a maintainer
    would apply by hand.
    """
    key = 'commits_7d_mean' if 'commits_7d_mean' in col_names else 'commits'
    if key not in col_names:
        return np.zeros(len(X_np))
    idx = col_names.index(key)
    return np.array([1.0 if seq[-1, idx] == 0.0 else 0.0 for seq in X_np])


def logreg_baseline_scores(X_tr_np, y_tr_np, X_te_np, pos_label=1):
    """
    Aggregate each (window, d) sequence → (3d,) = [window-mean, last-day, window-std].
    Fits LogisticRegression (no temporal structure) — lift over this is attributable
    to the LSTM's sequence modeling, not feature engineering alone.
    Returns P(pos_label) for each test sequence.
    """
    def agg(S):
        return np.concatenate([S.mean(axis=1), S[:, -1, :], S.std(axis=1)], axis=1)
    Xtr = agg(X_tr_np)
    Xte = agg(X_te_np)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1, solver='lbfgs')
    clf.fit(Xtr, y_tr_np)
    classes = list(clf.classes_)
    if pos_label not in classes:
        return np.zeros(len(Xte))
    return clf.predict_proba(Xte)[:, classes.index(pos_label)]


def compute_pr_auc(y_true, y_scores):
    """PR-AUC (average precision). Returns NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return average_precision_score(y_true, y_scores)


def compute_window_recall(df, state_col='state', prob_col='prob_positive',
                          horizon_days=14, threshold=0.3):
    """
    Fraction of INACTIVE/GONE onset events where model score >= threshold fires
    at least once in the `horizon_days` window immediately BEFORE the onset.

    Uses the raw `state` column (not the engineered label) so the metric is
    independent of label-window size and stays interpretable across horizons.

    NOTE: window-level recall is NOT directly comparable across different
    horizon_days values — use PR-AUC for cross-horizon comparison.
    """
    caught = total = 0
    at_risk = {'INACTIVE', 'GONE'}
    for _dev, grp in df.groupby('dev'):
        grp    = grp.sort_values('date').reset_index(drop=True)
        states = grp[state_col].values
        raw    = grp[prob_col].values.astype(float)
        scores = np.where(np.isnan(raw), 0.0, raw)
        for i in range(1, len(states)):
            if states[i] in at_risk and states[i - 1] not in at_risk:
                total += 1
                ws    = max(0, i - horizon_days)
                if len(scores[ws:i]) > 0 and scores[ws:i].max() >= threshold:
                    caught += 1
    return caught / total if total > 0 else float('nan')


# ---------------------------------------------------------------------------
# Evaluation suite — horizon sensitivity, feature ablation, LOPO, Dev-CV
# ---------------------------------------------------------------------------

def run_evaluation_suite(
    train_df,
    test_df,
    predictor_cols=None,
    window_size=90,
    horizons=(7, 14, 30),
    ablation_horizon=14,
    max_epochs=120,
    patience=40,
    lopo=True,
    dev_cv_folds=5,
):
    """
    Paper evaluation suite. Prints plain-text tables to stdout.

    Run from CLI:
        python Extractors/DemoAppV2.2.py --eval-suite

    Tables:
      1. Horizon sensitivity  (k in horizons; LSTM vs LogReg vs Naive)
      2. Feature group ablation  (k=ablation_horizon; one group at a time)
      3. Cross-validation  (LOPO and dev-fold-CV; k=ablation_horizon; LSTM)

    Returns dict of DataFrames with all results.
    """
    import torch.nn.functional as F
    from sklearn.model_selection import KFold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[EvalSuite] device={device}  "
          f"train={len(train_df)} rows  test={len(test_df)} rows")

    if predictor_cols is None:
        predictor_cols = [c for c in _BASE_COLS if c in train_df.columns]

    # ── inner helpers ─────────────────────────────────────────────────────────

    def _get_proba(model, X_tensor, pos_idx):
        """Softmax probability of pos_idx class for every sequence in X_tensor."""
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X_tensor), 256):
                b = X_tensor[s:s + 256].to(device)
                p = F.softmax(model(b), dim=1)
                out.extend(p[:, pos_idx].cpu().numpy().tolist())
        return np.array(out)

    def _labeled(raw_df, pcols, k):
        """Apply inactivity_window label engineering for horizon k."""
        devs = raw_df['dev'].unique().tolist()
        return genarate_responce_column(
            raw_df.copy(), tf_devs=devs,
            response_col=['state'], predictor_cols=list(pcols),
            window_size=window_size, mode='inactivity_window', shift_days=k,
        )  # → (df_k, resp_col, active_pcols)

    def _seqs(df_k, active_pcols, enc_col, le):
        """Encode labels with le, build rolling sequences, return numeric tensors."""
        df_k = df_k.copy()
        # Apply the same bool/string coercions run_prediction_pipeline_2 does
        # so that object-dtype columns (e.g. added_as_break = "yes"/"no") don't
        # cause torch.tensor to fail with "must be real number, not str".
        _bool_map = {"yes": 1, "no": 0, "True": 1, "False": 0, True: 1, False: 0}
        if "added_as_break" in df_k.columns and df_k["added_as_break"].dtype == object:
            df_k["added_as_break"] = df_k["added_as_break"].map(_bool_map)
        pcols_here = [c for c in active_pcols if c in df_k.columns]
        for c in pcols_here:
            if df_k[c].dtype == object:
                df_k[c] = pd.to_numeric(df_k[c], errors='coerce')
        df_k = df_k.fillna(0)
        rc    = enc_col.replace('_encoded', '')
        known = set(le.classes_)
        df_k[enc_col] = df_k[rc].astype(str).apply(
            lambda x: int(le.transform([x])[0]) if x in known else 0
        )
        X, y, devs, dates = build_rolling_sequences_with_meta(
            df_k, feature_cols=pcols_here, label_col=enc_col, window_size=window_size
        )
        return X, y, devs, dates, pcols_here

    def _state_merge(devs, dates, probs, te_df_k):
        """Merge model probs back to the state timeline for window-recall computation."""
        pred = pd.DataFrame({
            'dev':           list(devs),
            'date':          pd.to_datetime(list(dates)),
            'prob_positive': probs,
        })
        state = te_df_k[['dev', 'date', 'state']].copy()
        state['date'] = pd.to_datetime(state['date'])
        return state.merge(pred, on=['dev', 'date'], how='left')

    def _run_one(tr_raw, te_raw, pcols, k):
        """
        Train LSTM on tr_raw for horizon k; score baselines on te_raw.
        Returns (lstm_probs, naive_probs, lr_probs, y_te, te_devs, te_dates,
                 te_df_k, active_pcols)  or  None if te_raw yields no sequences.
        """
        tf_inner = tr_raw['dev'].unique().tolist()
        tr_k, resp_col, pcols_active = _labeled(tr_raw, pcols, k)
        te_k, _,        _            = _labeled(te_raw, pcols, k)

        model, le, _, active_pcols, enc_col, _ = run_prediction_pipeline_2(
            tr_k, pandas.DataFrame(), tf_inner, resp_col, pcols_active,
            window_size=window_size, epochs=max_epochs,
            hidden_size=128, num_layers=3, lr=0.001, dropout=0.25,
            patience=patience,
        )

        X_te, y_te, te_devs, te_dates, final_pcols = _seqs(te_k, active_pcols, enc_col, le)
        X_tr, y_tr, _,       _,        _           = _seqs(tr_k, active_pcols, enc_col, le)

        if len(X_te) == 0:
            return None

        classes  = list(le.classes_)
        pos_idx  = classes.index('1') if '1' in classes else 1
        y_te_np  = y_te.numpy()
        X_te_np  = X_te.numpy()

        lstm_probs  = _get_proba(model, X_te, pos_idx)
        naive_probs = naive_baseline_scores(X_te_np, final_pcols)
        lr_probs    = logreg_baseline_scores(
            X_tr.numpy(), y_tr.numpy(), X_te_np, pos_label=pos_idx
        )

        return lstm_probs, naive_probs, lr_probs, y_te_np, te_devs, te_dates, te_k, final_pcols

    def _metrics(probs, y_true, devs, dates, df_k, k):
        pr = compute_pr_auc(y_true, probs)
        wr = compute_window_recall(_state_merge(devs, dates, probs, df_k), horizon_days=k)
        return pr, wr

    # ── TABLE 1: Horizon sensitivity ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TABLE 1: Horizon sensitivity")
    print("=" * 60)

    horizon_rows = []
    if test_df.empty:
        print("  [SKIP] No test repos — cannot compute horizon / ablation tables.")
    else:
        for k in horizons:
            print(f"\n  [k={k}] training LSTM + baselines ...")
            res = _run_one(train_df, test_df, predictor_cols, k)
            if res is None:
                print(f"  [k={k}] skipped — no test sequences")
                continue
            lstm_p, naive_p, lr_p, y_te, te_devs, te_dates, te_k, _ = res
            for name, probs in [('LSTM', lstm_p), ('LogReg', lr_p), ('Naive', naive_p)]:
                pr, wr = _metrics(probs, y_te, te_devs, te_dates, te_k, k)
                horizon_rows.append({'k': k, 'Model': name, 'PR_AUC': pr, 'WinRecall_0.3': wr})
                print(f"    k={k}  {name:8s}  PR-AUC={pr:.4f}  WinRecall@0.3={wr:.4f}")

    # ── TABLE 2: Feature group ablation (k=ablation_horizon) ─────────────────
    print("\n" + "=" * 60)
    print(f"TABLE 2: Feature group ablation  (k={ablation_horizon}, LSTM only)")
    print("=" * 60)

    ablation_rows = []
    if test_df.empty:
        print("  [SKIP] No test repos.")
    else:
        group_defs = [
            ('Activity', _ACTIVITY_COLS),
            ('STN',      _STN_COLS),
            ('KD',       _KD_COLS),
            ('PH',       _PH_COLS),
            ('All',      _BASE_COLS),
        ]
        for grp_name, grp_cols in group_defs:
            pcols_sub = [c for c in grp_cols if c in train_df.columns]
            if not pcols_sub:
                print(f"  [{grp_name}] no columns present — skipping")
                continue
            print(f"\n  [{grp_name}] ({len(pcols_sub)} cols) training LSTM ...")
            res = _run_one(train_df, test_df, pcols_sub, ablation_horizon)
            if res is None:
                print(f"  [{grp_name}] skipped — no test sequences")
                continue
            lstm_p, _, _, y_te, te_devs, te_dates, te_k, _ = res
            pr, wr = _metrics(lstm_p, y_te, te_devs, te_dates, te_k, ablation_horizon)
            ablation_rows.append({'Group': grp_name, 'PR_AUC': pr, 'WinRecall_0.3': wr})
            print(f"    {grp_name:12s}  PR-AUC={pr:.4f}  WinRecall@0.3={wr:.4f}")

    # ── TABLE 3: Cross-validation (k=ablation_horizon, LSTM) ─────────────────
    print("\n" + "=" * 60)
    print(f"TABLE 3: Cross-validation  (k={ablation_horizon}, LSTM)")
    print("=" * 60)

    cv_rows = []

    # LOPO ─────────────────────────────────────────────────────────────────────
    if lopo:
        all_df    = pandas.concat([train_df, test_df], ignore_index=True) if not test_df.empty else train_df
        all_repos = all_df['_repo'].unique().tolist() if '_repo' in all_df.columns else []
        if not all_repos:
            print("  [LOPO] No _repo column found — skipping LOPO.")
        else:
            lopo_praucs = []
            print(f"\n  LOPO over {len(all_repos)} repos ...")
            for repo in all_repos:
                tr_lopo = all_df[all_df['_repo'] != repo]
                te_lopo = all_df[all_df['_repo'] == repo]
                if tr_lopo.empty or te_lopo.empty:
                    continue
                print(f"    hold-out: {repo}")
                res = _run_one(tr_lopo, te_lopo, predictor_cols, ablation_horizon)
                if res is None:
                    print(f"    {repo}  — no test sequences, skipped")
                    continue
                lstm_p, _, _, y_te, te_devs, te_dates, te_k, _ = res
                pr, _ = _metrics(lstm_p, y_te, te_devs, te_dates, te_k, ablation_horizon)
                lopo_praucs.append(pr)
                print(f"    {repo}  PR-AUC={pr:.4f}")
            lopo_mean = float(np.nanmean(lopo_praucs)) if lopo_praucs else float('nan')
            print(f"  LOPO mean PR-AUC: {lopo_mean:.4f}  (N={len(lopo_praucs)} repos)")
            cv_rows.append({'Method': 'LOPO', 'PR_AUC_mean': lopo_mean, 'N': len(lopo_praucs)})

    # 5-fold developer CV ──────────────────────────────────────────────────────
    devs_array    = np.array(train_df['dev'].unique())
    kf            = KFold(n_splits=dev_cv_folds, shuffle=True, random_state=42)
    dev_cv_praucs = []
    print(f"\n  {dev_cv_folds}-fold developer CV ({len(devs_array)} train devs) ...")
    for fold, (tr_idx, te_idx) in enumerate(kf.split(devs_array)):
        tr_devs_fold = set(devs_array[tr_idx])
        te_devs_fold = set(devs_array[te_idx])
        tr_fold = train_df[train_df['dev'].isin(tr_devs_fold)]
        te_fold = train_df[train_df['dev'].isin(te_devs_fold)]
        if tr_fold.empty or te_fold.empty:
            continue
        print(f"    Fold {fold + 1}/{dev_cv_folds}  "
              f"tr_devs={len(tr_devs_fold)}  te_devs={len(te_devs_fold)}")
        res = _run_one(tr_fold, te_fold, predictor_cols, ablation_horizon)
        if res is None:
            print(f"    Fold {fold + 1} — no test sequences, skipped")
            continue
        lstm_p, _, _, y_te, te_devs, te_dates, te_k, _ = res
        pr, _ = _metrics(lstm_p, y_te, te_devs, te_dates, te_k, ablation_horizon)
        dev_cv_praucs.append(pr)
        print(f"    Fold {fold + 1}  PR-AUC={pr:.4f}")
    dev_cv_mean = float(np.nanmean(dev_cv_praucs)) if dev_cv_praucs else float('nan')
    print(f"  Dev-CV mean PR-AUC: {dev_cv_mean:.4f}  ({dev_cv_folds}-fold)")
    cv_rows.append({'Method': f'Dev-CV ({dev_cv_folds}-fold)', 'PR_AUC_mean': dev_cv_mean, 'N': dev_cv_folds})

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n\n" + "#" * 60)
    print("### RESULTS SUMMARY")
    print("#" * 60)

    print("\n=== TABLE 1: HORIZON SENSITIVITY ===")
    print(f"{'k':>4}  {'Model':10}  {'PR-AUC':>8}  {'WinRecall@0.3':>14}")
    print(f"{'':>4}  {'':10}  {'':>8}  (WinRecall uses fixed threshold; NOT comparable across k)")
    for r in horizon_rows:
        print(f"{r['k']:>4}  {r['Model']:10}  {r['PR_AUC']:>8.4f}  {r['WinRecall_0.3']:>14.4f}")

    print(f"\n=== TABLE 2: FEATURE ABLATION (k={ablation_horizon}) ===")
    print(f"{'Group':12}  {'PR-AUC':>8}  {'WinRecall@0.3':>14}")
    for r in ablation_rows:
        print(f"{r['Group']:12}  {r['PR_AUC']:>8.4f}  {r['WinRecall_0.3']:>14.4f}")

    print(f"\n=== TABLE 3: CROSS-VALIDATION (k={ablation_horizon}, LSTM) ===")
    print(f"{'Method':25}  {'PR-AUC mean':>12}  {'N':>4}")
    for r in cv_rows:
        print(f"{r['Method']:25}  {r['PR_AUC_mean']:>12.4f}  {r['N']:>4}")

    print("\n[EvalSuite] Done.\n")
    return {
        'horizon':  pd.DataFrame(horizon_rows),
        'ablation': pd.DataFrame(ablation_rows),
        'cv':       pd.DataFrame(cv_rows),
    }


def run_hyperparameter_search(
    df,
    tf_devs,
    response_col,
    predictor_cols,
    window_size=90,
    max_epochs_per_config=120,
    patience=40,
    hp_results_path="hp_search_results.csv",
    epoch_curves_path=None,
    stop_flag_path=None,):
    """
    Grid search over LSTM hyperparameters.

    Saves a one-row-per-config summary to hp_results_path after every config.
    Saves full per-epoch curves to epoch_curves_path (one row per config×epoch).

    Early termination: create the file at stop_flag_path (default: same dir as
    hp_results_path with name 'hp_search_stop.flag') and the search will finish
    the current config then stop, returning the best model found so far.

    Grid: hidden_size × num_layers × lr × dropout = 36 configs total.
    NOTE: dropout only applies between LSTM layers — num_layers=1 always gets
    0 dropout (PyTorch constraint, handled silently in DeveloperLSTM).
    """
    from itertools import product as iproduct

    hp_results_path = Path(hp_results_path)
    if epoch_curves_path is None:
        epoch_curves_path = hp_results_path.parent / "hp_epoch_curves.csv"
    else:
        epoch_curves_path = Path(epoch_curves_path)
    if stop_flag_path is None:
        stop_flag_path = hp_results_path.parent / "hp_search_stop.flag"
    else:
        stop_flag_path = Path(stop_flag_path)

    # Grid: hidden_size × num_layers × lr × dropout  (2×2×2×2 = 16 configs)
    # 3 configs × ~80 avg epochs × 120 sec ≈ 12h. patience=40 stops before the
    # epoch-110 divergence observed in prior runs.
    param_grid = {
        'hidden_size': [128 ],
        'num_layers':  [ 3 ],
        'lr':          [0.001],
        'dropout':     [0.25],
    }
    keys    = list(param_grid.keys())
    configs = list(iproduct(*param_grid.values()))
    print(f"[HP Search] {len(configs)} configs, max {max_epochs_per_config} epochs "
          f"each with patience={patience}  (early stopping, so actual epochs will be less)")
    print(f"[HP Search] To stop early: create the file  {stop_flag_path}")

    hp_rows        = []
    curve_rows     = []
    best_f1        = -1.0
    best_model     = None
    best_le        = None
    best_test_df   = None
    best_pred_cols = None
    best_enc_col   = None

    try:
        for idx, values in enumerate(configs):
            config = dict(zip(keys, values))
            print(f"\n=== HP Config {idx+1}/{len(configs)}: {config} ===")

            model, le, test_df, pred_cols, enc_col, epoch_results = run_prediction_pipeline_2(
                df, pandas.DataFrame(), tf_devs, response_col, predictor_cols,
                window_size=window_size,
                epochs=max_epochs_per_config,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                lr=config['lr'],
                dropout=config['dropout'],
                patience=patience,
            )

            best_epoch_f1 = max((r['test_f1_macro'] for r in epoch_results), default=0.0)
            final         = epoch_results[-1]
            epochs_run    = len(epoch_results)

            # ── summary row ───────────────────────────────────────────────────
            row = {
                'config_id':                   idx,
                'hidden_size':                 config['hidden_size'],
                'num_layers':                  config['num_layers'],
                'lr':                          config['lr'],
                'dropout':                     config['dropout'],
                'epochs_run':                  epochs_run,
                'best_test_f1_macro':          best_epoch_f1,
                'final_test_f1_macro':         final['test_f1_macro'],
                'final_test_recall_macro':     final['test_recall_macro'],
                'final_test_precision_macro':  final.get('test_precision_macro', float('nan')),
                'final_test_recall_INACTIVE':  final.get('test_recall_INACTIVE', float('nan')),
                'final_test_loss':             final['test_loss'],
            }
            hp_rows.append(row)
            pd.DataFrame(hp_rows).to_csv(hp_results_path, index=False)

            # ── full per-epoch curves ─────────────────────────────────────────
            for er in epoch_results:
                curve_rows.append({
                    'config_id':   idx,
                    'hidden_size': config['hidden_size'],
                    'num_layers':  config['num_layers'],
                    'lr':          config['lr'],
                    'dropout':     config['dropout'],
                    **er,
                })
            pd.DataFrame(curve_rows).to_csv(epoch_curves_path, index=False)

            print(f"[HP] Config {idx+1} best F1={best_epoch_f1:.4f} "
                  f"({epochs_run} epochs) — saved to {hp_results_path}")

            if best_epoch_f1 > best_f1:
                best_f1        = best_epoch_f1
                best_model     = model
                best_le        = le
                best_test_df   = test_df
                best_pred_cols = pred_cols
                best_enc_col   = enc_col

            # ── sentinel early-stop ───────────────────────────────────────────
            if stop_flag_path.exists():
                print(f"[HP Search] Stop flag detected — finishing after config {idx+1}.")
                stop_flag_path.unlink(missing_ok=True)
                break

    except KeyboardInterrupt:
        print(f"\n[HP Search] KeyboardInterrupt — returning best model so far (F1={best_f1:.4f}).")

    hp_results_df = pd.DataFrame(hp_rows)
    if not hp_results_df.empty:
        print(f"\n=== HP Search Done. Best F1(macro)={best_f1:.4f} ===")
        print(hp_results_df.sort_values('best_test_f1_macro', ascending=False).to_string(index=False))

    return best_model, best_le, best_test_df, best_pred_cols, best_enc_col, hp_results_df

def timer():
    now = datetime.now()
    print(f"timebetween calls: {(now - LAST_CALLED[0]).total_seconds() if LAST_CALLED[0] else 'N/A'} seconds")
    LAST_CALLED[0] = now

def build_break_sequences(df, feature_cols, label_col,
                          dev_col="dev", date_col="break_start", window_size=5):
    """
    Builds rolling-window sequences from a break-level DataFrame (one row per break).

    For each developer, slides a window of `window_size` consecutive breaks
    and predicts the label of the break immediately after the window.

    Returns: (X_tensor [N, window, features], y_tensor [N], devs [N])
    """
    all_X, all_y, all_devs = [], [], []

    for dev, group in df.groupby(dev_col):
        group = group.sort_values(date_col)
        X = group[feature_cols].values.astype(float)
        y = group[label_col].values.astype(float)

        if len(group) <= window_size:
            continue

        for i in range(window_size, len(group)):
            all_X.append(X[i - window_size : i])
            all_y.append(y[i])
            all_devs.append(dev)

    if not all_X:
        return torch.zeros(0), torch.zeros(0), []

    X_tensor = torch.tensor(np.array(all_X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(all_y), dtype=torch.float32)
    return X_tensor, y_tensor, all_devs

def break_predictions_pipeline(
    breaks_df,
    window_size=5,
    epochs=200,
    hidden_size=64,
    num_layers=2,
    lr=0.001,
    dropout=0.2,
    patience=30,):
    """
    LSTM regression pipeline: given a window of past breaks, predict
    the length of the next break in days.

    Input:   break-level DataFrame from build_break_level_dataset()
             (one row per break, columns: dev, len, dates, th,
              commits, prs, issues, issue_activity, pr_activity)

    Response column: next_break_len = len.shift(-1) per developer,
                     so each row's target is the FOLLOWING break's length.

    Model:   DeveloperLSTM (output_size=1) with HuberLoss — robust to the
             long-tail distribution of break lengths.

    Metrics: MAE, RMSE, R²

    Returns: (model, scaler, results_df, feature_cols)
        results_df — dev, actual_next_break_len, predicted_next_break_len
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from torch.utils.data import DataLoader, TensorDataset

    df = breaks_df.copy()

    # ── parse break start date for ordering ──────────────────────────────────
    df["break_start"] = pd.to_datetime(
        df["dates"].str.split("/").str[0].str.strip()
    )
    df = df.sort_values(["dev", "break_start"]).reset_index(drop=True)

    # ── response: length of the NEXT break per developer ─────────────────────
    df["next_break_len"] = df.groupby("dev")["len"].shift(-1)
    df = df.dropna(subset=["next_break_len"]).reset_index(drop=True)

    feature_cols = ["len", "th", "commits", "prs", "issues", "issue_activity", "pr_activity"]
    label_col    = "next_break_len"

    # ── scale features ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))

    # ── train / test split by developer (70 / 30) ────────────────────────────
    devs      = df["dev"].unique()
    rng       = np.random.default_rng(42)
    test_devs = set(rng.choice(devs, size=max(1, int(len(devs) * 0.3)), replace=False))
    train_devs = set(devs) - test_devs

    train_df = df[df["dev"].isin(train_devs)].copy()
    test_df  = df[df["dev"].isin(test_devs)].copy()

    print(f"Train devs: {len(train_devs)}, Test devs: {len(test_devs)}")
    print(f"Train break rows: {len(train_df)}, Test break rows: {len(test_df)}")

    # ── build sequences ───────────────────────────────────────────────────────
    Xtr, ytr, _        = build_break_sequences(train_df, feature_cols, label_col, window_size=window_size)
    Xte, yte, te_devs  = build_break_sequences(test_df,  feature_cols, label_col, window_size=window_size)

    if len(Xtr) == 0:
        raise ValueError(
            f"No training sequences — window_size={window_size} is too large. "
            "Developers need at least window_size+1 breaks each."
        )

    print(f"Training sequences: {len(Xtr)}, Test sequences: {len(Xte)}")

    # ── model (output_size=1 for regression) ─────────────────────────────────
    model     = DeveloperLSTM(
        input_size  = Xtr.shape[2],
        hidden_size = hidden_size,
        num_layers  = num_layers,
        output_size = 1,
        dropout     = dropout,
    )
    criterion = nn.HuberLoss()        # robust to long-break outliers
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr.unsqueeze(1)),
        batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte.unsqueeze(1)),
        batch_size=32, shuffle=False
    )

    # ── training with early stopping (monitor MAE) ───────────────────────────
    best_mae         = float("inf")
    patience_counter = 0
    best_state_dict  = None

    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            criterion(model(Xb), yb).backward()
            optimizer.step()

        model.eval()
        preds_e, labels_e = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds_e.extend(model(Xb).squeeze(1).cpu().numpy())
                labels_e.extend(yb.squeeze(1).cpu().numpy())

        mae_e = mean_absolute_error(labels_e, preds_e)

        if mae_e < best_mae:
            best_mae         = mae_e
            patience_counter = 0
            best_state_dict  = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            rmse_e = np.sqrt(mean_squared_error(labels_e, preds_e))
            r2_e   = r2_score(labels_e, preds_e)
            print(f"Epoch {epoch+1:4d} | MAE {mae_e:.1f}d | RMSE {rmse_e:.1f}d | R² {r2_e:.3f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} — best test MAE: {best_mae:.1f}d")
            break

    # ── restore best weights and final eval ──────────────────────────────────
    model.load_state_dict(best_state_dict)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_preds.extend(model(Xb).squeeze(1).cpu().numpy())
            all_labels.extend(yb.squeeze(1).cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    mae  = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    r2   = r2_score(all_labels, all_preds)

    print(f"\n── Break Length Regression Results ─────────────────")
    print(f"  MAE:  {mae:.1f} days")
    print(f"  RMSE: {rmse:.1f} days")
    print(f"  R²:   {r2:.3f}")

    results_df = pd.DataFrame({
        "dev":                       te_devs,
        "actual_next_break_len":     all_labels,
        "predicted_next_break_len":  all_preds,
    })

    # Add org/repo so Dashboard can filter by repo
    dev_org_repo = breaks_df[["dev", "org", "repo"]].drop_duplicates("dev")
    results_df = results_df.merge(dev_org_repo, on="dev", how="left")

    # Save — append to global file, replacing rows for this org/repo on each run
    out_path = Path(__file__).resolve().parents[1] / "Organizations" / "break_prediction_df.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path)
        this_org  = dev_org_repo["org"].iloc[0]  if not dev_org_repo.empty else ""
        this_repo = dev_org_repo["repo"].iloc[0] if not dev_org_repo.empty else ""
        existing = existing[~((existing["org"] == this_org) & (existing["repo"] == this_repo))]
        results_df = pd.concat([existing, results_df], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    return model, scaler, results_df, feature_cols
#-----------------------
#visualization functions
#------------------------  

# ── state colours used by make_state_graph ────────────────────────────────────
_STATE_COLORS = {
    "ACTIVE":     "#2ecc71",   # green
    "NON_CODING": "#f1c40f",   # yellow
    "INACTIVE":   "#e74c3c",   # red
    "GONE":       "#95a5a6",   # grey
    "UNKNOWN":    "#bdc3c7",   # light grey
}

def _shade_states(ax, df, date_col, state_col, alpha=0.3):
    """Shade axis background by state, grouping consecutive equal-state rows."""
    if df.empty:
        return
    dates  = df[date_col].values
    states = df[state_col].values
    i = 0
    while i < len(states):
        j = i + 1
        while j < len(states) and states[j] == states[i]:
            j += 1
        color = _STATE_COLORS.get(str(states[i]), "#7f8c8d")
        end   = dates[j] if j < len(dates) else dates[-1] + np.timedelta64(2, "D")
        ax.axvspan(dates[i], end, color=color, alpha=alpha, zorder=1, linewidth=0)
        i = j

def make_state_graph_2(df, key_suffix=""):
    import matplotlib.patches as mpatches

    if df is None or df.empty:
        st.info("No data to display yet — run the pipeline first.")
        return

    # detect probability columns if they exist
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    has_probs = len(prob_cols) > 0

    devs = sorted(df["dev"].dropna().unique())
    rand_dev = random.choice(devs)
    dev  = st.selectbox("Select developer", devs, key = f"dev_box{rand_dev}")

    wdf = df[df["dev"] == dev].copy()
    wdf["date"]  = pd.to_datetime(wdf["date"])
    wdf = wdf.sort_values("date").reset_index(drop=True)
    wdf["state"] = wdf["state"].fillna("UNKNOWN")

    for col in ("commits", "prs", "issues", "issue_activity", "pr_activity"):
        wdf[col] = pd.to_numeric(wdf[col], errors="coerce").fillna(0)

    
    wdf["coding_total"]    = wdf["commits"] + wdf["prs"]
    wdf["noncoding_total"] = wdf["issues"] + wdf["issue_activity"] + wdf["pr_activity"]
    wdf["commits_log"]     = np.log1p(wdf["commits"])
    wdf["prs_log"]         = np.log1p(wdf["prs"])
    wdf["issues_log"]      = np.log1p(wdf["issues"])
    wdf["issue_activity_log"] = np.log1p(wdf["issue_activity"])
    wdf["pr_activity_log"] = np.log1p(wdf["pr_activity"])
    wdf["activity_log"]    = np.log1p(wdf["coding_total"])
    wdf["non_coding_log"]  = np.log1p(wdf["noncoding_total"])

    min_date = wdf["date"].min()
    max_date = wdf["date"].max()

    window = st.selectbox(
        "Pick Window in years",
        range(1,25),
        key=f"state_graph_window{key_suffix}",
    )
    window = window *365

    default_start  = (max_date - pd.Timedelta(days=window)).to_pydatetime().date()
    slider_max     = default_start
    win_start_date = st.slider(
        "Window start  (shows N year forward from this date)",
        min_value = min_date.to_pydatetime().date(),
        max_value = slider_max,
        value     = default_start,
        step      = timedelta(days=7),
        key       = f"msg_window{key_suffix}",
    )
    win_start = pd.Timestamp(win_start_date)
    win_end   = win_start + pd.Timedelta(days=window)

    mask  = (wdf["date"] >= win_start) & (wdf["date"] <= win_end)
    wdf_w = wdf[mask].copy()
    BG = "#1c1c2e"

    # ── figure layout: 3 panels always, 4th only if prob cols exist ───────────
    n_panels     = 4 if has_probs else 3
    height_ratios = [0.4, 2, 2, 2] if has_probs else [0.4, 2, 2]

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 11 if has_probs else 9),
        facecolor=BG,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.suptitle(f"Developer: {dev}", color="white", fontsize=12)
    fig.subplots_adjust(hspace=0.6)

    # Panel 0 – full timeline strip
    ax0 = axes[0]
    ax0.set_facecolor(BG)
    _shade_states(ax0, wdf, "date", "state", alpha=1.0)
    ax0.axvspan(win_start, win_end, color="black", alpha=0.5, zorder=2)
    ax0.set_xlim(min_date, max_date)
    ax0.set_yticks([])
    ax0.set_title("Full timeline  (white band = current window)", color="white", fontsize=9, pad=4)
    ax0.tick_params(colors="white", labelsize=8)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.xaxis.set_major_locator(mdates.YearLocator())
    for sp in ax0.spines.values():
        sp.set_visible(False)

    # Panel 1 – coding activity
    ax1 = axes[1]
    ax1.set_facecolor(BG)
    _shade_states(ax1, wdf_w, "date", "state", alpha=0.25)
    active = wdf_w[wdf_w["activity_log"] > 0]
    if not active.empty:
        ax1.bar(active["date"], active["commits_log"], width=1, color="Green", alpha=0.9, zorder=3)
        ax1.bar(active["date"], active["prs_log"],     width=1, color="Yellow", alpha=0.9, zorder=3)
    ax1.set_xlim(win_start, win_end)
    ax1.set_ylabel("log(commits + PRs)", color="white", fontsize=9)
    ax1.set_title("Coding Activity", color="white", fontsize=10, pad=4)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.legend(["commits_log", "prs_log"], loc="upper left", facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    for sp in ax1.spines.values():
        sp.set_color("#444")

    # Panel 2 – non-coding activity
    ax2 = axes[2]
    ax2.set_facecolor(BG)
    _shade_states(ax2, wdf_w, "date", "state", alpha=0.25)
    nc = wdf_w[wdf_w["non_coding_log"] > 0]
    #wdf["issues_log"]      = np.log1p(wdf["issues"])
    #wdf["issue_activity_log"] = np.log1p(wdf["issue_activity"])
    #wdf["pr_activity_log"] = np.log1p(wdf["pr_activity"])
    if not nc.empty:
        ax2.bar(nc["date"], nc["issues_log"], width=1, color="red", alpha=0.9, zorder=3)
        ax2.bar(nc["date"], nc["issue_activity_log"], width=1, color="red", alpha=0.9, zorder=3)
        ax2.bar(nc["date"], nc["pr_activity_log"], width=1, color="yellow", alpha=0.9, zorder=3)

    ax2.set_xlim(win_start, win_end)
    ax2.set_ylabel("log(issues + comments)", color="white", fontsize=9)
    ax2.set_title("Non-Coding Activity", color="white", fontsize=10, pad=4)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.legend(["issues_log", "issue_activity_log", "pr_activity_log"], loc="upper left", facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    for sp in ax2.spines.values():
        sp.set_color("#444")

    # Panel 3 – predicted state probabilities (only if prob cols exist) ────────
    if has_probs:
        ax3 = axes[3]
        ax3.set_facecolor(BG)

        # filter to window rows that actually have predictions
        prob_w = wdf_w.dropna(subset=prob_cols).sort_values("date")
        prob_w.to_csv("THIS_FILE.csv")

        #we have break_starts_in_14d and prob_0
        # prob_0 is the models yhat this is the reponces
        # the break_starts_in_14d in the true y value
        # we need to plot both of these

        # Detect binary (survival) vs multi-class (shifted_state) predictions
        is_binary = sorted(prob_cols) == ["prob_0", "prob_1"]

        if not prob_w.empty:
            if is_binary:
                # Binary mode: two line plots — P(not active) and P(active)
                ax3.plot(prob_w["date"], prob_w["prob_1"],
                        color="#e67e22", linewidth=1.5,
                        label="P(break starts in 14d)")
                ax3.plot(prob_w["date"], prob_w["break_starts_in_14d"].astype(float),
                        color="#2ecc71", linewidth=1.5,
                        label="True break in 14d")
                #ax3.plot(prob_w["date"], prob_w["prob_1"],
                #         color="#2ecc71", linewidth=1.5,
                #         label="P(active in 14d)")

                ax3.axhline(0.5, color="white", linestyle="--",
                            linewidth=0.8, alpha=0.5)
                ax3.legend(loc="upper left", facecolor=BG, labelcolor="white",
                           fontsize=8, framealpha=0)
            else:
                # Multi-class mode: stacked area shows all 4 state probabilities.
                # The areas sum to 1 at every point, giving an intuitive read of
                # where the model thinks the developer will be in N days.
                palette_mc   = [_STATE_COLORS.get(cl.replace("prob_", ""), "#888888")
                                for cl in prob_cols]
                dates        = prob_w["date"].values
                prob_matrix  = prob_w[prob_cols].values
                class_labels = [c.replace("prob_", "") for c in prob_cols]
                ax3.stackplot(dates, prob_matrix.T, labels=class_labels,
                              colors=palette_mc, alpha=0.35)

                # Overlay inactivity risk as a white dashed line — the single number
                # a project manager cares about: P(INACTIVE or GONE) in N days.
                _risk_col_p3 = next(
                    (c for c in ("inactivity_risk", "prob_INACTIVE") if c in prob_w.columns),
                    None
                )
                if _risk_col_p3:
                    _risk_lbl = ("Inactivity Risk  P(INACTIVE + GONE)"
                                 if _risk_col_p3 == "inactivity_risk"
                                 else "P(INACTIVE)")
                    ax3.plot(dates, prob_w[_risk_col_p3].values,
                             color="white", linewidth=1.8, linestyle="--",
                             alpha=0.85, label=_risk_lbl, zorder=5)

                ax3.legend(loc="upper left", facecolor=BG, labelcolor="white",
                           fontsize=8, framealpha=0)
        else:
            ax3.text(0.5, 0.5, "No predictions in this window",
                     ha="center", va="center", color="white",
                     transform=ax3.transAxes, fontsize=9)

        ax3.set_xlim(win_start, win_end)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Probability", color="white", fontsize=9)
        _title = ("Transition Probability  (LSTM — binary survival)"
                  if is_binary else "Predicted State Probabilities  (LSTM)")
        ax3.set_title(_title, color="white", fontsize=10, pad=4)
        ax3.tick_params(colors="white", labelsize=8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
        for sp in ax3.spines.values():
            sp.set_color("#444")

    # legend for state colours
    patches = [mpatches.Patch(color=c, label=s) for s, c in _STATE_COLORS.items()]
    fig.legend(handles=patches, loc="upper right", ncol=5,
               facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)

    st.pyplot(fig)
    plt.close(fig)
def view_df(df, name="DataFrame", max_rows=10_000):
    """
    HTML table viewer with CSV download.
    - CSV is written to disk (not embedded) — avoids holding it in RAM
    - Only renders `max_rows` rows in the browser table
    - Full dataset is always available via the download link
    """
    import tempfile, webbrowser, os, pathlib

    # ── Write CSV to a persistent temp file (not deleted on close) ──────────
    csv_file = tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".csv", encoding="utf-8",
        prefix=f"{name.replace(' ', '_')}_"
    )
    df.to_csv(csv_file, index=False)   # streams directly to disk, no in-memory string
    csv_file.close()
    csv_path = pathlib.Path(csv_file.name).as_uri()   # file:// URI for the <a> tag

    # ── Build preview table (truncated) ─────────────────────────────────────
    total_rows = len(df)
    preview = df.iloc[:max_rows]
    truncated = total_rows > max_rows
    caption = (
        f"Showing {max_rows:,} of {total_rows:,} rows — download for full dataset"
        if truncated else
        f"{total_rows:,} rows"
    )

    safe_name = name.replace('"', '').replace("'", "")

    # ── Build HTML pieces separately so we're not holding giant strings ──────
    style = """
<style>
  body { font-family: system-ui, 'Segoe UI', Arial; padding: 16px; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #ddd; padding: 6px; font-size: 13px; }
  th { position: sticky; top: 0; background: #fafafa; }
  .meta { color: #555; font-size: 13px; margin-bottom: 8px; }
  #dl-btn {
    display: inline-flex; align-items: center; gap: 6px;
    margin-bottom: 12px; padding: 7px 14px;
    background: #2563eb; color: #fff; border: none;
    border-radius: 6px; font-size: 14px; cursor: pointer;
    text-decoration: none;
  }
  #dl-btn:hover { background: #1d4ed8; }
</style>
"""

    # to_html on the preview slice only — much smaller string
    table_html = preview.to_html(index=False, escape=False)

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".html", encoding="utf-8"
    ) as f:
        f.write("<meta charset='utf-8'>\n")
        f.write(style)
        f.write(f"<h3>{safe_name}</h3>\n")
        f.write(f"<p class='meta'>{caption}</p>\n")
        # Download links directly to the CSV file — no data embedded in JS
        f.write(f'<a id="dl-btn" href="{csv_path}" download="{safe_name}.csv">&#8681; Download CSV</a>\n')
        f.write(table_html)

    webbrowser.open("file://" + f.name)

import pandas as pd
import tempfile
import webbrowser
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def view_na_graph(df, name="DataFrame"):
    """
    Calculates missing values, generates a Matplotlib graph, 
    embeds it in an HTML file, and opens it in the browser.
    """
    # 1. Calculate missing value counts and percentages
    na_counts = df.isna().sum()
    na_percent = (na_counts / len(df)) * 100

    # 2. Build a summary DataFrame and sort by most missing data
    na_summary = pd.DataFrame({
        'Column': na_percent.index,
        'Missing Percentage': na_percent.values,
        'Missing Count': na_counts.values
    }).sort_values(by='Missing Percentage', ascending=False)

    # 3. Create a Matplotlib Bar Chart
    # Make the figure wide enough to handle lots of columns
    plt.figure(figsize=(14, 7)) 
    plt.bar(na_summary['Column'], na_summary['Missing Percentage'], color='crimson')
    
    # Formatting
    plt.xticks(rotation=90, ha='right', fontsize=8) # Rotate labels so they don't overlap
    plt.ylabel('Percentage Missing (%)')
    plt.title(f"Missing Values Overview: {name} (Total Rows: {len(df):,})")
    plt.tight_layout() # Ensures labels aren't cut off

    # 4. Save the plot to an in-memory buffer (so we don't have to save a random image file)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close() # Close the plot to free memory
    buf.seek(0)
    
    # 5. Convert image to base64 string so we can embed it in HTML
    b64_img = base64.b64encode(buf.read()).decode('utf-8')

    # 6. Build the HTML page
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Missing Values - {name}</title>
        <style>
            body {{ font-family: system-ui, Arial, sans-serif; background: #f9fafb; padding: 20px; text-align: center; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: inline-block; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <img src="data:image/png;base64,{b64_img}" alt="NA Graph">
        </div>
    </body>
    </html>
    """

    # 7. Save to a persistent temporary HTML file and open it
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        f.write(html_content)
        html_path = f.name

    webbrowser.open("file://" + html_path)


def save_model_artifacts(model_folder, model, label_encoder, test_df, pred_cols_full, encoded_col):
    model_folder = Path(model_folder)
    os.makedirs(model_folder, exist_ok=True)

    # model weights
    torch.save(model.state_dict(), model_folder / "model_weights.pth")

    # save model config so we can rebuild the architecture on load
    torch.save({
        "input_size":  model.lstm.input_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers":  model.lstm.num_layers,
        "output_size": model.fc.out_features,
    }, model_folder / "model_config.pth")

    # label encoder
    joblib.dump(label_encoder, model_folder / "label_encoder.pkl")

    # test dataframe
    test_df.to_csv(model_folder / "test_df.csv", index=False)

    # predictor cols + encoded col (plain text/json)
    with open(model_folder / "pred_cols_full.json", "w") as f:
        json.dump(pred_cols_full, f)

    with open(model_folder / "encoded_col.txt", "w") as f:
        f.write(encoded_col)

    print(f"Model artifacts saved to: {model_folder}")

def load_model_artifacts(model_folder):
    model_folder = Path(model_folder)

    # rebuild model architecture from saved config
    config = torch.load(model_folder / "model_config.pth")
    model  = DeveloperLSTM(
        input_size  = config["input_size"],
        hidden_size = config["hidden_size"],
        num_layers  = config["num_layers"],
        output_size = config["output_size"],
    )
    model.load_state_dict(torch.load(model_folder / "model_weights.pth"))
    model.eval()

    # label encoder
    label_encoder = joblib.load(model_folder / "label_encoder.pkl")

    # test dataframe
    test_df = pd.read_csv(model_folder / "test_df.csv")
    test_df["date"] = pd.to_datetime(test_df["date"])

    # predictor cols + encoded col
    with open(model_folder / "pred_cols_full.json", "r") as f:
        pred_cols_full = json.load(f)

    with open(model_folder / "encoded_col.txt", "r") as f:
        encoded_col = f.read().strip()

    print(f"Model artifacts loaded from: {model_folder}")
    return model, label_encoder, test_df, pred_cols_full, encoded_col
def apply_increasing_noise(df, column_name, noise_scale=0.01):
    # FORCE the column to be numeric (this fixes the "1" vs 1 issue)
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    col_series = df[column_name].squeeze()
    
    # 1. Identify streaks
    group_id = (col_series != col_series.shift()).cumsum()
    
    # 2. Count consecutive occurrences
    streak_count = df.groupby(group_id).cumcount()
    
    # We use == 1.0 now to be safe with floats/ints
    df['streak'] = np.where(col_series == 1, streak_count, 0)
    
    # 3. Positive-Only Noise
    raw_noise = np.abs(np.random.normal(0, noise_scale, len(df)))
    
    # 4. Calculate Adjusted Value
    df['adjusted_value'] = np.where(
        col_series == 1,
        1.0 + (df['streak'] * raw_noise),
        0.0
    )
    
    # --- DEBUG PRINT ---
    active_rows = df[df[column_name] == 1]
    print(f"DEBUG: Found {len(active_rows)} rows matching '1'")
    if not active_rows.empty:
        print(active_rows[[column_name, 'streak', 'adjusted_value']].head(10))

    return df

def compute_enriched_predictors(
    enriched: pandas.DataFrame,
    commits_raw: pandas.DataFrame,
    perfile_commits: pandas.DataFrame,
    org_activity: pandas.DataFrame,
    repo_name: str,) -> pandas.DataFrame:
    """
    Appends ~28 new causal predictor columns to enriched.
    All features are backward-looking only (no label leakage).
    Groups: lagged activity, break history, code churn, static dev attrs,
            cross-repo activity, cyclic time features.
    """
    print("n\n\n\n\n\n THE FUCTION IS WORKING AND PRINTING \n\n\n\n\n\n\n\n")
    df = enriched.sort_values(["dev", "date"]).copy()
    df["date"] = pandas.to_datetime(df["date"]).dt.normalize()

    # ── Group A: Lagged activity ──────────────────────────────────────────────
    def _rmean(col, w):
        return df.groupby("dev")[col].transform(lambda s: s.rolling(w, min_periods=1).mean())

    def _rstd(col, w):
        return df.groupby("dev")[col].transform(
            lambda s: s.rolling(w, min_periods=1).std().fillna(0)
        )

    if "commits" in df.columns:
        df["commits_7d_mean"]        = _rmean("commits", 7)
        df["commits_30d_mean"]       = _rmean("commits", 30)
        df["commits_90d_mean"]       = _rmean("commits", 90)
        df["commits_7d_std"]         = _rstd("commits", 7)
        df["commits_30d_std"]        = _rstd("commits", 30)
        df["total_commits_lifetime"] = df.groupby("dev")["commits"].transform("cumsum")
    else:
        for _c in ["commits_7d_mean", "commits_30d_mean", "commits_90d_mean",
                   "commits_7d_std", "commits_30d_std", "total_commits_lifetime"]:
            df[_c] = 0

    df["coding_days_30d"] = (
        df.groupby("dev")["coding_day"].transform(lambda s: s.rolling(30, min_periods=1).sum())
        if "coding_day" in df.columns else 0
    )
    df["pr_activity_30d_mean"] = (
        _rmean("pr_activity", 30) if "pr_activity" in df.columns else 0
    )
    df["issue_activity_30d_mean"] = (
        _rmean("issue_activity", 30) if "issue_activity" in df.columns else 0
    )

    # ── Group B: Break history (causal — from state_causal_enc transitions) ───
    if "state_causal_enc" in df.columns:
        _enc           = df["state_causal_enc"].fillna(0).astype(int)
        _inactive      = _enc.isin({2, 3})
        _prev_inactive = df.groupby("dev")["state_causal_enc"].shift(1).fillna(0).astype(int).isin({2, 3})

        # onset: first day of an inactive/gone period
        df["_is_onset"] = (_inactive & ~_prev_inactive).astype(int)

        # days_since_last_break via causal forward-fill
        df["_onset_dt"]   = df["date"].where(df["_is_onset"].astype(bool), other=pandas.NaT)
        df["_last_onset"] = df.groupby("dev")["_onset_dt"].transform(lambda s: s.ffill())
        df["days_since_last_break"] = (
            (df["date"] - df["_last_onset"]).dt.days.fillna(0).clip(lower=0)
        )

        # length_of_last_break_days: length (days) of the most recently completed break
        df["_break_run_id"] = df.groupby("dev")["_is_onset"].transform("cumsum")
        _blk = (
            df.loc[_inactive]
              .groupby(["dev", "_break_run_id"])["date"]
              .transform("count")
              .reindex(df.index, fill_value=0)
              .astype(float)
        )
        _next_inactive = (
            df.groupby("dev")["state_causal_enc"].shift(-1)
              .fillna(-1).astype(int).isin({2, 3})
        )
        _break_end = _inactive & ~_next_inactive
        df["_break_end_len"] = _blk.where(_break_end & _inactive)
        df["length_of_last_break_days"] = (
            df.groupby("dev")["_break_end_len"]
              .transform(lambda s: s.ffill())
              .fillna(0)
        )

        df["n_breaks_past_90d"]  = df.groupby("dev")["_is_onset"].transform(
            lambda s: s.rolling(90,  min_periods=1).sum()
        )
        df["n_breaks_past_365d"] = df.groupby("dev")["_is_onset"].transform(
            lambda s: s.rolling(365, min_periods=1).sum()
        )
        df.drop(columns=["_is_onset", "_onset_dt", "_last_onset",
                          "_break_run_id", "_break_end_len"], errors="ignore", inplace=True)
    else:
        for _c in ["days_since_last_break", "length_of_last_break_days",
                   "n_breaks_past_90d", "n_breaks_past_365d"]:
            df[_c] = 0

    # ── Detect which commits_raw column maps to enriched['dev'] ──────────────
    # The tf_devs list stores entries like "author_name|David ..." or plain author_id
    # values. Read tf_devs.csv for this repo and inspect the prefix to determine the
    # right column — this is the canonical method used throughout the project.
    _dev_col = None
    _tf_devs_path = ORG_BASE / repo_name / "Results" / "tf_devs.csv"
    if _tf_devs_path.exists():
        try:
            _tf_raw = pandas.read_csv(_tf_devs_path).iloc[:, 0].dropna().astype(str).tolist()
            for _entry in _tf_raw:
                _entry = _entry.strip()
                if _entry.startswith("author_name|"):
                    _dev_col = "author_name"; break
                elif _entry.startswith("author_login|"):
                    _dev_col = "author_login"; break
                elif _entry.startswith("author_email|"):
                    _dev_col = "author_email"; break
            if _dev_col is None:
                _dev_col = "author_id"  # plain IDs, no prefix
        except Exception:
            _dev_col = None  # fall through to sample-based fallback below

    # Sample-based fallback (only used if tf_devs.csv is missing or unreadable)
    if _dev_col is None:
        _dev_sample = set(df["dev"].dropna().unique()[:100])
        for _cand in ("author_name", "author_login", "author_id", "author_email"):
            if _cand not in commits_raw.columns:
                continue
            if len(_dev_sample & set(commits_raw[_cand].dropna().unique())) > 0:
                _dev_col = _cand
                break

    # Build author_login → dev mapping so org_activity (login-only) can join
    _login_to_dev: dict = {}
    if _dev_col and _dev_col != "author_login" and "author_login" in commits_raw.columns:
        _login_to_dev = (
            commits_raw[["author_login", _dev_col]]
            .dropna(subset=["author_login", _dev_col])
            .drop_duplicates(subset=["author_login"])
            .set_index("author_login")[_dev_col]
            .to_dict()
        )

    # ── Group C: Code churn ───────────────────────────────────────────────────
    _has_churn = (
        not commits_raw.empty
        and _dev_col is not None
        and "additions_sum" in commits_raw.columns
        and "deletions_sum"  in commits_raw.columns
    )
    if _has_churn:
        _cr = commits_raw.copy()
        _cr["additions_sum"] = pandas.to_numeric(_cr["additions_sum"], errors="coerce").fillna(0)
        _cr["deletions_sum"] = pandas.to_numeric(_cr["deletions_sum"], errors="coerce").fillna(0)
        _cr["date"] = (
            pandas.to_datetime(_cr["created_at"], utc=True, errors="coerce")
            .dt.tz_localize(None).dt.normalize()
        )
        _cr = (
            _cr.groupby([_dev_col, "date"])
               .agg(lines_added=("additions_sum", "sum"),
                    lines_deleted=("deletions_sum", "sum"))
               .reset_index()
               .rename(columns={_dev_col: "dev"})
        )
        df = df.merge(_cr, on=["dev", "date"], how="left")
        df["lines_added_today"]   = pandas.to_numeric(df.pop("lines_added"),   errors="coerce").fillna(0)
        df["lines_deleted_today"] = pandas.to_numeric(df.pop("lines_deleted"), errors="coerce").fillna(0)
        df["churn_today"]         = df["lines_added_today"] + df["lines_deleted_today"]
        df["churn_ratio_today"]   = df["churn_today"] / (df["churn_today"] + 1)
        df["churn_30d_mean"]      = df.groupby("dev")["churn_today"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        )
    else:
        for _c in ["lines_added_today", "lines_deleted_today",
                   "churn_today", "churn_ratio_today", "churn_30d_mean"]:
            df[_c] = 0

    # ── Group D: Static developer features ───────────────────────────────────
    df["tenure_days"] = (
        df["date"] - df.groupby("dev")["date"].transform("min")
    ).dt.days.fillna(0).astype(int)

    # ── Group E: Cross-repo activity ──────────────────────────────────────────
    if (not org_activity.empty
            and "author_login" in org_activity.columns
            and "created_at"   in org_activity.columns):
        _oa = org_activity.copy()
        _oa["date"] = (
            pandas.to_datetime(_oa["created_at"], utc=True, errors="coerce")
            .dt.tz_localize(None).dt.normalize()
        )
        # Map author_login → dev so the join matches enriched['dev'] format
        if _login_to_dev:
            _oa["dev"] = _oa["author_login"].map(_login_to_dev)
            _oa = _oa.dropna(subset=["dev"])
        else:
            _oa = _oa.rename(columns={"author_login": "dev"})

        _org_total = (
            _oa.groupby(["dev", "date"]).size()
               .reset_index(name="org_commits_today")
        )
        _elsewhere = (
            _oa[_oa["repo"] != repo_name]
            if "repo" in _oa.columns else pandas.DataFrame()
        )
        if not _elsewhere.empty:
            _el_counts = (
                _elsewhere.groupby(["dev", "date"]).size()
                          .reset_index(name="_elsewhere")
            )
        else:
            _el_counts = pandas.DataFrame(columns=["dev", "date", "_elsewhere"])

        df = df.merge(_org_total, on=["dev", "date"], how="left")
        df = df.merge(_el_counts,  on=["dev", "date"], how="left")
        df["org_commits_today"]          = df["org_commits_today"].fillna(0)
        df["_elsewhere"]                 = df["_elsewhere"].fillna(0)
        _commits_col = df["commits"] if "commits" in df.columns else pandas.Series(0, index=df.index)
        df["this_repo_share_today"]      = _commits_col / df["org_commits_today"].clip(lower=1)
        df["org_active_elsewhere_today"] = (df["_elsewhere"] > 0).astype(int)
        df["org_active_elsewhere_7d"]    = df.groupby("dev")["org_active_elsewhere_today"].transform(
            lambda s: s.rolling(7, min_periods=1).sum()
        )
        df.drop(columns=["_elsewhere"], errors="ignore", inplace=True)
    else:
        for _c in ["org_commits_today", "this_repo_share_today",
                   "org_active_elsewhere_today", "org_active_elsewhere_7d"]:
            df[_c] = 0

    # ── Group F: Cyclic time features ─────────────────────────────────────────
    _dt = pandas.to_datetime(df["date"])
    df["dow_sin"]     = np.sin(2 * np.pi * _dt.dt.dayofweek / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * _dt.dt.dayofweek / 7)
    df["month_sin"]   = np.sin(2 * np.pi * _dt.dt.month / 12)
    df["month_cos"]   = np.cos(2 * np.pi * _dt.dt.month / 12)
    df["day_of_year"] = _dt.dt.dayofyear



    # ── Ensure all new columns exist with no NaNs ─────────────────────────────
    _new_cols = [
        "commits_7d_mean", "commits_30d_mean", "commits_90d_mean",
        "commits_7d_std", "commits_30d_std",
        "coding_days_30d", "pr_activity_30d_mean", "issue_activity_30d_mean",
        "total_commits_lifetime",
        "days_since_last_break", "length_of_last_break_days",
        "n_breaks_past_90d", "n_breaks_past_365d",
        "lines_added_today", "lines_deleted_today",
        "churn_today", "churn_ratio_today", "churn_30d_mean",
        "tenure_days",
        "org_commits_today", "this_repo_share_today",
        "org_active_elsewhere_today", "org_active_elsewhere_7d",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "day_of_year",
    ]
    for _c in _new_cols:
        if _c in df.columns:
            df[_c] = df[_c].fillna(0)

    view_na_graph(df, name="Final DataFrame")
    return df

def make_dev_acuracy_graph(df, prob_threshold=0.3, shift_days=14):
    """
    Per-developer accuracy analysis.

    Section 1 — Metrics table (sorted by TDR descending):
      • inactivity_windows : number of contiguous INACTIVE periods
      • detected_count     : periods where prob_1 >= threshold in the early-warning window
      • tdr_%              : detected_count / inactivity_windows × 100
      • total_flags        : total days where prob_1 >= threshold (false-alarm proxy)

    Section 2 — One probability timeline per developer (Panel 3 style),
    shown inside collapsible expanders ranked by TDR.
    """
    import matplotlib.patches as mpatches

    if df is None or df.empty:
        st.info("No data to display yet — run the pipeline first.")
        return

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        st.warning("No probability columns found — run evaluate_model first.")
        return

    is_binary = sorted(prob_cols) == ["prob_0", "prob_1"]
    BG = "#1c1c2e"

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["dev", "date"]).reset_index(drop=True)

    # ── Section 1: per-developer metrics ─────────────────────────────────────
    rows = []
    for dev, group in df.groupby("dev"):
        group  = group.sort_values("date").reset_index(drop=True)
        states = group["state"].fillna("UNKNOWN").values

        # For TDR we specifically want P(INACTIVE) — the class we're measuring detection of.
        # inactivity_risk = P(INACTIVE) + P(GONE) is also good (broader disengagement signal).
        if "prob_INACTIVE" in group.columns:
            probs = group["prob_INACTIVE"].astype(float).values
        elif "inactivity_risk" in group.columns:
            probs = group["inactivity_risk"].astype(float).values
        elif is_binary and "prob_1" in group.columns:
            probs = group["prob_1"].astype(float).values
        else:
            probs = group[prob_cols].max(axis=1).astype(float).values

        total_periods    = 0
        detected_periods = 0
        i = 0
        while i < len(states):
            if states[i] != "INACTIVE":
                i += 1
                continue
            j = i
            while j < len(states) and states[j] == "INACTIVE":
                j += 1
            win_len      = min(shift_days, j - i)
            window_probs = probs[i : i + win_len]
            all_nan      = np.all(np.isnan(window_probs))
            detected     = (not all_nan) and (np.nanmax(window_probs) >= prob_threshold)
            if detected:
                detected_periods += 1
            total_periods += 1
            i = j

        tdr         = detected_periods / total_periods if total_periods > 0 else float("nan")
        total_flags = int(np.nansum(probs >= prob_threshold))

        rows.append({
            "developer":          dev,
            "inactivity_windows": total_periods,
            "detected_count":     detected_periods,
            "tdr_%":              round(tdr * 100, 1) if not np.isnan(tdr) else float("nan"),
            "total_flags":        total_flags,
        })

    metrics_df = (
        pd.DataFrame(rows)
        .sort_values("tdr_%", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Per-Developer Detection Accuracy")
    st.caption(f"Threshold = {prob_threshold}  |  Early-warning window = {shift_days} days")
    st.dataframe(
        metrics_df.style.format({"tdr_%": "{:.1f}%"})
                        .background_gradient(subset=["tdr_%"], cmap="RdYlGn"),
        use_container_width=True,
    )

    # ── Section 2: per-developer probability timelines ────────────────────────
    st.subheader("Per-Developer Prediction Timelines")

    ranked_devs = metrics_df["developer"].tolist()
    for dev in ranked_devs:
        tdr_val = metrics_df.loc[metrics_df["developer"] == dev, "tdr_%"].values[0]
        tdr_str = f"{tdr_val:.1f}%" if not np.isnan(tdr_val) else "n/a"

        with st.expander(f"{dev}  —  TDR: {tdr_str}"):
            wdf    = df[df["dev"] == dev].copy().sort_values("date").reset_index(drop=True)
            prob_w = wdf.dropna(subset=prob_cols).sort_values("date")

            fig, ax = plt.subplots(figsize=(14, 3), facecolor=BG)
            ax.set_facecolor(BG)

            if not prob_w.empty:
                if is_binary:
                    ax.plot(prob_w["date"], prob_w["prob_1"],
                            color="#e67e22", linewidth=1.5,
                            label="P(break starts in 14d)")
                    if "break_starts_in_14d" in prob_w.columns:
                        ax.plot(prob_w["date"],
                                prob_w["break_starts_in_14d"].astype(float),
                                color="#2ecc71", linewidth=1.5,
                                label="True break in 14d")
                    ax.axhline(prob_threshold, color="white", linestyle="--",
                               linewidth=0.8, alpha=0.5,
                               label=f"threshold = {prob_threshold}")
                    ax.legend(loc="upper left", facecolor=BG, labelcolor="white",
                              fontsize=8, framealpha=0)
                else:
                    # Stacked area shows all 4 state probabilities summing to 1 —
                    # colour matches the state legend so it reads intuitively.
                    palette_mc   = [_STATE_COLORS.get(cl.replace("prob_", ""), "#888888")
                                    for cl in prob_cols]
                    class_labels = [c.replace("prob_", "") for c in prob_cols]
                    ax.stackplot(prob_w["date"].values, prob_w[prob_cols].values.T,
                                 labels=class_labels, colors=palette_mc, alpha=0.55)

                    # Overlay inactivity risk as a bold white dashed line — this is the
                    # single number a project manager cares about: P(INACTIVE or GONE in N days).
                    _risk_col_plot = next(
                        (c for c in ("inactivity_risk", "prob_INACTIVE") if c in prob_w.columns),
                        None
                    )
                    if _risk_col_plot:
                        _risk_label = ("Inactivity Risk  P(INACTIVE + GONE)"
                                       if _risk_col_plot == "inactivity_risk"
                                       else "P(INACTIVE in Nd)")
                        ax.plot(prob_w["date"], prob_w[_risk_col_plot],
                                color="white", linewidth=1.8, linestyle="--",
                                alpha=0.85, label=_risk_label, zorder=5)
                        ax.axhline(prob_threshold, color="#aaaaaa", linestyle=":",
                                   linewidth=0.8, alpha=0.6,
                                   label=f"threshold = {prob_threshold}")

                    ax.legend(loc="upper left", facecolor=BG, labelcolor="white",
                              fontsize=8, framealpha=0)
            else:
                ax.text(0.5, 0.5, "No predictions available",
                        ha="center", va="center", color="white",
                        transform=ax.transAxes, fontsize=9)

            ax.set_ylim(0, 1)
            ax.set_xlim(wdf["date"].min(), wdf["date"].max())
            _title = ("Transition Probability  (LSTM — binary onset)"
                      if is_binary else "Predicted State Probabilities N days ahead  (LSTM — 4-class)")
            ax.set_title(f"{dev}  —  {_title}", color="white", fontsize=10, pad=4)
            ax.set_ylabel("Probability", color="white", fontsize=9)
            ax.tick_params(colors="white", labelsize=8)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            for sp in ax.spines.values():
                sp.set_color("#444")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

def proceed_gate(prompt: str = "Proceed? (y/n): ") -> None:
    """Pause execution and wait for user confirmation. Exits on anything other than 'y'."""
    response = input(prompt).strip().lower()
    if response != "y":
        print("Aborted.")
        raise SystemExit(0)
#-----------------------
# main streamlit app
#------------------------

def main():
    
    st.set_page_config(page_title="Dev Inactivity Demo", layout="wide")
    #user input
    # we need to ask the user for a few things
    # we need the test train split (leave one dev or repo out)
    # we need to know if its dev mode
    # we need to be able to add anything to this in the future
    # we are using streamlit for this

    if 'list_of_repos' not in st.session_state:
        # Only pre-populate with training repos that have fully complete data collections.
        train_pairs = cfg.load_repo_split("train")
        st.session_state.list_of_repos = [
            f"{org}/{repo}" for org, repo in train_pairs
            if _is_repo_ready(org, repo)
        ]

    st.title("Developer Inactivity Prediction")
    st.write("Configure the settings for predicting developer inactivity.")
    st.write("Please provide the necessary inputs below:")

    repo_url = st.text_input("Enter GitHub Repository URL (e.g., https://github.com/user/repo) or (org/repo):")
    # if "add to queue" button is pressed
    if st.button("Add to Queue"):
        if repo_url.strip():  # Only add if input is not empty
            gitRepoName = repo_url.replace('https://github.com/', '').strip()
            # Add to session state list instead of local variable
            if gitRepoName not in st.session_state.list_of_repos:
                st.session_state.list_of_repos.append(gitRepoName)
                st.success(f"Added '{gitRepoName}' to queue!")
            else:
                st.warning(f"'{gitRepoName}' is already in the queue.")
        else:
            st.error("Please enter a repository URL or name.")

    if st.button("Clear Queue"):
        st.session_state.list_of_repos = []
        st.success("Queue cleared!")
    
    if st.button("⚠ Force Add All (skip collection check)", type="secondary"):
        # Adds ALL repos from repo_split.csv regardless of collection status.
        # Use when data collection is still running but you want to queue for processing.
        force_added, already_in = [], []
        for org in list_orgs():
            for repo in list_repos_for(org):
                repo_key = f"{org}/{repo}"
                if repo_key not in st.session_state.list_of_repos:
                    st.session_state.list_of_repos.append(repo_key)
                    force_added.append(repo_key)
                else:
                    already_in.append(repo_key)
        if force_added:
            st.success(f"Force-added {len(force_added)} repos: {', '.join(force_added)}")
        if already_in:
            st.info(f"{len(already_in)} repos already in queue.")
        if not force_added and not already_in:
            st.warning("No repos found in repo_split.csv.")


    if st.button("Add all to Queue"):
        # Add all repos (any split) in repo_split.csv that have fully complete data.
        added, skipped_collecting = [], []
        for org in list_orgs():
            for repo in list_repos_for(org):
                repo_key = f"{org}/{repo}"
                if not _is_repo_ready(org, repo):
                    skipped_collecting.append(repo_key)
                    continue
                if repo_key not in st.session_state.list_of_repos:
                    st.session_state.list_of_repos.append(repo_key)
                    added.append(repo_key)
        if added:
            st.success(f"Added {len(added)} repos: {', '.join(added)}")
        if skipped_collecting:
            st.info(f"Skipped {len(skipped_collecting)} still collecting: {', '.join(skipped_collecting)}")

    col_train, col_test = st.columns(2)

    with col_train:
        if st.button("Add Training Repos"):
            added, skipped_collecting = [], []
            for org, repo in cfg.load_repo_split("train"):
                repo_key = f"{org}/{repo}"
                if not _is_repo_ready(org, repo):
                    skipped_collecting.append(repo_key)
                    continue
                if repo_key not in st.session_state.list_of_repos:
                    st.session_state.list_of_repos.append(repo_key)
                    added.append(repo_key)
            if added:
                st.success(f"Added {len(added)} training repos: {', '.join(added)}")
            if skipped_collecting:
                st.info(f"Skipped {len(skipped_collecting)} still collecting: {', '.join(skipped_collecting)}")

    with col_test:
        if st.button("Add Test Repos"):
            added, skipped_collecting = [], []
            for org, repo in cfg.load_repo_split("test"):
                repo_key = f"{org}/{repo}"
                if not _is_repo_ready(org, repo):
                    skipped_collecting.append(repo_key)
                    continue
                if repo_key not in st.session_state.list_of_repos:
                    st.session_state.list_of_repos.append(repo_key)
                    added.append(repo_key)
            if added:
                st.success(f"Added {len(added)} test repos: {', '.join(added)}")
            if skipped_collecting:
                st.info(f"Skipped {len(skipped_collecting)} still collecting: {', '.join(skipped_collecting)}")

    st.caption(f"Selected repos: **{', '.join(st.session_state.list_of_repos)}**")

    # ── Training data readiness panel ────────────────────────────────────────
    # Shows the gap between repo_split.csv and what has actually been processed.
    # Training uses ALL repos on disk (not the queue), so this tells the user
    # exactly what's missing from the model's training set.
    with st.expander("Training data status", expanded=False):
        _all_train = cfg.load_repo_split("train")
        _all_test  = cfg.load_repo_split("test")

        _processed_train, _unprocessed_train, _uncollected_train = [], [], []
        for _o, _r in _all_train:
            _tl = ORG_BASE / _o / _r / "Results" / "all_users_labeled_timeline.csv"
            if _tl.exists():
                _processed_train.append(f"{_o}/{_r}")
            elif _is_repo_ready(_o, _r):
                _unprocessed_train.append(f"{_o}/{_r}")   # collected but not yet processed
            else:
                _uncollected_train.append(f"{_o}/{_r}")   # not even fully collected yet

        _processed_test, _unprocessed_test = [], []
        for _o, _r in _all_test:
            _tl = ORG_BASE / _o / _r / "Results" / "all_users_labeled_timeline.csv"
            if _tl.exists():
                _processed_test.append(f"{_o}/{_r}")
            else:
                _unprocessed_test.append(f"{_o}/{_r}")

        st.markdown(f"**Train repos** ({len(_processed_train)}/{len(_all_train)} processed)")
        if _processed_train:
            st.success("Processed (will be used by model): " + ", ".join(_processed_train))
        if _unprocessed_train:
            st.warning(
                f"{len(_unprocessed_train)} collected but not yet processed — add to queue and run Predictors: "
                + ", ".join(_unprocessed_train)
            )
            if st.button("Add unprocessed train repos to queue", key="_btn_add_unprocessed"):
                _added = []
                for _key in _unprocessed_train:
                    if _key not in st.session_state.list_of_repos:
                        st.session_state.list_of_repos.append(_key)
                        _added.append(_key)
                if _added:
                    st.success(f"Added: {', '.join(_added)}")
        if _uncollected_train:
            st.error(
                f"{len(_uncollected_train)} not yet collected (run extractor first): "
                + ", ".join(_uncollected_train)
            )

        st.markdown(f"**Test repos** ({len(_processed_test)}/{len(_all_test)} processed)")
        if _processed_test:
            st.success("Processed: " + ", ".join(_processed_test))
        if _unprocessed_test:
            st.warning("Not yet processed: " + ", ".join(_unprocessed_test))

        st.caption(
            "ℹ The Predictors step processes repos in the queue above. "
            "The model training step reads ALL processed repos from disk — not the queue. "
            "Run Predictors for any 'collected but not yet processed' repos above to include them in the next training run."
        )

    st.divider()
    st.subheader("Processing Pipeline")

    # ── Master overwrite toggle ──────────────────────────────────────────────
    # When OFF: every step loads from its cached output file if it exists.
    # When ON:  per-step levers appear so you can selectively re-run only what you need.
    master_overwrite = st.toggle(
        "Enable per-step overwrite controls",
        value=False,
        key="master_overwrite",
        help="Reveal individual levers to choose which pipeline steps to re-run.",
    )

    # ── Per-step overwrite levers (only shown when master is ON) ────────────
    # Defaults: all False → use cached files wherever possible
    ow_kd       = False   # KnowledgeDistribution / Truck Factor
    ow_timeline = False   # Basic timeline
    ow_label    = False   # Label developer activity (also regenerates state_causal_enc)
    ow_devnames = False   # Dev name lookup
    ow_stn      = False   # Social-Technical Network  ← the slow one
    ow_ph       = False   # Project Health
    ow_enriched = False   # Predictors merge step (final enriched timeline)

    if master_overwrite:
        st.caption(
            "Toggle a step ON to force it to re-run and overwrite its cached file. "
            "Steps left OFF will load from disk if the output already exists."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            ow_kd       = st.toggle("↻ Truck Factor / Knowledge Distribution", value=False, key="ow_kd",
                                    help="Recomputes doe.csv and truck_factor.json")
            ow_timeline = st.toggle("↻ Basic Timeline",                         value=False, key="ow_timeline",
                                    help="Recomputes user_timeline.csv")
            ow_label    = st.toggle("↻ Label Developer Activity",               value=False, key="ow_label",
                                    help="Recomputes all_users_labeled_timeline.csv (also regenerates state / state_causal_enc)")
        with col_b:
            ow_devnames = st.toggle("↻ Dev Name Lookup",                        value=False, key="ow_devnames",
                                    help="Recomputes dev_names.csv")
            ow_stn      = st.toggle("↻ Social-Technical Network",               value=False, key="ow_stn",
                                    help="Recomputes edge_list, metrics, and daily interaction CSVs (slow!)")
            ow_ph       = st.toggle("↻ Project Health",                         value=False, key="ow_ph",
                                    help="Recomputes project_health.json")
            ow_enriched = st.toggle("↻ Predictors / Enriched Timeline",         value=False, key="ow_enriched",
                                    help="Re-merges all predictor features onto the labeled timeline and overwrites all_users_labeled_timeline.csv")

    if st.button("Responce"):

        number_of_repos = len(st.session_state.list_of_repos)

        prediction_df = pd.DataFrame()


        for repo in st.session_state.list_of_repos:

            print(f"Processing repository: {repo}")
            
            #main
            main_folder = ORG_BASE / repo 
            os.makedirs(main_folder, exist_ok=True)
            # collection data
            collection_folder = main_folder / cfg.collection_folder
            os.makedirs(collection_folder, exist_ok=True)
            # TF_developers_folder
            TF_developers_folder = Path(main_folder, cfg.TF_developers_folder)
            os.makedirs(TF_developers_folder, exist_ok=True)
            # TIMELINE FOLDER
            timeline_folder = main_folder / cfg.timeline_folder
            os.makedirs(timeline_folder, exist_ok=True)
            # LABELED TIMELINE FOLDER
            labeled_timeline_folder = main_folder / cfg.labeled_timeline_folder
            os.makedirs(labeled_timeline_folder, exist_ok=True)

            #----------------------
            # Step 1: Load Data
            #---------------------- 
            print("\n\nStep 1: Loading raw data")
            raw_data_tables = load_users_activity(repo_full_name=repo)


            #----------------------
            # Step 2: Truck Factor
            #----------------------
            print("\n\nStep 2: Calculating Truck Factor")
            tf, tf_devs, author_map, DOE = kd.main(repo_full_name=repo, tables=raw_data_tables, overwrite=ow_kd)

            if not tf_devs:
                st.warning(f"No per-file commit data found for **{repo}** — skipping.")
                continue

            #----------------------
            # Basic Timeline
            #----------------------
            print("\n\nStep 3: Generating Basic Timeline")
            out = timeline(raw_data_tables, tf_devs, repo_full_name=repo, overwrite=ow_timeline)

            #----------------------
            # Label Timeline
            #----------------------
            print("\n\nStep 4: Labeling Developer Activity Timeline")
            user_labeled_timeline = label_developers_activity(repo=repo, tf_devs=tf_devs, over_write=ow_label)

            print("\n\nStep 4.5: Building developer name lookup")
            dev_names_path = ORG_BASE / repo / "Results" / "dev_names.csv"
            if not dev_names_path.exists() or ow_devnames:
                build_dev_names(repo_full_name=repo)
            else:
                st.info(f"[{repo}] dev_names: using cached file")

            print(user_labeled_timeline['dev'].unique())
            if prediction_df is not None and not prediction_df.empty:
                print(prediction_df["dev"].unique())
                print(f"{user_labeled_timeline['dev'].nunique()} + {prediction_df['dev'].nunique()} = {prediction_df['dev'].nunique() + user_labeled_timeline['dev'].nunique()}")
            #I want to combine all data set together long way
            prediction_df = pd.concat([prediction_df, user_labeled_timeline], ignore_index=True)
        #----------------------
        # Analyse Responce
        #----------------------
        # Persist across Streamlit reruns triggered by widget interaction
        st.session_state["prediction_df"] = prediction_df
        st.success(f"Pipeline complete — {len(prediction_df)} rows across {prediction_df['dev'].nunique()} developers.")
        # Collect comparison HTML paths for the inspection UI
        _comp_files = []
        for _repo in st.session_state.list_of_repos:
            _org2, _proj2 = _repo.split("/")
            _comp_dir = ORG_BASE / _org2 / _proj2 / "Results" / "state_comparisons"
            if _comp_dir.exists():
                _comp_files += sorted(_comp_dir.glob("*_state_comparison.html"))
        st.session_state["_state_comp_files"] = [str(p) for p in _comp_files]
        st.session_state["_state_comp_idx"]   = 0

    # ── State comparison inspection — outside button so it survives reruns ────
    if "_state_comp_files" in st.session_state and st.session_state["_state_comp_files"]:
        st.markdown("---")
        st.subheader("State Comparison Inspector")
        _files = st.session_state["_state_comp_files"]
        _idx   = st.session_state.get("_state_comp_idx", 0)
        _idx   = max(0, min(_idx, len(_files) - 1))

        _col_prev, _col_info, _col_next = st.columns([1, 4, 1])
        with _col_prev:
            if st.button("◀ Prev dev", key="comp_prev") and _idx > 0:
                st.session_state["_state_comp_idx"] = _idx - 1
                st.rerun()
        with _col_info:
            _dev_name = Path(_files[_idx]).stem.replace("_state_comparison", "")
            st.write(f"**{_dev_name}** — {_idx + 1} / {len(_files)}")
        with _col_next:
            if st.button("Next dev ▶", key="comp_next") and _idx < len(_files) - 1:
                st.session_state["_state_comp_idx"] = _idx + 1
                st.rerun()

        with open(_files[_idx], "r", encoding="utf-8") as _f:
            st.components.v1.html(_f.read(), height=650, scrolling=False)

        st.caption("Green = ACTIVE · Yellow = NON_CODING · Red = INACTIVE · Grey = GONE  "
                   "| Top strip = state (original, bidirectional)  "
                   "| Middle strip = state_causal (backward-only)")

    # ── interactive graph — lives OUTSIDE the button so it survives reruns ────
    if "prediction_df" in st.session_state:
        st.subheader("Developer Activity Explorer")
        make_state_graph_2(st.session_state["prediction_df"], key_suffix="_activity")


    if st.button("Predictors"):

        predictor_frames = []

        for repo in st.session_state.list_of_repos:

            print(f"\nProcessing repository: {repo}")

            # ── Cache check: skip entire enrichment if output already looks complete ──
            _enriched_path = ORG_BASE / repo / "Results" / "all_users_labeled_timeline.csv"
            _sentinel_cols = {"state_causal_enc", "repo_commits_7d", "files_worked_today",
                              "issue_interactions_today"}
            if _enriched_path.exists() and not ow_enriched:
                _existing_cols = set(pandas.read_csv(_enriched_path, nrows=0).columns.tolist())
                if _sentinel_cols.issubset(_existing_cols):
                    st.info(f"[{repo}] Enriched timeline: loading from cache (toggle ↻ Predictors / Enriched Timeline to recompute)")
                    predictor_frames.append(pandas.read_csv(_enriched_path))
                    continue

            # ── folder setup ────────────────────────────────────────────────
            main_folder = ORG_BASE / repo
            os.makedirs(main_folder, exist_ok=True)
            timeline_folder = main_folder / cfg.timeline_folder
            os.makedirs(timeline_folder, exist_ok=True)
            stn_folder = main_folder / cfg.social_network_metrics_folder
            os.makedirs(stn_folder, exist_ok=True)

            # ── Step 1: Load raw data ────────────────────────────────────────
            print("\n\nStep 1: Loading raw data")
            raw_data_tables = load_users_activity(repo_full_name=repo)

            with st.expander(f"Audit A — Raw data tables [{repo}]", expanded=False):
                _author_cols = {"author_id", "author_login", "author_name", "author_email"}
                for _key, _tbl in raw_data_tables.items():
                    if _tbl is None or _tbl.empty:
                        st.error(f"**{_key}**: EMPTY")
                        continue
                    _date_col = next((c for c in ("created_at", "committed_at") if c in _tbl.columns), None)
                    _date_range = (f"{pandas.to_datetime(_tbl[_date_col]).min().date()} → "
                                   f"{pandas.to_datetime(_tbl[_date_col]).max().date()}") if _date_col else "no date col"
                    st.write(f"**{_key}**: {len(_tbl):,} rows | cols: {list(_tbl.columns)} | {_date_range}")
                _pfc = raw_data_tables.get("perfile_commits", pandas.DataFrame())
                _missing_auth = _author_cols - set(_pfc.columns) if not _pfc.empty else _author_cols
                if _missing_auth:
                    st.warning(f"perfile_commits missing author columns: {_missing_auth} — KD features will be empty")

            # ── Step 2: Truck Factor ─────────────────────────────────────────
            print("\n\nStep 2: Calculating Truck Factor")
            tf, tf_devs, author_map, DOE = kd.main(
                repo_full_name=repo, tables=raw_data_tables,
                overwrite=ow_kd,
            )

            if not tf_devs:
                st.warning(f"No per-file commit data found for **{repo}** — skipping.")
                continue

            # ── Step 3: Basic Timeline ───────────────────────────────────────
            print("\n\nStep 3: Generating Basic Timeline")
            basic_tl = timeline(raw_data_tables, tf_devs, repo_full_name=repo, overwrite=ow_timeline)

            # ── Step 4: Label Developer Activity ────────────────────────────
            print("\n\nStep 4: Labeling Developer Activity Timeline")
            labeled_tl = label_developers_activity(
                repo=repo, tf_devs=tf_devs, over_write=ow_label,
            )
            

            with st.expander(f"Audit B — Labeled timeline [{repo}]", expanded=False):
                _b_expected = ["state", "state_causal", "state_causal_enc", "dev", "date",
                               "commits", "prs", "issues", "coding_day", "break_day"]
                _b_rows = []
                for _c in _b_expected:
                    _present = _c in labeled_tl.columns
                    _b_rows.append({
                        "column": _c,
                        "present": "✓" if _present else "✗ MISSING",
                        "pct_null": f"{labeled_tl[_c].isna().mean()*100:.1f}%" if _present else "—",
                    })
                st.write(f"Rows: {len(labeled_tl):,} | Devs: {labeled_tl['dev'].nunique() if 'dev' in labeled_tl.columns else '?'}")
                st.dataframe(pandas.DataFrame(_b_rows))
                _missing_label_cols = [c for c in ["state_causal", "state_causal_enc"] if c not in labeled_tl.columns]
                if _missing_label_cols:
                    st.warning(f"Missing {_missing_label_cols} — enable **↻ Label Developer Activity** and re-run Predictors.")


            # ── Step 4.5: Build dev name lookup ──────────────────────────────
            print("\n\nStep 4.5: Building developer name lookup")
            dev_names_path = ORG_BASE / repo / "Results" / "dev_names.csv"
            if not dev_names_path.exists() or ow_devnames:
                build_dev_names(repo_full_name=repo)
            else:
                st.info(f"[{repo}] dev_names: using cached file")

            # ── Step 5: Social-Technical Network + daily features ────────────
            # Cache: SocialTechnicalNetwork/{metrics, edge_list, daily}.csv
            # This is the slow step — skip entirely if cache exists and lever is OFF.
            print("\n\nStep 5: Building Social-Technical Network")
            stn_folder   = ORG_BASE / repo / cfg.social_technical_metrics_folder
            metrics_path = stn_folder / cfg.social_technical_metrics_file
            edge_path    = stn_folder / cfg.social_technical_edge_list_file
            daily_path   = stn_folder / cfg.social_technical_daily_interactions_file

            stn_cache_exists = all(p.exists() for p in [metrics_path, edge_path, daily_path])

            if stn_cache_exists and not ow_stn:
                st.info(f"[{repo}] STN: loading from cache (toggle ↻ Social-Technical Network to recompute)")
                metrics_df = pandas.read_csv(metrics_path)
                edge_df    = pandas.read_csv(edge_path)
                daily_df   = pandas.read_csv(daily_path)
            else:
                st.info(f"[{repo}] STN: computing... (this may take a while)")
                edge_df, metrics_df, daily_df, _stn_sim = stn.main(
                    repo, tf_devs, tables=raw_data_tables,
                )
                os.makedirs(stn_folder, exist_ok=True)
                metrics_df.to_csv(metrics_path, index=False)
                edge_df.to_csv(edge_path, index=False)
                daily_df.to_csv(daily_path, index=False)

            with st.expander(f"Audit C — STN daily_df [{repo}]", expanded=False):
                _c_expected = [
                    "dev", "date", "issue_interactions_today", "issue_unique_partners_today",
                    "issue_new_partners_today", "issue_threads_today", "pr_interactions_today",
                    "pr_unique_partners_today", "pr_new_partners_today", "pr_threads_today",
                    "total_unique_partners_today", "mention_out_today", "mention_in_today",
                    "solo_commit_day", "all_new_partners_today", "new_to_community_today", "regulars_today",
                ]
                if daily_df.empty:
                    st.error("daily_df is EMPTY — toggle ↻ Social-Technical Network to recompute.")
                else:
                    _c_rows = [{"column": _c, "present": "✓" if _c in daily_df.columns else "✗ MISSING",
                                "pct_nonzero": f"{(daily_df[_c] != 0).mean()*100:.1f}%" if _c in daily_df.columns and pandas.api.types.is_numeric_dtype(daily_df[_c]) else "—"}
                               for _c in _c_expected]
                    st.write(f"Rows: {len(daily_df):,} | Devs: {daily_df['dev'].nunique() if 'dev' in daily_df.columns else '?'}")
                    if "date" in daily_df.columns:
                        _dd = pandas.to_datetime(daily_df["date"])
                        st.write(f"Date range: {_dd.min().date()} → {_dd.max().date()}")
                    st.dataframe(pandas.DataFrame(_c_rows))
                    _c_missing = [r["column"] for r in _c_rows if r["present"] == "✗ MISSING"]
                    if _c_missing:
                        st.warning(f"Missing columns: {_c_missing} — STN cache may be stale. Toggle ↻ Social-Technical Network.")

            # ── Step 5b: Project Health weekly activity data ─────────────────
            # Cache: ProjectHealth/project_health.json
            print("\n\nStep 5b: Computing Project Health data")
            ph_path = ORG_BASE / repo / cfg.project_health_folder / cfg.project_health_file
            if not ph_path.exists() or ow_ph:
                _compute_project_health_data(raw_data_tables, repo, ph_path)
            else:
                st.info(f"[{repo}] Project Health: using cached file")

            # ── Step 5c: Knowledge-Distribution daily file-ownership features ─
            print("\n\nStep 5c: Computing daily file-ownership features (KD)")
            kd_daily_df = kd.compute_daily_kd_features(
                raw_data_tables.get("perfile_commits", pandas.DataFrame()),
                author_map,
            )

            with st.expander(f"Audit D — KD daily features [{repo}]", expanded=False):
                if kd_daily_df.empty:
                    st.error("kd_daily_df is EMPTY — perfile_commits may be missing or have wrong column names.")
                    _pfc2 = raw_data_tables.get("perfile_commits", pandas.DataFrame())
                    st.write(f"perfile_commits columns: {list(_pfc2.columns)}")
                else:
                    st.write(f"Rows: {len(kd_daily_df):,} | Devs: {kd_daily_df['dev'].nunique()}")
                    for _col in ["files_worked_today", "owned_files_today", "collab_files_today", "collab_commit_ratio"]:
                        if _col in kd_daily_df.columns:
                            st.write(f"  {_col}: non-zero={int((kd_daily_df[_col] != 0).sum()):,}  mean={kd_daily_df[_col].mean():.3f}")
                        else:
                            st.warning(f"  {_col}: MISSING")

            # ── Step 5d: Project-Health daily rolling repo totals ─────────────
            print("\n\nStep 5d: Computing daily project-health rolling totals")
            ph_daily_df = compute_daily_ph_features(raw_data_tables)

            with st.expander(f"Audit E — Project Health daily features [{repo}]", expanded=False):
                if ph_daily_df.empty:
                    st.error("ph_daily_df is EMPTY — check that commits have a 'created_at' column.")
                else:
                    st.write(f"Rows: {len(ph_daily_df):,}")
                    if "date" in ph_daily_df.columns:
                        _phd = pandas.to_datetime(ph_daily_df["date"])
                        st.write(f"Date range: {_phd.min().date()} → {_phd.max().date()}")
                    for _col in ["repo_commits_7d", "repo_prs_7d", "repo_issues_7d", "repo_active_devs_7d"]:
                        if _col in ph_daily_df.columns:
                            _nz = int((ph_daily_df[_col] != 0).sum())
                            st.write(f"  {_col}: non-zero={_nz:,}  mean={ph_daily_df[_col].mean():.1f}")
                            if _nz == 0:
                                st.warning(f"  {_col} is all zeros — date column mismatch likely.")
                        else:
                            st.warning(f"  {_col}: MISSING")

            
            # ── Step 5e: Load org-wide activity for cross-repo features ──────────
            _org_name = repo.split("/")[0]
            org_activity_path = ORG_BASE / _org_name / "_org_activity.csv"
            if org_activity_path.exists():
                org_activity_df = pandas.read_csv(org_activity_path)
            else:
                org_activity_df = pandas.DataFrame()
                st.warning(f"[{repo}] No _org_activity.csv found at {org_activity_path} — cross-repo features will be zero.")


            # ── Step 6: Merge all daily features onto labeled timeline ─────────
            print("\n\nStep 6: Merging all daily features onto labeled timeline")
            labeled_tl["date"] = pandas.to_datetime(labeled_tl["date"]).dt.normalize()
            enriched = labeled_tl.copy()

            # Drop any predictor columns that may already be on the loaded timeline
            # (from a prior Predictors run) to prevent _x/_y duplicate columns on merge.
            _predictor_cols_to_drop = [
                "Unnamed: 0",  # stray index column from CSVs saved without index=False
                "issue_interactions_today", "issue_unique_partners_today",
                "issue_new_partners_today", "issue_threads_today",
                "pr_interactions_today", "pr_unique_partners_today",
                "pr_new_partners_today", "pr_threads_today",
                "total_unique_partners_today", "mention_out_today", "mention_in_today",
                "solo_commit_day", "all_new_partners_today", "new_to_community_today",
                "regulars_today", "files_worked_today", "owned_files_today",
                "collab_files_today", "collab_commit_ratio",
                "repo_commits_7d", "repo_prs_7d", "repo_issues_7d", "repo_active_devs_7d",
                # ── expanded predictors ───────────────────────────────────────
                "commits_7d_mean", "commits_30d_mean", "commits_90d_mean",
                "commits_7d_std", "commits_30d_std",
                "coding_days_30d", "pr_activity_30d_mean", "issue_activity_30d_mean",
                "total_commits_lifetime",
                "days_since_last_break", "length_of_last_break_days",
                "n_breaks_past_90d", "n_breaks_past_365d",
                "lines_added_today", "lines_deleted_today",
                "churn_today", "churn_ratio_today", "churn_30d_mean",
                "tenure_days",
                "org_commits_today", "this_repo_share_today",
                "org_active_elsewhere_today", "org_active_elsewhere_7d",
                "dow_sin", "dow_cos", "month_sin", "month_cos", "day_of_year",
            ]
            enriched = enriched.drop(columns=[c for c in _predictor_cols_to_drop if c in enriched.columns])

            # 6a — STN social interaction features
            if not daily_df.empty:
                daily_df = daily_df.drop(columns=[c for c in daily_df.columns if c.startswith("Unnamed:")], errors="ignore")
                daily_df["date"] = pandas.to_datetime(daily_df["date"]).dt.normalize()
                enriched = pandas.merge(enriched, daily_df, on=["dev", "date"], how="left")
            stn_cols = [
                "issue_interactions_today", "issue_unique_partners_today",
                "issue_new_partners_today", "issue_threads_today",
                "pr_interactions_today", "pr_unique_partners_today",
                "pr_new_partners_today", "pr_threads_today",
                "total_unique_partners_today",
                "mention_out_today", "mention_in_today", "solo_commit_day",
                "all_new_partners_today", "new_to_community_today", "regulars_today",
            ]
            for col in stn_cols:
                if col in enriched.columns:
                    enriched[col] = enriched[col].fillna(0)

            # 6b — KD file-ownership features
            if not kd_daily_df.empty:
                kd_daily_df["date"] = pandas.to_datetime(kd_daily_df["date"]).dt.normalize()
                enriched = pandas.merge(enriched, kd_daily_df, on=["dev", "date"], how="left")
            for col in ["files_worked_today", "owned_files_today", "collab_files_today", "collab_commit_ratio"]:
                if col in enriched.columns:
                    enriched[col] = enriched[col].fillna(0)

            # 6c — Project-health rolling repo totals (date-only join; same value for all devs)
            if not ph_daily_df.empty:
                ph_daily_df["date"] = pandas.to_datetime(ph_daily_df["date"]).dt.normalize()
                enriched = pandas.merge(enriched, ph_daily_df, on=["date"], how="left")
            for col in ["repo_commits_7d", "repo_prs_7d", "repo_issues_7d", "repo_active_devs_7d"]:
                if col in enriched.columns:
                    enriched[col] = enriched[col].fillna(0)

            # ── Step 6d: Expanded predictor set ──────────────────────────────
            print("\n\nStep 6d: Computing expanded predictor set")
            enriched = compute_enriched_predictors(
                enriched,
                raw_data_tables.get("commits", pandas.DataFrame()),
                raw_data_tables.get("perfile_commits", pandas.DataFrame()),
                org_activity_df,
                repo,
            )

            
            with st.expander(f"Audit F2 — Expanded predictor distributions [{repo}]", expanded=False):
                _new_col_groups = {
                    "Lagged activity": [
                        "commits_7d_mean", "commits_30d_mean", "commits_90d_mean",
                        "commits_7d_std", "commits_30d_std", "coding_days_30d",
                        "pr_activity_30d_mean", "issue_activity_30d_mean", "total_commits_lifetime",
                    ],
                    "Break history": [
                        "days_since_last_break", "length_of_last_break_days",
                        "n_breaks_past_90d", "n_breaks_past_365d",
                    ],
                    "Code churn": [
                        "lines_added_today", "lines_deleted_today",
                        "churn_today", "churn_ratio_today", "churn_30d_mean",
                    ],
                    "Static / Tenure": ["tenure_days"],
                    "Cross-repo": [
                        "org_commits_today", "this_repo_share_today",
                        "org_active_elsewhere_today", "org_active_elsewhere_7d",
                    ],
                    "Time (cyclic)": ["dow_sin", "dow_cos", "month_sin", "month_cos", "day_of_year"],
                }
                for _grp, _cols in _new_col_groups.items():
                    st.markdown(f"**{_grp}**")
                    _rows = []
                    for _col in _cols:
                        if _col in enriched.columns:
                            _s = enriched[_col]
                            _nn  = f"{(1 - _s.isna().mean())*100:.1f}%"
                            _nz  = f"{(_s != 0).mean()*100:.1f}%" if pandas.api.types.is_numeric_dtype(_s) else "—"
                            _mn  = f"{_s.mean():.3f}" if pandas.api.types.is_numeric_dtype(_s) else "—"
                            _mx  = f"{_s.max():.1f}"  if pandas.api.types.is_numeric_dtype(_s) else "—"
                            _rows.append({"column": _col, "non-null %": _nn, "non-zero %": _nz, "mean": _mn, "max": _mx})
                        else:
                            _rows.append({"column": _col, "non-null %": "MISSING", "non-zero %": "—", "mean": "—", "max": "—"})
                    st.dataframe(pandas.DataFrame(_rows))
                    _all_zero = [r["column"] for r in _rows if r.get("non-zero %") == "0.0%"]
                    if _all_zero:
                        st.error(f"All-zero columns (check data linkage): {_all_zero}")
                    _missing = [r["column"] for r in _rows if r.get("non-null %") == "MISSING"]
                    if _missing:
                        st.warning(f"Missing columns (not computed): {_missing}")



            with st.expander(f"Audit F — Enriched timeline after merge [{repo}]", expanded=False):
                _f_all_expected = [
                    "commits", "prs", "issues", "state", "state_causal", "state_causal_enc",
                    "issue_interactions_today", "pr_interactions_today", "total_unique_partners_today",
                    "all_new_partners_today", "new_to_community_today", "regulars_today",
                    "files_worked_today", "owned_files_today", "collab_commit_ratio",
                    "repo_commits_7d", "repo_prs_7d", "repo_issues_7d", "repo_active_devs_7d",
                    # expanded predictors
                    "commits_7d_mean", "commits_30d_mean", "commits_90d_mean",
                    "tenure_days", "days_since_last_break", "n_breaks_past_90d",
                    "churn_today", "churn_30d_mean",
                    "org_active_elsewhere_today", "org_active_elsewhere_7d",
                    "dow_sin", "dow_cos", "day_of_year",
                ]
                st.write(f"Rows: {len(enriched):,} | Devs: {enriched['dev'].nunique()} | Columns: {len(enriched.columns)}")
                _f_rows = []
                for _c in _f_all_expected:
                    _present = _c in enriched.columns
                    _pct_null = f"{enriched[_c].isna().mean()*100:.1f}%" if _present else "—"
                    _pct_zero = f"{(enriched[_c] == 0).mean()*100:.1f}%" if _present and pandas.api.types.is_numeric_dtype(enriched[_c]) else "—"
                    _f_rows.append({"column": _c, "present": "✓" if _present else "✗ MISSING",
                                    "% null": _pct_null, "% zero": _pct_zero})
                st.dataframe(pandas.DataFrame(_f_rows))
                _f_missing = [r["column"] for r in _f_rows if r["present"] == "✗ MISSING"]
                if _f_missing:
                    st.error(f"These expected columns did not merge in: {_f_missing}")

            # ── Save enriched timeline to disk (overwrites basic labeled version) ─
            results_path = ORG_BASE / repo / "Results" / "all_users_labeled_timeline.csv"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            enriched.to_csv(results_path, index=False)

            predictor_frames.append(enriched)
            print(f"  {repo}: {len(enriched)} rows, {enriched['dev'].nunique()} devs")

            # ── Step 7: Write distrac/ parquet outputs (Stage 1) ─────────────
            print("\n\nStep 7: Writing distrac/ pipeline outputs (Stage 1)")
            _ph_data = None
            _ph_path = ORG_BASE / repo / cfg.project_health_folder / cfg.project_health_file
            if _ph_path.exists():
                with open(_ph_path) as _f:
                    _ph_data = json.load(_f)
            _dev_names_local = build_dev_names(repo_full_name=repo)
            dw.write_distrac_stage1(
                repo_full_name=repo,
                dev_names_df=_dev_names_local,
                tf=tf,
                tf_devs=tf_devs,
                doe_df=DOE,
                metrics_df=metrics_df,
                edge_df=edge_df,
                ph_data=_ph_data,
                commit_list_df=raw_data_tables.get("commits"),
            )

        if predictor_frames:
            predictor_df = pandas.concat(predictor_frames, ignore_index=True)

            st.session_state["predictor_df"] = predictor_df
            st.success(
                f"Predictors complete — {len(predictor_df)} rows across "
                f"{predictor_df['dev'].nunique()} developers with "
                f"{len(predictor_df.columns)} columns."
            )


    # ── Collect Developer Profiles ───────────────────────────────────────────
    st.divider()
    st.subheader("Developer Profiles")
    st.caption(
        "Downloads GitHub profile data (name, bio, avatar, contribution calendar) "
        "for every developer found in each repo's commit history. "
        "Saved to Organizations/{org}/{repo}/Developers/."
    )

    ow_profiles = st.toggle(
        "Overwrite existing profiles",
        value=False,
        key="ow_profiles",
        help="Re-download profiles that have already been collected.",
    )

    if st.button("Collect Developer Profiles"):
        from collect_developer_profiles import collect_developer_profiles
        _tokens_path = PROJECT_ROOT / "Resources" / "tokens.csv"
        try:
            token = pandas.read_csv(_tokens_path)["token"].iloc[0]
        except Exception as _te:
            st.error(f"Could not load GitHub token from {_tokens_path}: {_te}")
            st.stop()
        total_collected, total_skipped, total_failed = 0, 0, 0
        for repo in st.session_state.list_of_repos:
            with st.spinner(f"Collecting profiles for {repo}…"):
                result = collect_developer_profiles(
                    repo_full_name=repo,
                    token=token,
                    overwrite=ow_profiles,
                )
            total_collected += len(result["collected"])
            total_skipped   += len(result["skipped"])
            total_failed    += len(result["failed"])
            st.info(
                f"**{repo}** — "
                f"collected: {len(result['collected'])}  |  "
                f"skipped (cached): {len(result['skipped'])}  |  "
                f"failed: {len(result['failed'])}"
            )
            if result["failed"]:
                st.warning(f"Failed logins: {', '.join(result['failed'])}")
        st.success(
            f"Done — {total_collected} new profiles collected, "
            f"{total_skipped} skipped, {total_failed} failed."
        )

    st.divider()
    st.subheader("Step 6: Predict Inactivity")

    if "overwrite_prediction_model" not in st.session_state:
        st.session_state.overwrite_prediction_model = False
    st.session_state.overwrite_prediction_model = st.toggle(
        "Overwrite cached model (retrain from scratch)",
        value=st.session_state.overwrite_prediction_model,
        help="If enabled, the saved model weights are ignored and a new model is trained.",
        key=3,
    )

    # ── Forecast configuration — set OUTSIDE the button so settings persist ──
    _cfg_col1, _cfg_col2 = st.columns([2, 1])
    with _cfg_col1:
        _forecast_horizon = st.select_slider(
            "Forecast horizon (days ahead to predict)",
            options=[7, 14, 21, 30],
            value=st.session_state.get("forecast_horizon", 14),
            help=(
                "At each day D the model predicts the developer's state D + N days from now. "
                "Shorter horizons are easier to predict but give less lead time for intervention. "
                "Ablate this to find where predictive power degrades — that distance is publishable."
            ),
            key="forecast_horizon",
        )
    with _cfg_col2:
        _pred_mode = st.radio(
            "Prediction framing",
            options=["future_state", "inactivity_window", "survival_binary"],
            index=0,
            format_func=lambda m: {
                "future_state":       "4-class state (recommended)",
                "inactivity_window":  "binary at-risk window (pre-break + break)",
                "survival_binary":    "binary break-onset only (legacy)",
            }[m],
            help=(
                "**4-class state**: for each day D predict the developer's exact state at D+N "
                "(ACTIVE / NON_CODING / INACTIVE / GONE). Cleanest framing.\n\n"
                "**binary at-risk window**: label = 1 if the developer is currently INACTIVE/GONE "
                "*or* will enter INACTIVE/GONE within N days. Covers both the warning window "
                "before a break AND the break itself — most actionable for a project manager.\n\n"
                "**binary break-onset only**: legacy — predict only the N-day window before a break starts."
            ),
            key="prediction_mode",
        )

    if st.button("Run Inactivity Prediction"):
        timer()

        shift_days = _forecast_horizon
        pred_mode  = _pred_mode

        # ── A. Load TRAIN repos labeled timelines from disk ───────────────────
        train_pairs = cfg.load_repo_split("train")
        train_frames = []
        for _org, _repo in train_pairs:
            tl_path = ORG_BASE / _org / _repo / "Results" / "all_users_labeled_timeline.csv"
            if tl_path.exists():
                _df = pandas.read_csv(tl_path)
                _df["_repo"] = f"{_org}/{_repo}"
                train_frames.append(_df)
            else:
                st.warning(f"No labeled timeline for train repo {_org}/{_repo} — skipping.")

        # ── Load TEST repos labeled timelines from disk ───────────────────────
        test_pairs = cfg.load_repo_split("test")
        test_frames = []
        for _org, _repo in test_pairs:
            tl_path = ORG_BASE / _org / _repo / "Results" / "all_users_labeled_timeline.csv"
            if tl_path.exists():
                _df = pandas.read_csv(tl_path)
                _df["_repo"] = f"{_org}/{_repo}"
                test_frames.append(_df)
            else:
                st.warning(f"No labeled timeline for test repo {_org}/{_repo} — skipping.")

        if not train_frames:
            st.error("No training data found. Run the Predictors pipeline for train repos first.")
            st.stop()

        train_df    = pandas.concat(train_frames, ignore_index=True)
        test_df_all = pandas.concat(test_frames,  ignore_index=True) if test_frames else pandas.DataFrame()
        st.info(f"Training on {train_df['dev'].nunique()} developers from {len(train_frames)} train repos.")
        if not test_df_all.empty:
            st.info(f"Validation / inference on {test_df_all['dev'].nunique()} developers from {len(test_frames)} test repos.")

        # ── B. Feature / predictor setup ──────────────────────────────────────
        # Groups defined at module level: _ACTIVITY_COLS, _STN_COLS, _KD_COLS, _PH_COLS
        predictor_cols = [c for c in _BASE_COLS if c in train_df.columns]
        response_col = ['state']
        tf_devs = train_df['dev'].unique().tolist()

        st.info(f"**Framing:** {pred_mode}  |  **Horizon:** {shift_days} days ahead")

        # ── C. Generate response column for BOTH sets ─────────────────────────
        train_df, response_col, predictor_cols, = genarate_responce_column(
            train_df,
            tf_devs=tf_devs,
            response_col=response_col,
            predictor_cols=predictor_cols,
            window_size=90,
            mode=pred_mode,
            shift_days=shift_days,
        )
        if not test_df_all.empty:
            test_df_all, _, _ = genarate_responce_column(
                test_df_all,
                tf_devs=test_df_all['dev'].unique().tolist(),
                response_col=['state'],
                predictor_cols=predictor_cols,
                window_size=90,
                mode=pred_mode,
                shift_days=shift_days,
            )

        print(f"Active response column for testing: {response_col}")

        train_df_repocne = train_df["break_starts_in_14d"]
        
        df_1 = pd.DataFrame(train_df_repocne)
        result_1 = apply_increasing_noise(df_1, "break_starts_in_14d", noise_scale=0.05)

        train_df["response_col_noise"] = train_df_repocne


        if 1==0:

            return true

        # ── C2. Developer activity tier (replaces one-hot encoding) ───────────
        # Bucket each dev by their median daily commits into 4 tiers (0=low → 3=high).
        # Fit on train only; map onto test (unknown devs get tier 0).
        _dev_median = train_df.groupby("dev")["commits"].median()
        try:
            _tier_bins = pandas.qcut(_dev_median, q=4, labels=[0, 1, 2, 3], duplicates="drop")
        except ValueError:
            _tier_bins = pandas.cut(_dev_median, bins=4, labels=[0, 1, 2, 3])
        _tier_map = _tier_bins.to_dict()
        train_df["dev_activity_tier"] = train_df["dev"].map(_tier_map).fillna(0).astype(int)
        if not test_df_all.empty:
            test_df_all["dev_activity_tier"] = test_df_all["dev"].map(_tier_map).fillna(0).astype(int)
        if "dev_activity_tier" not in predictor_cols:
            predictor_cols.append("dev_activity_tier")

        # Save prepared data so the validation block (outside this button) can display it
        st.session_state["_train_ready"] = {
            "train_df":       train_df,
            "test_df_all":    test_df_all,
            "predictor_cols": predictor_cols,
            "response_col":   response_col,
            "shift_days":     shift_days,
            "pred_mode":      pred_mode,
            "_base_cols":     list(_BASE_COLS),
            "test_pairs":     test_pairs,
            "tf_devs":        tf_devs,
        }
        st.info("Data prepared — review the validation section below, then press **Proceed to Training**.")

    # ── Data validation + proceed gate — outside button so it survives reruns ──
    if "_train_ready" in st.session_state:
        _ready        = st.session_state["_train_ready"]
        _vt           = _ready["train_df"]
        _vtest        = _ready["test_df_all"]
        _vbase        = _ready["_base_cols"]
        _vpc          = _ready["predictor_cols"]
        _vrc          = _ready["response_col"]
        _rc_name      = _vrc[0] if isinstance(_vrc, list) else _vrc


        # Audit G — train vs test column cross-check (this mismatch causes the crash)
        with st.expander("Audit G — Train vs Test column cross-check", expanded=True):
            st.write(f"Train: {len(_vt):,} rows | {_vt['dev'].nunique()} devs")
            if not _vtest.empty:
                st.write(f"Test:  {len(_vtest):,} rows | {_vtest['dev'].nunique()} devs")
            _g_rows = []
            for _c in _vpc:
                _in_train = _c in _vt.columns
                _in_test  = _c in _vtest.columns if not _vtest.empty else None
                _g_rows.append({
                    "predictor_col": _c,
                    "in_train": "✓" if _in_train else "✗",
                    "in_test":  "✓" if _in_test else ("✗ ← WILL CRASH" if _in_test is False else "no test set"),
                    "train_pct_zero": f"{(_vt[_c] == 0).mean()*100:.1f}%" if _in_train and pandas.api.types.is_numeric_dtype(_vt[_c]) else "—",
                    "test_pct_zero":  f"{(_vtest[_c] == 0).mean()*100:.1f}%" if (_in_test and pandas.api.types.is_numeric_dtype(_vtest[_c])) else "—",
                })
            st.dataframe(pandas.DataFrame(_g_rows))
            _crash_risk = [r["predictor_col"] for r in _g_rows if "CRASH" in r["in_test"]]
            if _crash_risk:
                st.error(f"These columns are in train but missing from test — will cause KeyError: {_crash_risk}. "
                         f"Re-run Predictors with ↻ Enriched Timeline ON for all test repos.")

                # Class balance
        if _rc_name in _vt.columns:
            _bal = _vt[_rc_name].value_counts().sort_index()
            st.write("**Response column class balance (train):**", _bal.to_dict())

        # Full DataFrames (capped at 1000 rows)
        #st.caption(f"train_df: {len(_vt):,} rows total — showing first 1,000")
        #view_df(_vt.head(1000), "train_df — first 1,000 rows (before model)")
        #if not _vtest.empty:
            #st.caption(f"test_df: {len(_vtest):,} rows total — showing first 1,000")
            #view_df(_vtest.head(1000), "test_df — first 1,000 rows (before model)")

        st.markdown("---")
        if st.button("Proceed to Training", key="proceed_to_training"):
            _train_df     = _ready["train_df"]
            _test_df_all  = _ready["test_df_all"]
            _pred_cols    = _ready["predictor_cols"]
            _resp_col     = _ready["response_col"]
            _shift_days   = _ready["shift_days"]
            _pred_mode    = _ready.get("pred_mode", "future_state")
            _test_pairs   = _ready["test_pairs"]
            _tf_devs      = _ready["tf_devs"]

            st.info(f"Training: **{_pred_mode}** · horizon **{_shift_days}d** · "
                    f"{len(_pred_cols)} predictors · "
                    f"{_train_df[_resp_col[0] if isinstance(_resp_col, list) else _resp_col].nunique()} classes")

            # ── D. Train or load model ────────────────────────────────────────
            model_folder = ORG_BASE / cfg.model_folder
            os.makedirs(model_folder, exist_ok=True)
            model_exists = (model_folder / "model_weights.pth").exists()

            if model_exists and not st.session_state.overwrite_prediction_model:
                st.info(f"Loading existing model from {model_folder} ...")
                model, label_encoder, _, pred_cols_full, encoded_col = \
                    load_model_artifacts(model_folder)
                if not _test_df_all.empty:
                    encoded_test_df = _preprocess_df_for_model(
                        _test_df_all, label_encoder, _resp_col, encoded_col
                    )
                else:
                    encoded_test_df = pandas.DataFrame()
            else:
                hp_results_path    = str(ORG_BASE / "hp_search_results.csv")
                epoch_curves_path  = str(ORG_BASE / "hp_epoch_curves.csv")
                
                view_df(_train_df, name="Train DataFrame YOU NEED TO LOOK AT THIS")
                model, label_encoder, _, pred_cols_full, encoded_col, hp_results_df = \
                    run_hyperparameter_search(
                        _train_df, _tf_devs, _resp_col, _pred_cols,
                        window_size=90,
                        max_epochs_per_config=120,
                        patience=40,
                        hp_results_path=hp_results_path,
                        epoch_curves_path=epoch_curves_path,
                    )
                # HP search trains on train-only data; preprocess the real test set for inference
                encoded_test_df = _preprocess_df_for_model(
                    _test_df_all, label_encoder, _resp_col, encoded_col
                ) if not _test_df_all.empty else pandas.DataFrame()
                st.session_state["hp_results_df"] = hp_results_df
                st.session_state["epoch_curves_path"] = epoch_curves_path
                save_model_artifacts(model_folder, model, label_encoder,
                                     encoded_test_df, pred_cols_full, encoded_col)
                st.success(f"Model trained and saved to {model_folder}")

            # ── E. Run inference per TEST repo ────────────────────────────────
            all_predictions = []
            for _org, _repo in _test_pairs:
                repo_key   = f"{_org}/{_repo}"
                repo_slice = encoded_test_df[encoded_test_df["_repo"] == repo_key] \
                             if "_repo" in encoded_test_df.columns else pandas.DataFrame()
                if repo_slice.empty:
                    st.warning(f"No encoded data for {repo_key} — skipping inference.")
                    continue

                predictions = evaluate_model(
                    repo_slice, model, label_encoder,
                    predictor_cols=pred_cols_full,
                    encoded_col=encoded_col,
                    window_size=90,
                    shift_days=_shift_days,
                )

                per_repo_model_folder = ORG_BASE / _org / _repo / cfg.model_folder
                os.makedirs(per_repo_model_folder, exist_ok=True)
                predictions.to_csv(per_repo_model_folder / "test_df.csv", index=False)
                st.success(f"Saved predictions for {repo_key}  ({predictions['dev'].nunique()} devs)")
                all_predictions.append(predictions)

                dw.write_distrac_stage2(
                    repo_full_name=repo_key,
                    predictions_df=predictions,
                )

            if all_predictions:
                st.session_state["final_df"] = pandas.concat(all_predictions, ignore_index=True)

            del st.session_state["_train_ready"]   # clear so button reappears clean

    # ── HP search results — survives reruns, also shows if curves file exists ──
    _curves_path = st.session_state.get(
        "epoch_curves_path",
        str(ORG_BASE / "hp_epoch_curves.csv"),
    )
    if Path(_curves_path).exists():
        with st.expander("HP Search — Training Curves", expanded=False):
            if "hp_results_df" in st.session_state:
                st.dataframe(
                    st.session_state["hp_results_df"]
                    .sort_values("best_test_f1_macro", ascending=False)
                    .reset_index(drop=True)
                )
            plot_hp_epoch_curves(_curves_path)

    # ── interactive graph — lives OUTSIDE the button so it survives reruns ────
    if "final_df" in st.session_state:
        st.subheader("Developer Activity Explorer (Test Set + Predictions)")

        # final_df has StandardScaler-scaled activity columns (commits, prs, etc.).
        # Replace them with the raw counts from prediction_df so the graph shows
        # correct non-negative values.
        _raw_activity_cols = ["commits", "prs", "issues", "issue_activity", "pr_activity"]
        _final = st.session_state["final_df"].copy()
        _final["date"] = pd.to_datetime(_final["date"])

        if "prediction_df" in st.session_state:
            _clean = st.session_state["prediction_df"][["dev", "date"] + _raw_activity_cols].copy()
            _clean["date"] = pd.to_datetime(_clean["date"])
            _final = _final.drop(columns=[c for c in _raw_activity_cols if c in _final.columns])
            _final = _final.merge(_clean, on=["dev", "date"], how="left")
        make_state_graph_2(_final, key_suffix="_inactivity")

        make_dev_acuracy_graph(_final)
        

    if st.button("Run Break Length Prediction"):
        timer()

        breaks_df = build_break_level_dataset(st.session_state.list_of_repos)
        st.write(f"Loaded {len(breaks_df)} break events across {breaks_df['dev'].nunique()} developers")
        model, scaler, results_df, feature_cols = break_predictions_pipeline(
            breaks_df,
            window_size=5,
            epochs=200,
            hidden_size=128,
            num_layers=2,
            lr=0.001,
            patience=100,
        )

        st.success(
            f"MAE: {(results_df['actual_next_break_len'] - results_df['predicted_next_break_len']).abs().mean():.1f} days"
        )

        view_df( results_df, name = "results_df" )

if __name__ == "__main__":
    if "--eval-suite" in sys.argv:
        import warnings
        warnings.filterwarnings("ignore")

        print("[EvalSuite] Loading labeled timelines ...")
        train_pairs  = cfg.load_repo_split("train")
        train_frames = []
        for _org, _repo in train_pairs:
            tl = ORG_BASE / _org / _repo / "Results" / "all_users_labeled_timeline.csv"
            if tl.exists():
                _df = pandas.read_csv(tl)
                _df["_repo"] = f"{_org}/{_repo}"
                train_frames.append(_df)
            else:
                print(f"[EvalSuite] WARNING: missing train timeline {_org}/{_repo}")

        test_pairs  = cfg.load_repo_split("test")
        test_frames = []
        for _org, _repo in test_pairs:
            tl = ORG_BASE / _org / _repo / "Results" / "all_users_labeled_timeline.csv"
            if tl.exists():
                _df = pandas.read_csv(tl)
                _df["_repo"] = f"{_org}/{_repo}"
                test_frames.append(_df)
            else:
                print(f"[EvalSuite] WARNING: missing test timeline {_org}/{_repo}")

        if not train_frames:
            print("[EvalSuite] ERROR: No training data found. Aborting.")
            sys.exit(1)

        _train_df = pandas.concat(train_frames, ignore_index=True)
        _test_df  = pandas.concat(test_frames,  ignore_index=True) if test_frames else pandas.DataFrame()

        if _test_df.empty:
            print("[EvalSuite] No test repos — will run Dev-CV only (no LOPO, no horizon/ablation tables).")

        _pcols = [c for c in _BASE_COLS if c in _train_df.columns]
        print(f"[EvalSuite] train devs={_train_df['dev'].nunique()}  "
              f"test devs={_test_df['dev'].nunique() if not _test_df.empty else 0}  "
              f"feature cols={len(_pcols)}")

        run_evaluation_suite(
            _train_df, _test_df,
            predictor_cols=_pcols,
            lopo=not _test_df.empty,
        )
    else:
        main()

