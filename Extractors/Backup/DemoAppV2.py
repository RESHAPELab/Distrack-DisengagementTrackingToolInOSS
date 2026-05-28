#   conda activate osslab
#   streamlit run DemoAppV2.py


#   cd C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Extractors

from asyncio import Event
import json
from operator import index
from msilib import Table
from turtle import pd
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
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from dataclasses import dataclass
from github import Github, GithubException, UnknownObjectException, IncompletableObject

from ProjectHealthAnaysis import build_repo_health as project_health_main
from CommitExtractor import main as extract_repo_main

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, classification_report, confusion_matrix

sys.path.append('../')
import Settings as cfg
import Utilities as util
import KnowledgeDistribution
import SocialTechnicalNetwork as std

ORG_BASE = Path(r"C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY") / "Organizations"


#---------------------
# Data collection
#---------------------

#def get_data(devs, repo_full_name):
#    st.write(f"New data for {repo_full_name}...")
#    # this fuction need to find the file paths and then load all of the data
#    # we will put it into a list of tables and return it
#
#    project_root = Path("../")
#    orgs_dir = project_root / "Organizations"
#    organization, project = repo_full_name.split('/')
#    organization_folder = orgs_dir / organization / project
#
#    target_files = {
#        "issues": cfg.issue_list_file_name,
#        "issue_activity": cfg.issue_activity_file_name,
#        "prs_repo": cfg.PR_list_file_name,
#        "prs_comments": cfg.prs_comments_csv,
#        "commits": cfg.commit_list_file_name,
#        "perfile_commit": cfg.per_file_commits_path
#    }
#    #we need to make df for each of the target files
#    out = {k: pandas.DataFrame() for k in target_files.keys()}
#
#    for key, fname in target_files.items():
#        file_path = organization_folder / fname
#        print(f"Looking for file: {file_path}")
#        if file_path.exists():
#            try:
#                df = pandas.read_csv(file_path)
#                out[key] = pandas.concat([out[key], df], ignore_index=True)
#            except Exception as e:
#                print(f"[WARN] Failed reading {file_path}: {e}")
#        else:
#            print(f"[MISS] {file_path}")
#    st.write(f"New data start")
#
#    
#    return out

def load_users_activity(devs, repo_full_name):
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

    project_root = Path("../")
    orgs_dir = project_root / "Organizations"

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
    
    #we are making some test code and only looking at Rdatatable org for now
    org, repo = repo_full_name.split('/')
    organization_path = orgs_dir / org

    for repo in organization_path.iterdir():
        st.write(f"Processing repository: {repo}")
        if not repo.is_dir():
            continue
        for file_key, file_name in target_files.items():
            file_path = repo / file_name
            if file_path.exists():
                print(f"Found file: {file_path}")
                try:
                    # We need 5 different df that have the new data found appended on to them
                    df = pandas.read_csv(file_path)
                    if file_key == "issues":
                        issues = pandas.concat([issues, df], ignore_index=True)
                    elif file_key == "issue_activity":
                        issue_activity = pandas.concat([issue_activity, df], ignore_index=True)
                    elif file_key == "prs_repo":
                        prs_repo = pandas.concat([prs_repo, df], ignore_index=True)
                    elif file_key == "prs_comments":
                        prs_comments = pandas.concat([prs_comments, df], ignore_index=True)
                    elif file_key == "commit_list":
                        commits = pandas.concat([commits, df], ignore_index=True)
                    elif file_key == "perfile_commit":
                        perfile_commits = pandas.concat([perfile_commits, df], ignore_index=True)

                except Exception as e:
                    st.write(f"Error loading {file_path}: {e}")
            else:
                st.write(f"File not found: {file_path}")

    raw_data_tables = { "issues": issues, "issue_activity": issue_activity, 
                       "prs_repo": prs_repo, "prs_comments": prs_comments, 
                       "commits": commits , "perfile_commits": perfile_commits}

    #save the data to a temp file
    
    return raw_data_tables

def make_timeline(devs, repo_full_name, raw_data_tables):
    project_root = Path("../")
    orgs_dir = project_root / "Organizations"

    DAILY_COLS = ["dev", "date",
                "commits", "issues", "prs",
                "files_changed", "lines_added", "lines_removed",
                "prs_review", "prs_comment", "issues_commented", "issues_activity",

                "labeled", "closed", "commented", "mentioned", "subscribed", "referenced", "renamed", "issue_type_added", "unsubscribed", "pinned", "locked", "reopened", "assigned", "unlabeled", "connected", "milestoned", "comment_deleted", "unassigned", "unpinned", "demilestoned", "marked_as_duplicate", "transferred", "unmarked_as_duplicate", "unlocked", "parent_issue_added", "parent_issue_removed", "sub_issue_added", "sub_issue_removed", "disconnected", "nan", "unknown_event"]


    #TypeError: unsupported operand type(s) for +: 'WindowsPath' and 'str'
    #fix the folder pathing issue
    timeline_folder = Path(orgs_dir) / repo_full_name / cfg.timeline_folder_name

    timeline_path = timeline_folder / f"timeline.csv"

    #if timeline_path.is_file():
    #    st.write(f"Timeline already exists at {timeline_path}, loading existing timeline.")
    #    return pandas.read_csv(timeline_path)


    # make our daily df 
    # we already know devs in given in the fuction definition becuase
    # it needs to be the truck factory devs
    # we need to find max and min dates for each dev
    # we need to make a daily df with the columns we want from min date to max date for each user
    # find min date
    for table_name, table in raw_data_tables.items():
        table["created_at"] = pandas.to_datetime(table["created_at"], errors="coerce", utc=True).dt.date

    #'<=' not supported between instances of 'datetime.date' and 'float'
    # we need to convert all the created_at columns to datetime.date
    for table_name, table in raw_data_tables.items():
        raw_data_tables[table_name] = table[table["created_at"].notnull()]
        raw_data_tables[table_name]["created_at"] = pandas.to_datetime(raw_data_tables[table_name]["created_at"], errors="coerce", utc=True).dt.date

    min_date = min([table["created_at"].min() for table in raw_data_tables.values() if not table.empty])
    max_date = datetime.now(timezone.utc).date()

    date_range = pandas.date_range(start=min_date, end=max_date, freq="D").date
    daily_df = (pandas.MultiIndex.from_product([devs, date_range], names=["dev", "date"])
            .to_frame(index=False))
    
    for column in DAILY_COLS:
        if column not in ["dev", "date"]:
            daily_df[column] = 0


    # Important note: we only look at rows where the created_by is in devs
    # this could be incorrect if any files have different names in created_by
    #mask all rows that are not in devs or have a null created_by
    len_data = 0
    for table_name, table in raw_data_tables.items():

        mask = table["created_by"].isin(devs) & table["created_by"].notnull()
        raw_data_tables[table_name] = table[mask]

        len_data += len(raw_data_tables[table_name])
        st.write(f"{table_name}, original rows: {len(table)}, after dev filter: {len(raw_data_tables[table_name])}")

    
    row_count = 0 
    # we need to check if we have the right columns
    for index, row in raw_data_tables["commits"].iterrows():
        mask = (daily_df["date"] == row["created_at"]) & (daily_df["dev"] == row["created_by"] ) 
        #for commits we need to go to that day and that user and
        # we need to add one to the commits count
        # we need to add the files changed count
        # we need to add the lines added count
        # we need to add the lines removed count
        # we need to add the number of commits made that day 
        #commits += 1
        #files_changed += fileschanged_count
        #lines_added += additions_sum
        #lines_removed += deletions_sum
        # date format example: 2025-06-30 07:28:32+00:00
        # we have columns(repo,created_at,author_id,author_name,author_email,committer_id,sha,filename_list,fileschanged_count,additions_sum,deletions_sum)
        daily_df.loc[mask, "commits"]       += 1
        daily_df.loc[mask, "files_changed"] += int(row["fileschanged_count"])
        daily_df.loc[mask, "lines_added"]   += int(row["additions_sum"])
        daily_df.loc[mask, "lines_removed"] += int(row["deletions_sum"])
        row_count += 1
        if row_count % 1000 == 0:
            st.write(f"{row_count}/ {len_data}")

    st.write(f"next batch")
    daily_df.to_csv("demo_daily_df.csv", index=False)

    for index, row in raw_data_tables["prs_comments"].iterrows():
        mask = (daily_df["date"] == row["created_at"]) & (daily_df["dev"] == row["created_by"] ) 
        # for issues we have columns(repo,created_at,created_by,issue_number,title,state,closed_at,labels,assignees,milestone)
        # and we need to go to that day and that user
        # we need to add one to the issue count issues += 1
        # date format example: 2025-09-11T19:23:08Z
        # TODO: figure out what else we want to track from issues with out looking ahead
        daily_df.loc[mask, "issues"] += 1
        row_count += 1 
        if row_count % 1000 == 0:
            st.write(f"{row_count}/ {len_data}")

    st.write(f"next batch")
    daily_df.to_csv("demo_daily_df.csv", index=False)

    for index, row in raw_data_tables["prs_repo"].iterrows():
        mask = (daily_df["date"] == row["created_at"]) & (daily_df["dev"] == row["created_by"] ) 
        # we have columns(repo,created_at,created_by,PR_id,state,merged,closed_at,merged_at)
        # we just need to add one to the prs count for that day and user
        # date format example: 2025-09-10 16:02:11+00:00
        # TODO: figure out how we can feed the model the information about when a pr was merged_at
        daily_df.loc[mask, "prs"] += 1
        row_count += 1

    st.write(f"next batch")
    daily_df.to_csv("demo_daily_df.csv", index=False)


    for index, row in raw_data_tables["issues"].iterrows():
        mask = (daily_df["date"] == row["created_at"]) & (daily_df["dev"] == row["created_by"] ) 
        # for issues we have columns(repo,created_at,created_by,issue_number,title,state,closed_at,labels,assignees,milestone)
        # and we need to go to that day and that user
        # we need to add one to the issue count issues += 1
        # date format example: 2025-09-11T19:23:08Z
        # TODO: figure out what else we want to track from issues with out looking ahead
        daily_df.loc[mask, "issues"] += 1
        row_count += 1
        if row_count % 1000 == 0:
            st.write(f"{row_count}/ {len_data}")

    st.write(f"next batch")
    daily_df.to_csv("demo_daily_df.csv", index=False)
    unknown_seen = set()


    for index, row in raw_data_tables["issue_activity"].iterrows():
        mask = (daily_df["date"] == row["created_at"]) & (daily_df["dev"] == row["created_by"] ) 
        #columns (repo,issue_number,activity_id,item_type,event,body,created_at,actor)
        # we need to make n number of columns for each unique type of event
        # i think we added enough columns for this in the df declaration but throw an erro if not
        # make sure you label this so i can delete it later
        # for issue activity we need to go to that day and that user
        # we need add one to the event type count
        # date format example: 2025-09-09 06:07:02+00:00
        #TODO: figure out if item_type can be anything else besides "TimelineEvent"

        # we have many event types as you can see below 
        #item_types = table["item_type"].unique().tolist()
        #event_types = table["event"].unique().tolist()
        #Processing issue activity for dev jekyllbot on date 2021-08-19: item_types=['TimelineEvent'], event_types=['labeled', 'closed', 'commented', 'mentioned', 'subscribed', 'referenced', 'renamed', 'issue_type_added', 'unsubscribed', 'pinned', 'locked', 'reopened', 'assigned', 'unlabeled', 'connected', 'milestoned', 'comment_deleted', 'unassigned', 'unpinned', 'demilestoned', 'marked_as_duplicate', 'transferred', 'unmarked_as_duplicate', 'unlocked', 'parent_issue_added', 'parent_issue_removed', 'sub_issue_added', 'sub_issue_removed', 'disconnected']
        #event_type_counts = table['event'].value_counts()
        #Event type counts: {'commented': 42552, 'subscribed': 18094, 'labeled': 13302, 'mentioned': 12566, 'closed': 8678, 'locked': 4689, 'referenced': 3345, 'milestoned': 3254, 'demilestoned': 1557, 'assigned': 1305, 'unlabeled': 1229, 'renamed': 1145, 'reopened': 383, 'unassigned': 186, 'unsubscribed': 139, 'comment_deleted': 100, 'connected': 53, 'pinned': 33, 'unpinned': 30, 'marked_as_duplicate': 18, 'disconnected': 4, 'issue_type_added': 3, 'unlocked': 3, 'transferred': 1, 'unmarked_as_duplicate': 1, 'parent_issue_added': 1, 'parent_issue_removed': 1, 'sub_issue_added': 1, 'sub_issue_removed': 1}
        event = row["event"]

        if pandas.notna(event) and isinstance(event, str) and ":" in event:
            base, _ = event.split(":", 1)
            event = base  # e.g., "labeled:documentation" -> "labeled"

        # Treat NaN events as "NaN" and log once
        evt = (str(event) if pandas.notna(event) else "NaN")

        if evt not in daily_df.columns and evt not in DAILY_COLS:
            if evt not in unknown_seen:
                st.write(f"Unknown event type: {evt} — counting under 'unknown_event' and ignoring henceforth.")
                unknown_seen.add(evt)
            daily_df.loc[mask, "unknown_event"] += 1
            continue


        daily_df.loc[mask, event] += 1
        daily_df.loc[mask, "issues_activity"] += 1
        row_count += 1
        if row_count % 1000 == 0:
            st.write(f"{row_count}/ {len_data}")


    os.makedirs(timeline_folder, exist_ok=True)
    print(f"Saving timeline to {timeline_path}")
    daily_df.to_csv(timeline_path, index=False)
    st.write(f"Saved timeline to {timeline_path}")
    return daily_df

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
                 
#-----------------------
# inactivity labeling
#------------------------
def label_developers_activity(repo, process_all: bool = False) -> pandas.DataFrame:
    """
    main function for labeling developers
    sets up varables to call label timeline
    """
    
    # "../Organizations"
    organizationFolder = cfg.main_folder

    win = cfg.sliding_window_size

    repos_txt = '../' + cfg.repos_file
    repos_to_process = []
        
    if process_all is True:
        with open(repos_txt, 'r') as f:
            repos = f.readlines()
            repos_to_process = [r.strip() for r in repos if r.strip()]
            st.write(f"Found {len(repos)} repositories in {repos_txt}")
    else:
        repos_to_process = [repo]
    all_timelines = []
    all_diagnostics = []

    

    for repo in repos_to_process:
        organizationFolder = cfg.main_folder
        print("look here for repo:", repo)
        organization, project = repo.split('/')
        if Path(organizationFolder, organization).exists() == False:
            st.write(f"Organization folder not found: {Path(organizationFolder, organization)}")
            continue

        st.write(f"Start Identifying inactivity periods for {organization}/{project}...")

        organizationFolder = Path(organizationFolder) / organization / project

        commits =  pandas.read_csv(organizationFolder / "commit_list.csv", parse_dates=["created_at"], encoding="utf-8", header=0, sep=cfg.CSV_separator)

        tf_devs = pandas.read_csv(organizationFolder / cfg.TF_developers_file, sep=cfg.CSV_separator)
        
        print(f"Truck Factor developers loaded: {tf_devs}")
        pauses = write_pauses_table(commits, organizationFolder / "pauses_commits.csv", tf_devs, date_col="created_at")
        #make pauses to a csv file at this location C:\Users\samut\OneDrive\Documents\GitHub\developersInactivityAnalysisCOPY\Organizations\Rdatatable\data.table\Results

        print(f"{len(tf_devs)} Developers inactivity periods identified")
        st.write(f"{len(tf_devs)} Developers inactivity periods identified")

        output_folder = organizationFolder /  "Results"
        os.makedirs(output_folder, exist_ok=True)
        progress = st.progress(0)
        count = 0

        for dev in tf_devs:
            print(dev)
            if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
                column, dev = dev.split('|')
            progress.progress((count+1)/len(tf_devs))
            count= count+1

            timeline_folder = organizationFolder /  cfg.timeline_folder_name
            os.makedirs(timeline_folder, exist_ok=True)
        
            timeline_path = Path(timeline_folder) / f"timeline.csv"

            if timeline_path.is_file():
                #our is Timeline created at: ..\Organizations\atom\atom\Timelines\timeline.csv
                user_timeline = pandas.read_csv(timeline_path, sep=cfg.CSV_separator, parse_dates=["date"], index_col="date")
                user_timeline_change = user_timeline[user_timeline["dev"] == dev]
                
            else:
                #we need to make the timeline 
                print(f"Timeline not found at {timeline_path}, generating timeline...")
                continue

            breaks_folder = organizationFolder /  "Breaks"
            os.makedirs(breaks_folder, exist_ok=True)
            breaks_path =  Path(breaks_folder)/  f"{dev}_breaks.csv"              

            #if breaks_path.is_file():
            #    print(breaks_path)
            #    breaks_df = pandas.read_csv(breaks_path, sep=cfg.CSV_separator, index_col=0)  
            #    print(breaks_df)
            #
            #else:
            breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])
            #print all input varables
            breaks_df, diagnostics_df = identifyBreaks(pauses, dev=dev, window=win, debug_folder=output_folder)

            breaks_df.to_csv(breaks_path, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, index=False, lineterminator="\n")
                        
            #add label timeline'
            user_timeline = user_timeline[user_timeline["dev"] == dev]
            user_timeline = label_timeline(user_timeline, breaks_df)

            user_timeline = user_timeline.reset_index()
            all_timelines.append(user_timeline)

            all_diagnostics.append(diagnostics_df)

            out_csv = Path(output_folder) / f"{dev}_labeled_timeline.csv"
            user_timeline.to_csv(out_csv, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, lineterminator='\n', index_label='date')

            tf_devs_df = pandas.DataFrame(tf_devs, columns=["developer"])
            tf_devs_df.to_csv(Path(output_folder) / "tf_devs.csv", sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, quoting=None, lineterminator='\n')

            # WE NEED TO MAKE A MASTER USER TIMELINE that we use as the return value

        master_user_timeline = pandas.concat(all_timelines, ignore_index=True)
        master_diagnostics = pandas.concat(all_diagnostics, ignore_index=True)

    # append the diagnostics to the time line by joining on the win_end from diagnostics and the dates from timeline
    # master_diagnostics.win_end is object holding datetime.date (needs converting to pandas datetime + normalize).
    master_diagnostics["date"] = pandas.to_datetime(master_diagnostics["win_end"]).dt.normalize()
    master_user_timeline = master_user_timeline.merge(master_diagnostics, left_on=["dev", "date"], right_on=["dev", "date"], how="left")
    
    master_user_timeline.to_csv(Path(output_folder) / "all_users_labeled_timeline.csv", sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, lineterminator='\n', index_label='date')

    # visualize the breaks
    # for each developer in tf_devs we need to make a plot of their timeline with breaks marked
    devs = sorted(master_user_timeline["dev"].unique())
    print(f"Visualizing breaks for {len(devs)} developers...")
    print( devs)
    for dev in devs:
        plot_dev_timeline(master_user_timeline, dev)

    return master_user_timeline

def write_pauses_table(
        df: pandas.DataFrame,
        out_path: os.PathLike,
        tf_devs: list[str] | None = None,
        *,
        date_col: str = "created_at",
        tail_to_today: bool = True
    ) -> pandas.DataFrame:

    df[date_col] = pandas.to_datetime(df[date_col]).dt.normalize()


    rows = []

    count =0
    for dev in tf_devs:
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            print(f"found special dev {dev}")
            column, dev = dev.split('|')
            user_df = df[df[column] == dev]
        else:
            user_df = df[df["author_id"] == dev]

        active_days = sorted(user_df[date_col].dt.date.unique())
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
                current_row.append(f"{active_days[-1]}/{today}")
        if len(current_row) > 1:
            rows.append(current_row)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="",encoding="utf-8" ) as f:
        csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE).writerows(rows)
    out = pandas.DataFrame(rows)

    return out

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

    # We need to make coding_day,nc_day 
    # a coding day is something specific
    # coding activity both making a commit to a local repository and opening a pull request
    # that means in daily df commits > 0 AND prs > 0

    # coding activity: commit OR PR
    df["coding_day"] = ((df["commits"] > 0) | (df["prs"] > 0)).astype(int)

    # non-coding activity: any other event > 0 AND no coding
    noncoding_cols = [c for c in df.columns if c in [ "issues", "issue_activity", "prs_activity" ]]
    df["nc_day"] = ((df[noncoding_cols].sum(axis=1) > 0) & (df["coding_day"] == 0)).astype(int)


    df["break_day"] = None
    df["break_day"] = pandas.Series(False, index=df.index, dtype="boolean")
    df["th"] = pandas.Series(pandas.NA, index=df.index, dtype="Float64")
    df["len"] = pandas.Series(pandas.NA, index=df.index, dtype="Int64")
    df["index"] = pandas.Series(pandas.NA, index=df.index, dtype="Int64")

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

    gone_days = 365

    df["state"] = "ACTIVE"


    # Identify contiguous break windows (groups of consecutive True in break_day)
    bd = df["break_day"]
    group_id = (bd != bd.shift(1)).cumsum()

    df["event_day"] = ((df["coding_day"] >= 1) | (df["nc_day"] >= 1)).astype(int)

    # Precompute last event BEFORE a given date (global, across timeline)
    all_events_idx = df.index[df["event_day"]]

    for gid, block in df.groupby(group_id):

        if not block["break_day"].iloc[0]:
            continue  # not a break chunk

        # This is one contiguous break [start .. end] (inclusive)
        start_ts = block.index[0]
        end_ts   = block.index[-1]


        th_vals = df.loc[start_ts:end_ts, "th"]
        Tfov = int(round(th_vals.iloc[0]))

        # optional: get far-out threshold
        # Anchor silence to the last event (coding or non-coding) before the break starts
        prev_nc_idx = df.index[df["nc_day"] & (df.index < start_ts)]
        last_nc_before = prev_nc_idx.max()

        prev_activity_idx = df.index[df["event_day"] & (df.index < start_ts)]
        last_event_before = prev_activity_idx.max()

        #
        #last_nc = None  # most recent non-coding event inside this break
        #if last_nc_before is not None and (start_ts - last_nc_before).days <= Tfov:
        #    for d in range((start_ts - last_nc_before).days):
        #        current_day = last_nc_before + pandas.Timedelta(days=d + 1)
        #        df.at[current_day, "state"] = "NON_CODING"
        #    print(f"Processing break from {start_ts.date()} to {end_ts.date()} ")
        #    print("last_nc_before", last_nc_before)
        #    print("the most recent day was ", (start_ts - last_nc_before).days, "days ago")
        #    print("Tfov =", Tfov)
        #    view_df(df.loc[start_ts:end_ts], name="Break block before labeling")
        #    print("\n\n")
        

        # Walk day by day inside the break
        for d in block.index:

            # Non-coding event day => NON_CODING and update last_nc
            if bool(df.at[d, "nc_day"]):
                df.at[d, "state"] = "NON_CODING"
                last_nc_before = d
                continue

            # Silent day inside a break -> decide via Tfov and gone
            # Compute silence since the most relevant last event:
            # - Prefer last NC inside break; else use last event before break; else start-of-break as approximate anchor.
            if ((d - last_nc_before).days <= Tfov):
                df.at[d, "state"] = "NON_CODING"
                continue

            # No recent NC: INACTIVE vs GONE (since last ANY event)
            silent_days = (d - last_event_before).days
            if silent_days > gone_days:
                df.at[d, "state"] = "GONE"
            else:
                df.at[d, "state"] = "INACTIVE"
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

def identifyBreaks(pauses_dates_list, dev, window, debug_folder=None):
    '''
    Removes SURE BREAKS from windows to calculate Tfov
    and — with debug_folder — writes a per-window diagnostics CSV.
    '''
    pauses_dates_list = pauses_dates_list.values.tolist()

    breaks_df = pandas.DataFrame(columns=['len', 'dates', 'th'])
    diagnostics = []                             # NEW
    count = 0

    for row in pauses_dates_list:
        # print the first few characters of the row

        if str(row[0]).strip() != str(dev).strip():            # ⬅️  ignore other developers
            continue

        count += 1
        if count % 50 == 0:  # Print progress every 50 rows
            print(count)

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
            print(f"  FPS_dt={FPS_dt.date()}  LPE_dt={LPE_dt.date()}")
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

def timeline(tables, tf_devs):
    # given 3 files I need you to count the rows per day per user
    # you are given commits.csv issues.csv prs.csv
    # dev (string; developer id/handle)
    # date (date at daily granularity, e.g., YYYY-MM-DD)
    # commits (non-negative integer)
    # prs (non-negative integer)
    # issues (non-negative integer)
    
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


    for dev in tf_devs:
        print(dev)
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            print(f"found special dev {dev}")
            column, dev = dev.split('|')
            column = column
            name = dev
        else:
            column = "author_id"
            name = dev

    # Count rows per day per user
    user_activity = []
    for dev in tf_devs:
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            column, dev = dev.split('|')
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

        # Combine all dates and fill missing values with 0
        all_dates = daily_commits.index.union(daily_issues.index).union(daily_prs.index)
        
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

    return pandas.DataFrame(user_activity)


#-----------------------
# Developer timeline prediction
#------------------------
def plot_dev_timeline(
    df: pandas.DataFrame,
    dev: str,
    noncoding_cols=("issues", "issue_activity", "prs_activity"),
    *,
    height: int = 320,
    lookahead_days: int = 14,
    pred_proba_col: str | None = "rf_proba",   # e.g., 'rf_proba' or 'logit_proba'
    pred_threshold: float = 0.5,
):
    """
    Two-panel Streamlit viz:
      Top  : activity timeline (coding vs non-coding) with background state bands.
      Bottom: ground-truth lookahead windows and model predictions/probabilities.

    - 'lookahead_days' builds the windows you *intend* to predict.
    - 'pred_proba_col' is the model's per-day probability (0..1). If not found,
       it will try common fallbacks; if none, the bottom panel shows only truth.
    - 'pred_threshold' draws a dashed line and tick marks where prediction >= threshold.
    """

    # ---- guard: ensure required columns exist
    required = {"dev", "date", "commits", "prs", "state"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        return

    # ---- filter + normalize types
    if dev.startswith(("author_login", "author_name", "author_email")):
        _, dev = dev.split("|", 1)

    d = df[df["dev"].astype(str).str.strip() == str(dev).strip()].copy()
    if d.empty:
        st.warning(f"No rows for dev='{dev}'.")
        return

    d["date"] = pandas.to_datetime(d["date"], errors="coerce").dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)

    # ---- activity columns (numeric totals)
    d["coding_total"] = d[["commits", "prs"]].fillna(0).sum(axis=1)

    nc_present_cols = [c for c in noncoding_cols if c in d.columns]
    d["noncoding_sum_raw"] = d[nc_present_cols].fillna(0).sum(axis=1) if nc_present_cols else 0
    # only count non-coding when no coding that day
    d["noncoding_total"] = d["noncoding_sum_raw"].where(d["coding_total"] == 0, 0)

    # log-transform activity counts (for better viz scaling)
    d["coding_total"] = np.log1p(d["coding_total"])
    d["noncoding_total"] = np.log1p(d["noncoding_total"])


    # ---- background vertical bands by state (per day)
    d["next_date"] = d["date"] + pandas.Timedelta(days=1)
    state_domain = ["ACTIVE", "NON_CODING", "INACTIVE", "GONE"]
    state_range  = ["#5df15d", "#ffef5c", "#ff1c51", "#7a7a7a"]

    bg = (
        alt.Chart(d)
        .mark_rect(opacity=0.15)
        .encode(
            x=alt.X("date:T", title=None),
            x2="next_date:T",
            color=alt.Color("state:N", legend=None,
                            scale=alt.Scale(domain=state_domain, range=state_range)),
        )
    )

    line_coding = (
        alt.Chart(d)
        .mark_bar(strokeWidth=2, color="Green")
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("coding_total:Q", title="Daily activity (log count)"),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("coding_total:Q", title="coding (log)"),
                alt.Tooltip("state:N"),
            ],
        )
    )

    line_noncoding = (
        alt.Chart(d)
        .mark_bar(strokeDash=[4, 3], strokeWidth=2, opacity=0.9, color="yellow")
        .encode(
            x=alt.X("date:T"),
            y=alt.Y("noncoding_total:Q", title=None),
            tooltip=[
                alt.Tooltip("date:T"),
                alt.Tooltip("noncoding_total:Q", title="non-coding (log)"),
                alt.Tooltip("state:N"),
            ],
        )
    )

    chart_activity = alt.layer(bg, line_coding, line_noncoding).properties(height=height)

    # =========================
    # BOTTOM PANEL: TRUTH + PRED
    # =========================

    # --- 1) Build the *true* look-ahead windows (what you intend to predict)
    # Prefer explicit column if present; else derive transition days from state
    if "transition_to_inactive" in d.columns:
        trans = d["transition_to_inactive"].fillna(0).astype(int)
    else:
        # transition day = first INACTIVE after a non-INACTIVE day
        prev_state = d["state"].shift(1).fillna(d["state"].iloc[0])
        trans = ((d["state"] == "INACTIVE") & (prev_state != "INACTIVE")).astype(int)

    # mark 1 for the 'lookahead_days' *before* each transition (inclusive)
    rev = trans.iloc[::-1]
    y_true_window = rev.rolling(lookahead_days + 1, min_periods=1).max().iloc[::-1].astype(int)
    d["y_true_window"] = y_true_window
    d["y0"] = 0.0
    d["y1"] = 1.0

    truth_rect = (
        alt.Chart(d)
        .transform_filter(alt.datum.y_true_window == 1)
        .mark_rect(color="#ff1c51", opacity=0.2)  # light red windows
        .encode(
            x="date:T",
            x2="next_date:T",
            y="y0:Q",
            y2="y1:Q",
        )
    )

    # --- 2) Predicted probabilities + threshold + predicted-positive ticks
    # pick a prob column
    prob_col = None
    candidates = [pred_proba_col] if pred_proba_col else []
    candidates += [c for c in ["rf_proba", "logit_proba", "pred_proba", "rf_preds"] if c in d.columns]
    for c in candidates:
        if c in d.columns:
            prob_col = c
            break

    pred_layers = []
    if prob_col is not None:
        d["pred_proba"] = d[prob_col].clip(0, 1)

        pred_line = (
            alt.Chart(d)
            .mark_line(strokeWidth=1.5)
            .encode(
                x="date:T",
                y=alt.Y("pred_proba:Q", scale=alt.Scale(domain=[0, 1]), title="Pred. prob"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("pred_proba:Q", title=f"{prob_col}")],
            )
        )
        pred_layers.append(pred_line)

        # threshold rule
        rule_df = pandas.DataFrame({"y": [pred_threshold]})
        th_rule = alt.Chart(rule_df).mark_rule(strokeDash=[4, 3]).encode(y="y:Q")
        pred_layers.append(th_rule)

        # predicted-positive ticks (>= threshold)
        d["pred_pos"] = (d["pred_proba"] >= float(pred_threshold)).astype(int)
        d["tick_y"] = 0.95  # where to draw ticks in [0,1] space

        pred_ticks = (
            alt.Chart(d)
            .transform_filter(alt.datum.pred_pos == 1)
            .mark_tick(thickness=2, size=12)
            .encode(x="date:T", y="tick_y:Q")
        )
        pred_layers.append(pred_ticks)
    else:
        st.info("No prediction probability column found (looked for "
                f"{[pred_proba_col, 'rf_proba','logit_proba','pred_proba','rf_preds']}). "
                "Showing only ground-truth windows.")

    chart_eval = (
        alt.layer(truth_rect, *pred_layers)
        .properties(height=110)
        .resolve_scale(x="shared")
    )

    # =========================
    # COMPOSE + RENDER
    # =========================
    chart = alt.vconcat(chart_activity, chart_eval).resolve_scale(x="shared")
    st.subheader(f"Activity + Prediction Timeline — {dev}")
    st.caption(
        "Top: Solid = coding (commits+PRs), dashed = non-coding (issues/prs_activity when no coding). "
        "Background color = daily state.  "
        "Bottom: Red bands = TRUE look-ahead windows (next N days before an inactivity transition). "
        "Line = model probability; dashed = threshold; ticks = predicted positives."
    )
    st.altair_chart(chart, use_container_width=True)


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

def make_confusion_mats(pred_df, thr_rf=0.5, thr_logit=0.5, per_dev=False, date_col="date"):
    """
    Window-level confusion metrics.
    A 'window' = contiguous run where y_true==1. It's a TP if any prediction==1 within that run.
    Also reports FP flags = predicted positives on rows where y_true==0.

    Returns a dict with 'rf' and 'logit' DataFrames of metrics (overall or per-dev).
    """
    df = pred_df.copy()
    df = df.dropna(subset=["y_true", "rf_proba"])
    df["y_true"] = df["y_true"].astype(int)

    # threshold -> class labels
    df["rf_pred"]    = (df["rf_proba"]    >= thr_rf).astype(int)

    # choose grouping (per-dev or all together)
    group_cols = ["dev"] if per_dev and "dev" in df.columns else []
    sort_cols  = group_cols + ([date_col] if date_col in df.columns else [])

    if sort_cols:
        df = df.sort_values(sort_cols)

    def _metrics_for_model(g: pandas.DataFrame, model_col: str) -> pandas.Series:
        # mark contiguous positive windows
        mask = g["y_true"].eq(1)
        # a new window starts where we transition from 0/NaN to 1
        starts = mask & (~mask.shift(fill_value=False))
        run_id = starts.cumsum()                         # monotonically increasing id
        g = g.copy()
        g["pos_run_id"] = np.where(mask, run_id, np.nan)

        # window hits: any prediction==1 inside the window
        if mask.any():
            hits = (g.loc[mask]
                     .groupby("pos_run_id")[model_col]
                     .apply(lambda s: int((s == 1).any())))
            TP_windows = int(hits.sum())
            total_windows = int(hits.size)
        else:
            TP_windows = 0
            total_windows = 0

        FN_windows = int(total_windows - TP_windows)

        # false flags = predicted positives outside any positive window (i.e., where y_true==0)
        FP_flags = int(((g["y_true"] == 0) & (g[model_col] == 1)).sum())

        # optional: true-negative windows aren't well-defined here; we report window-accuracy per your definition
        denom = TP_windows + FN_windows
        window_accuracy = (TP_windows / denom) if denom > 0 else np.nan

        # (also handy) counts for context
        total_flags = int((g[model_col] == 1).sum())
        true_flags_inside_windows = int(((g["y_true"] == 1) & (g[model_col] == 1)).sum())

        return pandas.Series({
            "total_windows": total_windows,
            "Flagged_windows": TP_windows,
            "Missed_windows": FN_windows,
            "window_accuracy": window_accuracy,
            "Missed_flags": FP_flags,
            "total_flags": total_flags,
            "flags_inside_windows": true_flags_inside_windows,
        })

    # compute metrics per group, then aggregate if needed
    def _eval_model(model_col: str) -> pandas.DataFrame:
        if group_cols:
            per = df.groupby(group_cols, dropna=False).apply(_metrics_for_model, model_col).reset_index()
            # overall row (summing counts and recomputing accuracy)
            overall_counts = per[["total_windows","Flagged_windows","Missed_windows","Missed_flags","total_flags","flags_inside_windows"]].sum()
            denom = overall_counts["Flagged_windows"] + overall_counts["Missed_windows"]
            overall_acc = (overall_counts["Flagged_windows"] / denom) if denom > 0 else np.nan
            overall = pandas.DataFrame([{**{group_cols[0]: "__ALL__"}, **overall_counts.to_dict(), "window_accuracy": overall_acc}])
            return pandas.concat([per, overall], ignore_index=True)
        else:
            return _metrics_for_model(df, model_col).to_frame().T

    out = {
        "rf": _eval_model("rf_pred")
    }
    return out
    
def run_prediction_pipeline(df  , repo_key):
    
    devs= df["dev"].unique().tolist()

    y = build_response(df, 7, label_col = "state")
    df = df.merge(y[["dev", "date", "transition_to_inactive"]], on=["dev", "date"], how="left")

    label_col = "transition_to_inactive"
    date_col = "date"
    dev_col = "dev"


    df[label_col] = pandas.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    df[date_col] = pandas.to_datetime(df[date_col])

    # identify numeric columns for lag features
    num_cols = [ 'commits', 'prs', 'issues',
       'issue_activity', 'pr_activity', 'break_day',
       'th', 'len', 'win_th', 'last_th',
       'issue_total_interactions', 'issue_unique_partners',
       'issue_items_touched', 'issue_new_interactions',
       'issue_new_relationships', 'issue_total_relationships',
       'issue_repeat_partners', 'issue_avg_time_delta',
       'issue_min_response_time', 'issue_avg_response_time_distance_1',
       'issue_avg_interactions_per_user', 'issue_avg_distance_away',
       'pr_total_interactions', 'pr_unique_partners', 'pr_items_touched',
       'pr_new_interactions', 'pr_new_relationships', 'pr_total_relationships',
       'pr_repeat_partners', 'pr_avg_time_delta', 'pr_min_response_time',
       'pr_avg_response_time_distance_1', 'pr_avg_interactions_per_user',
       'pr_avg_distance_away']

    # remove these items from the list num_cols: 'win_pauses', 'partial_lengths', 'win_th', 'last_th'
    excluded_cols = {'win_pauses', 'partial_lengths', 'th', 'len', 'index' , 'coding_day', 'nc_day','break_day', 'Unnamed: 0'}
    num_cols = [c for c in num_cols if c not in excluded_cols]

    encoder = LabelEncoder()

    # Fit and transform the categorical column
    df['state_encoded'] = encoder.fit_transform(df['state'])
    df['break_day'] = encoder.fit_transform(df['break_day'])

    LAGS=5
    combined_results = []

    for dev in devs:
        for col in num_cols:
            for l in range(0, LAGS+1):
                df[f"{col}_lag{l}"] = df.groupby(dev_col, observed=True)[col].shift(l)

        test_df = df[ df["dev"] == dev ]
        train_df = df[ df["dev"] != dev ]

        Xtr = train_df[[c for c in df.columns if any(c.endswith(f"_lag{i}") for i in range(1, LAGS+1))]]
        Xtr['state_encoded'] = df['state_encoded']
        Xtr['break_day'] = df['break_day']
        Xtr = Xtr.dropna(axis=1, how='all')
        Xtr = Xtr.fillna(0)

        Xte = test_df[[c for c in df.columns if any(c.endswith(f"_lag{i}") for i in range(1, LAGS+1))]]
        Xte['state_encoded'] = df['state_encoded']
        Xte['break_day'] = df['break_day']
        Xte = Xte.dropna(axis=1, how='all')
        mask = ~Xte.isna().any(axis=1)
        Xte = Xte.fillna(0)

        ytr = train_df[label_col].astype(int)
        yte = test_df[label_col].astype(int)


        mask = ~Xte.isna().any(axis=1)


        len 
        
        rf_clf  = RandomForestClassifier(n_estimators=400, min_samples_split=5, min_samples_leaf=2,
                                        n_jobs=-1, random_state=42, class_weight="balanced_subsample")

        rf_clf.fit(Xtr,ytr)

        pred_df = pandas.DataFrame({
            "dev": test_df[dev_col].values,
            "date": test_df[date_col].values,
            "y_true": yte.values,
            "rf_proba": rf_clf.predict_proba(Xte)[:,1]        
            })


        combined_results.append(pred_df)

    combined_results = pandas.concat(combined_results, ignore_index=True)

    repo_key = "atom_atom"
    combined_results.to_csv(f"{repo_key}_results.csv", index=False)


    metrics = make_confusion_mats(combined_results, thr_rf=0.6, per_dev=False)

    print(metrics['rf'])
    return combined_results, metrics

#-----------------------
#visualization functions
#------------------------  

def view_df(df, name="DataFrame"):
    ''' Simple HTML table viewer for DataFrames '''
    import tempfile, webbrowser
    html = "\n".join([
        "<meta charset='utf-8'>",
        "<style>body{font-family:system-ui,Segoe UI,Arial}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:6px}th{position:sticky;top:0;background:#fafafa}</style>",
        f"<h3>{name}</h3>",
        df.to_html(index=False, escape=False),
    ])
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html", encoding="utf-8") as f:
        f.write(html)
        webbrowser.open("file://" + f.name)

def main():
    
    st.set_page_config(page_title="Dev Inactivity Demo", layout="wide")
    #user input
    # we need to ask the user for a few things
    # we need the test train split (leave one dev or repo out)
    # we need to know if its dev mode
    # we need to be able to add anything to this in the future
    # we are using streamlit for this
    dev_mode = st.checkbox("Developer Mode:", value=True)

    st.title("Developer Inactivity Prediction")
    st.write("Configure the settings for predicting developer inactivity.")
    st.write("Please provide the necessary inputs below:")

    # make a text box that users can write repos urls into
    repo_url = st.text_input("Enter GitHub Repository URL (e.g., https://github.com/user/repo) or (org/repo):")
    list_of_repos = []
    # if "add to queue" button is pressed
    if st.button("Add to Queue"):
        gitRepoName = repo_url.replace('https://github.com/', '').strip()
        # parse the repo url to get user and repo name
        list_of_repos.append(gitRepoName)

    st.caption(f"Selected repos: **{', '.join(list_of_repos)}**")
    # (Later buttons can use `repo_key` and `paths`, e.g., Update, Label, Predict)

    st.divider()

    if st.button("Step 1: Find Core Developers"):

        for repo in list_of_repos:
            
            tf, tf_devs = KnowledgeDistribution.main(repo_full_name=repo)
            #print our tf_devs as a table
            st.write(f"For {repo}:")
            st.write(tf_devs)

            tf_devs_list_path = Path(cfg.main_folder) / repo /cfg.core_devs_file_name
            # save the list of core developers to a file
            tf_devs_list = tf_devs[0].tolist()

            tf_devs_list_path.write_text("\n".tf_devs_list)

    st.divider()

    if st.button("Step 2: Get Raw Data"):
        if  st.session_state.tf_devs is None:
            st.write("Please find core developers first.")
            return None
        raw_data_tables = load_users_activity( st.session_state.tf_devs, repo_full_name=repo_key)

        #test_raw_data = get_data(st.session_state.tf_devs, repo_full_name=repo_key)

        st.session_state.raw_data_tables = raw_data_tables

    st.divider()

    if st.button("Step 3: Timeline and Visualize Activity"):

        timeline_folder = Path(cfg.main_folder) / repo_key / cfg.timeline_folder_name
    
        os.makedirs(timeline_folder, exist_ok=True)
    
        timeline_path = Path(timeline_folder) / f"timeline.csv"

        out = timeline(st.session_state.raw_data_tables, st.session_state.tf_devs)
        # we need to save this time line in the right spot
        out.to_csv(timeline_path, sep=cfg.CSV_separator)

        st.write(out)

    st.divider()
    # add a toggle button
    process_all = st.checkbox("Process All Data Without User Input", value=False)

    if st.button("Step 4: Label Activity Data"):
        
        user_labeled_timeline = label_developers_activity(repo=repo_key , process_all=process_all)
        st.session_state.labeled_data = user_labeled_timeline

    st.divider()

    if st.button("Step 5: Add Social Network Metrics"):
        issue_interactions_interactions_metrics, pr_interactions_interactions_metrics, combined_interactions = std.main(repo_key, st.session_state.tf_devs, tables= st.session_state.raw_data_tables)
        # we need to merge on dev and date
        #rename out columns from_user day to dev and date

        issue_interactions = issue_interactions_interactions_metrics.rename(columns={
            "from_user": "dev",
            "day": "date",
            "author_id": "author_id",
            "author_name": "author_name",
            "author_login": "author_login",
            "author_email": "author_email",
            "total_interactions": "issue_total_interactions",
            "unique_partners": "issue_unique_partners",
            "items_touched": "issue_items_touched",
            "new_interactions": "issue_new_interactions",
            "new_relationships": "issue_new_relationships",
            "total_relationships": "issue_total_relationships",
            "repeat_partners": "issue_repeat_partners",
            "avg_time_delta": "issue_avg_time_delta",
            "min_response_time": "issue_min_response_time",
            "avg_response_time_distance_1": "issue_avg_response_time_distance_1",
            "avg_interactions_per_user": "issue_avg_interactions_per_user",
            "avg_distance_away": "issue_avg_distance_away"
        })
        pr_interactions = pr_interactions_interactions_metrics.rename(columns={
            "from_user": "dev",
            "day": "date",
            "author_id": "author_id",
            "author_name": "author_name",
            "author_login": "author_login",
            "author_email": "author_email",
            "total_interactions": "pr_total_interactions",
            "unique_partners": "pr_unique_partners",
            "items_touched": "pr_items_touched",
            "new_interactions": "pr_new_interactions",
            "new_relationships": "pr_new_relationships",
            "total_relationships": "pr_total_relationships",
            "repeat_partners": "pr_repeat_partners",
            "avg_time_delta": "pr_avg_time_delta",
            "min_response_time": "pr_min_response_time",
            "avg_response_time_distance_1": "pr_avg_response_time_distance_1",
            "avg_interactions_per_user": "pr_avg_interactions_per_user",
            "avg_distance_away": "pr_avg_distance_away"
        })
        issue_interactions["date"] = pandas.to_datetime(issue_interactions["date"], utc=True).dt.tz_convert(None).dt.normalize()
        pr_interactions["date"] = pandas.to_datetime(pr_interactions["date"], utc=True).dt.tz_convert(None).dt.normalize()

        dev = st.session_state.tf_devs[0]
        
        if dev.startswith("author_login") or dev.startswith("author_name") or dev.startswith("author_email"):
            column, dev = dev.split('|')
        else:
            column = "author_id"
                
        #we need to merge issue_interactions on column and labeled_data on "dev"
        out = pandas.merge(
            st.session_state.labeled_data,
            issue_interactions,
            left_on=["dev", "date"],
            right_on=[column, "date"],
            how="left",
            suffixes=('', '_issue')
        )

        out = pandas.merge(
            out,
            pr_interactions,
            left_on=["dev", "date"],
            right_on=[column, "date"],
            how="left",
            suffixes=('', '_pr')
        )

        # check if this could is all na values
        out["issue_total_interactions"] = out["issue_total_interactions"].fillna(0)
        out["pr_total_interactions"] = out["pr_total_interactions"].fillna(0)
        # print the sum of issue_total_interactions and pr_total_interactions
        print("Sum of issue_total_interactions:", out["issue_total_interactions"].sum())
        print("Sum of pr_total_interactions:", out["pr_total_interactions"].sum())

        results_path =  cfg.main_folder + '/' + repo_key
        social_technical_metrics_folder = Path(results_path, cfg.social_technical_metrics_folder)
        timeline_combined = Path(social_technical_metrics_folder ,cfg.social_technical_metrics_combined)
        print("Saving labeled timeline with social-technical metrics to:", timeline_combined)
        out.to_csv(timeline_combined, sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, lineterminator='\n')
        st.session_state.labeled_data_with_social_technical_metrics = out

    st.divider()
    st.subheader("Step 6: Predict Inactivity")
    if st.button("Run Inactivity Prediction"):

        social_technical_metrics_folder = Path(cfg.main_folder) / repo_key / cfg.social_technical_metrics_folder
        timeline_combined = Path(social_technical_metrics_folder / cfg.social_technical_metrics_combined)
        #read csv
        print("Reading timeline with social-technical metrics from:", timeline_combined)   
        df = pandas.read_csv(timeline_combined, dtype=str)

        response = build_response(df, N=7)

        data = df.merge(response[["dev", "date", "transition_to_inactive"]], on=["dev", "date"], how="left")

        # load our predictions
        csv_data, confusion = run_prediction_pipeline(df, repo_key)

        data['date'] = pandas.to_datetime(data['date'])
        csv_data['date'] = pandas.to_datetime(csv_data['date'])


        data = data.merge(csv_data[["dev", "date", "rf_proba"]], on=["dev", "date"], how="left")

        # if rf_proba > 0.7 we make rf_proba_tf 1 and 0 otherwise
        data['rf_preds'] = data['rf_proba'].apply(lambda x: 1 if x > 0.7 else 0)

        view_df(data.head(), name="Data with Predictions")
        # remove columns that are not needed
        # needed columns are dev, date, state, transition_to_inactive, rf_proba, rf_preds
        data = data[["dev", "date", "commits", "prs", "issues", "issue_activity", "pr_activity", "state", "transition_to_inactive", "rf_proba", "rf_preds"]]

        data.to_csv(f"final_data_with_predictions.csv", index=False)

        for dev in st.session_state.tf_devs:
            print("the dev for the plot is", dev)
            plot_dev_timeline(
                data,
                dev=dev,    # or just "someUser"
                lookahead_days=14,              # whatever N you trained for
                pred_proba_col="rf_proba",      # or "logit_proba"
                pred_threshold=0.5
            )

if __name__ == "__main__":
    main()

