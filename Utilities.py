### IMPORT SECTION
import os
import random

import numpy
import pandas
from datetime import datetime, timezone
from github import Github, GithubException
import time
from requests.exceptions import Timeout
from tqdm import tqdm               # pip install tqdm

import Settings as cfg
from pathlib import Path
from typing import Sequence, Union
import csv
import json
import portalocker



# ### MODULE FUNCTIONS
# def waitRateLimit(ghub):
#     exception_thrown = True
#     while (exception_thrown):
#         exception_thrown = False
#         try:
#             search_limit = ghub.get_rate_limit().search.remaining
#             core_limit = ghub.get_rate_limit().core.remaining

#             S_reset = ghub.get_rate_limit().search.reset
#             ttw = 0
#             now = datetime.now(timezone.utc)
#             if (search_limit <= 5):
#                 S_reset = ghub.get_rate_limit().search.reset
#                 #ttw = (S_reset - now).total_seconds() + 300
#                 ttw = 3600
#                 print('Waiting {} for limit reset', ttw)
#             if (core_limit <= 500):
#                 C_reset = ghub.get_rate_limit().core.reset
#                 #ttw = (C_reset - now).total_seconds() + 300
#                 ttw = 3600
#                 print('Waiting {} for limit reset', ttw)
#             time.sleep(ttw)### originally ttw. Setting 1 hour for testing exceptions
#         except GithubException as ghe:
#             print('Exception Occurred While Getting Rate Limits: Github', ghe)
#             exception_thrown = True
#             pass
#         except Timeout as toe:
#             print('Exception Occurred While Getting Rate Limits: Timeout', toe)
#             exception_thrown = True
#             pass
#         except:
#             print('Execution Interrupted: Unknown Reason')
#             raise

# modified by @bateman to handle multiple tokens and reduce the waiting time

    

import random
import time

def _norm_time(s):
    return pandas.to_datetime(s, utc=True, errors="coerce")

def getRandomToken():
    """Return a random token from the tokens list"""
    tokensList, tokens_df  = getTokensList()
    
    token = random.choice(tokensList)
    return token


def getSpisificToken(id):
    """Return a specific token from the tokens list by its index"""
    tokensList, tokens_df = getTokensList()
    
    if id < 0 or id >= len(tokensList):
        raise IndexError("Token index out of range.")
    
    token = tokensList[id]
    return token


def getTokensList():
    """Return the tokens list and the tokens dataframe."""
    tokensList = []
    # Read the tokens from the file
    filename = cfg.tokens_file
    tokens_df = pandas.read_csv(Path(filename), sep=cfg.CSV_separator, na_values=cfg.CSV_missing) 
    # Check if the tokens file is empty
    if tokens_df.empty:
        raise ValueError("The tokens file is empty. Please add valid tokens.")
    # Convert the tokens column to a list
    tokensList = tokens_df["token"].tolist()
    # Remove any leading or trailing whitespace characters from each token
    
    return tokensList, tokens_df




def updateIsEmpty(token: str, tokens_df: pandas.DataFrame):
    """
    Flip the `is_empty` flag for one token and flush to disk.
    """
    ghub = Github(token)
    rl=ghub.get_rate_limit()
    if rl.core.remaining < 50:
        #set the spesific tokens is_empty flag to True
        tokens_df.loc[tokens_df["token"] == token, "is_empty"] = True
        # Save the updated DataFrame back to the CSV file
        return tokens_df, True
    else:
        tokens_df.loc[tokens_df["token"] == token, "is_empty"] = False 
        #flush the changes to disk       
        # Save the updated DataFrame back to the CSV file
        return tokens_df, False

def _mask(tok: str) -> str:
    if not tok: return "<none>"
    t = tok.strip()
    return (t[:4] + "…" + t[-4:]) if len(t) > 8 else "<short>"

class BadToken(Exception):
    pass

def getSameToken(ghub, same_token, position=0):
    """
    Reuse *same* token unless Core limit < threshold.
    If below, wait for the official reset moment (+ `safety_margin` s)
    and show a live countdown with tqdm.
    
    Returns
    -------
    (ghub, same_token, core_remaining, reset_datetime)
    """
    try:
        rl      = ghub.get_rate_limit().core
        core    = rl.remaining
        reset_t = rl.reset   # tz-aware UTC datetime
        if core >= 100:
            # Plenty of calls left – just return.
            return ghub, same_token, core, reset_t
    except GithubException as e:
        if getattr(e, "status", None) == 401:
                print(f"[GITHUB 401] token_idx={position} token={_mask(same_token)} is invalid")
                raise BadToken(f"401 for token_idx={position} token={_mask(same_token)}") from e
        raise

    # -----------------------------------------------------------------
    #  We need to wait
    # -----------------------------------------------------------------
    # Make sure *now* is in the same timezone as `reset_t`
    now = datetime.now(tz=reset_t.tzinfo or timezone.utc)
    wait_sec = int((reset_t - now).total_seconds()) + 2
    wait_min = int(wait_sec / 60)

    if wait_sec < 0:                      # already past, avoid negative sleep
        wait_sec = 0

    tqdm.write(f"⏳ Token {position} rate-limited — waiting {wait_min} min")

    try:
        for _ in tqdm(range(wait_sec),
                      position=position,
                      desc=f"⏳ reset tok-{position}",
                      leave=False,
                      dynamic_ncols=True):
            time.sleep(1)
    except KeyboardInterrupt:
        tqdm.write("Interrupted by user, exiting...")
        exit(0)
    tqdm.write(f"✅ Token {position} reset complete")
    # Re-authenticate (optional but safest)
    ghub = Github(same_token) if same_token else ghub

    # Fetch new quota
    rl = ghub.get_rate_limit().core
    return ghub, same_token, rl.remaining, rl.reset


def getNextToken( ghub= None, last_token=None):
    """Return the next avalbe token from the tokens list and if not find the samlles time reset."""
    #the same df with out last token
    tokensList, tokens_df = getTokensList()

    tokens_df = tokens_df[tokens_df['token'] != last_token]
    
    #update the that are empty
    for token in tokensList:
        tokens_df, t_f = updateIsEmpty(token, tokens_df)  # mark each token as empty
        if t_f == False:
            
            core_limit = ghub.get_rate_limit().core.remaining
            reset_time = ghub.get_rate_limit().core.reset
            return ghub, last_token, core_limit, reset_time
    
    lowest_time = Github(last_token).get_rate_limit().core.reset # Initialize with the maximum possible timestamp

    for token in tokensList:
        ghub = Github(token)
        reset = ghub.get_rate_limit().core.reset
        #make reset not a timezone aware timestamp

        #add time values to a times_list
        if lowest_time > reset:
            lowest_time = reset
            next_token = token  
    
    ghub = Github(next_token)
    core_limit = ghub.get_rate_limit().core.remaining
    reset_time = ghub.get_rate_limit().core.reset
    

    #find how long it is until the next reset
    time_to_wait = lowest_time - pandas.Timestamp.now(tz="utc")
    print(f"Waiting for {time_to_wait}")
    #wait until the next reset
    time_to_wait = time_to_wait.total_seconds()
    if time_to_wait > 0:
        time.sleep(time_to_wait + 1)
    
    #change reset_time to a timezone aware timestamp
    reset_time = pandas.Timestamp(lowest_time, tz="mtz")
    return ghub, next_token, core_limit, reset_time


# Utilities.py  (replace the old add)

def ensure_csv(path: Union[str, Path],
               columns: Sequence[str],
               sep: str = ",",
               encoding: str = "utf-8",
               create_parents: bool = True) -> Path:
    """
    Ensure a CSV exists at `path` with exactly the header `columns` (in order).
    - If file doesn't exist: create it atomically and write the header.
    - If file exists but is empty: write the header.
    - If file exists with data: validate header; raise if it doesn't match.

    Returns the Path to the CSV.
    """
    p = Path(path)
    if create_parents:
        p.parent.mkdir(parents=True, exist_ok=True)
    # change sep to string if it is not a string
    sep = str(sep)
    # 1) Create atomically with header if it doesn't exist
    try:
        with p.open("x", encoding=encoding, newline="") as f:  # exclusive create
            writer = csv.writer(f, delimiter=sep)
            writer.writerow(list(columns))
        return p
    except FileExistsError:
        pass  # already exists

    # 2) If exists but empty -> write header
    if p.stat().st_size == 0:
        with p.open("w", encoding=encoding, newline="") as f:
            writer = csv.writer(f, delimiter=sep)
            writer.writerow(list(columns))
        return p

    # 3) Validate existing header
    with p.open("r", encoding="utf-8-sig", newline="") as f:  # utf-8-sig handles potential BOM
        reader = csv.reader(f, delimiter=sep)
        try:
            existing = next(reader)
        except StopIteration:
            existing = []

    expected = list(columns)
    if existing != expected:
        raise ValueError(
            f"CSV schema mismatch at {p}.\n"
            f"  expected: {expected}\n"
            f"  found   : {existing}"
        )

    return p


def add(df: pandas.DataFrame, row) -> None:
    """
    Mutate `df` in place by appending `row` as a new record,
    preserving column order and dtypes where possible.
    """
    # Build a Series so we can cast element-wise if desired
    new = pandas.Series(row, index=df.columns)

    # Optional dtype coercion
    for col in df.columns:
        try:
            new[col] = new[col].astype(df[col].dtype, copy=False)
        except (ValueError, TypeError, AttributeError):
            pass          # keep whatever works

    # Append in place
    df.loc[len(df)] = new

def daysBetween(d1, d2):
    """Returns the number of days between two dates"""
    from datetime import datetime
    try:
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except:
        d1 = datetime.strftime("%Y-%m-%d")
        d2 = datetime.strftime("%Y-%m-%d")
        return abs((d2 - d1).days)


def getLastPageRead(log_file):
    """Reads the last read results page during the commit extraction"""
    with open(log_file,'r') as reader:
        log_line = reader.readline()
    lp=int(log_line.split(':')[-1])
    return lp

def getLastActivitiesPageRead(log_file):
    """Reads the last read results page during the activities extraction"""
    with open(log_file,'r') as reader:
        last_line = reader.readline()
    split_string=last_line.split(',')
    last_issues_page=int(split_string[0].split(':')[-1])
    last_issue=split_string[1].split(':')[-1]
    last_item_page=int(split_string[2].split(':')[-1])
    return last_issues_page, last_issue, last_item_page

def daterange(start_date, end_date):
    """Iterates on the dates in a range"""
    from datetime import timedelta  # , date
    from datetime import datetime

    # Fix: add %z to the datetime.strptime to avoid the following error:
    # "ValueError: unconverted data remains: +00:00"
    if type(start_date) == str:
        start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S%z')
    if type(end_date) == str:
        end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S%z')

    for n in range(int((end_date - start_date).days + 2)):
        yield start_date + timedelta(n)

def getFarOutThreshold(values):
    ''' NOT USED FUNCTION'''
    q_3rd = numpy.percentile(values,75)
    q_1st = numpy.percentile(values,25)
    iqr = q_3rd-q_1st
    th = q_3rd + 3*iqr
    return th

def parse_TF_results(results_folder, destination_folder):
    # Read TF from the Reports Folder
    tf_report = open(os.path.join(results_folder,'TF_report.txt'), 'r', encoding="utf8")

    record = False
    for line in tf_report:
        if (record == True):
            dev = line.replace('\n', '').split(';')
            if (len(dev) == 3):
                add(TF_devs, dev)
        if (line.startswith('TF = ')):
            TF_header = tf_report.readline().replace('TF authors (', '').replace('):\n', '').split(';')
            TF_devs = pandas.DataFrame(columns=TF_header)
            record = True
    print(TF_devs)
    TF_devs.to_csv(os.path.join(destination_folder,'TF_devs_names.csv'),
                   sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, index=False, quoting=None, lineterminator='\n')

def map_name_login(results_folder, repo_name, destination_folder):
    # Find TF login from Names
    TF_devs = pandas.read_csv(os.path.join(results_folder,'TF_devs_names.csv'), sep=cfg.CSV_separator)
    TF_names_list = TF_devs.Developer.tolist()
    TF_name_login_map = pandas.DataFrame(columns=['name', 'login'])

    token = 'Insert a Valid Token'
    g = Github(token)

    repo = g.get_repo(repo_name)
    contributors = repo.get_contributors()

    for contributor in contributors:
        dev_name = contributor.name
        for name in TF_names_list:
            if (dev_name == name):
                add(TF_name_login_map, [name, contributor.login])
        if (len(TF_name_login_map) == len(TF_names_list)):
            break
    TF_name_login_map.to_csv(os.path.join(destination_folder,'TF_devs.csv'),
                             sep=cfg.CSV_separator, na_rep=cfg.CSV_missing, index=False, quoting=None, lineterminator='\n')

def unmask_TF_routine():
    repos = getReposList()
    for r in repos:
        working_folder = os.path.join(cfg.main_folder, cfg.TF_report_folder,r)
        if ('TF_devs_names.csv' not in os.listdir(working_folder)):
            parse_TF_results(working_folder, working_folder)
        if ('TF_devs.csv' not in os.listdir(working_folder)):
            map_name_login(working_folder,r,working_folder)

def checkTFCoverage(projectName, devs):
    ''' Checks for the given project, the % of TFs in the given devs list '''
    TFfolder = os.path.join(cfg.TF_report_folder.split('/')[1], projectName)
    TFs_file = os.path.join(TFfolder, cfg.TF_developers_file)

    TFs = pandas.read_csv(TFs_file, sep=cfg.CSV_separator)['login'].tolist()
    TFs = [d.lower() for d in TFs]
    num_TFs = len(TFs)

    devs = [d.lower() for d in devs]
    num_devs = len(devs)

    intersection = len(set(TFs).intersection(set(devs)))
    perc = (intersection/num_TFs) * 100

    return num_TFs, num_devs, perc


def getReposList():
    """Return the list of repositories to analyze."""
    repos_list = []
    # Read the repositories from the file
    filename = cfg.repos_file
    with open(Path(filename), 'r') as file:
        for line in file:
            repo = line.strip()
            if repo and not repo.startswith('#'):  # Ignore empty lines and comments
                repos_list.append(repo)
    if not repos_list:
        raise ValueError("The repositories file is empty. Please add valid repository names.")
    return repos_list


def save_json(data, filename, type):
    #saves data at in a json file
    # at a spesific line
    #{
    #"repo": "Rdatatable/data.table",
    #"updated_at": "2024-10-22T10:20:25Z",
    #"streams": {
    #"issues_with_timeline": { "after": None, "processed": 0, "total": None, "complete": False },
    #"prs_with_comments": { "after": None, "processed": 0, "total": None, "complete": False },
    #.....
    # type = "prs_with_comments" or "issues_with_timeline" or "commits"
    # we need to update that streams line only


    import json
    #load the json file
    if type != "data_cursor":
        with open(filename, 'r') as file:
            json_data = json.load(file)
            json_data["streams"][type].update(data)
            json_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        #save the json file
        try:
            with portalocker.Lock(filename, timeout=10, flags=portalocker.LOCK_EX) as file:
                file.seek(0)
                file.truncate()
                json.dump(json_data, file, indent=4)
        except portalocker.exceptions.LockException:
            raise
    else:
        
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)



def append_rows_csv(file_path, rows, sep=cfg.CSV_separator):
    """Append multiple rows to a CSV file."""
    import csv
    if not rows:
        return
    file_exists = os.path.isfile(file_path)
    if file_exists:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            existing_header = next(csv.reader(f, delimiter=sep), None)
        if existing_header and list(rows[0].keys()) != existing_header:
            raise ValueError(
                f"Schema mismatch in {file_path}:\n"
                f"  existing header : {existing_header}\n"
                f"  new row keys    : {list(rows[0].keys())}\n"
                "Delete or migrate the CSV before writing with the new schema."
            )
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=sep)
        if not file_exists:
            # Write header if file does not exist
            writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(row.values())


### MAIN FUNCTION
def main():
    print("Utilities Activated")
#    reposList = getReposList()
#    for gitRepoName in reposList:
#        organization, project = gitRepoName.split('/')
#        devs_file = os.path.join(cfg.A80api_report_folder.split('/')[1], project, cfg.A80api_developers_file)
#        devs = pandas.read_csv(devs_file, sep=cfg.CSV_separator)['login'].tolist()
#
#        TFs, DVS, INTR = checkTFCoverage(project, devs)
#        print('Project: {}, TFs: {}, A80API: {}, TF in A80API: {}'.format(project, TFs, DVS, INTR))

if __name__ == "__main__":
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_FOLDER)

    main()

