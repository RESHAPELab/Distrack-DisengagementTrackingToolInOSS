### GitHub Settings
items_per_page = 100  # The number of results in each page of the GitHub results. Max: 100
tokens_file = "../Resources/tokens.csv"
repos_file  = "../Resources/repositories.txt"
repo_split_file = "../Resources/repo_split.csv"  # defines train/test repo assignments


def load_repo_split(split: str | None = None) -> list[tuple[str, str]]:
    """
    Read Resources/repo_split.csv and return a list of (org, repo) tuples.

    Parameters
    ----------
    split : "train", "test", or None (returns all rows)

    The CSV has three columns: org, repo, split
    Edit Resources/repo_split.csv to change which repos are in each group.
    """
    import csv
    from pathlib import Path
    path = Path(__file__).resolve().parent / "Resources" / "repo_split.csv"
    results = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split is None or row["split"].strip().lower() == split.lower():
                results.append((row["org"].strip(), row["repo"].strip()))
    return results

 # The relative path of the file containing the list of the github tokens
from pathlib import Path as _Path
_repo_root = _Path(__file__).resolve().parent

main_file_path = str(_repo_root)

#All folders

main_folder = str(_repo_root / "Organizations")
collection_folder = "/RawData"
TF_developers_folder = "TruckFactor"
user_timelines_folder = "/UserTimelines"
timeline_folder = "/Timelines"
labeled_timeline_folder = "/Labeled_Timelines"
social_network_metrics_folder = "SocialTechnicalNetwork"
project_health_metrics_folder = "Project_Health_Metrics"
project_health_folder         = "ProjectHealth"
project_health_file           = "project_health.json"


timeline_file = "timeline_combined.csv"
social_technical_metrics_combined = "timeline_combined.csv"

dev_health_metrics_folder = "DevHealthMetrics"

photo_folder= "Images"


# All files

next_page = "next_page_PR.txt"
last_page ="last_flushed_page_PR.txt"
next_page_commits = "next_page_commits.txt"  # The file where the next page of the commits will be stored
last_page_commits = "last_flushed_page_commits.txt"  # The file where the last flushed page of the commits will be stored
next_page_issues = "next_page_issues.txt"  # The file where the next page of the issues will be stored
last_page_issues = "last_flushed_page_issues.txt"  # The file where the last flushed page of the issues will be stored

per_file_commits_path = "per_file_commits.csv"  # The file where the per-file commits will be stored
repo_tree_path = "repo_tree.csv"  # The file where the repo tree snapshot will be stored

data_cursor = "data_cursor.json"  # The file where the next page of the issues will be stored

excluded_csv = "excluded_data_points.csv"  # The file where the excluded repos will be listed

author_map_file = "author_map.csv"
DOE_file = "DOE.csv"
TF_developers_file = "TruckFactor.csv"

dev_health_metrics = "dev_health_metrics.csv"
folder_summary_df = "folder_summary_df.csv"
dev_summary_df = "dev_summary_df.csv"


### Extraction Settings
data_collection_date = "2025-08-26"  # The max date to consider for the commits and activities extraction
main_folder = str(_repo_root / "Organizations")  # The main folder where results will be archived
logs_folder = str(_repo_root / "logs")  # The folder where the logs will be archived
results_folder = "/Results"  # The folder where the results will be archived
temp_data_folder = "../temp_data"  # The folder where temporary data will be stored

model_path = "../PredictionModel/model.joblib"

supported_modes = ['tf', 'a80', 'a80mod', 'a80api']

TF_report_folder = "../Organizations/.tf_cache"  # The folder where the TF/core developers are archived
truck_factor_file = "truck_factor.json" # The file where the TF/core developers are listed as <name;login>ù
## WARNING: The correct path to save the <TF_developers_file> is <TF_report_folder>/<organization/mainRepo>/<TF_developers_file>

A80_report_folder = "../A80_Results"  # The folder where the TF/core developers are archived
A80_developers_file = "A80_devs.csv" # The file where the TF/core developers are listed as <name;login>

modTh = 5

pauses_list_file_name = "coding_pauses.csv"  # The file where the lists of devs' pauses durations will be archived
pauses_dates_file_name = "pauses_coredevs.csv"  # The file where the lists of devs' pauses boundary dates will be archived

model_folder = "model_folder"

commit_history_table_file_name = "commit_history_table.csv"
coding_history_table_file_name = "coding_history_table.csv"
issue_comments_list_file_name = "issues_comments_repo.csv"
issue_events_list_file_name = "issues_events_repo.csv"
issue_list_file_name = "issues_repo.csv"
issue_timeline_file_name = "issues_timeline_repo.csv"
PR_list_file_name= "prs_repo.csv"
prs_comments_csv = "prs_comments.csv"



issue_list_file_name = "issues.csv"  
issue_activity_file_name = "issue_activity.csv"  
PR_list_file_name= "prs_repo.csv" 
prs_comments_csv = "prs_comments.csv"  
commit_list_file_name = "commit_list.csv"  
#core_commit_coverage = 0.8 

### Files Settings
CSV_separator = ","  # Character for cell separation in the used files
CSV_missing = "NA"  # Character for the missing values in the used files

### Breaks Identification
timeline_folder_name = 'Timelines'  # The folder where the other repo activities will be archived
breaks_folder_name = 'Dev_Breaks'  # The folder where the breaks list for each developer will be archived
sliding_window_size = 90  # The size in days of the sliding window
shift = 7  # The number of days to shift the sliding window of

### Breaks Labeling
labeled_breaks_folder_name = breaks_folder_name + '/Labeled_Breaks'  # The folder where the labeled breaks list for each developer will be archived
gone_threshold = 365  # Threshold to label a break as 'GONE'

### Statistics
chains_folder_name = 'Chains'  # The folder where the %age of the transitions for each organization will be archived

### CONSTANTS
A = 'ACTIVE'  # Label of the Active status: Developers contribute commits
NC = 'NON_CODING'  # Label of the Non-coding status: Developers do not contribute commits, but show other activity signals
I = 'INACTIVE'  # Label of the Inactive status: Developers do not show any activity signal
G = 'GONE'  # Label of the Gone status: Developers have been Inactive for longer than <gone_threshold>



#key_folders = ['Activities_Plots', 'Dead&Resurrected_Users', 'Hibernated&Unfrozen_Users', 'Sleeping&Awaken_Users', 'DevStats_Plots', 'Longer_Breaks']

#commit page storage


social_technical_metrics_file = "social_technical_metrics.csv"
social_technical_metrics_folder = "SocialTechnicalNetwork"
social_technical_edge_list_file = "edge_list.csv"
social_technical_daily_interactions_file = "daily_interactions.csv"


social_technical_nodes_file = "stn_nodes.csv"
social_technical_links_file = "stn_links.csv"
social_technical_html_file  = "stn_network.html"

# Hugging Face deployment — set via environment variables in production
import os as _os
USE_HF_STORAGE  = _os.environ.get("USE_HF_STORAGE",  "false").lower() == "true"
HF_DATASET_REPO = _os.environ.get("HF_DATASET_REPO", "Coupur/distrack-data")