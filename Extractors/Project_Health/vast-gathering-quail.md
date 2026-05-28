# DISTRAC: Project Readiness Plan
## Context

The user has a working rough-draft research pipeline for analyzing developer inactivity in OSS projects. They've been writing stream-of-consciousness ideas about what needs to be fixed and have asked me to synthesize everything into one clear document: what the project does, how it works end-to-end, and a prioritized list of concrete improvements to get it ready for developer testing.

This plan does NOT implement anything yet. It is a synthesis and roadmap document.

---

## What This Project Is

DISTRAC is a research tool for OSS project maintainers and researchers. It answers: *"If this developer goes inactive for N weeks, what concretely changes in the project?"*

**The three research questions it answers:**
1. What are this developer's inactivity patterns?
2. When will they next go inactive, and for how long?
3. What is the impact of that inactivity on the project?

---

## How the Project Works (End-to-End)

### Stage 1: Data Collection (`Extractor_2_electric_boogaloo/CommitExtractorV3.py`)
- **Input:** GitHub API tokens (`Resources/tokens.csv`), repo list (`Resources/repositories.txt`)
- **Process:** GraphQL + REST API calls to GitHub; parallel extraction with SimpleScheduler
- **Output per repo** in `Organizations/{org}/{repo}/`:
  - `commit_list.csv` — commits with author info
  - `issues.csv` + `issue_activity.csv`
  - `prs_repo.csv` + `prs_comments.csv`
  - `per_file_commits.csv` — per-file changes per commit
  - `repo_tree.csv` — file tree snapshot
  - `data_cursor.json` — tracks extraction progress
- **Tracks progress** so re-runs only collect new data

### Stage 2: Processing Pipeline (`Extractors/DemoAppV2.2.py`)
- **Input:** Raw CSVs from Stage 1
- **Process:**
  1. `build_dev_names()` → canonical dev identity from author_id/login/name/email
  2. `KnowledgeDistribution.main()` → TF (Truck Factor) devs list + DOE file expertise scores
  3. `SocialTechnicalNetwork.main()` → collaboration graph, edge list, node metrics
  4. `timeline()` → daily activity aggregation per dev
  5. `label_developers_activity()` → ACTIVE/NON_CODING/INACTIVE/GONE states per dev per day
  6. `identifyBreaks()` → Q3+3*IQR sliding window break detection
  7. `build_response()` → N-day lookahead labels for ML training
  8. `run_prediction_pipeline()` → LSTM binary classifier (predicts transition to inactivity)
- **Output per repo:**
  - `Results/all_users_labeled_timeline.csv`
  - `Results/dev_names.csv`
  - `KnowledgeDistribution/doe.csv` + `truck_factor.json`
  - `SocialTechnicalNetwork/edge_list.csv` + `social_technical_metrics.csv`
  - `ProjectHealth/project_health.json`
  - `model_folder/test_df.pkl` — LSTM predictions with `prob_1`

### Stage 3: Dashboard (`Extractors/Dashboard.py`)
- **Input:** All Stage 2 outputs + `Resources/repo_split.csv` (test repos only)
- **5 panels:**
  1. **Break Risk Overview** — table of all TF devs with `prob_1` risk score; user selects a dev
  2. **Break Simulation** — network impact if selected dev departs
  3. **Social-Technical Network** — D3 interactive collaboration graph
  4. **Knowledge Distribution** — file expertise matrix (DOE scores)
  5. **Project Health** — activity projection charts (commits/PRs/issues during absence)

---

## All Ideas Extracted and Organized

The following is a synthesized, deduplicated working list from all the "new idea" restarts in the user's message:

---

### THEME 1: Data Pipeline Stability & Standardization

**The core insight:** "If we know how we are going to use the data then we can implement the checks we need at data collection."

#### Concrete tasks:

**1.1 — Enhance the Repos Tracking CSV** (`Resources/repo_split.csv`)
Add columns to track the full lifecycle of each repo:
- `org`, `repo` — already exists
- `split` — train/test, already exists
- `collection_status` — not_started / in_progress / complete / error
- `collected_up_to` — date string (e.g., "2025-12-31") — what date range we collected
- `last_collected_at` — timestamp of last extraction run
- `processing_status` — not_started / labeled / modeled / complete
- `tf_count` — truck factor count (filled after processing)
- `dev_count` — number of TF devs (filled after processing)
- `notes` — any manual flags

**1.2 — Define a global data cutoff date**
Current: `Settings.py` has `data_collection_date = "2025-08-26"` but it's not enforced in collection.
Fix: Add `DATA_CUTOFF_DATE = "2025-12-31"` to Settings.py and filter all raw CSVs to `created_at <= DATA_CUTOFF_DATE` at load time (in `load_users_activity()` and `label_developers_activity()`).

**1.3 — Remove bot users at data collection**
`SocialTechnicalNetwork.py` already has `_is_bot(login)`. Apply this filter in `CommitExtractorV3.py` before writing rows to CSV, not just in the network analysis.

**1.4 — Data Validation Pipeline (new script: `validate_data.py`)**
A script that, for each repo in the tracking CSV, checks:
- All 6 required raw CSVs exist and are non-empty
- Column schemas match expected (using `Utilities.ensure_csv()` logic)
- Date range is consistent (`created_at` within expected window)
- No all-NaN author columns
- Writes a `data_status.json` per repo with pass/fail per check
- Prints a summary table across all repos

**1.5 — Auto-generated data description file**
After validation passes, write `Organizations/{org}/{repo}/DATA_DESCRIPTION.md` with:
- Row counts per CSV
- Date range covered
- List of unique developers found
- TF dev list (if processed)
- Any validation warnings

---

### THEME 2: Code Integration & Choke Points

**The core insight:** "We have 4 main files that are working together... I want to read and understand the flow of the code to see any problems."

#### Known choke points (from code analysis):

**2.1 — `load_users_activity()` iterates ALL repos in an org**
- Bug: Line 277 loops over all subdirectories in the org folder, concatenating data from all repos, not just the target repo.
- Fix: Add `if repo_path.name != target_repo: continue` filter.

**2.2 — TruckFactor data loading crashes (NameError)**
- Bug: Line 383-386 in `DemoAppV2.2.py` references `_tf_data["tf_list"]` but `_tf_data` is never initialized when `tf_devs=None`.
- Fix: Initialize `_tf_data = {}` before the conditional block.

**2.3 — Probability threshold inconsistency**
- Bug: `thr_rf=0.6` in `make_confusion_mats()` but `prob_threshold=0.3` in prediction display.
- Fix: Centralize in Settings.py as `cfg.prob_threshold = 0.5` and use everywhere.

**2.4 — `test_df.pkl`/`test_df.csv` path discovery is fragile**
- Current: `_find_test_df()` searches for either `.pkl` or `.csv` in model_folder.
- Fix: Standardize to always use `.csv`; save as CSV in pipeline.

**2.5 — Impact column is "TBD"**
- Line 529: `"Impact": "TBD"` — never calculated.
- Fix: Use `get_impact_level()` which already exists; it needs `risk_score` from DOE data and `on_truck_factor` flag. Wire this up in `build_risk_table()`.

**2.6 — `data_collection_date` in Settings.py is hardcoded to "2025-08-26"**
- This gets stale. Replace with a derivable value or explicit config users must set.

**2.7 — One data loader, one data saver per section**
Consolidate the repeated CSV loading patterns. Create:
- `load_repo_data(repo_full_name) → dict[str, pd.DataFrame]` — one function that loads all raw CSVs for a repo
- `load_processed_data(repo_full_name) → dict` — loads all Stage 2 outputs

---

### THEME 3: Dashboard UX — Input & Interaction Redesign

**The core insight:** "I want to rethink the way that we are having users interact with our dashboard."

#### Concrete components to build:

**3.1 — New Date + Window Selector (HTML component)**
Replace the current `select_slider` with a self-contained HTML component that:
- Shows a timeline with all commits plotted as a histogram (the "background")
- Has a draggable **current date** point (primary selector)
- Has a draggable **window start** point that follows the current date by default
- When the window point is clicked/dragged, it sets the window length independently
- Displays: current date, window length in days, date range label
- Outputs: `selected_date`, `window_days` to Python session state
- **Visual analogy:** stock trading chart with a time window selector

**3.2 — Developer Selection Table (replaces selectbox)**
Replace the simple selectbox in the Break Risk Overview with a styled interactive table that shows:
- Developer display name
- Role (Maintainer/Collaborator/Bridge/Peripheral)
- Transition probability % (from `prob_1`)
- Risk level badge (High/Medium/Low)
- Next break length estimate
- TF indicator (✓ if on truck factor)
- Click row to select developer

**3.3 — Unified Variable Selector (combining 3.1 + 3.2)**
Design goal: Users can rapidly explore developer + date + window combinations.
- Auto-populate defaults on load: first test repo, today's date, highest-risk developer
- Date selector at top
- Developer table below
- Selecting a dev updates all 5 panels immediately

---

### THEME 4: Project Health Panel

**The core insight:** The panel is planned (see `DISTRAC_project_health_plan.md`) but not yet integrated.

**4.1 — Build Iteration 1 of Project Health**
Per the plan file at `Extractors/Project_Health/DISTRAC_project_health_plan.md`:
- Goal: show developer's avg weekly commits/PRs/issues + project gap during absence
- Components: 3 metric cards + bar charts with absence projection + "% of repo total"
- Mock data first (fictional "Alex Chen" on fictional repo)
- Output: `project_health_v1.html`
- Use Chart.js from cdnjs

**4.2 — Connect `project_health.json` generator**
Currently `_load_ph_data()` in Dashboard.py reads `ProjectHealth/project_health.json`.
Need to verify `ProjectHealthMetrics.py` exists and generates this file correctly from `all_users_labeled_timeline.csv`.

---

### THEME 5: Developer Guide Document

**The core insight:** "We need a way that describes how to run all our scripts in one place and what we need to do to get the project ready for developers."

**5.1 — Write `DISTRAC_Developer_Guide.md`**
This document already exists as a file in the repo but content is unknown.
It should cover:
1. **Setup:** Python environment, conda `osslab`, GitHub tokens
2. **Step-by-step pipeline:**
   - Step 0: Add repos to `Resources/repositories.txt` + `Resources/repo_split.csv`
   - Step 1: Run data collection (`CommitExtractorV3.py`) — explain this is slow, resume-safe
   - Step 2: Run processing pipeline (`DemoAppV2.2.py`) — what buttons to click, what order
   - Step 3: Run dashboard (`Dashboard.py`) — `streamlit run`
3. **Data directory structure** — explain `Organizations/{org}/{repo}/` layout
4. **Expected outputs** — what files should exist after each step
5. **Common errors** — rate limits, missing tokens, schema mismatches
6. **Data cutoff** — how to set and why

---

## Prioritized Implementation Order

### Phase 1 — Foundation (do before anything else)
These are blocking: everything else breaks without them.
1. **Fix 2.1** — `load_users_activity()` org iteration bug
2. **Fix 2.2** — TruckFactor NameError crash
3. **Fix 2.3** — Centralize prob_threshold in Settings.py
4. **Task 1.1** — Enhance repos tracking CSV (add status columns)
5. **Task 1.2** — Define and enforce DATA_CUTOFF_DATE

### Phase 2 — Stability
Make the pipeline reproducible and verifiable.
6. **Task 1.4** — Data validation script (`validate_data.py`)
7. **Task 1.5** — Auto-generate `DATA_DESCRIPTION.md` per repo
8. **Fix 2.5** — Wire up Impact column in risk table
9. **Fix 2.7** — Consolidate data loaders

### Phase 3 — Dashboard UX
Make the app usable for testing.
10. **Task 3.2** — Developer selection table
11. **Task 3.1** — Date + window timeline selector
12. **Task 3.3** — Auto-populate defaults on load

### Phase 4 — Features
Complete the missing panel.
13. **Task 4.1** — Project Health panel iteration 1 (mock data)
14. **Task 4.2** — Connect `project_health.json` generator
15. **Task 5.1** — Write full `DISTRAC_Developer_Guide.md`

---

## Files to Create or Modify

| File | Action | Priority |
|------|--------|----------|
| `Extractors/Dashboard.py` | Fix 2.1, 2.2, 2.3, 2.5, 2.7; add 3.2, 3.3 | P1 |
| `Settings.py` | Add DATA_CUTOFF_DATE, prob_threshold | P1 |
| `Resources/repo_split.csv` | Add new tracking columns | P1 |
| `validate_data.py` (new) | Data validation pipeline | P2 |
| `Extractors/Dashboard.py` | Date+window selector component | P3 |
| `Extractors/Project_Health/project_health_v1.html` (new) | PH panel iteration 1 | P4 |
| `DISTRAC_Developer_Guide.md` | Full developer guide | P4 |

---

## How to Verify Everything Works

1. Run `python validate_data.py` — all test repos should pass all checks
2. Run `streamlit run Extractors/Dashboard.py` — dashboard loads without errors
3. Select any test repo → Break Risk Overview table renders with real risk scores (not "TBD")
4. Select a developer → all 4 existing panels render without errors
5. Project Health panel renders with chart (even on mock data)
6. Date selector outputs correct `selected_date` and `window_days` to Python

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `Extractor_2_electric_boogaloo/CommitExtractorV3.py` | Stage 1: GitHub data collection |
| `Extractors/DemoAppV2.2.py` | Stage 2: Processing + ML pipeline |
| `Extractors/Dashboard.py` | Stage 3: Interactive dashboard |
| `Extractors/KnowledgeDistribution.py` | TF + DOE computation |
| `Extractors/SocialTechnicalNetwork.py` | Network graph + simulation |
| `Settings.py` | All config constants |
| `Utilities.py` | Shared helpers (CSV I/O, tokens, time) |
| `Resources/repo_split.csv` | Repo registry (train/test) |
| `Resources/tokens.csv` | GitHub API tokens |
| `Extractors/Project_Health/DISTRAC_project_health_plan.md` | PH panel build plan |
| `DISTRAC_Developer_Guide.md` | (To be written) How to run everything |
