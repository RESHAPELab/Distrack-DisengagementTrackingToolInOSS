---
title: Distrack
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Distrack — Developer Disengagement Tracking Tool

[![DOI](https://zenodo.org/badge/183011533.svg)](https://zenodo.org/badge/latestdoi/183011533)

Distrack is a research tool for extracting, analyzing, and visualizing developer activity in open-source software (OSS) projects on GitHub. It identifies disengagement patterns, computes project health metrics, and trains predictive models to flag developers at risk of leaving a project.

---

## (STOP!!) Which path are you on?

There are two ways into this project depending on what you want out of it. If you're reviewing the project and just want to see it work, the short path is all you need.

| | **Try it** (Short & Fast) | **Reproduce it** (Long) |
|---|---|---|
| **Goal** | See the tool running on real data | Rebuild the full study from scratch |
| **Who** | Reviewers, users evaluating the tool | Developers extending or replicating the work |
| **Data** | Pre-computed snapshot from Zenodo | You collect & process all 57 repos yourself |
| **GitHub tokens** | Not needed | Required |
| **Disk** | A few GB (only the repos you browse) | ~13 GB full dataset |
| **Where** | [Quick Start](#quick-start--30-minute-guide) — below | [`REPRODUCTION.md`](REPRODUCTION.md) — full walkthrough |

In the short path we will just be setting up the dashboard with data we download while the long path uses the backend to recollect and process data for the front end insted of downloading. 
---

## IMPORTANT parts

In this Replication package there are a few spots that are the most important. 

(1) The front end application
(2)

## What is Distrack?

Distrack is many diffrent parts to a research project. There are two main parts to it a backend data pipeline and a font end dashboard application. Distrack answers one question about an OSS project: **which contributors are drifting away, and how soon might they leave?** You point it at a GitHub project (or load one of the pre-analyzed projects), and it gives you:

- **Per-developer activity timelines** — when each contributor was active, slowing down, or gone and a forecast of who is likely to stop contributing.
- **Disengagement simulations** — Simulation of the devlopers effect on a repo giveing users insights on how a devloper departure amy effect it.  

Behind the scense distrack is a four-stage pipeline. Each stage writes its output to disk so the next stage can pick it up, and the dashboard reads the final results.

```
  GitHub API
      │
      ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. COLLECT   │──▶│ 2. ANALYZE   │──▶│ 3. MODEL     │──▶│ 4. SERVE      │
│ commits, PRs │   │ truck factor │   │ break detect │   │ FastAPI +     │
│ issues, tree │   │ STN, health  │   │ + prediction │   │ Streamlit UI  │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
  Data Collection/    Analysis/          Dashboard/         Dashboard/
                                         DemoAppV2.*.py      app.py, *_api.py
```

1. **Collect** (Backend: `Data Collection/`) — pulls commits, pull requests, issues, and the repo file tree from the GitHub API for each project in `Resources/repositories.txt`.
2. **Analyze** (Backend: `Analysis/`) — computes knowledge distribution (truck factor), social-technical network metrics, and project health indicators.
3. **Model** (Backend: `Dashboard/DemoAppV2.*.py`) —  builds per-developer state timelines, detects activity breaks, and predicts disengagement risk. Two model variants ship with the project:
   - **`DemoAppV2.2.py`** — classification-based pipeline with a full evaluation suite (baselines, PR-AUC, window recall).
   - **`DemoAppV2.3.py`** — survival-analysis pipeline (Cox time-varying model) producing conditional survival probabilities and risk bands.
4. **Serve** (Frontend: `Dashboard/`) — Front end `distrac_writer.py` writes results as parquet, `distrac_api.py` (FastAPI) serves them, and `app.py` / `Dashboard.py` (Streamlit) render the UI.

The full reproduction path through all four stages is documented in [`REPRODUCTION.md`](REPRODUCTION.md).

---


## Quick Start — 30-minute guide

Run the dashboard on the pre-computed dataset. No GitHub tokens, no data collection.

### 1. Set up the environment

```bash
conda activate osslab          # or: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Developed on Python 3.10+.

### 2. Get the pre-computed data

Download the snapshot from [Zenodo](https://zenodo.org/badge/latestdoi/183011533) and unzip it into the project root so the layout looks like:

```
Distrack/
└── Organizations/
    └── <org>/<repo>/distrac/   ← parquet files for each analyzed project
```

This is the output of the collection + analysis stages, so you skip straight to browsing results.

### 3. Run the dashboard

```bash
streamlit run Dashboard/app.py
```

Open the **Dashboard** page and pick any of the pre-analyzed repositories. You'll get the activity timelines, disengagement risk scores, and project-health views without running the pipeline.

> Want to analyze a *new* repository, or rebuild the dataset from raw GitHub data? That's the 2-day path — see [`REPRODUCTION.md`](REPRODUCTION.md).

---

## Reproducing the full study — 2-day guide

The full reproduction (collect → analyze → model → serve, including GitHub tokens, rate-limit handling, and the issues you're likely to hit) is in its own document so this README stays readable:

➡️ **[`REPRODUCTION.md`](REPRODUCTION.md)**

At a glance, the four stages are:

1. **Collect** raw GitHub data for the 57 projects in `Resources/repositories.txt` (requires tokens; ~13 GB output).
2. **Analyze** each project to compute truck factor, social-technical network, and health metrics.
3. **Model** developer states and disengagement risk via the `DemoAppV2.*.py` pipeline.
4. **Serve** the results through the FastAPI backend and Streamlit dashboard.

---

## Project Structure

```
Distrack/
├── Settings.py              # global configuration (paths, filenames, GitHub settings)
├── Utilities.py             # shared GitHub API helpers and rate-limit handling
├── reset_issue_collection.py# utility to reset issue collection state
│
├── Data Collection/         # GitHub data extraction
│   ├── CommitExtractorV3.py         # main extractor: commits, PRs, issues, repo tree
│   ├── PullRequestsExtractor.py     # legacy PR extraction
│   ├── NonMergedCommitsExtractor.py # non-merged PR commits
│   ├── collect_developer_profiles.py# GitHub developer profile data
│   └── SimpleScheduler.py           # parallel task scheduler for extraction
│
├── Analysis/                # metric computation modules
│   ├── KnowledgeDistribution.py     # truck factor and knowledge concentration
│   ├── ProjectHealthMetrics.py      # project health indicators
│   ├── SocialTechnicalNetwork.py    # social-technical network metrics
│   └── SocialTechnicalNetworkV2.py  # enhanced STN analysis
│
├── Dashboard/               # Streamlit UI and FastAPI backend
│   ├── app.py               # entry point (Streamlit multi-page navigation)
│   ├── DemoAppV2.2.py       # pipeline + prediction UI (classification + eval suite)
│   ├── DemoAppV2.3.py       # pipeline + prediction UI (survival analysis)
│   ├── Dashboard.py         # results viewer (reads parquet output)
│   ├── distrac_api.py       # FastAPI server for dashboard data
│   └── distrac_writer.py    # writes analysis output as parquet files
│
├── PredictionModel/         # trained ML model (model.joblib)
├── Resources/               # project config (tokens gitignored)
│   └── repositories.txt     # list of OSS projects analyzed (org/repo format)
├── Docs/                    # research papers
├── static/                  # frontend assets (index.html)
├── requirements.txt
├── REPRODUCTION.md          # full collect → analyze → model → serve walkthrough
└── README.md
```

---

## Output

The pipeline writes structured parquet files to `Organizations/<org>/<repo>/distrac/`:

| File | Contents |
|---|---|
| `developers.parquet` | per-developer activity timelines |
| `break_predictions.parquet` | disengagement risk scores |
| `truck_factor.parquet` | knowledge ownership metrics |
| `knowledge_doe.parquet` | degree of expertise distribution |
| `stn_metrics.parquet` | social-technical network node metrics |
| `stn_edges.parquet` | social-technical network edges |
| `activity_weekly.parquet` | weekly aggregated activity |
| `project_health.json` | overall project health summary |
| `manifest.json` | signals pipeline completion |

---

## Data

The `Organizations/` folder (13 GB, gitignored) contains the full extracted dataset for the 57 projects in the study. To regenerate it, follow [`REPRODUCTION.md`](REPRODUCTION.md). A pre-computed snapshot is available on [Zenodo](https://zenodo.org/badge/latestdoi/183011533).

---
## Research Context

I included a Docs file that shows off all of the work proceeding the project. Its in chronological order and shows off the progress of the project. The project started with Igor reading "Will you come back to contribute" from fabio where we get the state labler. Igor proposed a project to NSF in 2023 where we use this new labeleling algorithum to make predictions on the state of a user. The proposal was rejected for the 120,000$ they asked for. Then a NAU internal grant called the TRIF Faculty Research and Creative Activity Support Grants program (FGS) at Northern Arizona University for around 20-25K. Work started in Decmenber of 2024 and an undergraduate assistant started in Jan of 2025. The first inital results paper was made 1 year later and submitted on Jan 5th 2026 to the 48th International Conference on Software Engineering for the ACM Student Research Competition reciving 2nd place. Then in May of 2026 a Tool demonstartion paper was submited to International Conference on Software Maintenance and Evolution. This is where the project stands with the papers and publication traces below.

- *Will you come back to contribute? Investigating the inactivity of OSS developers* (2021) — [`Docs/1. 2021_2103.04656v3.pdf`](Docs/1.%202021_2103.04656v3.pdf)
- *Sustainability Breaks in OSS* — NSF Small Grant (2023) — [`Docs/2. 2023_NSF__Small__Sustainability_Breaks.pdf`](Docs/2.%202023_NSF__Small__Sustainability_Breaks.pdf)
- *2024 NAU Grant_Sustainability_Breaks* - [`Docs/3. 2024_NAU_Grant_Sustainability_Breaks.pdf`](Docs\3. 2024_NAU_Grant_Sustainability_Breaks.pdf)
- *Early Forecasting of Developer Inactivity in Open Source Projects*— ICSE 2026 — [`Docs\4. 2025_SRC2026_Sam.pdf`](Docs\4. 2025_SRC2026_Sam.pdf)
- *DisTrac: Disengagement Tracking Tool in Open Source Projects* — ICSME 2026 — [`Docs/5. 2026_ICSME2026_Distrac_Sam.pdf`](Docs/5.%202026_ICSME2026_Distrac_Sam.pdf)

---
## Citation

If you use Distrack in your research, please cite:

```bibtex
@inproceedings{distrack2026,
  title     = {Distrac: A Disengagement Tracking Tool for OSS},
  author    = {Wu, Samuel and ...},
  booktitle = {Proceedings of ICSME 2026},
  year      = {2026}
}
```
