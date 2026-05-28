# Reproducing Distrack from Scratch

This is the full, end-to-end guide: collect raw GitHub data, compute all metrics, train the prediction models, and serve the dashboard. Budget about **two days** of mostly-unattended runtime for the 57-project dataset — collection is the long pole and is gated by the GitHub API.

If you only want to *see the tool working*, don't do any of this — use the pre-computed Zenodo snapshot and the 30-minute Quick Start in [`README.md`](README.md) instead.

---

## Overview

Each stage writes to disk, so you can stop and resume between stages. All per-project output lands under `Organizations/<org>/<repo>/`.

---

## Stage 0 — Environment & prerequisites

```bash
conda activate osslab            # or a fresh venv on Python 3.10+
pip install -r requirements.txt
```

You'll need:

- **Python 3.10+**
- **~13 GB free disk** for the full dataset
- **GitHub personal access tokens** (see Stage 1) — effectively required at this scale
- A `Settings.py` you've reviewed (paths, filenames, GitHub settings live here)

---

## Stage 1 — Collect raw GitHub data

### 1a. Add tokens

Without tokens the GitHub API caps you at **60 requests/hour**, which is far too slow for 57 repos. Create `Resources/tokens.txt` with one classic personal access token per line:

```
ghp_yourtoken1here
ghp_yourtoken2here
```

Classic tokens need the **`repo`** and **`read:user`** scopes. More tokens = more throughput, since the collector rotates across them to stay under rate limits.

### 1b. Choose the repositories

`Resources/repositories.txt` holds the 57 projects from the study, one `org/repo` per line. Edit it to change the project set. Start with **one small repo** for your first run to validate the whole pipeline before committing to the full set.

### 1c. Run the extractor

```bash
python "Data Collection/CommitExtractorV3.py"
```

This reads `repositories.txt` and extracts commits, PRs, issues, and the repo file tree for each project, writing to `Organizations/<org>/<repo>/`. `SimpleScheduler.py` handles parallel extraction across projects.

> **Issues you'll hit here**
> - **Rate limiting** — the single biggest slowdown. Add more tokens; expect long pauses when limits are exhausted.
> - **Large repos stalling** — very active projects (thousands of contributors) take a long time on issue/PR collection. Let them run overnight.
> - **Interrupted issue collection** — use `reset_issue_collection.py` to reset issue-collection state for a project before re-running, so you don't double-count or get stuck mid-stream.
> - **Disk** — the full set is ~13 GB. Check space before launching the whole list.

Optionally collect contributor profile metadata:

```bash
python "Data Collection/collect_developer_profiles.py"
```

---

## Stage 2 — Compute metrics

The analysis modules turn raw extracted data into the metrics the model and dashboard consume:

- `Analysis/KnowledgeDistribution.py` — truck factor and knowledge concentration
- `Analysis/SocialTechnicalNetwork.py` / `SocialTechnicalNetworkV2.py` — social-technical network metrics and edges
- `Analysis/ProjectHealthMetrics.py` — project health indicators

The supported way to run these is through the **Pipeline** page of the dashboard:

```bash
streamlit run Dashboard/app.py
```

Open **Pipeline**, select a repository, and run the analysis steps. The pipeline is modular with **per-step caching and overwrite toggles**, so you can re-run a single stage without redoing everything.

> **Issues you'll hit here**
> - **Overwrite toggle vs. cache** — caching keys off existing output files. If a step seems to ignore your "overwrite" toggle and keeps serving stale results, confirm the cached file is actually being regenerated (see *Known issues* below — the break-detection cache has a known quirk in `DemoAppV2.3.py`).
> - **Duplicate `date` columns / index collisions** — merging timelines across sources can produce duplicate date columns or index conflicts; if a merge errors or balloons in row count, check for a `date` that exists as both a column and an index.
> - _[TODO: add the exact direct-invocation commands here if you want a non-dashboard CLI path for the analysis modules.]_

---

## Stage 3 — Model developer disengagement

This stage builds per-developer state timelines (`ACTIVE` / `NON_CODING` / `INACTIVE` / `GONE`), detects activity breaks, and predicts disengagement risk. It runs inside the `DemoAppV2.*.py` pipeline (Pipeline page). Predictors are grouped into three families plus core activity features:

- **Project Health (PH)**
- **Social-Technical Network (STN)**
- **Knowledge Distribution (KD)**

There are two model variants — pick based on what you're reproducing:

| | `DemoAppV2.2.py` | `DemoAppV2.3.py` |
|---|---|---|
| **Approach** | Classification | Survival analysis |
| **Model** | Temporal classifier + baselines | Cox time-varying (`lifelines`) |
| **Outputs** | Class predictions, eval metrics | Conditional survival probabilities, risk bands (LOW/MEDIUM/HIGH/CRITICAL) |
| **Evaluation** | Full suite: baselines, PR-AUC, window recall | Concordance index, survival diagnostics |
| **Break-detection cache** | Respects the overwrite toggle | **Always reuses cached breaks if present** (see Known issues) |

Trained model artifacts are written to `PredictionModel/` (e.g. `model.joblib`), and predictions to `Organizations/<org>/<repo>/distrac/break_predictions.parquet`.

> **Issues you'll hit here**
> - **Break detection alignment** — the original (forward-sliding, bidirectional) algorithm produces the *response* variable, and a causal (backward-only) variant produces the *predictor* — keep these distinct so you don't leak look-ahead information into the model.
> - **Sparse positives** — survival framing in `V2.3` reshapes each at-risk run into its own subject to avoid the 1-day-ahead positive sparsity of a single-clock setup; if you switch formulations, expect very different base hazard behavior.
> - _[TODO: note the exact entry point / button used to trigger model training if it isn't obvious from the Pipeline page.]_

---

## Stage 4 — Serve and view

Write results as parquet (if not already produced by the pipeline):

```bash
# writes analysis output as parquet
python Dashboard/distrac_writer.py
```

Start the FastAPI backend:

```bash
uvicorn Dashboard.distrac_api:app --reload    # http://localhost:8000
```

Start the dashboard:

```bash
streamlit run Dashboard/app.py
```

- **Pipeline** page — run/re-run collection, analysis, and modeling for a repository.
- **Dashboard** page — browse results for any analyzed repository.

A `manifest.json` in each project's `distrac/` folder signals that the pipeline completed for that project.

---

## Known issues & gotchas

A running list of things that have bitten us. Add to this as you go.

- **`DemoAppV2.3.py` break cache ignores overwrite.** The "original breaks" branch reuses the cached `*_breaks.csv` whenever the file exists, regardless of the overwrite toggle (`V2.2` gated this on `not over_write`). If you need to force regeneration in `V2.3`, delete the cached breaks file manually or restore the `and not over_write` condition.
- **Duplicate `date` column/index on merge.** Timeline merges can fail or duplicate rows when `date` is present as both a column and an index — reset/rename before merging.
- **Streamlit custom components.** Custom HTML/JS components have unreliable initialization and cross-component communication; prefer native Streamlit primitives for anything load-bearing.
- **GitHub rate limits.** The dominant cost of Stage 1 — more tokens is the only real fix.
- _[TODO: parquet engine notes (pyarrow vs fastparquet), any platform-specific path issues, conda env pins.]_

---

## Output reference

See the output table in [`README.md`](README.md#output) for what each parquet/JSON file under `Organizations/<org>/<repo>/distrac/` contains.
