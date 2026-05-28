"""
distrac_writer.py  —  Pipeline → Dashboard output contract.

Writes the `distrac/` folder for one repo after the pipeline has run.
This is the ONLY interface between the pipeline and the dashboard.

Usage
-----
Call write_distrac_stage1() at the end of the "Predictors" button handler,
once per repo.

Call write_distrac_stage2() at the end of the "Run Inactivity Prediction"
button handler, once per repo.  This writes manifest.json last, which is
the signal that tells the dashboard all data is ready.
"""

import datetime
import json
from pathlib import Path

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import Settings as cfg

ORG_BASE      = PROJECT_ROOT / "Organizations"
PIPELINE_VERSION = "1.0"

# File extensions considered "important" for knowledge distribution
_IMPORTANT_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".scala",
    ".go", ".rs", ".cpp", ".c", ".h", ".cs",
    ".rb", ".php", ".swift", ".m",
    ".r", ".R", ".sql", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".json", ".xml",
    ".md", ".rst",
}

_EXT_LANG_MAP = {
    ".py": "Python",   ".js": "JavaScript",  ".ts": "TypeScript",
    ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".java": "Java",   ".kt": "Kotlin",       ".scala": "Scala",
    ".go": "Go",       ".rs": "Rust",         ".cpp": "C++",
    ".c": "C",         ".h": "C/C++",         ".cs": "C#",
    ".rb": "Ruby",     ".php": "PHP",          ".swift": "Swift",
    ".r": "R",         ".R": "R",              ".sql": "SQL",
    ".sh": "Shell",    ".bash": "Shell",       ".yaml": "YAML",
    ".yml": "YAML",    ".md": "Markdown",      ".rst": "reStructuredText",
    ".toml": "TOML",   ".xml": "XML",          ".json": "JSON",
}


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ext_to_lang(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    return _EXT_LANG_MAP.get(ext, "Other")


def _build_login_to_dev_id(dev_names_df: pd.DataFrame) -> dict[str, str]:
    """Build a {login → dev_id} lookup from the dev_names DataFrame."""
    if dev_names_df.empty or "login" not in dev_names_df.columns:
        return {}
    result = {}
    for _, row in dev_names_df.iterrows():
        login  = str(row.get("login", "")).strip()
        dev_id = str(row.get("dev_id", "")).strip()
        if login and dev_id and login != "nan" and dev_id != "nan":
            result[login] = dev_id
    return result


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add any missing columns as None so parquet write never fails."""
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — all non-model outputs (call after Predictors button)
# ─────────────────────────────────────────────────────────────────────────────

def write_distrac_stage1(
    repo_full_name: str,
    dev_names_df: pd.DataFrame,
    tf: int,
    tf_devs: list[str],
    doe_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    ph_data: dict | None = None,
    commit_list_df: pd.DataFrame | None = None,) -> Path:
    """
    Write developers, truck_factor, knowledge_doe, stn_metrics, stn_edges,
    activity_weekly, activity_baselines, and commit_history parquet files.

    Parameters
    ----------
    repo_full_name : "org/repo"
    dev_names_df   : output of build_dev_names()
    tf             : truck factor integer from kd.main()
    tf_devs        : list of TF developer IDs from kd.main()
    doe_df         : DOE DataFrame from kd.main()
    metrics_df     : network node metrics from stn.main()
    edge_df        : network edge list from stn.main()
    ph_data        : dict loaded from project_health.json (or None)
    commit_list_df : raw commit_list DataFrame (raw_data_tables["commits"])

    Returns
    -------
    Path to the distrac/ folder.
    """
    org, repo = repo_full_name.split("/", 1)
    distrac_dir = ORG_BASE / org / repo / "distrac"
    distrac_dir.mkdir(parents=True, exist_ok=True)

    login_to_dev_id = _build_login_to_dev_id(dev_names_df)
    tf_dev_set      = set(tf_devs)

    # ── 1. developers.parquet ─────────────────────────────────────────────────
    devs = dev_names_df.copy()
    devs["org"]       = org
    devs["repo"]      = repo
    devs["is_tf_dev"] = devs["dev_id"].isin(tf_dev_set) if "dev_id" in devs.columns else False

    if not metrics_df.empty and "user" in metrics_df.columns and "role" in metrics_df.columns:
        login_role    = dict(zip(metrics_df["user"].astype(str), metrics_df["role"].astype(str)))
        devs["role"]  = devs["login"].map(login_role).fillna("Unknown") if "login" in devs.columns else "Unknown"
    else:
        devs["role"] = "Unknown"

    dev_cols = ["org", "repo", "dev_id", "login", "name", "email", "is_tf_dev", "role"]
    _ensure_cols(devs, dev_cols)[dev_cols].to_parquet(distrac_dir / "developers.parquet", index=False)
    print(f"  [distrac] developers.parquet — {len(devs)} rows")

    # ── 2. truck_factor.parquet ───────────────────────────────────────────────
    pd.DataFrame([{"org": org, "repo": repo, "tf": int(tf)}]).to_parquet(
        distrac_dir / "truck_factor.parquet", index=False
    )
    print(f"  [distrac] truck_factor.parquet — tf={tf}")

    # ── 3. knowledge_doe.parquet ──────────────────────────────────────────────
    if doe_df is not None and not doe_df.empty:
        doe_out = doe_df.copy()
        # KD writes 'developer' column; rename to dev_id for consistency
        if "developer" in doe_out.columns and "dev_id" not in doe_out.columns:
            doe_out = doe_out.rename(columns={"developer": "dev_id"})
        doe_out["org"]          = org
        doe_out["repo"]         = repo
        doe_out["lang"]         = doe_out["file_path"].apply(_ext_to_lang) if "file_path" in doe_out.columns else "Other"
        doe_out["is_important"] = doe_out["file_path"].apply(
            lambda p: Path(p).suffix.lower() in _IMPORTANT_EXT
        ) if "file_path" in doe_out.columns else False
        doe_cols = ["org", "repo", "dev_id", "file_path", "DOE", "lang", "is_important"]
        _ensure_cols(doe_out, doe_cols)[doe_cols].to_parquet(distrac_dir / "knowledge_doe.parquet", index=False)
        print(f"  [distrac] knowledge_doe.parquet — {len(doe_out)} rows")
    else:
        pd.DataFrame(columns=["org", "repo", "dev_id", "file_path", "DOE", "lang", "is_important"]).to_parquet(
            distrac_dir / "knowledge_doe.parquet", index=False
        )
        print(f"  [distrac] knowledge_doe.parquet — empty (no DOE data)")

    # ── 4. stn_metrics.parquet ────────────────────────────────────────────────
    if metrics_df is not None and not metrics_df.empty:
        stn_out = metrics_df.copy()
        # STN uses 'user' (login); resolve to dev_id
        if "user" in stn_out.columns:
            stn_out["dev_id"] = stn_out["user"].map(login_to_dev_id).fillna(stn_out["user"])
        elif "dev_id" not in stn_out.columns:
            stn_out["dev_id"] = ""
        stn_out["org"]  = org
        stn_out["repo"] = repo
        stn_cols = [
            "org", "repo", "dev_id",
            "betweenness_centrality", "degree", "weighted_degree",
            "is_articulation_point", "is_community_bridge", "communities_spanned",
            "issue_focus_pct", "pr_focus_pct",
        ]
        _ensure_cols(stn_out, stn_cols)[stn_cols].to_parquet(distrac_dir / "stn_metrics.parquet", index=False)
        print(f"  [distrac] stn_metrics.parquet — {len(stn_out)} rows")
    else:
        pd.DataFrame(columns=[
            "org", "repo", "dev_id", "betweenness_centrality", "degree",
            "weighted_degree", "is_articulation_point", "is_community_bridge",
            "communities_spanned", "issue_focus_pct", "pr_focus_pct",
        ]).to_parquet(distrac_dir / "stn_metrics.parquet", index=False)
        print(f"  [distrac] stn_metrics.parquet — empty (no STN data)")

    # ── 5. stn_edges.parquet ─────────────────────────────────────────────────
    if edge_df is not None and not edge_df.empty:
        edges_out = edge_df.copy()
        # SocialTechnicalNetwork.py uses developer_a/b; normalise to source/target
        if "developer_a" in edges_out.columns and "source" not in edges_out.columns:
            edges_out = edges_out.rename(columns={"developer_a": "source", "developer_b": "target"})
        if "weight_total" in edges_out.columns and "weight" not in edges_out.columns:
            edges_out = edges_out.rename(columns={"weight_total": "weight"})
        if "source" in edges_out.columns:
            edges_out["source_dev_id"] = edges_out["source"].map(login_to_dev_id).fillna(edges_out["source"])
        else:
            edges_out["source_dev_id"] = ""
        if "target" in edges_out.columns:
            edges_out["target_dev_id"] = edges_out["target"].map(login_to_dev_id).fillna(edges_out["target"])
        else:
            edges_out["target_dev_id"] = ""
        # Ensure weight_reviews always exists — older STN runs omit it
        if "weight_reviews" not in edges_out.columns:
            edges_out["weight_reviews"] = 0.0
        if "weight" not in edges_out.columns:
            # Try weight_total as fallback
            edges_out["weight"] = edges_out.get("weight_total", pd.Series(0.0, index=edges_out.index))
        edges_out["org"]  = org
        edges_out["repo"] = repo
        edge_cols = ["org", "repo", "source_dev_id", "target_dev_id", "weight", "weight_reviews"]
        _ensure_cols(edges_out, edge_cols)[edge_cols].to_parquet(distrac_dir / "stn_edges.parquet", index=False)
        print(f"  [distrac] stn_edges.parquet — {len(edges_out)} rows")
    else:
        pd.DataFrame(columns=["org", "repo", "source_dev_id", "target_dev_id", "weight", "weight_reviews"]).to_parquet(
            distrac_dir / "stn_edges.parquet", index=False
        )
        print(f"  [distrac] stn_edges.parquet — empty (no STN edge data)")

    # ── 6. project_health.json (raw copy used by departure-simulation HTML) ─────
    if ph_data:
        ph_out = distrac_dir / "project_health.json"
        with open(ph_out, "w", encoding="utf-8") as _f:
            json.dump(ph_data, _f, ensure_ascii=False, indent=2)
        print(f"  [distrac] project_health.json — written")

    # ── 6b. activity_weekly.parquet + activity_baselines.parquet ──────────────
    if ph_data and "developers" in ph_data:
        weeks   = ph_data.get("weeks", [])
        n_weeks = len(weeks)

        week_rows = []
        for dev_id, d in ph_data["developers"].items():
            wc = d.get("weekly_commits", [])
            wp = d.get("weekly_prs",     [])
            wi = d.get("weekly_issues",  [])
            for i, label in enumerate(weeks):
                week_rows.append({
                    "org":        org,
                    "repo":       repo,
                    "dev_id":     dev_id,
                    "week_index": i - n_weeks,      # -16 … -1
                    "week_label": label,
                    "commits":    float(wc[i]) if i < len(wc) else 0.0,
                    "prs":        float(wp[i]) if i < len(wp) else 0.0,
                    "issues":     float(wi[i]) if i < len(wi) else 0.0,
                })
        pd.DataFrame(week_rows).to_parquet(distrac_dir / "activity_weekly.parquet", index=False)
        print(f"  [distrac] activity_weekly.parquet — {len(week_rows)} rows")

        repo_totals  = ph_data.get("repo_totals", {})
        baseline_rows = []
        for dev_id, d in ph_data["developers"].items():
            baseline_rows.append({
                "org":                   org,
                "repo":                  repo,
                "dev_id":                dev_id,
                "generated_at":          ph_data.get("generated_at"),
                "baseline_commits":      d.get("baseline_commits", 0.0),
                "baseline_prs":          d.get("baseline_prs",     0.0),
                "baseline_issues":       d.get("baseline_issues",  0.0),
                "repo_commits_per_week": repo_totals.get("commits_per_week", 0.0),
                "repo_prs_per_week":     repo_totals.get("prs_per_week",     0.0),
                "repo_issues_per_week":  repo_totals.get("issues_per_week",  0.0),
            })
        pd.DataFrame(baseline_rows).to_parquet(distrac_dir / "activity_baselines.parquet", index=False)
        print(f"  [distrac] activity_baselines.parquet — {len(baseline_rows)} rows")
    else:
        pd.DataFrame(columns=[
            "org", "repo", "dev_id", "week_index", "week_label", "commits", "prs", "issues",
        ]).to_parquet(distrac_dir / "activity_weekly.parquet", index=False)
        pd.DataFrame(columns=[
            "org", "repo", "dev_id", "generated_at", "baseline_commits",
            "baseline_prs", "baseline_issues", "repo_commits_per_week",
            "repo_prs_per_week", "repo_issues_per_week",
        ]).to_parquet(distrac_dir / "activity_baselines.parquet", index=False)
        print(f"  [distrac] activity_weekly/baselines.parquet — empty (no project health data)")

    # ── 7. commit_history.parquet ─────────────────────────────────────────────
    if commit_list_df is not None and not commit_list_df.empty:
        ch = commit_list_df.copy()

        # Parse dates
        if "created_at" in ch.columns:
            ch["date"] = pd.to_datetime(ch["created_at"], utc=True, errors="coerce").dt.normalize().dt.date
        else:
            ch["date"] = None

        # Resolve developer identity to dev_id
        if "author_id" in ch.columns:
            ch["dev_id"] = ch["author_id"].astype(str)
        elif "author_login" in ch.columns:
            ch["dev_id"] = ch["author_login"].map(login_to_dev_id).fillna(ch["author_login"].astype(str))
        else:
            ch["dev_id"] = None

        ch["org"]  = org
        ch["repo"] = repo

        # Per-developer daily counts
        dev_daily = (
            ch.dropna(subset=["date", "dev_id"])
            .groupby(["org", "repo", "dev_id", "date"])
            .size()
            .reset_index(name="commit_count")
        )

        # Repo-level daily counts (dev_id = None → repo summary row)
        repo_daily = (
            ch.dropna(subset=["date"])
            .assign(dev_id=None)
            .groupby(["org", "repo", "date"])
            .size()
            .reset_index(name="commit_count")
        )
        repo_daily["dev_id"] = None

        commit_hist = pd.concat([dev_daily, repo_daily], ignore_index=True)
        commit_hist.to_parquet(distrac_dir / "commit_history.parquet", index=False)
        print(f"  [distrac] commit_history.parquet — {len(commit_hist)} rows")
    else:
        pd.DataFrame(columns=["org", "repo", "dev_id", "date", "commit_count"]).to_parquet(
            distrac_dir / "commit_history.parquet", index=False
        )
        print(f"  [distrac] commit_history.parquet — empty (no commit data)")

    print(f"[distrac_writer] Stage 1 complete for {repo_full_name}")
    return distrac_dir


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — model predictions + manifest (call after Inactivity Prediction)
# ─────────────────────────────────────────────────────────────────────────────

def write_distrac_stage2(
    repo_full_name: str,
    predictions_df: pd.DataFrame,
    break_len_df: pd.DataFrame | None = None,) -> Path:
    """
    Write break_predictions.parquet and manifest.json.

    manifest.json is written last and signals to the dashboard that all
    distrac/ data is complete and ready to load.

    Parameters
    ----------
    repo_full_name  : "org/repo"
    predictions_df  : per-dev daily predictions from evaluate_model()
                      Must contain: dev (or dev_id), date, prob_1, prob_0, state
    break_len_df    : optional regression predictions (predicted_next_break_len per dev)
                      If None, the column is left as NaN.
    """
    org, repo = repo_full_name.split("/", 1)
    distrac_dir = ORG_BASE / org / repo / "distrac"
    distrac_dir.mkdir(parents=True, exist_ok=True)

    # ── 8. break_predictions.parquet ─────────────────────────────────────────
    pred_out = predictions_df.copy()
    pred_out["org"]  = org
    pred_out["repo"] = repo

    # Normalise column name: 'dev' → 'dev_id'
    if "dev" in pred_out.columns and "dev_id" not in pred_out.columns:
        pred_out = pred_out.rename(columns={"dev": "dev_id"})

    # Merge break-length regression predictions if available
    if break_len_df is not None and not break_len_df.empty:
        bl = break_len_df.copy()
        if "dev" in bl.columns and "dev_id" not in bl.columns:
            bl = bl.rename(columns={"dev": "dev_id"})
        # Take the last prediction per dev (most recent)
        last_bl = (
            bl.sort_values("dev_id")
            .groupby("dev_id")["predicted_next_break_len"]
            .last()
            .reset_index()
        )
        pred_out = pred_out.merge(last_bl, on="dev_id", how="left")
    else:
        if "predicted_next_break_len" not in pred_out.columns:
            pred_out["predicted_next_break_len"] = None

    # Normalise response label column name
    if "break_label" not in pred_out.columns:
        # Accept old binary column, new future_state columns, or generic fallbacks
        _state_shifted_cols = [c for c in pred_out.columns if c.startswith("state_shifted_")]
        for candidate in (["true_onset_in_7d", "inactivity_window_14d", "break_starts_in_14d"] + _state_shifted_cols + ["response", "label"]):
            if candidate in pred_out.columns:
                pred_out = pred_out.rename(columns={candidate: "break_label"})
                break
        if "break_label" not in pred_out.columns:
            pred_out["break_label"] = None

    # Binary model (inactivity_window): prob_1 = P(at-risk), used directly.
    # 4-class model (future_state):     P(INACTIVE) + P(GONE).
    # Survival model:                   risk_30d = 1 - S_30.
    _is_binary_model   = "prob_1" in pred_out.columns and "prob_INACTIVE" not in pred_out.columns
    _is_survival_model = "S_7" in pred_out.columns or "risk_30d" in pred_out.columns
    if _is_survival_model:
        pred_out["inactivity_risk"] = pred_out["risk_30d"].fillna(0).clip(0, 1) \
                                      if "risk_30d" in pred_out.columns \
                                      else (1 - pred_out["S_30"].fillna(1)).clip(0, 1)
    elif _is_binary_model:
        pred_out["inactivity_risk"] = pred_out["prob_1"].fillna(0).clip(0, 1)
    else:
        _p_inactive = pred_out["prob_INACTIVE"].fillna(0) if "prob_INACTIVE" in pred_out.columns \
                      else pd.Series(0.0, index=pred_out.index)
        _p_gone     = pred_out["prob_GONE"].fillna(0)     if "prob_GONE"     in pred_out.columns \
                      else pd.Series(0.0, index=pred_out.index)
        pred_out["inactivity_risk"] = (_p_inactive + _p_gone).clip(0, 1)

    pred_cols = [
        "org", "repo", "dev_id", "date",
        "inactivity_risk",
        # binary / 4-class LSTM columns
        "prob_0", "prob_1",
        "prob_ACTIVE", "prob_NON_CODING", "prob_INACTIVE", "prob_GONE",
        "state", "commits", "prs", "issues", "issue_activity", "pr_activity",
        "break_label", "predicted_next_break_len",
        # survival model columns
        "S_7", "S_14", "S_30", "S_60", "S_90",
        "risk_30d", "risk_band", "true_onset_in_7d",
    ]
    _ensure_cols(pred_out, pred_cols)[pred_cols].to_parquet(
        distrac_dir / "break_predictions.parquet", index=False
    )
    print(f"  [distrac] break_predictions.parquet — {len(pred_out)} rows")

    # ── 9. manifest.json ─────────────────────────────────────────────────────
    # Count rows in every parquet file to give the dashboard a quick sanity check
    row_counts: dict[str, int] = {}
    for fname in [
        "developers", "break_predictions", "stn_metrics", "stn_edges",
        "knowledge_doe", "truck_factor", "activity_weekly",
        "activity_baselines", "commit_history",
    ]:
        p = distrac_dir / f"{fname}.parquet"
        if p.exists():
            try:
                row_counts[fname] = len(pd.read_parquet(p, columns=[]))
            except Exception:
                row_counts[fname] = -1
        else:
            row_counts[fname] = 0

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_cutoff":      cfg.data_collection_date,
        "repo":             repo_full_name,
        "row_counts":       row_counts,
    }
    with open(distrac_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[distrac_writer] Stage 2 complete for {repo_full_name} — manifest written.")
    return distrac_dir
