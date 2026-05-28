#   conda activate osslab
#   streamlit run Extractors/Dashboard.py

import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
from pathlib import Path
from datetime import timedelta
import datetime
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import Settings as cfg

ORG_BASE = PROJECT_ROOT / "Organizations"


# ─────────────────────────────────────────────────────────────────────────────
# DISTRAC PARQUET LOADER  (single cached entry-point for all panel data)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_repo(repo_full: str) -> dict:
    """
    Load all pre-computed distrac/ parquet files for one repo.

    Returns a dict with keys:
        developers, predictions, stn_metrics, stn_edges, doe, tf,
        ph_weekly, ph_baselines, commits, _manifest

    Returns an EMPTY dict {} if manifest.json does not exist, meaning the
    Predictors + Inactivity Prediction pipeline has not been run for this repo.
    Callers must check `if not load_repo(repo)` before accessing keys.
    """
    base = ORG_BASE / repo_full / "distrac"
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return {}

    files = {
        "developers":   "developers.parquet",
        "predictions":  "break_predictions.parquet",
        "stn_metrics":  "stn_metrics.parquet",
        "stn_edges":    "stn_edges.parquet",
        "doe":          "knowledge_doe.parquet",
        "tf":           "truck_factor.parquet",
        "ph_weekly":    "activity_weekly.parquet",
        "ph_baselines": "activity_baselines.parquet",
        "commits":      "commit_history.parquet",
    }
    result: dict = {}
    for key, fname in files.items():
        p = base / fname
        if p.exists():
            try:
                result[key] = pd.read_parquet(p)
            except Exception as e:
                print(f"[load_repo] Failed to read {p}: {e}")
                result[key] = pd.DataFrame()
        else:
            result[key] = pd.DataFrame()

    with open(manifest_path) as f:
        result["_manifest"] = json.load(f)

    return result


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


# ─────────────────────────────────────────────────────────────────────────────
# REPO DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def list_repos_for(org: str) -> list[str]:
    """All test repos for this org that exist on disk."""
    test_pairs = cfg.load_repo_split("test")
    repos = sorted(
        [repo for o, repo in test_pairs if o == org],
        key=str.casefold,
    )
    return [r for r in repos if (ORG_BASE / org / r).is_dir()]

@st.cache_data(show_spinner=False)
def list_orgs() -> list[str]:
    """Orgs from the test split in repo_split.csv that have a repo folder on disk."""
    test_pairs = cfg.load_repo_split("test")
    return sorted(
        {org for org, repo in test_pairs if (ORG_BASE / org / repo).is_dir()},
        key=str.casefold,
    )

@st.cache_data(show_spinner=False)
def _load_stn_metrics(repo_full_name: str) -> pd.DataFrame:
    """Load STN node metrics.
    Tries distrac/stn_metrics.parquet first; falls back to CSV.
    Adds a 'user' column (login) so existing panel code keeps working."""
    repo_data = load_repo(repo_full_name)
    if repo_data and not repo_data.get("stn_metrics", pd.DataFrame()).empty:
        df = repo_data["stn_metrics"].copy()
        # Panel code expects 'user' column (login). Add it from developers if needed.
        if "user" not in df.columns:
            devs = repo_data.get("developers", pd.DataFrame())
            if not devs.empty and "dev_id" in devs.columns and "login" in devs.columns:
                id_to_login = dict(zip(devs["dev_id"], devs["login"]))
                df["user"] = df["dev_id"].map(id_to_login).fillna(df["dev_id"])
            else:
                df["user"] = df.get("dev_id", "")
        return df
    # Fallback: old CSV
    path = ORG_BASE / repo_full_name / cfg.social_technical_metrics_folder / cfg.social_technical_metrics_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_edge_list(repo_full_name: str) -> pd.DataFrame:
    """Load STN edge list.
    Tries distrac/stn_edges.parquet first; falls back to CSV.
    Adds 'source'/'target' columns (logins) so existing panel code keeps working."""
    repo_data = load_repo(repo_full_name)
    if repo_data and not repo_data.get("stn_edges", pd.DataFrame()).empty:
        df = repo_data["stn_edges"].copy()
        # Panel code expects 'source'/'target' columns (login).
        if "source" not in df.columns and "source_dev_id" in df.columns:
            devs = repo_data.get("developers", pd.DataFrame())
            if not devs.empty and "dev_id" in devs.columns and "login" in devs.columns:
                id_to_login = dict(zip(devs["dev_id"], devs["login"]))
                df["source"] = df["source_dev_id"].map(id_to_login).fillna(df["source_dev_id"])
                df["target"] = df["target_dev_id"].map(id_to_login).fillna(df["target_dev_id"])
            else:
                df["source"] = df["source_dev_id"]
                df["target"] = df["target_dev_id"]
        return df
    # Fallback: old CSV
    path = ORG_BASE / repo_full_name / cfg.social_technical_metrics_folder / cfg.social_technical_edge_list_file
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _build_stn_html(repo_full_name: str, selected_date, window_days: int = 30) -> str:
    """Cached wrapper around get_html_for_streamlit.
    Re-computed only when repo or date changes; fast for same repo+date pair.
    """
    _ext_dir = str(Path(__file__).resolve().parent)
    if _ext_dir not in sys.path:
        sys.path.insert(0, _ext_dir)
    from SocialTechnicalNetworkV2 import get_html_for_streamlit
    return get_html_for_streamlit(
        repo_full_name=repo_full_name,
        as_of_date=selected_date,
        window_days=window_days,
    )


def render_stn_panel(repo_full_name: str, selected_dev_id: str | None,
                     selected_date=None):
    """
    Render the Social-Technical Network panel.
    Shows:
      A) Collaboration metrics table for all TF developers
      B) Top-10 collaborators bar chart for the selected developer
      C) Network graph (spring layout) restricted to TF developers
    """
    import networkx as nx

    metrics_df = _load_stn_metrics(repo_full_name)
    edge_df    = _load_edge_list(repo_full_name)
    org, repo  = repo_full_name.split("/", 1)

    if metrics_df.empty:
        st.info(
            "No Social-Technical Network data found for this repo.  \n"
            "Run the **Predictors** step on the Pipeline page to generate it."
        )
        return

    # ── resolve display names ────────────────────────────────────────────────
    def _dn(login: str) -> str:
        # metrics uses the `user` login column — look up via dev_names by login field
        names_df = _load_dev_names(repo_full_name)
        if not names_df.empty:
            row = names_df[names_df["login"] == login]
            if not row.empty:
                for field in ("name", "login"):
                    val = row.iloc[0].get(field, "")
                    if pd.notna(val) and str(val).strip():
                        return str(val).strip()
        return login

    
    # ── Sub-section B: top collaborators for selected dev ───────────────────
    if selected_dev_id:
        # selected_dev_id is the create_developer_id key; metrics uses login
        # look up the login from dev_names
        names_df = _load_dev_names(repo_full_name)
        selected_login = None
        if not names_df.empty:
            row = names_df[names_df["dev_id"] == selected_dev_id]
            if not row.empty:
                selected_login = row.iloc[0].get("login")
        if not selected_login:
            # fallback: try using the ID directly
            selected_login = selected_dev_id.split("|")[-1] if "|" in selected_dev_id else selected_dev_id



    # ── Sub-section C: D3 interactive network ────────────────────────────────
    with st.spinner("Building interaction network…"):
        html_str = _build_stn_html(repo_full_name, selected_date, window_days=90)
    components.html(html_str, height=650, scrolling=False)


@st.cache_data(show_spinner=False)
def _load_dev_names(repo_full_name: str) -> pd.DataFrame:
    """Load developer identity table.
    Tries distrac/developers.parquet first; falls back to Results/dev_names.csv."""
    repo_data = load_repo(repo_full_name)
    if repo_data and not repo_data.get("developers", pd.DataFrame()).empty:
        df = repo_data["developers"].copy()
        # Ensure backward-compatible columns
        if "raw_id" not in df.columns:
            df["raw_id"] = df.get("dev_id", "")
        needed = ["dev_id", "name", "login", "email", "raw_id"]
        for c in needed:
            if c not in df.columns:
                df[c] = None
        return df[needed]
    # Fallback: old CSV
    path = ORG_BASE / repo_full_name / "Results" / "dev_names.csv"
    if not path.exists():
        return pd.DataFrame(columns=["dev_id", "name", "login", "email", "raw_id"])
    return pd.read_csv(path)


# ── Role colors (shared across panels) ───────────────────────────────────────
ROLE_COLORS = {
    "Maintainer":  "#8B1C1C",   # dark red   — critical
    "Collaborator": "#8B6914",  # dark gold  — moderate
    "Bridge":      "#4B4BA0",   # slate blue — high
    "Peripheral":  "#555555",   # gray       — low
}
ROLE_RISK = {
    "Maintainer":  "critical",
    "Collaborator": "moderate",
    "Bridge":      "high",
    "Peripheral":  "low",
}


def _dn_login(login: str, repo_full_name: str) -> str:
    """Resolve a raw login string to a display name via dev_names.csv."""
    names_df = _load_dev_names(repo_full_name)
    if not names_df.empty:
        row = names_df[names_df["login"] == login]
        if not row.empty:
            for field in ("name", "login"):
                val = row.iloc[0].get(field, "")
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
    return login


def get_display_name(dev_id: str, repo_full_name: str) -> str:
    """Return the most readable name for a dev_id.
    Priority: name > login > email > stripped dev_id fallback.
    Falls back gracefully if dev_names.csv has not been generated yet.
    """
    df = _load_dev_names(repo_full_name)
    if not df.empty:
        row = df[df["dev_id"] == dev_id]
        if not row.empty:
            r = row.iloc[0]
            for field in ("name", "login", "email"):
                val = r.get(field, "")
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
    return dev_id.split("|")[-1] if "|" in dev_id else dev_id


@st.cache_data(show_spinner=False)
def load_commit_history(repo_full: str) -> pd.DataFrame:
    """Daily commit-count histogram for the whole repo.
    Returns DataFrame with columns: date (datetime.date), count (int).
    Tries distrac/commit_history.parquet first; falls back to raw commit_list.csv.
    """
    repo_data = load_repo(repo_full)
    if repo_data and not repo_data.get("commits", pd.DataFrame()).empty:
        df = repo_data["commits"]
        # Repo-level rows have dev_id == None
        repo_rows = df[df["dev_id"].isna()][["date", "commit_count"]].copy()
        repo_rows = repo_rows.rename(columns={"commit_count": "count"})
        repo_rows["date"] = pd.to_datetime(repo_rows["date"]).dt.date
        return repo_rows.sort_values("date").reset_index(drop=True)
    # Fallback: raw CSV
    path = ORG_BASE / repo_full / "commit_list.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "count"])
    df = pd.read_csv(path, parse_dates=["created_at"], usecols=["created_at"])
    daily = df.set_index("created_at").resample("D").size().reset_index(name="count")
    daily["date"] = daily["created_at"].dt.date
    return daily[["date", "count"]]


@st.cache_data(show_spinner=False)
def load_dev_commit_history(repo_full: str, dev_id: str) -> pd.DataFrame:
    """Daily commit-count histogram filtered to one developer.
    Returns same shape as load_commit_history.
    Tries distrac/commit_history.parquet first; falls back to raw CSV.
    """
    repo_data = load_repo(repo_full)
    if repo_data and not repo_data.get("commits", pd.DataFrame()).empty:
        df = repo_data["commits"]
        sub = df[df["dev_id"].astype(str) == str(dev_id)][["date", "commit_count"]].copy()
        if not sub.empty:
            sub = sub.rename(columns={"commit_count": "count"})
            sub["date"] = pd.to_datetime(sub["date"]).dt.date
            return sub.sort_values("date").reset_index(drop=True)

    # Fallback: raw CSV
    path = ORG_BASE / repo_full / "commit_list.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "count"])
    needed = ["created_at", "author_id", "author_login"]
    try:
        df = pd.read_csv(path, parse_dates=["created_at"], usecols=lambda c: c in needed)
    except Exception:
        return pd.DataFrame(columns=["date", "count"])

    if "author_id" in df.columns:
        sub = df[df["author_id"].astype(str) == str(dev_id)]
    else:
        sub = pd.DataFrame()

    if sub.empty and "author_login" in df.columns:
        names_df = _load_dev_names(repo_full)
        if not names_df.empty:
            row = names_df[names_df["dev_id"] == dev_id]
            if not row.empty:
                login = row.iloc[0].get("login", "")
                if login:
                    sub = df[df["author_login"].astype(str) == str(login)]

    if sub.empty:
        return pd.DataFrame(columns=["date", "count"])

    daily = sub.set_index("created_at").resample("D").size().reset_index(name="count")
    daily["date"] = daily["created_at"].dt.date
    return daily[["date", "count"]]


def _build_timeline_payload(
    repo_full: str,
    selected_date: datetime.date,
    window_days: int,
    hist_df: pd.DataFrame,
    dev_hist_df: pd.DataFrame | None = None,) -> dict:
    """Assemble the JSON payload for the timeline HTML component."""
    window_start = selected_date - datetime.timedelta(days=window_days)

    def _to_pairs(df: pd.DataFrame) -> list:
        """Convert date/count DataFrame to [unix_ms, count] pairs for Chart.js."""
        if df.empty:
            return []
        result = []
        for _, r in df.iterrows():
            d = r["date"]
            if isinstance(d, datetime.date):
                ts = int(datetime.datetime.combine(d, datetime.time.min).timestamp() * 1000)
            else:
                ts = int(pd.Timestamp(d).timestamp() * 1000)
            result.append([ts, int(r["count"])])
        return result

    return {
        "histogram":         _to_pairs(hist_df),
        "dev_histogram":     _to_pairs(dev_hist_df) if dev_hist_df is not None and not dev_hist_df.empty else None,
        "selected_date":     selected_date.isoformat(),
        "window_start":      window_start.isoformat(),
        "window_days":       window_days,
        "label_date":        selected_date.strftime("%b %d, %Y"),
        "label_window_start": window_start.strftime("%b %d, %Y"),
        "label_window_days": f"{window_days} days",
        "repo":              repo_full,
    }


@st.cache_data
def load_commit_dates(repo_full: str):
    """Return the earliest commit date for a repo. Delegates to load_commit_history."""
    hist_df = load_commit_history(repo_full)
    if hist_df.empty:
        return datetime.date.today() - datetime.timedelta(days=365)
    return hist_df["date"].min()

def _find_test_df(repo_full_name: str) -> Path | None:
    """Look for per-repo predictions — checks distrac/ parquet first, then legacy CSV/pkl."""
    # New path: distrac/break_predictions.parquet
    parquet_path = ORG_BASE / repo_full_name / "distrac" / "break_predictions.parquet"
    if parquet_path.exists():
        return parquet_path
    # Legacy paths
    base = ORG_BASE / repo_full_name / cfg.model_folder
    if (base / "test_df.csv").exists():
        return base / "test_df.csv"
    if (base / "test_df.pkl").exists():
        return base / "test_df.pkl"
    return None

def repo_data_status(repo_full_name: str) -> dict:
    """Check which pre-computed output files exist for a repo.
    Checks distrac/ manifest first (new path); falls back to legacy file checks."""
    base = ORG_BASE / repo_full_name
    manifest_path = base / "distrac" / "manifest.json"

    if manifest_path.exists():
        distrac_base = base / "distrac"
        manifest = {}
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            pass
        generated_at = manifest.get("generated_at", "")
        return {
            "distrac":        True,
            "generated_at":   generated_at,
            "timeline":       True,
            "model":          (distrac_base / "break_predictions.parquet").exists(),
            "truck_factor":   (distrac_base / "truck_factor.parquet").exists(),
            "knowledge_dist": (distrac_base / "knowledge_doe.parquet").exists(),
        }

    # Legacy fallback
    return {
        "distrac":        False,
        "generated_at":   "",
        "timeline":       (base / "Results" / "all_users_labeled_timeline.csv").exists(),
        "model":          _find_test_df(repo_full_name) is not None,
        "truck_factor":   (base / "KnowledgeDistribution" / "truck_factor.json").exists(),
        "knowledge_dist": (base / "KnowledgeDistribution" / "doe.csv").exists(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATE COLORS  (used by activity chart)
# ─────────────────────────────────────────────────────────────────────────────

_STATE_COLORS = {
    "ACTIVE":     "#2ecc71",
    "NON_CODING": "#f1c40f",
    "INACTIVE":   "#e74c3c",
    "GONE":       "#95a5a6",
    "UNKNOWN":    "#bdc3c7",
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


_MEAN_LEAD_TIME_DAYS = 13.5   # from Extractors/tdr_report.txt — mean lead time to break detection


# ─────────────────────────────────────────────────────────────────────────────
# T1: RISK LEVEL  (prob_1 from LSTM → human label)
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_level(prob_1: float) -> tuple[str, str]:
    """Convert LSTM probability to a risk label and hex color.
    Thresholds: High >= 0.6 | Medium >= 0.3 | Low < 0.3
    """
    if prob_1 >= 0.6:
        return "High",   "#e74c3c"
    elif prob_1 >= 0.3:
        return "Medium", "#f1c40f"
    else:
        return "Low",    "#2ecc71"


# ─────────────────────────────────────────────────────────────────────────────
# T2: IMPACT LEVEL  (knowledge-distribution risk_score → human label)
# ─────────────────────────────────────────────────────────────────────────────

def get_impact_level(risk_score: float, on_truck_factor: bool) -> tuple[str, str]:
    """Bucket a knowledge-distribution risk score into a human label."""
    critical = getattr(cfg, "impact_critical_threshold", 15)
    moderate = getattr(cfg, "impact_moderate_threshold", 5)
    if on_truck_factor and risk_score >= critical:
        return "Critical", "#e74c3c"
    elif risk_score >= moderate:
        return "Moderate", "#f1c40f"
    else:
        return "Low",      "#2ecc71"


# ─────────────────────────────────────────────────────────────────────────────
# BREAK LENGTH PREDICTIONS  (from LSTM regression model output)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_break_prediction_df(repo_full_name: str = "") -> pd.DataFrame:
    """Load break-length predictions.
    Tries distrac/break_predictions.parquet first; falls back to org-level CSV."""
    if repo_full_name:
        repo_data = load_repo(repo_full_name)
        if repo_data and not repo_data.get("predictions", pd.DataFrame()).empty:
            df = repo_data["predictions"].copy()
            # Normalise to the column name the legacy code expects
            if "dev_id" in df.columns and "dev" not in df.columns:
                df = df.rename(columns={"dev_id": "dev"})
            return df
    # Fallback: old org-level CSV
    path = ORG_BASE / "break_prediction_df.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_next_break_length(dev: str, org: str, repo: str) -> str:
    """
    Return the predicted next break length for a developer as a display string.
    Tries distrac/break_predictions.parquet first; falls back to legacy CSV.
    Returns 'N/A' if no data exists.
    """
    repo_full = f"{org}/{repo}"
    df = _load_break_prediction_df(repo_full)
    if df.empty:
        return "N/A"
    dev_col = "dev" if "dev" in df.columns else "dev_id"
    sub = df[df[dev_col] == dev]
    if sub.empty:
        return "N/A"
    if "predicted_next_break_len" not in sub.columns:
        return "N/A"
    # Take the most recent non-null prediction
    valid = sub.dropna(subset=["predicted_next_break_len"])
    if valid.empty:
        return "N/A"
    pred = valid.iloc[-1]["predicted_next_break_len"]
    try:
        return f"~{int(round(float(pred)))} days"
    except (ValueError, TypeError):
        return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# T3: BREAK RISK OVERVIEW TABLE
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_table(repo_full_name: str, selected_date: datetime.date) -> pd.DataFrame | None:
    """
    Assemble the risk table for one repo — all developers in the test set.
    Returns None if model predictions are missing.
    Columns: dev_id, Contributor, Risk Level, risk_color,
             Next Break Length, Expected Timeframe, Impact, impact_color, prob_1
    """
    org, repo = repo_full_name.split("/", 1)

    # ── Try parquet first (fast, typed, no file-format ambiguity) ────────────
    repo_data = load_repo(repo_full_name)
    if repo_data and not repo_data.get("predictions", pd.DataFrame()).empty:
        preds = repo_data["predictions"].copy()
        if "dev_id" in preds.columns and "dev" not in preds.columns:
            preds = preds.rename(columns={"dev_id": "dev"})
        test_df = preds
    else:
        # ── Fallback: legacy CSV / pkl ────────────────────────────────────────
        model_path = _find_test_df(repo_full_name)
        if model_path is None:
            return None
        if model_path.suffix == ".pkl":
            test_df = pd.read_pickle(model_path)
        else:
            test_df = pd.read_csv(model_path)

    test_df["date"] = pd.to_datetime(test_df["date"])

    # ── DIAGNOSTIC ──────────────────────────────────────────────────────────
    print(f"\n[build_risk_table] Loading: {model_path}")
    print(f"[build_risk_table] Shape: {test_df.shape}")
    prob_cols_found = [c for c in test_df.columns if c.startswith("prob_")]
    print(f"[build_risk_table] prob_* columns found: {prob_cols_found}")
    if not prob_cols_found:
        print("[build_risk_table] !! prob_1 is MISSING — test_df.csv was saved before")
        print("[build_risk_table] !! evaluate_model() ran. Re-run the pipeline on the")
        print("[build_risk_table] !! Pipeline page to regenerate test_df.csv with predictions.")
        st.warning(
            f"**`test_df.csv` is missing probability columns.**  \n"
            "The file was saved before the model ran predictions.  \n"
            "**Fix:** Go to the Pipeline page and re-run the prediction pipeline for this repo.  \n"
            f"File: `{model_path}`"
        )
        return None
    # ────────────────────────────────────────────────────────────────────────

    cutoff = pd.Timestamp(selected_date)
    current_probs = (
        test_df.dropna(subset=["prob_1"])
        .loc[lambda df: df["date"] <= cutoff]
        .sort_values("date")
        .groupby("dev")["prob_1"]
        .last()
    )



    rows = []
    for dev in current_probs.index:
        prob                   = float(current_probs[dev])
        risk_label, risk_color = get_risk_level(prob)
        rows.append({
            "dev_id":             dev,
            "Contributor":        get_display_name(dev, f"{org}/{repo}"),
            "Risk Level":         risk_label,
            "risk_color":         risk_color,
            "Next Break Length":  get_next_break_length(dev, org, repo),
            "Expected Timeframe": f"~{_MEAN_LEAD_TIME_DAYS:.0f} days",
            "Impact":             "TBD",
            "impact_color":       "transparent",
            "prob_1":             prob,
        })

    return pd.DataFrame(rows).sort_values("prob_1", ascending=False).reset_index(drop=True)


def render_risk_table(repo_full_name: str, selected_date: datetime.date):
    """
    Render the Break Risk Overview table as a selectable dataframe.
    Returns (event, risk_df). event has .selection.rows for the selected row index.
    Both empty/None if no data.
    """
    risk_df = build_risk_table(repo_full_name, selected_date)

    if risk_df is None or risk_df.empty:
        st.info(
            "No model predictions found for this repo yet.  \n"
            "Run the **Predictors** pipeline on the Pipeline page to generate predictions."
        )
        return None, pd.DataFrame()

    # Filter to TF developers if tf_devs.csv exists
    tf_devs_path = ORG_BASE / repo_full_name / "Results" / "tf_devs.csv"
    if tf_devs_path.exists():
        tf_devs = pd.read_csv(tf_devs_path)
        if "developer" in tf_devs.columns:
            risk_df = risk_df[risk_df["dev_id"].isin(tf_devs["developer"])]

    if risk_df.empty:
        st.warning("No TF developer predictions found for this repo.")
        return None, pd.DataFrame()

    # Enrich with Role from STN metrics
    metrics_df = _load_stn_metrics(repo_full_name)
    names_df   = _load_dev_names(repo_full_name)
    if not metrics_df.empty and "user" in metrics_df.columns and "role" in metrics_df.columns:
        # Build login→role map from STN metrics
        login_role = dict(zip(metrics_df["user"], metrics_df["role"]))
        # Map dev_id → login via dev_names
        def _get_role(dev_id):
            if not names_df.empty:
                row = names_df[names_df["dev_id"] == dev_id]
                if not row.empty:
                    login = row.iloc[0].get("login", "")
                    if login and login in login_role:
                        return login_role[login]
            return "Unknown"
        risk_df["Role"] = risk_df["dev_id"].apply(_get_role)
    else:
        risk_df["Role"] = "Unknown"

    # Add Prob% column
    risk_df["Prob (%)"] = (risk_df["prob_1"] * 100).round(1).astype(str) + "%"

    def _color_cell(val):
        colors = {
            "High": "#e74c3c", "Medium": "#f1c40f", "Low": "#2ecc71",
        }
        bg = colors.get(val, "transparent")
        text = "white" if val == "High" else "black"
        return f"background-color: {bg}; color: {text}; font-weight: bold"

    display_cols = ["Contributor", "Role", "Risk Level", "Prob (%)", "Next Break Length", "Expected Timeframe"]
    styled = (
        risk_df[display_cols]
        .style
        .map(_color_cell, subset=["Risk Level"])
    )
    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="dev_selector_table",
    )

    selected_dev = None

    selected_rows = event.selection.rows  # list of selected row indices, e.g. [2]

    if selected_rows:
        selected_index = selected_rows[0]
        selected_dev = risk_df.iloc[selected_index]["Contributor"]

    return event, risk_df, selected_dev

# ─────────────────────────────────────────────────────────────────────────────
# T8: ACTIVITY CHART
# ─────────────────────────────────────────────────────────────────────────────



def _make_state_graph(df: pd.DataFrame, repo_full_name: str = ""):
    import matplotlib.patches as mpatches

    if df is None or df.empty:
        st.info("No data to display yet.")
        return

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    has_probs = len(prob_cols) > 0

    raw_devs = sorted(df["dev"].dropna().unique())
    # Build display-name → raw dev_id mapping
    dev_display_map = {get_display_name(d, repo_full_name): d for d in raw_devs}
    display_names   = sorted(dev_display_map.keys())
    selected_display = st.selectbox("Select developer", display_names, key="chart_dev")
    dev = dev_display_map[selected_display]

    wdf = df[df["dev"] == dev].copy()
    wdf["date"]  = pd.to_datetime(wdf["date"])
    wdf = wdf.sort_values("date").reset_index(drop=True)
    wdf["state"] = wdf["state"].fillna("UNKNOWN")

    for col in ("commits", "prs", "issues", "issue_activity", "pr_activity"):
        wdf[col] = pd.to_numeric(wdf[col], errors="coerce").fillna(0)

    wdf["coding_total"]       = wdf["commits"] + wdf["prs"]
    wdf["noncoding_total"]    = wdf["issues"] + wdf["issue_activity"] + wdf["pr_activity"]
    wdf["commits_log"]        = np.log1p(wdf["commits"])
    wdf["prs_log"]            = np.log1p(wdf["prs"])
    wdf["issues_log"]         = np.log1p(wdf["issues"])
    wdf["issue_activity_log"] = np.log1p(wdf["issue_activity"])
    wdf["pr_activity_log"]    = np.log1p(wdf["pr_activity"])
    wdf["activity_log"]       = np.log1p(wdf["coding_total"])
    wdf["non_coding_log"]     = np.log1p(wdf["noncoding_total"])

    min_date = wdf["date"].min()
    max_date = wdf["date"].max()

    window = st.selectbox("Window (years)", range(1, 25), key="chart_window") * 365
    default_start  = (max_date - pd.Timedelta(days=window)).to_pydatetime().date()
    win_start_date = st.slider(
        "Window start",
        min_value=min_date.to_pydatetime().date(),
        max_value=default_start,
        value=default_start,
        step=timedelta(days=7),
        key="chart_slider",
    )
    win_start = pd.Timestamp(win_start_date)
    win_end   = win_start + pd.Timedelta(days=window)
    mask      = (wdf["date"] >= win_start) & (wdf["date"] <= win_end)
    wdf_w     = wdf[mask].copy()

    BG            = "#1c1c2e"
    n_panels      = 4 if has_probs else 3
    height_ratios = [0.4, 2, 2, 2] if has_probs else [0.4, 2, 2]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(14, 11 if has_probs else 9),
        facecolor=BG, gridspec_kw={"height_ratios": height_ratios},
    )
    fig.suptitle(f"Developer: {selected_display}", color="white", fontsize=12)
    fig.subplots_adjust(hspace=0.6)

    # Panel 0 — full timeline strip
    ax0 = axes[0]
    ax0.set_facecolor(BG)
    _shade_states(ax0, wdf, "date", "state", alpha=1.0)
    ax0.axvspan(win_start, win_end, color="black", alpha=0.5, zorder=2)
    ax0.set_xlim(min_date, max_date)
    ax0.set_yticks([])
    ax0.set_title("Full timeline  (dark band = current window)", color="white", fontsize=9, pad=4)
    ax0.tick_params(colors="white", labelsize=8)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.xaxis.set_major_locator(mdates.YearLocator())
    for sp in ax0.spines.values():
        sp.set_visible(False)

    # Panel 1 — coding activity
    ax1 = axes[1]
    ax1.set_facecolor(BG)
    _shade_states(ax1, wdf_w, "date", "state", alpha=0.25)
    active = wdf_w[wdf_w["activity_log"] > 0]
    if not active.empty:
        ax1.bar(active["date"], active["commits_log"], width=1, color="#2ecc71", alpha=0.9, zorder=3)
        ax1.bar(active["date"], active["prs_log"],     width=1, color="#f1c40f", alpha=0.9, zorder=3)
    ax1.set_xlim(win_start, win_end)
    ax1.set_ylabel("log(commits + PRs)", color="white", fontsize=9)
    ax1.set_title("Coding Activity", color="white", fontsize=10, pad=4)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.legend(["commits", "PRs"], loc="upper left", facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
    for sp in ax1.spines.values():
        sp.set_color("#444")

    # Panel 2 — non-coding activity
    ax2 = axes[2]
    ax2.set_facecolor(BG)
    _shade_states(ax2, wdf_w, "date", "state", alpha=0.25)
    nc = wdf_w[wdf_w["non_coding_log"] > 0]
    if not nc.empty:
        ax2.bar(nc["date"], nc["issues_log"],         width=1, color="#e74c3c", alpha=0.9, zorder=3)
        ax2.bar(nc["date"], nc["issue_activity_log"], width=1, color="#c0392b", alpha=0.9, zorder=3)
        ax2.bar(nc["date"], nc["pr_activity_log"],    width=1, color="#f1c40f", alpha=0.9, zorder=3)
    ax2.set_xlim(win_start, win_end)
    ax2.set_ylabel("log(issues + comments)", color="white", fontsize=9)
    ax2.set_title("Non-Coding Activity", color="white", fontsize=10, pad=4)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.legend(["issues", "issue activity", "PR activity"], loc="upper left", facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    for sp in ax2.spines.values():
        sp.set_color("#444")

    # Panel 3 — break probability (only if model predictions exist)
    if has_probs:
        ax3 = axes[3]
        ax3.set_facecolor(BG)
        prob_w    = wdf_w.dropna(subset=prob_cols).sort_values("date")
        is_binary = sorted(prob_cols) == ["prob_0", "prob_1"]

        if not prob_w.empty:
            if is_binary:
                ax3.plot(prob_w["date"], prob_w["prob_1"],
                         color="#e67e22", linewidth=1.5, label="P(break in 14d)")
                if "break_starts_in_14d" in prob_w.columns:
                    ax3.plot(prob_w["date"], prob_w["break_starts_in_14d"].astype(float),
                             color="#2ecc71", linewidth=1.5, label="True break in 14d")
                ax3.axhline(0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
                ax3.legend(loc="upper left", facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
            else:
                palette_mc   = [_STATE_COLORS.get(cl.replace("prob_", ""), "#888") for cl in prob_cols]
                class_labels = [c.replace("prob_", "") for c in prob_cols]
                ax3.stackplot(prob_w["date"].values, prob_w[prob_cols].values.T,
                              labels=class_labels, colors=palette_mc, alpha=0.25)
                ax3.legend(loc="upper left", facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
        else:
            ax3.text(0.5, 0.5, "No predictions in this window",
                     ha="center", va="center", color="white", transform=ax3.transAxes, fontsize=9)

        ax3.set_xlim(win_start, win_end)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Probability", color="white", fontsize=9)
        ax3.set_title("Break Probability  (LSTM)", color="white", fontsize=10, pad=4)
        ax3.tick_params(colors="white", labelsize=8)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax3.get_xticklabels(), rotation=30, ha="right")
        for sp in ax3.spines.values():
            sp.set_color("#444")

    patches = [mpatches.Patch(color=c, label=s) for s, c in _STATE_COLORS.items()]
    fig.legend(handles=patches, loc="upper right", ncol=5,
               facecolor=BG, labelcolor="white", fontsize=8, framealpha=0)
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# DEPARTURE SIMULATION PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_simulation_panel(repo_full_name: str, selected_dev_id: str | None):
    """
    Render the departure simulation panel for the selected developer.

    Uses the pre-computed edge list and metrics from SocialTechnicalNetwork.
    Three sections:
      1) Developer profile — role, betweenness, articulation point, degree
      2) Network impact    — connected components before/after, isolated users
      3) Most affected collaborators table
    """
    # Import STN from same directory
    _ext_dir = str(Path(__file__).resolve().parent)
    if _ext_dir not in sys.path:
        sys.path.insert(0, _ext_dir)
    import SocialTechnicalNetwork as stn

    if not selected_dev_id:
        st.info("Select a contributor above to run the departure simulation.")
        return

    edge_df    = _load_edge_list(repo_full_name)
    metrics_df = _load_stn_metrics(repo_full_name)

    if edge_df.empty or metrics_df.empty:
        st.warning(
            "No Social-Technical Network data found.  \n"
            "Run the STN pipeline step first to generate it."
        )
        return

    # Ensure backward-compat: older CSVs lack weight_reviews column
    if "weight_reviews" not in edge_df.columns:
        edge_df["weight_reviews"] = 0

    # Resolve dev_id → login
    selected_login = None
    names_df = _load_dev_names(repo_full_name)
    if not names_df.empty:
        sel_row = names_df[names_df["dev_id"] == selected_dev_id]
        if not sel_row.empty:
            selected_login = sel_row.iloc[0].get("login")
    if not selected_login:
        selected_login = selected_dev_id.split("|")[-1] if "|" in selected_dev_id else selected_dev_id

    # Check login exists in network
    if "user" in metrics_df.columns and selected_login not in metrics_df["user"].values:
        dev_display = get_display_name(selected_dev_id, repo_full_name)
        st.warning(
            f"**{dev_display}** has no recorded social interactions in the network.  \n"
            "They may not have participated in any issue or PR threads."
        )
        return

    dev_display = get_display_name(selected_dev_id, repo_full_name)
    st.markdown(f"#### Simulating departure of: **{dev_display}**")

    with st.spinner("Running simulation..."):
        # Use full network history (no temporal filter) since raw threads aren't cached here
        result = stn.simulate_departure(
            departing_developer=selected_login,
            edge_df=edge_df,
            metrics_df=metrics_df,
            issue_threads={},
            pr_threads={},
            lookback_days=None,
        )

    if "error" in result:
        st.warning(result["error"])
        return

    # ── Section 1: Developer profile ─────────────────────────────────────────
    st.markdown("**Developer Profile**")

    role = None
    if "role" in metrics_df.columns:
        dev_row_m = metrics_df[metrics_df["user"] == selected_login]
        if not dev_row_m.empty:
            role = dev_row_m.iloc[0]["role"]

    role_color = ROLE_COLORS.get(role, "#555555") if role else "#555555"
    risk_label = ROLE_RISK.get(role, "unknown") if role else "unknown"

    st.markdown(
        f"<div style='display:inline-block;background:{role_color};padding:6px 16px;"
        f"border-radius:8px;color:white;font-weight:bold;font-size:1.1em'>"
        f"{role or 'Unknown'} &nbsp;<span style='font-weight:normal;font-size:0.85em'>"
        f"Risk: {risk_label}</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")  # spacing

    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.metric("Degree", result["degree"])
    with pc2:
        st.metric("Weighted Degree", result["weighted_degree"])
    with pc3:
        st.metric("Betweenness", f"{result['betweenness_centrality']:.4f}")
    with pc4:
        ap_val = result["is_articulation_point"]
        st.metric("Articulation Point", "Yes" if ap_val else "No",
                  help="If Yes, removing this developer will disconnect the network into separate groups.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.metric("Issue Focus", f"{result['issue_focus_pct']*100:.0f}%")
    with fc2:
        st.metric("PR Focus", f"{result['pr_focus_pct']*100:.0f}%")
    with fc3:
        bridge_txt = "Yes" if result["is_community_bridge"] else "No"
        st.metric("Community Bridge", bridge_txt,
                  help=f"Connects {result['communities_spanned']} communities.")

    if result["is_articulation_point"]:
        st.error("This developer is an **articulation point** — their departure will disconnect the network.")
    elif result["is_community_bridge"]:
        st.warning("This developer is a **community bridge** — their departure degrades cross-team collaboration.")

    # ── Section 2: Network impact ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Network Impact After Departure**")

    c_before = result["components_before"]
    c_after  = result["components_after"]
    splits   = result["new_splits"]
    isolated = result["isolated_users"]
    groups   = result["disconnected_groups"]

    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        delta_str = f"+{splits}" if splits > 0 else str(splits)
        st.metric("Connected Components", c_after, delta=delta_str,
                  delta_color="inverse" if splits > 0 else "off")
    with nc2:
        st.metric("Isolated Developers", len(isolated))
    with nc3:
        st.metric("Disconnected Groups", len(groups))

    if isolated:
        names = [_dn_login(l, repo_full_name) for l in isolated]
        st.markdown(f"**Isolated developers:** {', '.join(names)}")

    if groups:
        for i, grp in enumerate(groups, 1):
            grp_names = [_dn_login(l, repo_full_name) for l in grp]
            st.markdown(f"**Group {i}** ({len(grp)} devs): {', '.join(grp_names)}")

    # ── Section 3: Most affected collaborators ────────────────────────────────
    st.markdown("---")
    st.markdown("**Most Affected Collaborators**")

    affected = result.get("most_affected", [])
    if affected:
        aff_rows = []
        for a in affected:
            aff_rows.append({
                "Collaborator":   _dn_login(a["user"], repo_full_name),
                "Shared Threads": a["lost_weight"],
                "Their Total":    a["their_total_wd"],
                "% Network Lost": f"{a['pct_wd_lost']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(aff_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No collaborators found for this developer in the network.")


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE DISTRIBUTION PANEL
# ─────────────────────────────────────────────────────────────────────────────

_KD_IMPORTANT_EXT: set = {
    ".py", ".r", ".rmd", ".rnw",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".scala",
    ".cpp", ".c", ".h", ".hpp", ".cc",
    ".cs", ".go", ".rs", ".swift",
    ".rb", ".php", ".sh", ".bash",
    ".sql", ".vue", ".svelte",
}

_KD_LANG_MAP: dict = {
    ".py": "Python", ".r": "R", ".rmd": "R", ".rnw": "R",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java", ".kt": "Kotlin", ".scala": "Scala",
    ".cpp": "C++", ".c": "C", ".h": "C", ".hpp": "C++", ".cc": "C++",
    ".cs": "C#", ".go": "Go", ".rs": "Rust", ".swift": "Swift",
    ".rb": "Ruby", ".php": "PHP",
    ".sh": "Shell", ".bash": "Shell", ".sql": "SQL",
    ".vue": "Vue", ".svelte": "Svelte",
    ".csv": "CSV", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".md": "Markdown", ".txt": "Text", ".html": "HTML", ".css": "CSS",
}


@st.cache_data(show_spinner=False)
def _load_knowledge_data(repo_full_name: str):
    """Load DOE CSV and truck-factor JSON for a repo."""
    base     = ORG_BASE / repo_full_name / "KnowledgeDistribution"
    doe_path = base / "doe.csv"
    tf_path  = base / cfg.truck_factor_file
    if not doe_path.exists() or not tf_path.exists():
        return 0, [], pd.DataFrame()
    doe_df = pd.read_csv(doe_path)
    with open(tf_path, "r") as f:
        tf_data = json.load(f)
    return int(tf_data.get("tf", 0)), tf_data.get("tf_list", []), doe_df


def _kd_shorten_dev_id(dev_id: str) -> str:
    if not dev_id:
        return "unknown"
    if "|" in dev_id:
        return dev_id.split("|", 1)[1].split("@")[0]
    return str(dev_id)


_KD_MAX_FILES = 2000   # cap to keep the HTML payload under ~3 MB

def _kd_build_files_json(doe_df: pd.DataFrame) -> list:
    if doe_df.empty:
        return []
    files = []
    for file_path, group in doe_df.groupby("file_path"):
        expertise: dict = {}
        for _, row in group.iterrows():
            short   = _kd_shorten_dev_id(str(row["developer"]))
            doe_val = float(row["DOE"])
            expertise[short] = round(expertise.get(short, 0.0) + doe_val, 3)
        expertise = {k: v for k, v in expertise.items() if v > 0}
        if not expertise:
            continue
        ext       = Path(str(file_path)).suffix.lower()
        lang      = _KD_LANG_MAP.get(ext, "Other")
        important = ext in _KD_IMPORTANT_EXT
        files.append({"path": str(file_path), "expertise": expertise,
                      "lang": lang, "important": important})

    if len(files) <= _KD_MAX_FILES:
        return files

    # Cap: keep all important files first, then fill with non-important up to limit
    important_files = [f for f in files if f["important"]]
    other_files     = [f for f in files if not f["important"]]

    if len(important_files) > _KD_MAX_FILES:
        # Too many important files — keep the ones with the highest max DOE
        important_files.sort(
            key=lambda f: max(f["expertise"].values()), reverse=True
        )
        return important_files[:_KD_MAX_FILES]

    remaining = _KD_MAX_FILES - len(important_files)
    other_files.sort(key=lambda f: max(f["expertise"].values()), reverse=True)
    return important_files + other_files[:remaining]


def _build_knowledge_html(repo_full_name: str, tf: int, tf_list: list,
                           doe_df: pd.DataFrame, selected_dev: str ) -> str:
    files_json    = _kd_build_files_json(doe_df)
    tf_devs_short = [_kd_shorten_dev_id(d) for d in (tf_list or [])]
    selected_dev_short = _kd_shorten_dev_id(selected_dev) if selected_dev else None
    meta = {
        "repo":         repo_full_name,
        "tf":           int(tf),
        "tf_devs":      tf_devs_short,
        "file_count":   len(files_json),
        "selected_dev": selected_dev_short,   # shortened login passed to JS
    }
    data_block = (
        f"    window.REPO_META  = {json.dumps(meta, ensure_ascii=False)};\n"
        f"    window.REPO_FILES = {json.dumps(files_json, ensure_ascii=False)};\n"
    )
    return _knowledge_html_template().replace("/* __DATA_INJECTION__ */", data_block)


def render_knowledge_panel(repo_full_name: str, selected_dev: str, height: int = 800  ) -> None:
    """Render the Knowledge Distribution dashboard inside Streamlit."""

    # ── Stage 1: file paths ──────────────────────────────────────────────────
    base     = ORG_BASE / repo_full_name / "KnowledgeDistribution"
    doe_path = base / "doe.csv"
    tf_path  = base / cfg.truck_factor_file


    tf, tf_list, doe_df = _load_knowledge_data(repo_full_name)

    if doe_df.empty:
        st.error("doe_df is EMPTY — stopping here.")
        st.info(
            "No Knowledge Distribution data found for this repo.  \n"
            "Run the **Predictors** step on the Pipeline page to generate it."
        )
        return
        
    files_json = _kd_build_files_json(doe_df)
    if not files_json:
        st.error("files_json is EMPTY — all DOE values may be ≤ 0, or column names differ.")

    # ── Stage 4: HTML injection check + size ────────────────────────────
    PLACEHOLDER = "/* __DATA_INJECTION__ */"
    template = _knowledge_html_template()
    placeholder_found = PLACEHOLDER in template
    
    if doe_df.empty:
        return

    html_str = _build_knowledge_html(repo_full_name, tf, tf_list, doe_df, selected_dev)
    components.html(html_str, height=height, scrolling=True)


# ─────────────────────────────────────────────────────────────────────────────
# PROJECT HEALTH PANEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_ph_data(repo_full_name: str):
    """Load project_health.json for a repo.  Returns None if not yet computed."""
    path = ORG_BASE / repo_full_name / cfg.project_health_folder / cfg.project_health_file
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _build_ph_html(ph_data: dict, dev_id: str, display_name: str,
                   break_weeks: int) -> str:
    """Inject real data into the Project Health HTML template."""
    devs     = ph_data.get("developers", {})
    dev_data = devs.get(dev_id)

    # fuzzy fall-back: substring match
    if dev_data is None:
        for k, v in devs.items():
            if dev_id in k or k in dev_id:
                dev_data = v
                break

    if dev_data is None:
        return ""

    n_weeks     = len(ph_data["weeks"])
    weeks       = ph_data["weeks"]
    proj_labels = [f"Week +{i+1}" for i in range(break_weeks)]
    all_labels  = weeks + proj_labels
    is_absence  = [False] * n_weeks + [True] * break_weeks

    commits = dev_data["weekly_commits"] + [0] * break_weeks
    prs     = dev_data["weekly_prs"]     + [0] * break_weeks
    issues  = dev_data["weekly_issues"]  + [0] * break_weeks

    bc = dev_data["baseline_commits"]
    bp = dev_data["baseline_prs"]
    bi = dev_data["baseline_issues"]

    totals = ph_data.get("repo_totals", {})
    tc = totals.get("commits_per_week", 1) or 1
    tp = totals.get("prs_per_week",     1) or 1
    ti = totals.get("issues_per_week",  1) or 1

    def _pct(v, t):  return round(v / t * 100) if t else 0
    def _badge(p):
        if p < 10:  return "low",  "Low"
        if p < 25:  return "med",  "Medium"
        if p < 40:  return "high", "High"
        return            "crit", "Critical"

    c_cls, c_lbl = _badge(_pct(bc, tc))
    p_cls, p_lbl = _badge(_pct(bp, tp))
    i_cls, i_lbl = _badge(_pct(bi, ti))

    lost_c = round(bc * break_weeks, 1)
    lost_p = round(bp * break_weeks, 1)
    lost_i = round(bi * break_weeks, 1)
    max_pct = max(_pct(bc, tc), _pct(bp, tp), _pct(bi, ti))

    summary = (
        f"{display_name} contributes ~{max_pct}% of repo activity. "
        f"A {break_weeks}-week absence is projected to reduce activity by "
        f"~{lost_c} commits, ~{lost_p} PRs, and ~{lost_i} issues."
    )

    payload = {
        "dev_name":    display_name,
        "repo":        ph_data.get("repo", ""),
        "break_weeks": break_weeks,
        "snap_date":   ph_data.get("generated_at", "")[:10],
        "labels":      all_labels,
        "is_absence":  is_absence,
        "commits":     commits,
        "prs":         prs,
        "issues":      issues,
        "baseline_c":  bc,  "pct_c": _pct(bc, tc),  "badge_c": c_cls,  "label_c": c_lbl,
        "baseline_p":  bp,  "pct_p": _pct(bp, tp),  "badge_p": p_cls,  "label_p": p_lbl,
        "baseline_i":  bi,  "pct_i": _pct(bi, ti),  "badge_i": i_cls,  "label_i": i_lbl,
        "summary":     summary,
    }

    data_block = f"const PH_DATA = {json.dumps(payload)};"
    return _ph_html_template().replace("/* __PH_DATA__ */", data_block)


def render_project_health_panel(repo_full_name: str, selected_dev_id: str | None,
                                 break_weeks: int = 2, height: int = 530) -> None:
    """Render the Project Health impact panel inside Streamlit."""
    ph_data = _load_ph_data(repo_full_name)

    if ph_data is None:
        st.info(
            "No Project Health data found for this repo.  \n"
            "Run the **Predictors** step on the Pipeline page to generate it."
        )
        return

    if not selected_dev_id:
        st.info("Select a contributor above to view their project health impact.")
        return

    devs     = ph_data.get("developers", {})
    dev_data = devs.get(selected_dev_id)
    if dev_data is None:
        for k in devs:
            if selected_dev_id in k or k in selected_dev_id:
                selected_dev_id = k
                dev_data = devs[k]
                break

    if dev_data is None:
        st.info("No weekly activity data found for this developer.")
        return

    display_name = get_display_name(selected_dev_id, repo_full_name)
    html_str = _build_ph_html(ph_data, selected_dev_id, display_name, break_weeks)
    if html_str:
        components.html(html_str, height=height, scrolling=False)


def _ph_html_template() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px; background: transparent; color: #2c3142; padding: 4px 0 12px;
}
.ph-header { margin-bottom: 14px; }
.ph-header h2 { font-size: 16px; font-weight: 600; margin: 0 0 3px; color: #2c3142; }
.ph-header p  { font-size: 12px; color: #6b7280; margin: 0; }

.ph-cards {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
  margin-bottom: 14px;
}
.ph-card {
  background: #f8f9fc; border: 1px solid #e8eaf0;
  border-radius: 8px; padding: 12px 14px;
}
.ph-card-label { font-size: 11px; color: #6b7280; margin-bottom: 3px; }
.ph-card-value { font-size: 24px; font-weight: 600; color: #2c3142; margin-bottom: 2px; }
.ph-card-sub   { font-size: 11px; color: #9ca3af; }
.ph-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 2px 8px; border-radius: 20px; margin-top: 6px;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.badge-crit { background: #fff0f0; color: #b91c1c; border: 1px solid #fecaca; }
.badge-high { background: #fff4ed; color: #c2410c; border: 1px solid #fed7aa; }
.badge-med  { background: #fefce8; color: #92400e; border: 1px solid #fde68a; }
.badge-low  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

.ph-charts {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
  margin-bottom: 12px;
}
.ph-chart-wrap {
  background: #fff; border: 1px solid #e8eaf0;
  border-radius: 8px; padding: 10px 10px 6px;
}
.ph-chart-title { font-size: 12px; font-weight: 600; color: #2c3142; margin-bottom: 8px; }
.ph-legend {
  display: flex; gap: 12px; margin-top: 6px;
  font-size: 10px; color: #9ca3af;
}
.ph-legend span { display: flex; align-items: center; gap: 4px; }
.ph-leg-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }

.ph-summary {
  font-size: 12px; color: #6b7280; line-height: 1.6;
  padding: 8px 12px; border-left: 3px solid #4178f0;
  background: #f0f4ff; border-radius: 0 6px 6px 0;
}
</style>
</head>
<body>

<div class="ph-header">
  <h2 id="ph-title">Project Health</h2>
  <p  id="ph-sub"></p>
</div>

<div class="ph-cards">
  <div class="ph-card">
    <div class="ph-card-label">Avg weekly commits</div>
    <div class="ph-card-value" id="val-c">—</div>
    <div class="ph-card-sub"   id="sub-c">—</div>
    <span class="ph-badge" id="badge-c"></span>
  </div>
  <div class="ph-card">
    <div class="ph-card-label">Avg weekly PRs</div>
    <div class="ph-card-value" id="val-p">—</div>
    <div class="ph-card-sub"   id="sub-p">—</div>
    <span class="ph-badge" id="badge-p"></span>
  </div>
  <div class="ph-card">
    <div class="ph-card-label">Avg weekly issues</div>
    <div class="ph-card-value" id="val-i">—</div>
    <div class="ph-card-sub"   id="sub-i">—</div>
    <span class="ph-badge" id="badge-i"></span>
  </div>
</div>

<div class="ph-charts">
  <div class="ph-chart-wrap">
    <div class="ph-chart-title">Weekly commits</div>
    <div style="position:relative;height:160px"><canvas id="cChart"></canvas></div>
    <div class="ph-legend">
      <span><span class="ph-leg-dot" style="background:#378ADD"></span>Historical</span>
      <span><span class="ph-leg-dot" style="background:#F09595"></span>Projected absence</span>
    </div>
  </div>
  <div class="ph-chart-wrap">
    <div class="ph-chart-title">Weekly PRs opened</div>
    <div style="position:relative;height:160px"><canvas id="prChart"></canvas></div>
    <div class="ph-legend">
      <span><span class="ph-leg-dot" style="background:#378ADD"></span>Historical</span>
      <span><span class="ph-leg-dot" style="background:#F09595"></span>Projected absence</span>
    </div>
  </div>
  <div class="ph-chart-wrap">
    <div class="ph-chart-title">Weekly issues</div>
    <div style="position:relative;height:160px"><canvas id="issChart"></canvas></div>
    <div class="ph-legend">
      <span><span class="ph-leg-dot" style="background:#378ADD"></span>Historical</span>
      <span><span class="ph-leg-dot" style="background:#F09595"></span>Projected absence</span>
    </div>
  </div>
</div>

<div class="ph-summary" id="ph-summary"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
/* __PH_DATA__ */

document.getElementById('ph-title').textContent =
  'Project Health — ' + PH_DATA.dev_name;
document.getElementById('ph-sub').textContent =
  'Predicted absence: ' + PH_DATA.break_weeks + ' week(s)'
  + '  ·  Data as of ' + PH_DATA.snap_date
  + '  ·  ' + PH_DATA.repo;

function setCard(valId, subId, badgeId, baseline, pct, badgeCls, badgeLbl) {
  document.getElementById(valId).textContent  = baseline.toFixed(1);
  document.getElementById(subId).textContent  = pct + '% of repo total';
  var el = document.getElementById(badgeId);
  el.textContent  = badgeLbl + ' impact';
  el.className    = 'ph-badge badge-' + badgeCls;
}
setCard('val-c','sub-c','badge-c', PH_DATA.baseline_c, PH_DATA.pct_c, PH_DATA.badge_c, PH_DATA.label_c);
setCard('val-p','sub-p','badge-p', PH_DATA.baseline_p, PH_DATA.pct_p, PH_DATA.badge_p, PH_DATA.label_p);
setCard('val-i','sub-i','badge-i', PH_DATA.baseline_i, PH_DATA.pct_i, PH_DATA.badge_i, PH_DATA.label_i);

document.getElementById('ph-summary').textContent = PH_DATA.summary;

function barColors(isAbsence) {
  return isAbsence.map(a => a ? '#F09595' : '#378ADD');
}
function makeChart(id, data, maxY) {
  new Chart(document.getElementById(id), {
    type: 'bar',
    data: {
      labels: PH_DATA.labels,
      datasets: [{ data: data, backgroundColor: barColors(PH_DATA.is_absence),
                   borderRadius: 3, borderSkipped: false }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: {
          label: ctx => PH_DATA.is_absence[ctx.dataIndex]
            ? ' Projected absence' : ' ' + ctx.parsed.y
        }}
      },
      scales: {
        x: { grid: { display: false },
             ticks: { font: { size: 9 }, color: '#888780', maxRotation: 45, autoSkip: true, maxTicksLimit: 8 }},
        y: { grid: { color: 'rgba(136,135,128,0.12)' },
             ticks: { font: { size: 9 }, color: '#888780', stepSize: 1 },
             min: 0, max: maxY }
      }
    }
  });
}

var maxC = Math.max(...PH_DATA.commits.filter((_,i) => !PH_DATA.is_absence[i]), 1);
var maxP = Math.max(...PH_DATA.prs.filter((_,i)     => !PH_DATA.is_absence[i]), 1);
var maxI = Math.max(...PH_DATA.issues.filter((_,i)  => !PH_DATA.is_absence[i]), 1);

makeChart('cChart',  PH_DATA.commits, Math.ceil(maxC * 1.3));
makeChart('prChart', PH_DATA.prs,     Math.ceil(maxP * 1.3));
makeChart('issChart',PH_DATA.issues,  Math.ceil(maxI * 1.3));
</script>
</body>
</html>"""


def _timeline_html_template() -> str:
    return """<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 12px; background: transparent; color: #2c3142;
    padding: 4px 0 4px;
    }
    .tl-wrap {
    position: relative; width: 100%;
    }
    .tl-canvas-wrap {
    position: relative; width: 100%; height: 120px;
    }
    .tl-info {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    margin-top: 6px; font-size: 11px; color: #6b7280;
    }
    .tl-info-item { display: flex; align-items: center; gap: 4px; }
    .tl-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
    .tl-toggle {
    margin-left: auto; font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 4px;
    background: #f0f4ff; color: #4178f0;
    border: 1px solid #c7d8fb; cursor: pointer;
    }
    .tl-toggle:hover { background: #e0eaff; }
    .tl-empty {
    height: 120px; display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: #9ca3af;
    border: 1px dashed #e0e0e0; border-radius: 6px;
    }
    </style>
    </head>
    <body>
    <div class="tl-wrap">
    <div class="tl-canvas-wrap">
        <div id="tl-empty-msg" class="tl-empty" style="display:none">No commit history available for this repo.</div>
        <canvas id="tl-chart" style="display:block;width:100%;height:100%"></canvas>
    </div>
    <div class="tl-info">
        <div class="tl-info-item">
        <div class="tl-dot" style="background:#4178f0"></div>
        <span id="lbl-snapshot"></span>
        </div>
        <div class="tl-info-item">
        <div class="tl-dot" style="background:#f59e0b; opacity:0.6"></div>
        <span id="lbl-window"></span>
        </div>
        <div class="tl-info-item" style="color:#9ca3af">
        <span id="lbl-repo"></span>
        </div>
        <button class="tl-toggle" id="tl-toggle-btn" style="display:none" onclick="toggleView()">Dev view</button>
    </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    /* __TIMELINE_DATA__ */

    var currentView = 'repo';
    var chart = null;

    function toDate(ms) { return new Date(ms); }

    function buildColors(pairs, windowStartMs, selectedMs) {
    return pairs.map(function(p) {
        var t = p[0];
        if (t >= windowStartMs && t <= selectedMs) return 'rgba(245,158,11,0.55)';
        return 'rgba(180,190,210,0.4)';
    });
    }

    function buildChart(pairs) {
    var canvas = document.getElementById('tl-chart');
    var emptyMsg = document.getElementById('tl-empty-msg');

    if (!pairs || pairs.length === 0) {
        canvas.style.display = 'none';
        emptyMsg.style.display = 'flex';
        return;
    }
    canvas.style.display = 'block';
    emptyMsg.style.display = 'none';

    var selectedMs   = new Date(TIMELINE_DATA.selected_date).getTime();
    var windowStMs   = new Date(TIMELINE_DATA.window_start).getTime();
    var labels       = pairs.map(function(p){ return toDate(p[0]); });
    var counts       = pairs.map(function(p){ return p[1]; });
    var bgColors     = buildColors(pairs, windowStMs, selectedMs);

    if (chart) { chart.destroy(); }

    chart = new Chart(canvas, {
        type: 'bar',
        data: {
        labels: labels,
        datasets: [{
            data: counts,
            backgroundColor: bgColors,
            borderWidth: 0,
            borderRadius: 1,
            barPercentage: 1.0,
            categoryPercentage: 1.0,
        }]
        },
        options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: { display: false },
            tooltip: {
            callbacks: {
                title: function(items) {
                var d = new Date(items[0].parsed.x);
                return d.toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'});
                },
                label: function(item) { return ' ' + item.parsed.y + ' commits'; }
            }
            }
        },
        scales: {
            x: {
            type: 'time',
            time: { unit: 'month', displayFormats: { month: 'MMM yy' } },
            grid: { display: false },
            ticks: { font: { size: 9 }, color: '#9ca3af', maxTicksLimit: 10, maxRotation: 0 },
            border: { display: false },
            },
            y: {
            display: false,
            grid: { display: false },
            }
        }
        }
    });
    }

    function init() {
    // Set info labels
    document.getElementById('lbl-snapshot').textContent =
        'Snapshot: ' + TIMELINE_DATA.label_date;
    document.getElementById('lbl-window').textContent =
        'Window: ' + TIMELINE_DATA.label_window_start + ' \u2013 ' + TIMELINE_DATA.label_date +
        ' (' + TIMELINE_DATA.label_window_days + ')';
    document.getElementById('lbl-repo').textContent = TIMELINE_DATA.repo;

    if (TIMELINE_DATA.dev_histogram) {
        document.getElementById('tl-toggle-btn').style.display = 'inline-block';
    }

    buildChart(TIMELINE_DATA.histogram);
    }

    function toggleView() {
    var btn = document.getElementById('tl-toggle-btn');
    if (currentView === 'repo') {
        currentView = 'dev';
        btn.textContent = 'Repo view';
        buildChart(TIMELINE_DATA.dev_histogram || []);
    } else {
        currentView = 'repo';
        btn.textContent = 'Dev view';
        buildChart(TIMELINE_DATA.histogram);
    }
    }

    init();
    </script>
    </body>
    </html>"""


def _build_timeline_html(payload: dict) -> str:
    """Inject timeline payload into the HTML template."""
    data_block = f"const TIMELINE_DATA = {json.dumps(payload)};"
    return _timeline_html_template().replace("/* __TIMELINE_DATA__ */", data_block)


def _knowledge_html_template() -> str:
    return r"""<!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --surface: #ffffff;
        --surface2: #f8f9fc;
        --border: #e8eaf0;
        --border-soft: #f0f1f5;
        --text: #2c3142;
        --text-mid: #6b7280;
        --text-dim: #9ca3af;
        --accent: #4178f0;
        --accent-light: #eef2fd;
        --red: #e53e3e;       --red-light: #fff5f5;    --red-border: #fed7d7;
        --orange: #d97706;    --orange-light: #fffbeb;  --orange-border: #fde68a;
        --yellow: #b45309;    --yellow-light: #fefce8;  --yellow-border: #fef08a;
        --green: #16a34a;     --green-light: #f0fdf4;   --green-border: #bbf7d0;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
        --radius: 10px;
        --radius-sm: 6px;
    }

    body {
        font-family: 'Inter', sans-serif;
        background: transparent;
        color: var(--text);
        font-size: 13px;
        padding: 4px 0 16px;
    }

    .repo-header {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 14px; flex-wrap: wrap;
    }
    .repo-name {
        font-family: 'JetBrains Mono', monospace; font-size: 13px;
        font-weight: 600; color: var(--text); flex: 1;
    }
    .stat-pill {
        font-size: 11px; font-weight: 600; padding: 3px 10px;
        border-radius: 20px; background: var(--surface2);
        border: 1px solid var(--border); color: var(--text-mid);
    }
    .stat-pill.tf     { background: var(--accent-light); color: var(--accent); border-color: #c7d8fb; }
    .stat-pill.danger { background: var(--red-light); color: var(--red); border-color: var(--red-border); }

    .section-label {
        font-size: 11px; font-weight: 600; letter-spacing: 0.07em;
        text-transform: uppercase; color: var(--text-dim);
        margin-bottom: 8px; padding-left: 2px;
    }

    .risk-section { margin-bottom: 16px; }
    .risk-scroll {
        display: flex; gap: 8px; overflow-x: auto;
        padding-bottom: 4px; scrollbar-width: thin;
        scrollbar-color: var(--border) transparent;
    }
    .risk-scroll::-webkit-scrollbar { height: 3px; }
    .risk-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .risk-chip {
        display: flex; flex-direction: column; gap: 3px;
        padding: 9px 12px; background: var(--surface);
        border: 1px solid var(--border); border-top: 2px solid transparent;
        border-radius: var(--radius-sm); cursor: pointer; min-width: 148px;
        flex-shrink: 0; transition: box-shadow 0.15s, transform 0.15s;
        box-shadow: var(--shadow-sm);
    }
    .risk-chip:hover { box-shadow: var(--shadow-md); transform: translateY(-1px); }
    .risk-chip.active { background: var(--accent-light); border-color: #c7d8fb; border-top-color: var(--accent); }
    .risk-chip.risk-red    { border-top-color: var(--red); }
    .risk-chip.risk-red.active    { background: var(--red-light);    border-color: var(--red-border); }
    .risk-chip.risk-orange { border-top-color: var(--orange); }
    .risk-chip.risk-orange.active { background: var(--orange-light); border-color: var(--orange-border); }
    .risk-chip.risk-yellow { border-top-color: var(--yellow); }
    .risk-chip.risk-yellow.active { background: var(--yellow-light); border-color: var(--yellow-border); }

    .chip-top { display: flex; align-items: center; justify-content: space-between; }
    .chip-rank { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); }
    .chip-badge {
        font-size: 9px; font-weight: 600; letter-spacing: 0.04em;
        padding: 1px 6px; border-radius: 20px; text-transform: uppercase;
    }
    .badge-red    { background: var(--red-light);    color: var(--red);    border: 1px solid var(--red-border); }
    .badge-orange { background: var(--orange-light); color: var(--orange); border: 1px solid var(--orange-border); }
    .badge-yellow { background: var(--yellow-light); color: var(--yellow); border: 1px solid var(--yellow-border); }
    .badge-green  { background: var(--green-light);  color: var(--green);  border: 1px solid var(--green-border); }
    .chip-name {
        font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 500;
        color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px;
    }
    .chip-path { font-size: 10px; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    .workspace { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }

    .explorer-card, .metrics-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: var(--radius); box-shadow: var(--shadow-sm); overflow: hidden;
    }
    .card-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 11px 14px; border-bottom: 1px solid var(--border-soft);
    }
    .card-header-title { font-size: 12px; font-weight: 600; color: var(--text); }
    .card-header-sub   { font-size: 11px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }

    .tree-body { padding: 6px 0; max-height: 500px; overflow-y: auto; }
    .tree-body::-webkit-scrollbar { width: 4px; }
    .tree-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

    .tree-node {
        display: flex; align-items: center; padding: 4px 12px 4px 0;
        cursor: pointer; user-select: none; transition: background 0.1s; position: relative;
    }
    .tree-node:hover { background: var(--surface2); }
    .tree-node.selected { background: var(--accent-light); }
    .tree-node.selected::before {
        content: ''; position: absolute; left: 0; top: 0; bottom: 0;
        width: 2px; background: var(--accent);
    }
    .indent-unit { width: 18px; flex-shrink: 0; }
    .tree-arrow {
        width: 14px; height: 14px; display: flex; align-items: center;
        justify-content: center; font-size: 8px; color: var(--text-dim);
        transition: transform 0.15s; flex-shrink: 0;
    }
    .tree-arrow.open { transform: rotate(90deg); }
    .tree-icon { font-size: 13px; margin-right: 6px; flex-shrink: 0; }
    .tree-label {
        font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text);
        flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .tree-node.folder-node > .tree-label { font-weight: 500; }
    .tree-label.dim { color: var(--text-dim); }
    .tree-risk-dot { width: 6px; height: 6px; border-radius: 50%; margin-right: 10px; flex-shrink: 0; }

    .metrics-card.empty-state {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; min-height: 220px; padding: 32px; text-align: center;
    }
    .empty-icon { font-size: 26px; margin-bottom: 10px; opacity: 0.35; }
    .empty-text { font-size: 12px; color: var(--text-dim); line-height: 1.7; }

    .metrics-body { padding: 16px; animation: fadeUp 0.18s ease; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:translateY(0)} }

    .metrics-path { font-size: 11px; color: var(--text-dim); margin-bottom: 12px; }
    .metrics-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
    .lang-tag {
        font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
        padding: 2px 8px; border-radius: 20px;
        background: var(--accent-light); color: var(--accent); border: 1px solid #c7d8fb;
        text-transform: uppercase;
    }

    .risk-alert {
        display: flex; align-items: flex-start; gap: 9px;
        padding: 10px 12px; border-radius: var(--radius-sm);
        margin-bottom: 16px; font-size: 12px; line-height: 1.5;
    }
    .risk-alert.alert-red    { background: var(--red-light);    border: 1px solid var(--red-border);    color: var(--red); }
    .risk-alert.alert-orange { background: var(--orange-light); border: 1px solid var(--orange-border); color: var(--orange); }
    .risk-alert.alert-yellow { background: var(--yellow-light); border: 1px solid var(--yellow-border); color: var(--yellow); }
    .risk-alert.alert-green  { background: var(--green-light);  border: 1px solid var(--green-border);  color: var(--green); }

    .alert-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }
    .alert-dot-red    { background: var(--red); }
    .alert-dot-orange { background: var(--orange); }
    .alert-dot-yellow { background: var(--yellow); }
    .alert-dot-green  { background: var(--green); }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
    .pulse { animation: pulse 1.8s infinite; }

    .section-title {
        font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px;
    }

    .exp-rows { display: flex; flex-direction: column; gap: 8px; margin-bottom: 4px; }
    .exp-row  { display: flex; align-items: center; gap: 8px; }
    .exp-name { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 500; width: 80px; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .exp-track { flex: 1; height: 5px; background: var(--border-soft); border-radius: 3px; overflow: hidden; }
    .exp-fill  { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
    .exp-val   { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--text-dim); width: 36px; text-align: right; flex-shrink: 0; }

    .divider { height: 1px; background: var(--border-soft); margin: 14px 0; }

    .folder-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
    .stat-box { background: var(--surface2); border: 1px solid var(--border-soft); border-radius: var(--radius-sm); padding: 10px 12px; }
    .stat-num   { font-size: 20px; font-weight: 700; color: var(--text); line-height: 1; margin-bottom: 3px; }
    .stat-label { font-size: 10px; color: var(--text-dim); }

    .risk-breakdown { display: flex; flex-direction: column; gap: 7px; }
    .breakdown-row { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--text-mid); }
    .breakdown-left { display: flex; align-items: center; gap: 8px; }
    .breakdown-dot  { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .breakdown-count { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-dim); }

    .watch-row { display: flex; align-items: center; gap: 7px; cursor: pointer; padding: 3px 0; }
    .watch-row:hover .watch-name { color: var(--accent); }
    .watch-name { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text); transition: color 0.12s; }
    </style>
    </head>
    <body>

    <div class="repo-header">
    <div class="repo-name" id="repoName">—</div>
    <div class="stat-pill tf"     id="tfPill">TF —</div>
    <div class="stat-pill"        id="filePill">— files</div>
    <div class="stat-pill danger" id="riskPill">— at risk</div>
    </div>

    <div class="risk-section">
    <div class="section-label">⚠ At-Risk Files — Important &amp; Undermaintained</div>
    <div class="risk-scroll" id="riskList"></div>
    </div>

    <div class="workspace">
    <div class="explorer-card">
        <div class="card-header">
        <div class="card-header-title">File Explorer</div>
        <div class="card-header-sub" id="repoSubtitle">—</div>
        </div>
        <div class="tree-body" id="treeContainer"></div>
    </div>
    <div class="metrics-card empty-state" id="metricsCard">
        <div class="empty-icon">📂</div>
        <div class="empty-text">Click any file or folder<br>to view knowledge metrics</div>
    </div>
    </div>

    <script>
    /* __DATA_INJECTION__ */

    const PALETTE = [
    '#4178f0','#16a34a','#d97706','#9333ea','#0891b2',
    '#db2777','#65a30d','#ea580c','#7c3aed','#0284c7',
    ];
    const devColorCache = {};
    let colorIdx = 0;
    function devColor(name) {
    if (!devColorCache[name]) devColorCache[name] = PALETTE[colorIdx++ % PALETTE.length];
    return devColorCache[name];
    }

    const RISK_ORDER = { red:0, orange:1, yellow:2, green:3 };
    const DOT   = { red:'#e53e3e', orange:'#d97706', yellow:'#b45309', green:'#16a34a' };
    const ICONS = {
    Python:'🐍', R:'📊', JavaScript:'🟨', TypeScript:'🔷',
    Java:'☕', 'C++':'⚙', C:'⚙', 'C#':'🔷', Go:'🐹', Rust:'🦀',
    Ruby:'💎', PHP:'🐘', Shell:'🖥', SQL:'🗄', Vue:'💚', Svelte:'🔶',
    CSV:'📋', JSON:'📋', YAML:'📋', Markdown:'📝', HTML:'🌐', CSS:'🎨', Other:'📄',
    };

    function classifyRisk(expertise) {
    const sorted = Object.entries(expertise).sort((a,b) => b[1]-a[1]);
    const total  = sorted.reduce((t,[,v]) => t+v, 0);
    if (sorted.length === 1)
        return { level:'red',    label:'Sole Expert',      desc:`Only ${sorted[0][0]} understands this file` };
    if (sorted[0][1] / total > 0.80)
        return { level:'orange', label:'Sole Maintainer',  desc:`${sorted[0][0]} dominates — others have minimal knowledge` };
    if (sorted.length === 2)
        return { level:'yellow', label:'Narrow Expertise', desc:`Only ${sorted[0][0]} & ${sorted[1][0]} have meaningful knowledge` };
    return { level:'green', label:'Broad Coverage', desc:`${sorted.length} contributors share knowledge` };
    }

    const META  = window.REPO_META  || { repo:'unknown', tf:0, tf_devs:[], file_count:0 };
    const DEV   = META.selected_dev || null;   // shortened login, e.g. "JeanMeche"

    // All files with risk classification
    const ALL_FILES = (window.REPO_FILES || []).map(f => ({ ...f, risk: classifyRisk(f.expertise) }));

    // If a developer is selected, keep only files where they have any DOE
    const FILES = DEV
        ? ALL_FILES.filter(f => Object.prototype.hasOwnProperty.call(f.expertise, DEV) && f.expertise[DEV] > 0)
        : ALL_FILES;

    const atRisk = FILES
    .filter(f => f.important && f.risk.level !== 'green')
    .sort((a,b) => {
        if (RISK_ORDER[a.risk.level] !== RISK_ORDER[b.risk.level])
        return RISK_ORDER[a.risk.level] - RISK_ORDER[b.risk.level];
        return Math.max(...Object.values(b.expertise)) - Math.max(...Object.values(a.expertise));
    });

    document.getElementById('repoName').textContent     = META.repo;
    document.getElementById('repoSubtitle').textContent  = DEV ? `${META.repo.split('/')[1]}/ — ${DEV}` : META.repo.split('/')[1] + '/';
    document.getElementById('tfPill').textContent        = `TF ${META.tf}`;
    document.getElementById('filePill').textContent      = DEV ? `${FILES.length} / ${ALL_FILES.length} files` : `${FILES.length} files`;
    document.getElementById('riskPill').textContent      = `${atRisk.length} at risk`;

    function buildTree(files) {
    const root = { name: META.repo.split('/')[1], children:{}, files:[], path:'' };
    files.forEach(f => {
        const parts = f.path.split('/');
        let node = root;
        for (let i=0; i<parts.length-1; i++) {
        const k = parts[i];
        if (!node.children[k])
            node.children[k] = { name:k, children:{}, files:[], path:parts.slice(0,i+1).join('/') };
        node = node.children[k];
        }
        node.files.push(f);
    });
    return root;
    }

    const tree   = buildTree(FILES);
    let expanded = new Set();
    let selected = null;

    function collectFiles(node) {
    let out = [...node.files];
    Object.values(node.children).forEach(c => out = out.concat(collectFiles(c)));
    return out;
    }
    function worstRisk(node) {
    return collectFiles(node).reduce(
        (w,f) => RISK_ORDER[f.risk.level] < RISK_ORDER[w] ? f.risk.level : w, 'green'
    );
    }

    function renderRiskList() {
    const el = document.getElementById('riskList');
    el.innerHTML = '';
    atRisk.forEach((file, i) => {
        const name = file.path.split('/').pop();
        const dir  = file.path.split('/').slice(0,-1).join('/');
        const chip = document.createElement('div');
        chip.className = `risk-chip risk-${file.risk.level}`;
        chip.dataset.path = file.path;
        chip.innerHTML = `
        <div class="chip-top">
            <span class="chip-rank">${String(i+1).padStart(2,'0')}</span>
            <span class="chip-badge badge-${file.risk.level}">${file.risk.label}</span>
        </div>
        <div class="chip-name">${name}</div>
        <div class="chip-path">${dir}/</div>
        `;
        chip.addEventListener('click', () => navigateTo(file.path));
        el.appendChild(chip);
    });
    }

    function renderTree() {
    const c = document.getElementById('treeContainer');
    c.innerHTML = '';
    function renderNode(node, depth) {
        const isRoot = depth === 0;
        const isOpen = isRoot || expanded.has(node.path);
        const kids   = Object.values(node.children);
        const wr     = worstRisk(node);
        if (!isRoot) {
        const el = document.createElement('div');
        el.className = 'tree-node folder-node' + (selected === node.path ? ' selected' : '');
        el.dataset.path = node.path;
        const indent = Array(depth).fill('<div class="indent-unit"></div>').join('');
        const arrow  = (kids.length || node.files.length)
            ? `<div class="tree-arrow ${isOpen?'open':''}">▶</div>`
            : `<div class="tree-arrow"></div>`;
        el.innerHTML = `
            <div style="display:flex">${indent}</div>
            ${arrow}
            <div class="tree-icon">📂</div>
            <div class="tree-label">${node.name}</div>
            <div class="tree-risk-dot" style="background:${DOT[wr]}"></div>
        `;
        el.addEventListener('click', e => {
            e.stopPropagation();
            expanded.has(node.path) ? expanded.delete(node.path) : expanded.add(node.path);
            selectNode(node.path, node, 'folder');
        });
        c.appendChild(el);
        }
        if (isOpen) {
        kids.sort((a,b) => a.name.localeCompare(b.name)).forEach(ch => renderNode(ch, depth+1));
        node.files
            .sort((a,b) => a.path.split('/').pop().localeCompare(b.path.split('/').pop()))
            .forEach(file => {
            const name = file.path.split('/').pop();
            const el   = document.createElement('div');
            el.className = 'tree-node' + (selected === file.path ? ' selected' : '');
            el.dataset.path = file.path;
            const indent2 = Array(depth+1).fill('<div class="indent-unit"></div>').join('');
            el.innerHTML = `
                <div style="display:flex">${indent2}</div>
                <div class="tree-arrow"></div>
                <div class="tree-icon">${ICONS[file.lang] || '📄'}</div>
                <div class="tree-label${file.important ? '' : ' dim'}">${name}</div>
                <div class="tree-risk-dot" style="background:${DOT[file.risk.level]};opacity:${file.important?1:0.3}"></div>
            `;
            el.addEventListener('click', e => { e.stopPropagation(); selectNode(file.path, file, 'file'); });
            c.appendChild(el);
            });
        }
    }
    renderNode(tree, 0);
    }

    function navigateTo(path) {
    path.split('/').forEach((_,i,arr) => { if(i>0) expanded.add(arr.slice(0,i).join('/')); });
    const file = FILES.find(f => f.path === path);
    if (file) selectNode(path, file, 'file');
    document.querySelectorAll('.risk-chip').forEach(el =>
        el.classList.toggle('active', el.dataset.path === path));
    }
    function selectNode(path, data, type) {
    selected = path;
    renderTree();
    renderMetrics(path, data, type);
    }

    function renderMetrics(path, data, type) {
    const card = document.getElementById('metricsCard');
    card.className = 'metrics-card';
    if (type === 'file') {
        const name   = path.split('/').pop();
        const dir    = path.split('/').slice(0,-1).join('/');
        const sorted = Object.entries(data.expertise).sort((a,b) => b[1]-a[1]);
        const maxVal = sorted[0]?.[1] || 1;
        card.innerHTML = `
        <div class="card-header">
            <div class="card-header-title">${name}</div>
            <div class="metrics-tags" style="margin:0">
            <span class="lang-tag">${data.lang}</span>
            <span class="chip-badge badge-${data.risk.level}">${data.risk.label}</span>
            </div>
        </div>
        <div class="metrics-body">
            <div class="metrics-path">${dir}/</div>
            <div class="risk-alert alert-${data.risk.level}">
            <div class="alert-dot alert-dot-${data.risk.level}${data.risk.level==='red'?' pulse':''}"></div>
            <span>${data.risk.desc}</span>
            </div>
            <div class="section-title">Degree of Expertise (DOE)</div>
            <div class="exp-rows">
            ${sorted.map(([u,v]) => `
                <div class="exp-row">
                <div class="exp-name" style="color:${devColor(u)}" title="${u}">${u}</div>
                <div class="exp-track">
                    <div class="exp-fill" style="width:${(v/maxVal*100).toFixed(1)}%;background:${devColor(u)}"></div>
                </div>
                <div class="exp-val">${v.toFixed(2)}</div>
                </div>
            `).join('')}
            </div>
            ${!data.important ? `<div class="divider"></div><div style="font-size:11px;color:var(--text-dim)">ℹ This file type is not ranked in the at-risk list</div>` : ''}
        </div>
        `;
    } else {
        const allFiles  = path ? FILES.filter(f => f.path.startsWith(path+'/')) : FILES;
        const important = allFiles.filter(f => f.important);
        const byRisk    = { red:[], orange:[], yellow:[], green:[] };
        important.forEach(f => byRisk[f.risk.level].push(f));
        const langs = [...new Set(important.map(f => f.lang))];
        const name  = path ? path.split('/').pop() : META.repo.split('/')[1];
        const risky = important.filter(f => f.risk.level !== 'green');
        card.innerHTML = `
        <div class="card-header">
            <div class="card-header-title">📂 ${name}/</div>
            <div class="metrics-tags" style="margin:0">
            ${langs.map(l => `<span class="lang-tag">${l}</span>`).join('')}
            </div>
        </div>
        <div class="metrics-body">
            <div class="folder-stats">
            <div class="stat-box"><div class="stat-num">${allFiles.length}</div><div class="stat-label">total files</div></div>
            <div class="stat-box"><div class="stat-num" style="color:var(--red)">${risky.length}</div><div class="stat-label">at-risk files</div></div>
            </div>
            <div class="section-title">Risk Breakdown</div>
            <div class="risk-breakdown">
            ${byRisk.red.length    ? `<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--red)"></div><span>Sole Expert</span></div><span class="breakdown-count">${byRisk.red.length}</span></div>` : ''}
            ${byRisk.orange.length ? `<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--orange)"></div><span>Sole Maintainer</span></div><span class="breakdown-count">${byRisk.orange.length}</span></div>` : ''}
            ${byRisk.yellow.length ? `<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--yellow)"></div><span>Narrow Expertise</span></div><span class="breakdown-count">${byRisk.yellow.length}</span></div>` : ''}
            ${byRisk.green.length  ? `<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--green)"></div><span>Broad Coverage</span></div><span class="breakdown-count">${byRisk.green.length}</span></div>` : ''}
            ${!important.length    ? '<div style="color:var(--text-dim);font-size:11px">No important files here</div>' : ''}
            </div>
            ${risky.length ? `
            <div class="divider"></div>
            <div class="section-title">Files to Watch</div>
            <div style="display:flex;flex-direction:column;gap:4px">
                ${risky.map(f => `
                <div class="watch-row" onclick="navigateTo('${f.path}')">
                    <div class="breakdown-dot" style="background:${DOT[f.risk.level]}"></div>
                    <span class="watch-name">${f.path.split('/').pop()}</span>
                </div>
                `).join('')}
            </div>
            ` : ''}
        </div>
        `;
    }
    }

    renderRiskList();
    renderTree();
    </script>
    </body>
    </html>"""


# ─────────────────────────────────────────────────────────────────────────────
# SELECTION AREA  (date/window + developer selector)
# ─────────────────────────────────────────────────────────────────────────────

def render_selection_area(repo_full: str):
    """
    Render the combined date/window selector and developer risk table.
    Returns (selected_date, window_days, selected_dev_id, risk_df).
    Auto-populates:  date = today, window = 30 days, developer = highest prob_1 row.
    Resets auto-selection when repo changes.
    """
    # ── Reset state when repo changes ────────────────────────────────────────
    if st.session_state.get("_last_repo") != repo_full:
        st.session_state["_last_repo"]    = repo_full
        st.session_state["_dev_sel_row"]  = [0]

    # ── Date & window controls ────────────────────────────────────────────────

    window_days = st.select_slider(
        "Lookback window",
        options=[7, 14, 30, 60, 90],
        value=30,
        key="sel_window",
    )

    min_date = load_commit_dates(repo_full)

    # Max date = updated_at from the data collection cursor (when data was last fetched)
    max_date = None
    _cursor_path = ORG_BASE / repo_full / cfg.data_cursor
    if _cursor_path.exists():
        try:
            with open(_cursor_path, "r") as _f:
                _cursor_data = json.load(_f)
            _updated_at = _cursor_data.get("updated_at", "")
            if _updated_at:
                max_date = datetime.datetime.fromisoformat(
                    _updated_at.replace("Z", "+00:00")
                ).date()
        except Exception:
            pass
    if max_date is None:
        # Fallback: latest date in the commit history
        _hist = load_commit_history(repo_full)
        max_date = _hist["date"].max() if not _hist.empty else min_date

    min_ord  = min_date.toordinal()
    max_ord  = max_date.toordinal()
    selected_ord = st.select_slider(
        "Select snapshot date",
        options=range(min_ord, max_ord + 1),
        value=max_ord,
        format_func=lambda o: datetime.date.fromordinal(o).strftime("%b %d, %Y"),
        key="sel_date_slider",
    )
    selected_date = datetime.date.fromordinal(selected_ord)

    # ── Timeline visualizer ───────────────────────────────────────────────────
    # Use the previously stored dev selection (from last rerun) to show dev histogram
    prev_sel_rows = st.session_state.get("_dev_sel_row", [0])
    hist_df = load_commit_history(repo_full)
    dev_hist_df = None
    # We'll populate dev_hist_df after we know the current selection below;
    # for now peek at the cached row to get the dev_id from risk_df if available
    _peek_risk = build_risk_table(repo_full, selected_date)
    if _peek_risk is not None and not _peek_risk.empty:
        _peek_idx = min(prev_sel_rows[0] if prev_sel_rows else 0, len(_peek_risk) - 1)
        _peek_dev = _peek_risk.iloc[_peek_idx]["dev_id"]
        dev_hist_df = load_dev_commit_history(repo_full, _peek_dev)
    tl_payload = _build_timeline_payload(repo_full, selected_date, window_days, hist_df, dev_hist_df)
    components.html(_build_timeline_html(tl_payload), height=180, scrolling=False)

    # ── Developer risk table ──────────────────────────────────────────────────
    st.caption("Click any row to select a developer to explore — table sorted by break risk (highest first).")
    event, risk_df, selected_dev = render_risk_table(repo_full, selected_date)

    if event is None or risk_df.empty:
        return selected_date, window_days, None, risk_df

    # Auto-selection: use user click if present, else fall back to session default
    sel_rows = event.selection.rows if event.selection.rows else st.session_state.get("_dev_sel_row", [0])
    if event.selection.rows:
        st.session_state["_dev_sel_row"] = event.selection.rows

    selected_row_idx = sel_rows[0] if sel_rows else 0
    selected_row_idx = min(selected_row_idx, len(risk_df) - 1)
    row = risk_df.iloc[[selected_row_idx]]
    selected_dev = row["dev_id"].values[0] if not row.empty else None


    # Auto-select caption
    if not event.selection.rows:
        display = row["Contributor"].values[0] if not row.empty else "—"

    return selected_date, window_days, selected_dev, risk_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="DISTRAC Dashboard", layout="wide")

    # ── Sidebar: repo selector ───────────────────────────────────────────────
    with st.sidebar:

        orgs = list_orgs()
        if not orgs:
            st.error(f"No organizations found under:\n`{ORG_BASE}`")
            st.stop()

        org        = st.selectbox("Organization", orgs,            key="sidebar_org")
        repos      = list_repos_for(org)
        if not repos:
            st.warning("No repos found for this organization.")
            st.stop()

        repo       = st.selectbox("Repository",   repos,           key="sidebar_repo")
        repo_full  = f"{org}/{repo}"

        if st.button("Clear cache", help="Reload all files from disk (use after running the pipeline)"):
            st.cache_data.clear()
            st.rerun()

        # Show which pre-computed files are available
        st.divider()
        st.caption("Data status:")
        status = repo_data_status(repo_full)
        icons  = {True: "True", False: "False"}
        if status.get("distrac"):
            st.markdown("✅ **distrac/ outputs ready**")
            generated_at = status.get("generated_at", "")
            if generated_at:
                try:
                    import datetime as _dt
                    ts = _dt.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                    age_h = (_dt.datetime.now(_dt.timezone.utc) - ts).total_seconds() / 3600
                    age_str = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h/24:.0f}d ago"
                    st.caption(f"Generated: {age_str}")
                except Exception:
                    st.caption(f"Generated: {generated_at[:10]}")
        else:
            st.markdown("Warning: distrac/ not built — run pipeline")
        st.markdown(f"{icons[status['timeline']]} Labeled timeline")
        st.markdown(f"{icons[status['model']]} Model predictions")
        st.markdown(f"{icons[status['truck_factor']]} Truck Factor")
        st.markdown(f"{icons[status['knowledge_dist']]} Knowledge distribution")

    # ── Main content ─────────────────────────────────────────────────────────
    st.title(f"DISTRAC  —  {repo_full}")

    # ── Selection area: date/window/developer ────────────────────────────────
    selected_date, window_days, selected_dev, risk_df = render_selection_area(repo_full)

    if selected_dev is None:
        st.info("No developer predictions available. Run the pipeline first.")
        return

    # Derive break length from risk table for the selected developer
    row = risk_df[risk_df["dev_id"] == selected_dev]
    break_weeks = 2
    if not row.empty:
        raw_break = str(row["Next Break Length"].iloc[0])
        nums = [int(s) for s in raw_break.split() if s.isdigit()]
        if nums:
            break_weeks = max(1, round(nums[0] / 7)) if "day" in raw_break else nums[0]

    st.header(f"{selected_dev}")

    # ── Project Health panel ──────────────────────────────────────────────────
    st.divider()
    render_project_health_panel(repo_full, selected_dev, break_weeks=break_weeks)

    # ── Social-Technical Network panel ──────────────────────────────────────
    render_stn_panel(repo_full, selected_dev, selected_date)

    # ── Knowledge Distribution panel ─────────────────────────────────────────
    render_knowledge_panel(repo_full_name = repo_full, selected_dev = selected_dev )


if __name__ == "__main__":
    main()


