"""
DISTRAC Dashboard v2 — FastAPI backend
Run: python Extractors/distrac_api.py
Open: http://localhost:8000
"""
import json
import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import Settings as cfg

ORGS_DIR = Path(cfg.main_folder)
STATIC_DIR = ROOT / "static"


def _resolve_file(local_path: Path) -> Path:
    """In HF mode, download the file from HF Datasets and return the cached local path.
    Falls back to the original path on any error so the caller's .exists() check handles it.
    """
    if not cfg.USE_HF_STORAGE:
        return local_path
    try:
        rel = local_path.relative_to(ROOT).as_posix()
        # The dataset was uploaded from ./Organizations, so the root in HF
        # is the org level — strip the "Organizations/" prefix.
        if rel.startswith("Organizations/"):
            rel = rel[len("Organizations/"):]
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(
            repo_id=cfg.HF_DATASET_REPO,
            filename=rel,
            repo_type="dataset",
        ))
    except Exception:
        return local_path

app = FastAPI(title="DISTRAC API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _repo_dir(org: str, repo: str) -> Path:
    return ORGS_DIR / org / repo


def _distrac(org: str, repo: str) -> Path:
    return _repo_dir(org, repo) / "distrac"


def _available_repos() -> list[dict]:
    repos = []
    for org, repo in cfg.load_repo_split("test"):
        if cfg.USE_HF_STORAGE:
            repos.append({"org": org, "repo": repo, "label": f"{org}/{repo}"})
        elif _distrac(org, repo).exists():
            repos.append({"org": org, "repo": repo, "label": f"{org}/{repo}"})
    return repos


def _load_parquet(org: str, repo: str, name: str) -> pd.DataFrame:
    path = _resolve_file(_distrac(org, repo) / f"{name}.parquet")
    if not path.exists():
        raise HTTPException(404, f"Data not found: {name}")
    return pd.read_parquet(path)


def _load_test_df(org: str, repo: str) -> pd.DataFrame:
    path = _resolve_file(_repo_dir(org, repo) / "model_folder" / "test_df.csv")
    if not path.exists():
        raise HTTPException(404, "Prediction data not found")
    # Read header first so optional columns degrade gracefully
    available = pd.read_csv(path, nrows=0).columns.tolist()
    want = {
        "dev", "date", "commits", "prs", "issues", "issue_activity", "pr_activity", "state",
        # multiclass future_state model (new)
        "inactivity_risk", "prob_INACTIVE", "prob_GONE", "prob_ACTIVE", "prob_NON_CODING",
        # binary model columns
        "prob_1", "prob_0",
        # ground-truth response column — load all possible names, normalise below
        "inactivity_window_14d",   # new: pre-break + break at-risk window
        "break_starts_in_14d",     # legacy: onset-only
        "break_label",             # normalised name written by distrac_writer
    }
    usecols = [c for c in available if c in want]
    df = pd.read_csv(path, usecols=usecols)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "inactivity_risk" not in df.columns and {"prob_INACTIVE", "prob_GONE"}.issubset(df.columns):
        df["inactivity_risk"] = df["prob_INACTIVE"].fillna(0) + df["prob_GONE"].fillna(0)
    if "inactivity_risk" not in df.columns and "prob_1" in df.columns:
        df["inactivity_risk"] = df["prob_1"].fillna(0)
    return df


def _clean(obj):
    """Recursively replace NaN/inf with None for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _build_dev_map(test_df: pd.DataFrame, developers: pd.DataFrame) -> dict[str, str]:
    """Return mapping: test_df 'dev' value → canonical dev_id.

    Prefers entries that have a real GitHub login over synthetic
    'author_name|...' entries that are created as fallback identifiers.
    """
    dev_unique = developers.drop_duplicates("dev_id")
    by_id = set(dev_unique["dev_id"])

    # Build name → dev_id, preferring rows with a real login
    # Sort so rows with login come last (they will overwrite synthetic ones)
    sorted_devs = dev_unique.sort_values(
        "login", na_position="first", key=lambda s: s.fillna("")
    )
    by_name: dict[str, str] = {}
    for _, row in sorted_devs.iterrows():
        n = row.get("name")
        if pd.notna(n) and n:
            by_name[str(n)] = str(row["dev_id"])

    mapping: dict[str, str] = {}
    for d in test_df["dev"].unique():
        if d in by_id:
            mapping[d] = d
        elif d in by_name:
            mapping[d] = by_name[d]
    return mapping


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/repos")
def list_repos():
    return _available_repos()


@app.get("/api/repos/{org}/{repo}/overview")
def repo_overview(org: str, repo: str):
    ch = _load_parquet(org, repo, "commit_history")
    # aggregate all developers per day
    daily = ch.groupby("date", as_index=False)["commit_count"].sum()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    weekly = (
        daily.set_index("date")["commit_count"]
        .resample("W-MON")
        .sum()
        .reset_index()
    )
    weekly.columns = ["date", "commits"]
    weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
    return _clean({
        "weekly_commits": weekly.to_dict("records"),
        "total_commits": int(daily["commit_count"].sum()),
        "first_commit": daily["date"].min().strftime("%Y-%m-%d"),
        "last_commit":  daily["date"].max().strftime("%Y-%m-%d"),
    })


@app.get("/api/repos/{org}/{repo}/developers")
def repo_developers(org: str, repo: str):
    devs_df   = _load_parquet(org, repo, "developers")
    test_df   = _load_test_df(org, repo)
    dev_map   = _build_dev_map(test_df, devs_df)

    # keep unique dev_id rows from developers
    devs_unique = devs_df.drop_duplicates("dev_id").set_index("dev_id")

    results = []
    for raw_dev, dev_id in dev_map.items():
        sub = test_df[test_df["dev"] == raw_dev].sort_values("date")
        if sub.empty:
            continue
        last_row   = sub.iloc[-1]
        total_commits = int(sub["commits"].sum())
        current_state = str(last_row["state"])
        # inactivity_risk = P(INACTIVE) + P(GONE) — primary risk signal from future_state model.
        # Falls back to prob_1 for backward compatibility with old binary-model outputs.
        _risk_col  = next((c for c in ("inactivity_risk", "prob_1") if c in sub.columns), None)
        risk_series = sub[_risk_col].dropna() if _risk_col else pd.Series(dtype=float)
        depart_prob = float(risk_series.iloc[-1]) if len(risk_series) else None

        if dev_id in devs_unique.index:
            row       = devs_unique.loc[dev_id]
            login     = row.get("login") or dev_id
            name      = row.get("name") or ""
            is_tf     = bool(row.get("is_tf_dev", False))
            role      = str(row.get("role") or "")
        else:
            login, name, is_tf, role = dev_id, "", False, ""

        results.append({
            "dev_id":        dev_id,
            "login":         login,
            "name":          name,
            "total_commits": total_commits,
            "current_state": current_state,
            "depart_prob":   depart_prob,
            "role":          role,
            "is_truck_factor": is_tf,
        })

    # sort: truck factor first, then by depart_prob desc
    results.sort(key=lambda d: (-int(d["is_truck_factor"]),
                                 -(d["depart_prob"] or 0)))
    return _clean(results)


@app.get("/api/repos/{org}/{repo}/developers/{dev_id}/timeline")
def dev_timeline(org: str, repo: str, dev_id: str):
    test_df  = _load_test_df(org, repo)
    devs_df  = _load_parquet(org, repo, "developers")
    dev_map  = _build_dev_map(test_df, devs_df)

    # find the raw_dev key that maps to this dev_id
    raw_dev = next((k for k, v in dev_map.items() if v == dev_id), None)
    if raw_dev is None:
        raise HTTPException(404, f"Developer {dev_id} not found in timeline data")

    sub = test_df[test_df["dev"] == raw_dev].sort_values("date")
    # deduplicate by date (take last row per date)
    sub = sub.drop_duplicates("date", keep="last")
    timeline = []
    for _, r in sub.iterrows():
        entry = {
            "date":           str(r["date"]),
            "commits":        int(r["commits"])        if pd.notna(r.get("commits"))        else 0,
            "prs":            int(r["prs"])            if pd.notna(r.get("prs"))            else 0,
            "issues":         int(r["issues"])         if pd.notna(r.get("issues"))         else 0,
            "pr_activity":    int(r["pr_activity"])    if pd.notna(r.get("pr_activity"))    else 0,
            "issue_activity": int(r["issue_activity"]) if pd.notna(r.get("issue_activity")) else 0,
            "state":          str(r["state"]),
        }
        # Include all available probability columns for the frontend to display
        for _pcol in ("inactivity_risk", "prob_INACTIVE", "prob_GONE",
                      "prob_ACTIVE", "prob_NON_CODING",
                      "prob_1", "prob_0"):           # prob_1/0 = legacy backward-compat
            if pd.notna(r.get(_pcol)):
                entry[_pcol] = float(r[_pcol])
        # Ground-truth response label — pick whichever column is present.
        # Priority: new at-risk window (pre-break + break) > legacy onset-only > normalised name.
        _truth_col = next(
            (c for c in ("inactivity_window_14d", "break_starts_in_14d", "break_label")
             if pd.notna(r.get(c))),
            None
        )
        if _truth_col is not None:
            entry["break_starts_in_14d"] = int(r[_truth_col])
        # Code churn — already aggregated daily in test_df
        if pd.notna(r.get("lines_added_today")):
            entry["lines_added"] = int(r["lines_added_today"])
        if pd.notna(r.get("lines_deleted_today")):
            entry["lines_deleted"] = int(r["lines_deleted_today"])
        timeline.append(entry)

    devs_unique = devs_df.drop_duplicates("dev_id").set_index("dev_id")
    login = devs_unique.loc[dev_id, "login"] if dev_id in devs_unique.index else dev_id
    return _clean({"dev_id": dev_id, "login": login, "timeline": timeline})


@app.get("/api/repos/{org}/{repo}/developers/{dev_id}/avatar")
def dev_avatar(org: str, repo: str, dev_id: str):
    devs_df = _load_parquet(org, repo, "developers")
    devs_unique = devs_df.drop_duplicates("dev_id").set_index("dev_id")
    if dev_id not in devs_unique.index:
        raise HTTPException(404, "Developer not found")
    login = str(devs_unique.loc[dev_id, "login"])
    avatar_path = _resolve_file(_repo_dir(org, repo) / "Developers" / login / "avatar.png")
    if not avatar_path.exists():
        raise HTTPException(404, "Avatar not found")
    return FileResponse(str(avatar_path), media_type="image/png")


@app.get("/api/repos/{org}/{repo}/activity-window")
def activity_window(org: str, repo: str,
                    start_date: Optional[str] = None,
                    end_date:   Optional[str] = None):
    """Repo-wide activity totals and per-week averages for a date window.
    Reads raw CSV files so ALL developers are included, not just truck-factor ones.
    """
    import datetime as _dt
    # Derive n_weeks from calendar range, not from data row count
    if start_date and end_date:
        sd_d = _dt.date.fromisoformat(start_date)
        ed_d = _dt.date.fromisoformat(end_date)
        n_weeks = max(1, round((ed_d - sd_d).days / 7))
    else:
        n_weeks = 1

    repo_dir = _repo_dir(org, repo)
    sd_ts = pd.Timestamp(start_date, tz="UTC") if start_date else None
    ed_ts = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=6) if end_date else None

    def _count_raw(fname: str) -> int:
        path = _resolve_file(repo_dir / fname)
        if not path.exists():
            return 0
        try:
            df = pd.read_csv(path, usecols=["created_at"], low_memory=False)
            ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce").dropna()
            if sd_ts is not None:
                ts = ts[ts >= sd_ts]
            if ed_ts is not None:
                ts = ts[ts <= ed_ts]
            return len(ts)
        except Exception:
            return 0

    tc = _count_raw("commit_list.csv")
    tp = _count_raw("prs_repo.csv")
    ti = _count_raw("issues.csv")

    return _clean({
        "n_weeks":              n_weeks,
        "avg_commits_per_week": round(tc / n_weeks, 1),
        "avg_prs_per_week":     round(tp / n_weeks, 1),
        "avg_issues_per_week":  round(ti / n_weeks, 1),
    })


@app.get("/api/repos/{org}/{repo}/project-health")
def project_health(org: str, repo: str):
    path = _resolve_file(_repo_dir(org, repo) / "ProjectHealth" / "project_health.json")
    if not path.exists():
        raise HTTPException(404, "Project health data not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/repos/{org}/{repo}/stn")
def social_network(org: str, repo: str):
    metrics = _load_parquet(org, repo, "stn_metrics")
    devs    = _load_parquet(org, repo, "developers").drop_duplicates("dev_id")
    id_to_login = devs.set_index("dev_id")["login"].to_dict()
    # developers.parquet has the role column; build dev_id → role map
    id_to_role  = devs.set_index("dev_id")["role"].to_dict()

    nodes = []
    for _, r in metrics.iterrows():
        login = id_to_login.get(r["dev_id"], r["dev_id"])
        role  = id_to_role.get(r["dev_id"], "")
        nodes.append({
            "id":          login,
            "dev_id":      r["dev_id"],
            "role":        str(role or ""),
            "degree":      int(r["degree"]) if pd.notna(r.get("degree")) else 0,
            "betweenness": float(r["betweenness_centrality"]) if pd.notna(r.get("betweenness_centrality")) else 0.0,
        })

    edges_df = _load_parquet(org, repo, "stn_edges")
    mask = (
        edges_df["source_dev_id"].notna() & (edges_df["source_dev_id"] != "") &
        edges_df["target_dev_id"].notna() & (edges_df["target_dev_id"] != "") &
        (edges_df["source_dev_id"] != edges_df["target_dev_id"])
    )
    top_edges = edges_df[mask].nlargest(500, "weight")
    edge_list = [
        {
            "source": id_to_login.get(r["source_dev_id"], r["source_dev_id"]),
            "target": id_to_login.get(r["target_dev_id"], r["target_dev_id"]),
            "weight": int(r["weight"]),
        }
        for _, r in top_edges.iterrows()
    ]

    return _clean({"nodes": nodes, "edges": edge_list})


@app.get("/api/repos/{org}/{repo}/knowledge")
def knowledge(org: str, repo: str):
    doe_df = _load_parquet(org, repo, "knowledge_doe")
    tf_df  = _load_parquet(org, repo, "truck_factor")
    devs   = _load_parquet(org, repo, "developers").drop_duplicates("dev_id")

    tf_count  = int(tf_df["tf"].iloc[0]) if len(tf_df) else 0
    tf_list   = devs[devs["is_tf_dev"] == True]["login"].dropna().tolist()
    id_to_login = devs.set_index("dev_id")["login"].to_dict()

    # for each file, compute ownership concentration
    doe_df["login"] = doe_df["dev_id"].map(id_to_login).fillna(doe_df["dev_id"])
    file_groups = doe_df.groupby("file_path")

    files = []
    for fpath, group in file_groups:
        total_doe  = group["DOE"].sum()
        if total_doe <= 0:
            continue
        top_idx   = group["DOE"].idxmax()
        top_row   = group.loc[top_idx]
        top_login = top_row["login"]
        ownership = float(top_row["DOE"] / total_doe)
        n_devs    = len(group)
        alternatives = max(0, n_devs - 1)
        is_important = bool(top_row.get("is_important", False))
        lang = str(top_row.get("lang", "Other") or "Other")
        files.append({
            "path":        str(fpath),
            "ownership":   ownership,
            "top_dev":     str(top_login),
            "alternatives": alternatives,
            "is_important": is_important,
            "lang":        lang,
        })

    # sort: important first, then by ownership desc
    files.sort(key=lambda f: (-int(f["is_important"]), -f["ownership"]))
    files = files[:500]  # cap for frontend performance

    return _clean({
        "truck_factor": tf_count,
        "tf_list":      tf_list,
        "files":        files,
    })


# ── Departure-simulation HTML helpers (ported from Dashboard.py) ─────────────

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
_KD_MAX_FILES = 2000


def _kd_shorten_dev_id(dev_id: str) -> str:
    if not dev_id:
        return "unknown"
    if "|" in dev_id:
        return dev_id.split("|", 1)[1].split("@")[0]
    return str(dev_id)


def _kd_build_dev_map(devs_df: pd.DataFrame) -> dict:
    """Build dev_id → display name mapping with priority: name > login > email."""
    dev_map: dict = {}
    for _, row in devs_df.iterrows():
        dev_id = str(row.get("dev_id", "")).strip()
        if not dev_id or dev_id in ("nan", "none", ""):
            continue
        for field in ("name", "login", "email"):
            val = row.get(field, "")
            if pd.notna(val) and str(val).strip() not in ("", "nan", "none"):
                dev_map[dev_id] = str(val).strip()
                break
    return dev_map


def _kd_resolve_dev(dev_id: str, dev_map: dict) -> str:
    """Resolve a dev_id to a human-readable name, falling back to _kd_shorten_dev_id."""
    return dev_map.get(dev_id) or _kd_shorten_dev_id(dev_id)


def _kd_build_files_json(doe_df: pd.DataFrame, dev_map: dict | None = None) -> list:
    if doe_df.empty:
        return []
    files = []
    for file_path, group in doe_df.groupby("file_path"):
        expertise: dict = {}
        for _, row in group.iterrows():
            short   = _kd_resolve_dev(str(row.get("dev_id", "")), dev_map or {})
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
    important_files = [f for f in files if f["important"]]
    other_files     = [f for f in files if not f["important"]]
    if len(important_files) > _KD_MAX_FILES:
        important_files.sort(key=lambda f: max(f["expertise"].values()), reverse=True)
        return important_files[:_KD_MAX_FILES]
    remaining = _KD_MAX_FILES - len(important_files)
    other_files.sort(key=lambda f: max(f["expertise"].values()), reverse=True)
    return important_files + other_files[:remaining]


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
        --surface: #ffffff; --surface2: #f8f9fc;
        --border: #e8eaf0; --border-soft: #f0f1f5;
        --text: #2c3142; --text-mid: #6b7280; --text-dim: #9ca3af;
        --accent: #4178f0; --accent-light: #eef2fd;
        --red: #e53e3e;    --red-light: #fff5f5;    --red-border: #fed7d7;
        --orange: #d97706; --orange-light: #fffbeb;  --orange-border: #fde68a;
        --yellow: #b45309; --yellow-light: #fefce8;  --yellow-border: #fef08a;
        --green: #16a34a;  --green-light: #f0fdf4;   --green-border: #bbf7d0;
        --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
        --radius: 10px; --radius-sm: 6px;
    }
    body { font-family: 'Inter', sans-serif; background: transparent; color: var(--text); font-size: 13px; padding: 4px 0 16px; }
    .repo-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
    .repo-name { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; color: var(--text); flex: 1; }
    .stat-pill { font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; background: var(--surface2); border: 1px solid var(--border); color: var(--text-mid); }
    .stat-pill.tf     { background: var(--accent-light); color: var(--accent); border-color: #c7d8fb; }
    .stat-pill.danger { background: var(--red-light); color: var(--red); border-color: var(--red-border); }
    .section-label { font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; padding-left: 2px; }
    .risk-section { margin-bottom: 16px; }
    .risk-scroll { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 4px; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
    .risk-scroll::-webkit-scrollbar { height: 3px; }
    .risk-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    .risk-chip { display: flex; flex-direction: column; gap: 3px; padding: 9px 12px; background: var(--surface); border: 1px solid var(--border); border-top: 2px solid transparent; border-radius: var(--radius-sm); cursor: pointer; min-width: 148px; flex-shrink: 0; transition: box-shadow 0.15s, transform 0.15s; box-shadow: var(--shadow-sm); }
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
    .chip-badge { font-size: 9px; font-weight: 600; letter-spacing: 0.04em; padding: 1px 6px; border-radius: 20px; text-transform: uppercase; }
    .badge-red    { background: var(--red-light);    color: var(--red);    border: 1px solid var(--red-border); }
    .badge-orange { background: var(--orange-light); color: var(--orange); border: 1px solid var(--orange-border); }
    .badge-yellow { background: var(--yellow-light); color: var(--yellow); border: 1px solid var(--yellow-border); }
    .badge-green  { background: var(--green-light);  color: var(--green);  border: 1px solid var(--green-border); }
    .chip-name { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 500; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }
    .chip-path { font-size: 10px; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .workspace { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }
    .explorer-card, .metrics-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow-sm); overflow: hidden; }
    .card-header { display: flex; align-items: center; justify-content: space-between; padding: 11px 14px; border-bottom: 1px solid var(--border-soft); }
    .card-header-title { font-size: 12px; font-weight: 600; color: var(--text); }
    .card-header-sub   { font-size: 11px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
    .tree-body { padding: 6px 0; max-height: 500px; overflow-y: auto; }
    .tree-body::-webkit-scrollbar { width: 4px; }
    .tree-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    .tree-node { display: flex; align-items: center; padding: 4px 12px 4px 0; cursor: pointer; user-select: none; transition: background 0.1s; position: relative; }
    .tree-node:hover { background: var(--surface2); }
    .tree-node.selected { background: var(--accent-light); }
    .tree-node.selected::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: var(--accent); }
    .indent-unit { width: 18px; flex-shrink: 0; }
    .tree-arrow { width: 14px; height: 14px; display: flex; align-items: center; justify-content: center; font-size: 8px; color: var(--text-dim); transition: transform 0.15s; flex-shrink: 0; }
    .tree-arrow.open { transform: rotate(90deg); }
    .tree-icon { font-size: 13px; margin-right: 6px; flex-shrink: 0; }
    .tree-label { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text); flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tree-node.folder-node > .tree-label { font-weight: 500; }
    .tree-label.dim { color: var(--text-dim); }
    .tree-risk-dot { width: 6px; height: 6px; border-radius: 50%; margin-right: 10px; flex-shrink: 0; }
    .metrics-card.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 220px; padding: 32px; text-align: center; }
    .empty-icon { font-size: 26px; margin-bottom: 10px; opacity: 0.35; }
    .empty-text { font-size: 12px; color: var(--text-dim); line-height: 1.7; }
    .metrics-body { padding: 16px; animation: fadeUp 0.18s ease; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(5px)} to{opacity:1;transform:translateY(0)} }
    .metrics-path { font-size: 11px; color: var(--text-dim); margin-bottom: 12px; }
    .metrics-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
    .lang-tag { font-size: 10px; font-weight: 600; letter-spacing: 0.05em; padding: 2px 8px; border-radius: 20px; background: var(--accent-light); color: var(--accent); border: 1px solid #c7d8fb; text-transform: uppercase; }
    .risk-alert { display: flex; align-items: flex-start; gap: 9px; padding: 10px 12px; border-radius: var(--radius-sm); margin-bottom: 16px; font-size: 12px; line-height: 1.5; }
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
    .section-title { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 10px; }
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
    .activity-strip { margin-top: 14px; border-top: 1px solid var(--border-soft); padding-top: 10px; }
    .activity-title { font-size: 11px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--text-dim); margin-bottom: 8px; }
    .activity-row { display: flex; align-items: center; gap: 8px; padding: 5px 0; border-bottom: 1px solid var(--border-soft); font-size: 11.5px; }
    .activity-row:last-child { border-bottom: none; }
    .activity-sha { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--accent); text-decoration: none; flex-shrink: 0; }
    .activity-sha:hover { text-decoration: underline; }
    .activity-date { color: var(--text-dim); flex-shrink: 0; min-width: 68px; }
    .activity-author { color: var(--text); font-weight: 500; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .activity-diff { font-family: 'JetBrains Mono', monospace; font-size: 10px; flex-shrink: 0; }
    .activity-diff .add { color: var(--green); }
    .activity-diff .del { color: var(--red); }
    .activity-empty { color: var(--text-dim); font-size: 11.5px; font-style: italic; padding: 6px 0; }
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
    const PALETTE = ['#4178f0','#16a34a','#d97706','#9333ea','#0891b2','#db2777','#65a30d','#ea580c','#7c3aed','#0284c7'];
    const devColorCache = {};
    let colorIdx = 0;
    function devColor(name) {
        if (!devColorCache[name]) devColorCache[name] = PALETTE[colorIdx++ % PALETTE.length];
        return devColorCache[name];
    }
    const RISK_ORDER = { red:0, orange:1, yellow:2, green:3 };
    const DOT   = { red:'#e53e3e', orange:'#d97706', yellow:'#b45309', green:'#16a34a' };
    const ICONS = { Python:'🐍', R:'📊', JavaScript:'🟨', TypeScript:'🔷', Java:'☕', 'C++':'⚙', C:'⚙', 'C#':'🔷', Go:'🐹', Rust:'🦀', Ruby:'💎', PHP:'🐘', Shell:'🖥', SQL:'🗄', Vue:'💚', Svelte:'🔶', CSV:'📋', JSON:'📋', YAML:'📋', Markdown:'📝', HTML:'🌐', CSS:'🎨', Other:'📄' };
    function classifyRisk(expertise) {
        const sorted = Object.entries(expertise).sort((a,b) => b[1]-a[1]);
        const total  = sorted.reduce((t,[,v]) => t+v, 0);
        if (sorted.length === 1) return { level:'red',    label:'Sole Expert',      desc:`Only ${sorted[0][0]} understands this file` };
        if (sorted[0][1] / total > 0.80) return { level:'orange', label:'Sole Maintainer',  desc:`${sorted[0][0]} dominates — others have minimal knowledge` };
        if (sorted.length === 2) return { level:'yellow', label:'Narrow Expertise', desc:`Only ${sorted[0][0]} & ${sorted[1][0]} have meaningful knowledge` };
        return { level:'green', label:'Broad Coverage', desc:`${sorted.length} contributors share knowledge` };
    }
    const META  = window.REPO_META  || { repo:'unknown', tf:0, tf_devs:[], file_count:0 };
    const DEV   = META.selected_dev || null;
    const ALL_FILES = (window.REPO_FILES || []).map(f => ({ ...f, risk: classifyRisk(f.expertise) }));
    const FILES = DEV ? ALL_FILES.filter(f => Object.prototype.hasOwnProperty.call(f.expertise, DEV) && f.expertise[DEV] > 0) : ALL_FILES;
    const atRisk = FILES.filter(f => f.important && f.risk.level !== 'green').sort((a,b) => {
        if (RISK_ORDER[a.risk.level] !== RISK_ORDER[b.risk.level]) return RISK_ORDER[a.risk.level] - RISK_ORDER[b.risk.level];
        return Math.max(...Object.values(b.expertise)) - Math.max(...Object.values(a.expertise));
    });
    document.getElementById('repoName').textContent    = META.repo;
    document.getElementById('repoSubtitle').textContent = DEV ? `${META.repo.split('/')[1]}/ — ${DEV}` : META.repo.split('/')[1] + '/';
    document.getElementById('tfPill').textContent       = `TF ${META.tf}`;
    document.getElementById('filePill').textContent     = DEV ? `${FILES.length} / ${ALL_FILES.length} files` : `${FILES.length} files`;
    document.getElementById('riskPill').textContent     = `${atRisk.length} at risk`;
    function buildTree(files) {
        const root = { name: META.repo.split('/')[1], children:{}, files:[], path:'' };
        files.forEach(f => {
            const parts = f.path.split('/');
            let node = root;
            for (let i=0; i<parts.length-1; i++) {
                const k = parts[i];
                if (!node.children[k]) node.children[k] = { name:k, children:{}, files:[], path:parts.slice(0,i+1).join('/') };
                node = node.children[k];
            }
            node.files.push(f);
        });
        return root;
    }
    const tree = buildTree(FILES);
    let expanded = new Set();
    let selected = null;
    function collectFiles(node) { let out=[...node.files]; Object.values(node.children).forEach(c=>out=out.concat(collectFiles(c))); return out; }
    function worstRisk(node) { return collectFiles(node).reduce((w,f)=>RISK_ORDER[f.risk.level]<RISK_ORDER[w]?f.risk.level:w,'green'); }
    function renderRiskList() {
        const el = document.getElementById('riskList');
        el.innerHTML = '';
        atRisk.forEach((file,i) => {
            const name=file.path.split('/').pop(), dir=file.path.split('/').slice(0,-1).join('/');
            const chip=document.createElement('div');
            chip.className=`risk-chip risk-${file.risk.level}`; chip.dataset.path=file.path;
            chip.innerHTML=`<div class="chip-top"><span class="chip-rank">${String(i+1).padStart(2,'0')}</span><span class="chip-badge badge-${file.risk.level}">${file.risk.label}</span></div><div class="chip-name">${name}</div><div class="chip-path">${dir}/</div>`;
            chip.addEventListener('click',()=>navigateTo(file.path));
            el.appendChild(chip);
        });
    }
    function renderTree() {
        const c=document.getElementById('treeContainer'); c.innerHTML='';
        function renderNode(node,depth) {
            const isRoot=depth===0, isOpen=isRoot||expanded.has(node.path);
            const kids=Object.values(node.children), wr=worstRisk(node);
            if (!isRoot) {
                const el=document.createElement('div');
                el.className='tree-node folder-node'+(selected===node.path?' selected':'');
                el.dataset.path=node.path;
                const indent=Array(depth).fill('<div class="indent-unit"></div>').join('');
                const arrow=(kids.length||node.files.length)?`<div class="tree-arrow ${isOpen?'open':''}">▶</div>`:`<div class="tree-arrow"></div>`;
                el.innerHTML=`<div style="display:flex">${indent}</div>${arrow}<div class="tree-icon">📂</div><div class="tree-label">${node.name}</div><div class="tree-risk-dot" style="background:${DOT[wr]}"></div>`;
                el.addEventListener('click',e=>{e.stopPropagation();expanded.has(node.path)?expanded.delete(node.path):expanded.add(node.path);selectNode(node.path,node,'folder');});
                c.appendChild(el);
            }
            if (isOpen) {
                kids.sort((a,b)=>a.name.localeCompare(b.name)).forEach(ch=>renderNode(ch,depth+1));
                node.files.sort((a,b)=>a.path.split('/').pop().localeCompare(b.path.split('/').pop())).forEach(file=>{
                    const name=file.path.split('/').pop(), el=document.createElement('div');
                    el.className='tree-node'+(selected===file.path?' selected':''); el.dataset.path=file.path;
                    const indent2=Array(depth+1).fill('<div class="indent-unit"></div>').join('');
                    el.innerHTML=`<div style="display:flex">${indent2}</div><div class="tree-arrow"></div><div class="tree-icon">${ICONS[file.lang]||'📄'}</div><div class="tree-label${file.important?'':' dim'}">${name}</div><div class="tree-risk-dot" style="background:${DOT[file.risk.level]};opacity:${file.important?1:0.3}"></div>`;
                    el.addEventListener('click',e=>{e.stopPropagation();selectNode(file.path,file,'file');});
                    c.appendChild(el);
                });
            }
        }
        renderNode(tree,0);
    }
    function navigateTo(path) {
        path.split('/').forEach((_,i,arr)=>{if(i>0)expanded.add(arr.slice(0,i).join('/'));});
        const file=FILES.find(f=>f.path===path);
        if(file) selectNode(path,file,'file');
        document.querySelectorAll('.risk-chip').forEach(el=>el.classList.toggle('active',el.dataset.path===path));
    }
    function selectNode(path,data,type) { selected=path; renderTree(); renderMetrics(path,data,type); }
    function renderMetrics(path,data,type) {
        const card=document.getElementById('metricsCard'); card.className='metrics-card';
        if (type==='file') {
            const name=path.split('/').pop(), dir=path.split('/').slice(0,-1).join('/');
            const sorted=Object.entries(data.expertise).sort((a,b)=>b[1]-a[1]), maxVal=sorted[0]?.[1]||1;
            card.innerHTML=`<div class="card-header"><div class="card-header-title">${name}</div><div class="metrics-tags" style="margin:0"><span class="lang-tag">${data.lang}</span><span class="chip-badge badge-${data.risk.level}">${data.risk.label}</span></div></div><div class="metrics-body"><div class="metrics-path">${dir}/</div><div class="risk-alert alert-${data.risk.level}"><div class="alert-dot alert-dot-${data.risk.level}${data.risk.level==='red'?' pulse':''}"></div><span>${data.risk.desc}</span></div><div class="section-title">Degree of Expertise (DOE)</div><div class="exp-rows">${sorted.map(([u,v])=>`<div class="exp-row"><div class="exp-name" style="color:${devColor(u)}" title="${u}">${u}</div><div class="exp-track"><div class="exp-fill" style="width:${(v/maxVal*100).toFixed(1)}%;background:${devColor(u)}"></div></div><div class="exp-val">${v.toFixed(2)}</div></div>`).join('')}</div>${!data.important?`<div class="divider"></div><div style="font-size:11px;color:var(--text-dim)">ℹ This file type is not ranked in the at-risk list</div>`:''}<div class="activity-strip"><div class="activity-title">Recent Commits</div><div id="activity-content"><span class="activity-empty">Loading…</span></div></div></div>`;
        loadFileActivity(path);
        } else {
            const allFiles=path?FILES.filter(f=>f.path.startsWith(path+'/')):FILES;
            const important=allFiles.filter(f=>f.important), byRisk={red:[],orange:[],yellow:[],green:[]};
            important.forEach(f=>byRisk[f.risk.level].push(f));
            const langs=[...new Set(important.map(f=>f.lang))], name=path?path.split('/').pop():META.repo.split('/')[1];
            const risky=important.filter(f=>f.risk.level!=='green');
            card.innerHTML=`<div class="card-header"><div class="card-header-title">📂 ${name}/</div><div class="metrics-tags" style="margin:0">${langs.map(l=>`<span class="lang-tag">${l}</span>`).join('')}</div></div><div class="metrics-body"><div class="folder-stats"><div class="stat-box"><div class="stat-num">${allFiles.length}</div><div class="stat-label">total files</div></div><div class="stat-box"><div class="stat-num" style="color:var(--red)">${risky.length}</div><div class="stat-label">at-risk files</div></div></div><div class="section-title">Risk Breakdown</div><div class="risk-breakdown">${byRisk.red.length?`<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--red)"></div><span>Sole Expert</span></div><span class="breakdown-count">${byRisk.red.length}</span></div>`:''} ${byRisk.orange.length?`<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--orange)"></div><span>Sole Maintainer</span></div><span class="breakdown-count">${byRisk.orange.length}</span></div>`:''} ${byRisk.yellow.length?`<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--yellow)"></div><span>Narrow Expertise</span></div><span class="breakdown-count">${byRisk.yellow.length}</span></div>`:''} ${byRisk.green.length?`<div class="breakdown-row"><div class="breakdown-left"><div class="breakdown-dot" style="background:var(--green)"></div><span>Broad Coverage</span></div><span class="breakdown-count">${byRisk.green.length}</span></div>`:''} ${!important.length?'<div style="color:var(--text-dim);font-size:11px">No important files here</div>':''}</div>${risky.length?`<div class="divider"></div><div class="section-title">Files to Watch</div><div style="display:flex;flex-direction:column;gap:4px">${risky.map(f=>`<div class="watch-row" onclick="navigateTo('${f.path}')"><div class="breakdown-dot" style="background:${DOT[f.risk.level]}"></div><span class="watch-name">${f.path.split('/').pop()}</span></div>`).join('')}</div>`:''}</div>`;
        }
    }
    function loadFileActivity(filePath) {
        const strip = document.getElementById('activity-content');
        if (!strip) return;
        const [org, repo] = (META.repo || '').split('/');
        if (!org || !repo) return;
        fetch(`/api/repos/${org}/${repo}/file-activity?path=${encodeURIComponent(filePath)}`)
            .then(r => r.json())
            .then(data => {
                if (!data.commits || !data.commits.length) {
                    strip.innerHTML = '<span class="activity-empty">No commit history found for this file.</span>';
                    return;
                }
                strip.innerHTML = data.commits.map(c =>
                    `<div class="activity-row">` +
                    `<a class="activity-sha" href="${c.url}" target="_blank">${c.sha}</a>` +
                    `<span class="activity-date">${c.date}</span>` +
                    `<span class="activity-author">${c.author}</span>` +
                    `<span class="activity-diff"><span class="add">+${c.adds}</span>&nbsp;<span class="del">-${c.dels}</span></span>` +
                    `</div>`
                ).join('');
            })
            .catch(() => { strip.innerHTML = '<span class="activity-empty">Could not load activity.</span>'; });
    }
    renderRiskList();
    renderTree();
    </script>
    </body>
    </html>"""


def _build_kd_html(repo_full_name: str, tf: int, tf_devs_short: list,
                   files_json: list, selected_dev_short: str | None) -> str:
    meta = {
        "repo":         repo_full_name,
        "tf":           int(tf),
        "tf_devs":      tf_devs_short,
        "file_count":   len(files_json),
        "selected_dev": selected_dev_short,
    }
    data_block = (
        f"    window.REPO_META  = {json.dumps(meta, ensure_ascii=False)};\n"
        f"    window.REPO_FILES = {json.dumps(files_json, ensure_ascii=False)};\n"
    )
    return _knowledge_html_template().replace("/* __DATA_INJECTION__ */", data_block)


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


def _build_ph_html(ph_data: dict, dev_id: str, display_name: str,
                   break_weeks: int, login: str = "") -> str:
    devs     = ph_data.get("developers", {})
    dev_data = devs.get(dev_id)
    # project_health.json keys developers by numeric author_id or "author_login|{login}",
    # but dev_id is a GitHub node ID — try login-based fallbacks.
    if dev_data is None and login:
        dev_data = devs.get(f"author_login|{login}") or devs.get(login)
    if dev_data is None:
        for k, v in devs.items():
            if dev_id in k or k in dev_id:
                dev_data = v
                break
    if dev_data is None and login:
        for k, v in devs.items():
            if login.lower() in k.lower() or k.lower() in login.lower():
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
    max_pct      = max(_pct(bc, tc), _pct(bp, tp), _pct(bi, ti))
    summary = (
        f"{display_name} contributes ~{max_pct}% of repo activity. "
        f"A {break_weeks}-week absence is projected to reduce activity by "
        f"~{round(bc*break_weeks,1)} commits, ~{round(bp*break_weeks,1)} PRs, "
        f"and ~{round(bi*break_weeks,1)} issues."
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


def _build_ph_html_from_history(org: str, repo: str, dev_id: str,
                                display_name: str, break_weeks: int) -> str:
    """Fallback for developers absent from project_health.json (e.g. already departed).
    Builds the panel from commit_history using 16 weeks ending at their last active date."""
    try:
        ch = _load_parquet(org, repo, "commit_history")
    except Exception:
        return ""

    ch["date"] = pd.to_datetime(ch["date"], utc=True, errors="coerce")
    dev_ch = ch[ch["dev_id"] == dev_id].dropna(subset=["date"])
    if dev_ch.empty:
        return ""

    n_weeks   = 16
    # Add 1 day so the last commit day falls inside the final week bucket (< we).
    last_date = dev_ch["date"].max() + pd.Timedelta(days=1)
    week_starts = [last_date - pd.Timedelta(weeks=(n_weeks - i)) for i in range(n_weeks)]
    week_labels = [d.strftime("%b %d") for d in week_starts]

    def _weekly_sum(df, col="commit_count"):
        out = []
        for ws in week_starts:
            we = ws + pd.Timedelta(weeks=1)
            out.append(int(df.loc[(df["date"] >= ws) & (df["date"] < we), col].sum()))
        return out

    weekly_commits = _weekly_sum(dev_ch)
    repo_weekly    = _weekly_sum(ch)

    bc       = round(sum(weekly_commits) / n_weeks, 1)
    repo_avg = round(sum(repo_weekly)    / n_weeks, 1) or 1
    if bc == 0:
        return ""

    fake_ph = {
        "repo":         f"{org}/{repo}",
        "generated_at": last_date.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "weeks":        week_labels,
        "repo_totals":  {"commits_per_week": repo_avg, "prs_per_week": 1, "issues_per_week": 1},
        "developers": {
            dev_id: {
                "weekly_commits": weekly_commits,
                "weekly_prs":     [0] * n_weeks,
                "weekly_issues":  [0] * n_weeks,
                "baseline_commits": bc,
                "baseline_prs":     0.0,
                "baseline_issues":  0.0,
            }
        },
    }
    return _build_ph_html(fake_ph, dev_id, display_name, break_weeks)


def _raw_repo_weekly_avg(org: str, repo: str,
                         sd: "pd.Timestamp", ed: "pd.Timestamp",
                         n_weeks: int) -> tuple:
    """Count all-developer commits/PRs/issues from raw CSVs within [sd, ed].

    Returns (commits_per_week, prs_per_week, issues_per_week).
    Uses raw source files so non-truck-factor developers are included.
    """
    repo_dir = _repo_dir(org, repo)

    def _count(fname: str) -> int:
        path = _resolve_file(repo_dir / fname)
        if not path.exists():
            return 0
        try:
            df = pd.read_csv(path, usecols=["created_at"], low_memory=False)
            ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
            return int(((ts >= sd) & (ts <= ed)).sum())
        except Exception:
            return 0

    n = max(n_weeks, 1)
    tc = round(_count("commit_list.csv") / n, 1) or 1
    tp = round(_count("prs_repo.csv")    / n, 1) or 1
    ti = round(_count("issues.csv")      / n, 1) or 1
    return tc, tp, ti


def _build_ph_from_window(org: str, repo: str, dev_id: str, display_name: str,
                          start_date: str, end_date: str, break_weeks: int) -> str:
    """Build a PH panel filtered to a specific date window.

    Uses activity_weekly.parquet + activity_baselines.parquet to reconstruct absolute
    dates for each stored week, then filters to [start_date, end_date].
    Falls back to commit_history if the window lies outside the stored 16-week range.
    """
    import datetime as _dt
    try:
        aw  = _load_parquet(org, repo, "activity_weekly")
        ab  = _load_parquet(org, repo, "activity_baselines")
    except Exception:
        return ""

    if aw.empty or ab.empty:
        return ""

    # --- Map week_index → absolute date using the generated_at anchor ---
    # generated_at is the same for all devs (one snapshot), grab first non-null.
    gen_at_raw = ab["generated_at"].dropna().iloc[0] if not ab["generated_at"].dropna().empty else None
    if gen_at_raw is None:
        return ""
    gen_at = pd.Timestamp(gen_at_raw, tz="UTC") if not isinstance(gen_at_raw, pd.Timestamp) \
             else gen_at_raw.tz_localize("UTC") if gen_at_raw.tzinfo is None else gen_at_raw

    aw = aw.copy()
    aw["abs_date"] = aw["week_index"].apply(lambda wi: gen_at + pd.Timedelta(weeks=int(wi)))

    sd = pd.Timestamp(start_date, tz="UTC")
    ed = pd.Timestamp(end_date,   tz="UTC") + pd.Timedelta(days=6)   # inclusive week end

    window = aw[(aw["abs_date"] >= sd) & (aw["abs_date"] <= ed)]

    # No stored data for this window → fall back to raw commit history
    if window.empty:
        return _build_ph_html_from_history(org, repo, dev_id, display_name, break_weeks)

    # --- Build ph_data dict from filtered rows ---
    dev_rows  = window[window["dev_id"] == dev_id].sort_values("abs_date")
    all_rows  = window.sort_values("abs_date")

    if dev_rows.empty:
        return _build_ph_html_from_history(org, repo, dev_id, display_name, break_weeks)

    n_weeks      = len(dev_rows)
    week_labels  = [r.strftime("%b %d") for r in dev_rows["abs_date"]]
    wc = dev_rows["commits"].fillna(0).tolist()
    wp = dev_rows["prs"].fillna(0).tolist()
    wi = dev_rows["issues"].fillna(0).tolist()

    bc = round(sum(wc) / n_weeks, 1)
    bp = round(sum(wp) / n_weeks, 1)
    bi = round(sum(wi) / n_weeks, 1)
    if bc + bp + bi == 0:
        return _build_ph_html_from_history(org, repo, dev_id, display_name, break_weeks)

    # Repo totals — count from raw CSVs so ALL developers are included, not just
    # truck-factor devs stored in activity_weekly.parquet.
    tc, tp, ti = _raw_repo_weekly_avg(org, repo, sd, ed, n_weeks)

    ph_data = {
        "repo":         f"{org}/{repo}",
        "generated_at": ed.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "weeks":        week_labels,
        "repo_totals":  {"commits_per_week": tc, "prs_per_week": tp, "issues_per_week": ti},
        "developers": {
            dev_id: {
                "weekly_commits": [float(v) for v in wc],
                "weekly_prs":     [float(v) for v in wp],
                "weekly_issues":  [float(v) for v in wi],
                "baseline_commits": bc,
                "baseline_prs":     bp,
                "baseline_issues":  bi,
            }
        },
    }
    return _build_ph_html(ph_data, dev_id, display_name, break_weeks, login=display_name)


# ── Departure-simulation endpoints (return full HTML panels) ─────────────────

@app.get(
    "/api/repos/{org}/{repo}/departure/{dev_id}/project-health-panel",
    response_class=HTMLResponse,
)
def departure_ph_panel(
    org: str, repo: str, dev_id: str,
    break_weeks: int = 2,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
):
    ph_path = _resolve_file(_distrac(org, repo) / "project_health.json")
    if not ph_path.exists():
        ph_path = _resolve_file(_repo_dir(org, repo) / "ProjectHealth" / "project_health.json")
    try:
        devs_df = _load_parquet(org, repo, "developers").drop_duplicates("dev_id")
        row     = devs_df[devs_df["dev_id"] == dev_id]
        display = str(row.iloc[0].get("login") or dev_id) if not row.empty else dev_id
    except Exception:
        display = dev_id.split("|")[-1] if "|" in dev_id else dev_id

    # When a date window is supplied, compute PH from the stored weekly activity data.
    if start_date and end_date:
        html = _build_ph_from_window(org, repo, dev_id, display, start_date, end_date, break_weeks)
        if html:
            return HTMLResponse(content=html)
        # fall through to standard logic if window yields nothing

    html = ""
    if ph_path.exists():
        ph_data = json.loads(ph_path.read_text(encoding="utf-8"))
        # Recompute repo_totals from raw CSVs: project_health.json may have been cached
        # when only truck-factor developers were stored, making the repo total equal to
        # the TF dev's total (100% for repos with TF=1).
        try:
            _n_weeks = len(ph_data.get("weeks", [])) or 16
            _gen_at_str = ph_data.get("generated_at", "")
            if _gen_at_str:
                _gen_at = pd.Timestamp(_gen_at_str, tz="UTC") \
                          if "+" in _gen_at_str or "Z" in _gen_at_str \
                          else pd.Timestamp(_gen_at_str).tz_localize("UTC")
                _ph_sd = _gen_at - pd.Timedelta(weeks=_n_weeks)
                _ph_ed = _gen_at
                _rtc, _rtp, _rti = _raw_repo_weekly_avg(org, repo, _ph_sd, _ph_ed, _n_weeks)
                ph_data["repo_totals"] = {
                    "commits_per_week": _rtc,
                    "prs_per_week":     _rtp,
                    "issues_per_week":  _rti,
                }
        except Exception:
            pass  # keep existing repo_totals if something goes wrong
        html = _build_ph_html(ph_data, dev_id, display, break_weeks, login=display)
    if not html:
        # Developer absent from the current-window project_health.json (likely already departed).
        # Fall back to commit_history to show their last 16 weeks of active contribution.
        html = _build_ph_html_from_history(org, repo, dev_id, display, break_weeks)
    if not html:
        raise HTTPException(404, f"No project health data for developer {dev_id}")
    return HTMLResponse(content=html)


# In-memory cache for on-demand KD recomputation: (org, repo, as_of_date) → (doe_df, tf_count, tf_devs_short)
_kd_cache: dict = {}


def _compute_kd_for_date(org: str, repo: str, as_of_date_str: str):
    """Recompute KD (DOE + truck factor) from per_file_commits.csv filtered to committed_at <= as_of_date.
    Results are cached in _kd_cache so repeated requests for the same date are instant.
    Returns (doe_df, tf_count, tf_devs_short) or (None, None, None) if CSV not available.
    """
    cache_key = (org, repo, as_of_date_str)
    if cache_key in _kd_cache:
        return _kd_cache[cache_key]

    csv_path = _resolve_file(_repo_dir(org, repo) / cfg.per_file_commits_path)
    if not csv_path.exists():
        _kd_cache[cache_key] = (None, None, None)
        return (None, None, None)

    ext_dir = str(ROOT / "Extractors")
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    try:
        from KnowledgeDistribution import (
            calculate_doe_metrics, calculate_doe,
            build_authors_map_from_doe, runTruckFactor,
        )
    except ImportError:
        _kd_cache[cache_key] = (None, None, None)
        return (None, None, None)

    try:
        df = pd.read_csv(csv_path, sep=cfg.CSV_separator if hasattr(cfg, "CSV_separator") else ",",
                         low_memory=False)
        if "committed_at" not in df.columns:
            raise ValueError("committed_at column missing")
        df["committed_at"] = pd.to_datetime(df["committed_at"], utc=True, errors="coerce")
        cutoff = pd.Timestamp(as_of_date_str, tz="UTC") + pd.Timedelta(days=1)
        df = df[df["committed_at"] < cutoff].copy()
        if df.empty:
            _kd_cache[cache_key] = (None, None, None)
            return (None, None, None)

        metrics_df  = calculate_doe_metrics(df)
        doe_df      = calculate_doe(metrics_df)
        authors_map = build_authors_map_from_doe(doe_df)
        tf_count, tf_list = runTruckFactor(doe_df, authors_map)

        # Rename 'developer' → 'dev_id' to match what _kd_build_files_json expects.
        if "developer" in doe_df.columns and "dev_id" not in doe_df.columns:
            doe_df = doe_df.rename(columns={"developer": "dev_id"})

        # Add lang and is_important columns.
        if "file_path" in doe_df.columns:
            doe_df["lang"]         = doe_df["file_path"].apply(lambda p: _KD_LANG_MAP.get(Path(str(p)).suffix.lower(), "Other"))
            doe_df["is_important"] = doe_df["file_path"].apply(lambda p: Path(str(p)).suffix.lower() in _KD_IMPORTANT_EXT)

        result = (doe_df, tf_count, tf_list)
    except Exception as exc:
        print(f"[KD on-demand] error for {org}/{repo} as_of={as_of_date_str}: {exc}")
        result = (None, None, None)

    _kd_cache[cache_key] = result
    return result


@app.get(
    "/api/repos/{org}/{repo}/departure/{dev_id}/knowledge-panel",
    response_class=HTMLResponse,
)
def departure_kd_panel(org: str, repo: str, dev_id: str, as_of_date: Optional[str] = None):
    devs_df = _load_parquet(org, repo, "developers").drop_duplicates("dev_id")
    dev_map = _kd_build_dev_map(devs_df)
    selected_short = _kd_resolve_dev(dev_id, dev_map) if dev_id else None

    # On-demand recompute when a date is requested.
    if as_of_date:
        doe_df, tf_count, tf_list_raw = _compute_kd_for_date(org, repo, as_of_date)
        if doe_df is not None:
            files_json = _kd_build_files_json(doe_df, dev_map)
            if files_json:
                tf_devs_short = [_kd_resolve_dev(str(d), dev_map) for d in (tf_list_raw or [])]
                return HTMLResponse(content=_build_kd_html(
                    f"{org}/{repo}", tf_count, tf_devs_short, files_json, selected_short,
                ))
        # Fall through to static parquets if recompute failed or CSV missing.

    doe_df = _load_parquet(org, repo, "knowledge_doe")
    tf_df  = _load_parquet(org, repo, "truck_factor")
    tf_count = int(tf_df["tf"].iloc[0]) if len(tf_df) else 0
    tf_col = next((c for c in ("is_tf_dev", "is_truck_factor") if c in devs_df.columns), None)
    if tf_col is not None:
        tf_devs_short = [
            _kd_resolve_dev(str(row["dev_id"]), dev_map)
            for _, row in devs_df[devs_df[tf_col] == True].iterrows()
            if pd.notna(row.get("dev_id"))
        ]
    else:
        tf_devs_short = []
    files_json = _kd_build_files_json(doe_df, dev_map)
    if not files_json:
        raise HTTPException(404, "No knowledge distribution data found — run the Predictors step first")
    return HTMLResponse(content=_build_kd_html(
        f"{org}/{repo}", tf_count, tf_devs_short, files_json, selected_short,
    ))


@app.get("/api/repos/{org}/{repo}/file-activity")
def file_activity(org: str, repo: str, path: str, limit: int = 8):
    """Return the most recent commits touching a specific file path."""
    csv_path = _resolve_file(_repo_dir(org, repo) / cfg.per_file_commits_path)
    if not csv_path.exists():
        return {"commits": [], "total": 0}
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["file_path"] == path].copy()
    if df.empty:
        return {"commits": [], "total": 0}
    df["committed_at"] = pd.to_datetime(df["committed_at"], utc=True, errors="coerce")
    df = df.sort_values("committed_at", ascending=False).drop_duplicates("sha").head(limit)
    devs_df = _load_parquet(org, repo, "developers").drop_duplicates("dev_id")
    dev_map = _kd_build_dev_map(devs_df)
    commits = []
    for _, row in df.iterrows():
        dev_id = str(row.get("author_id", "")).strip()
        name = (dev_map.get(dev_id)
                or str(row.get("author_name") or "").strip()
                or str(row.get("author_login") or "").strip()
                or "unknown")
        sha = str(row.get("sha", ""))
        commits.append({
            "sha":   sha[:7],
            "date":  str(row["committed_at"])[:10] if pd.notna(row.get("committed_at")) else "",
            "author": name,
            "adds":  int(row.get("additions", 0) or 0),
            "dels":  int(row.get("deletions", 0) or 0),
            "url":   f"https://github.com/{org}/{repo}/commit/{sha}",
        })
    return {"commits": commits, "total": int(len(df))}


@app.get(
    "/api/repos/{org}/{repo}/departure/{dev_id}/stn-panel",
    response_class=HTMLResponse,
)
def departure_stn_panel(
    org: str, repo: str, dev_id: str,
    date: Optional[str] = None, window_days: int = 30,
):
    for _d in (str(ROOT / "Analysis"), str(ROOT / "Dashboard"), str(ROOT / "Extractors")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
    try:
        from SocialTechnicalNetworkV2 import get_html_for_streamlit
    except ImportError:
        raise HTTPException(500, "SocialTechnicalNetworkV2 module not available")
    import datetime as _dt
    as_of = _dt.date.fromisoformat(date) if date else _dt.date.today()

    # Resolve dev_id → login so the STN can auto-select the right node
    initial_node = None
    tf_logins    = []
    try:
        devs_df = _load_parquet(org, repo, "developers").drop_duplicates("dev_id")
        row = devs_df[devs_df["dev_id"] == dev_id]
        if not row.empty:
            login = row.iloc[0].get("login")
            if login and str(login).strip() and str(login).lower() not in ("nan", "none", ""):
                initial_node = str(login).strip()
        # Collect truck-factor developer logins for priority display
        tf_col = next((c for c in ("is_tf_dev", "is_truck_factor", "truck_factor") if c in devs_df.columns), None)
        if tf_col:
            tf_logins = devs_df[devs_df[tf_col] == True]["login"].dropna().astype(str).tolist()
    except Exception:
        pass
    if initial_node is None:
        initial_node = dev_id

    html = get_html_for_streamlit(
        repo_full_name=f"{org}/{repo}",
        as_of_date=as_of,
        window_days=window_days,
        initial_node=initial_node,
        tf_devs=tf_logins,
    )
    return HTMLResponse(content=html)


# ── static file serving ───────────────────────────────────────────────────────

if STATIC_DIR.exists():
    # Serve everything in static/ but catch root explicitly
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"DISTRAC API starting — http://localhost:8000")
    print(f"  orgs dir : {ORGS_DIR}")
    print(f"  static   : {STATIC_DIR}")
    uvicorn.run("distrac_api:app", host="0.0.0.0", port=8000,
                reload=True, reload_dirs=[str(ROOT / "Extractors")])
