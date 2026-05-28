"""
OSS Developer Departure Simulation — Standalone Streamlit App
Combines the scenario-input flow with Dashboard.py's simulation panels.

Run:
    cd <project_root>
    streamlit run departure_sim.py
"""

import json
import sys
import datetime
from datetime import timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ── Locate project root robustly ─────────────────────────────────────────────
# departure_sim.py may be executed from the project root (standalone) or from
# Extractors/ (via app.py st.Page).  Walk up from __file__ until we find the
# Settings.py anchor that lives in the project root.
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, here.parent.parent]:
        if (candidate / "Settings.py").exists():
            return candidate
    return here.parent   # fallback: best-effort

PROJECT_ROOT = _find_project_root()
_ext_dir = str(PROJECT_ROOT / "Extractors")
if _ext_dir not in sys.path:
    sys.path.insert(0, _ext_dir)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import Dashboard panel functions ─────────────────────────────────────────
from Dashboard import (
    list_orgs,
    list_repos_for,
    repo_data_status,
    load_commit_history,
    build_risk_table,
    get_display_name,
    render_stn_panel,
    render_simulation_panel,
    render_project_health_panel,
    render_knowledge_panel,
    ORG_BASE,
)
import Settings as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers  (departure_sim-specific — lightweight, CSV-backed)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_cursor_max_date(repo_full: str) -> datetime.date:
    path = ORG_BASE / repo_full / cfg.data_cursor
    if path.exists():
        try:
            data = json.loads(path.read_text())
            ts = data.get("updated_at", "")
            if ts:
                return datetime.datetime.fromisoformat(
                    ts.replace("Z", "+00:00")
                ).date()
        except Exception:
            pass
    hist = load_commit_history(repo_full)
    return hist["date"].max() if not hist.empty else datetime.date.today()


@st.cache_data(show_spinner=False)
def _load_commit_min_date(repo_full: str) -> datetime.date:
    hist = load_commit_history(repo_full)
    return hist["date"].min() if not hist.empty else datetime.date(2000, 1, 1)


@st.cache_data(show_spinner=False)
def _load_weekly_commits(repo_full: str) -> pd.DataFrame:
    """Return weekly commit counts for the whole repo."""
    hist = load_commit_history(repo_full)
    if hist.empty:
        return pd.DataFrame(columns=["week", "commits"])
    hist["date"] = pd.to_datetime(hist["date"])
    hist["week"] = hist["date"].dt.to_period("W").dt.start_time
    weekly = hist.groupby("week")["count"].sum().reset_index(name="commits")
    weekly["week"] = pd.to_datetime(weekly["week"])
    return weekly


# ─────────────────────────────────────────────────────────────────────────────
# Input page
# ─────────────────────────────────────────────────────────────────────────────

def render_input_page(repo_full: str) -> None:
    org, repo = repo_full.split("/", 1)

    st.title("Developer Departure Simulation")
    st.caption(f"{repo_full} · Predict who might leave and simulate their departure")

    # ── Load date bounds ────────────────────────────────────────────────────
    max_date = _load_cursor_max_date(repo_full)
    min_date = _load_commit_min_date(repo_full)

    # ── Lookback window slider (above chart) ────────────────────────────────
    max_window = max(30, (max_date - min_date).days)
    window_days = st.slider(
        "Lookback window (days)",
        min_value=30,
        max_value=min(max_window, 730),
        value=min(365, max_window),
        step=30,
        key="window_days",
        help="How far back from the prediction date to analyse commits",
    )

    # ── Commit activity bar chart ───────────────────────────────────────────
    st.subheader("Commit activity")

    weekly = _load_weekly_commits(repo_full)

    # We need current_date to filter; read it from session_state if already set
    # (first run it won't be set yet — use max_date as default)
    current_date_val = st.session_state.get("current_date", max_date)
    start_date_preview = current_date_val - timedelta(days=window_days)

    chart_data = weekly[
        (weekly["week"] >= pd.Timestamp(start_date_preview))
        & (weekly["week"] <= pd.Timestamp(current_date_val))
    ]

    if chart_data.empty:
        st.info("No commit data in this window.")
    else:
        chart = (
            alt.Chart(chart_data)
            .mark_bar(color="#4C78A8", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y", labelAngle=-30)),
                y=alt.Y("commits:Q", title="Commits per week"),
                tooltip=[
                    alt.Tooltip("week:T", title="Week", format="%b %d, %Y"),
                    alt.Tooltip("commits:Q", title="Commits"),
                ],
            )
            .properties(height=200)
        )
        st.altair_chart(chart, use_container_width=True)

    st.caption(
        f"{start_date_preview} → {current_date_val}  ·  "
        f"{int(chart_data['commits'].sum()):,} commits in window"
    )

    # ── Prediction date slider (below chart — aligns with x-axis) ──────────
    current_date = st.slider(
        "Prediction date",
        min_value=min_date,
        max_value=max_date,
        value=max_date,
        format="YYYY-MM-DD",
        key="current_date",
        help="Slide to change the 'as-of' date. The chart and developer risk list update automatically.",
    )

    start_date = current_date - timedelta(days=window_days)

    # ── Developer risk list ─────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Developers at risk — as of {current_date}")

    risk_df = build_risk_table(repo_full, current_date)

    if risk_df is None or risk_df.empty:
        st.warning(
            "No model predictions found for this repo yet.  \n"
            "Run the **Predictors** pipeline on the Pipeline page to generate predictions."
        )
        return

    # selectbox (reliable session_state write)
    dev_options = risk_df["dev_id"].tolist()
    dev_labels = {
        row.dev_id: f"{row.Contributor}  —  {row.prob_1:.1%} departure risk"
        for row in risk_df.itertuples()
    }

    selected_dev_id = st.selectbox(
        "Select a developer to simulate",
        options=dev_options,
        format_func=lambda x: dev_labels.get(x, x),
        key="selected_developer",
    )

    # Card-style developer grid
    st.markdown("##### All tracked developers")
    hcols = st.columns([3, 1, 2, 2])
    hcols[0].markdown("**Developer**")
    hcols[1].markdown("**Role**")
    hcols[2].markdown("**Departure risk**")
    hcols[3].markdown("**Last prediction**")

    for row in risk_df.itertuples():
        is_selected = row.dev_id == selected_dev_id
        prefix = "▶ " if is_selected else "  "
        name_md = f"{prefix}**{row.Contributor}**" if is_selected else f"{prefix}{row.Contributor}"

        cols = st.columns([3, 1, 2, 2])
        cols[0].markdown(name_md)
        cols[1].markdown(f"`{getattr(row, 'Role', '—')}`")
        cols[2].progress(float(row.prob_1), text=f"{row.prob_1:.1%}")
        # Next break length as proxy for last-prediction context
        cols[3].markdown(getattr(row, "Next Break Length", "—"))

    # ── Selection summary + action ──────────────────────────────────────────
    st.divider()
    selected_name = dev_labels.get(selected_dev_id, selected_dev_id).split("  —")[0]

    row = risk_df[risk_df["dev_id"] == selected_dev_id]
    selected_prob = float(row["prob_1"].iloc[0]) if not row.empty else 0.0
    break_raw = str(row["Next Break Length"].iloc[0]) if not row.empty else ""
    nums = [int(s) for s in break_raw.split() if s.isdigit()]
    break_weeks = max(1, round(nums[0] / 7)) if nums and "day" in break_raw else (nums[0] if nums else 2)

    st.markdown(
        f"**Selected:** {selected_name} &nbsp;|&nbsp; "
        f"**Date:** {current_date} &nbsp;|&nbsp; "
        f"**Window:** {window_days} days &nbsp;|&nbsp; "
        f"**Risk:** {selected_prob:.1%}"
    )

    if st.button("Simulate Departure", type="primary"):
        st.session_state.page = "simulation"
        st.session_state.selected_name = selected_name
        st.session_state.selected_date = current_date
        st.session_state.break_weeks = break_weeks
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation page
# ─────────────────────────────────────────────────────────────────────────────

def render_simulation_page(repo_full: str) -> None:
    dev_id  = st.session_state.get("selected_developer", None)
    name    = st.session_state.get("selected_name", dev_id or "Unknown")
    date    = st.session_state.get("selected_date", datetime.date.today())
    window  = st.session_state.get("window_days", 365)
    bweeks  = st.session_state.get("break_weeks", 2)

    if st.button("← Back to scenario"):
        st.session_state.page = "input"
        st.rerun()

    st.title(f"Simulating departure: {name}")
    st.caption(f"{repo_full} · As of {date} · {window}-day lookback window · ~{bweeks}-week absence projected")

    tab1, tab2, tab3 = st.tabs(["Social Network", "Project Health", "Knowledge Distribution"])

    with tab1:
        st.subheader("Social / Technical Network")
        # D3 interactive network graph — developer highlighted + departure impact
        render_stn_panel(repo_full, dev_id, date)
        

    with tab2:
        st.subheader("Project Health Impact")
        # Weekly contribution chart + projected absence impact
        render_project_health_panel(repo_full, dev_id, break_weeks=bweeks)

    with tab3:
        st.subheader("Knowledge Distribution")
        # D3 treemap of files at risk
        render_knowledge_panel(repo_full_name=repo_full, selected_dev=dev_id)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str | None:
    """Render org/repo selector in sidebar. Returns repo_full or None."""
    with st.sidebar:
        st.header("Repository")

        orgs = list_orgs()
        if not orgs:
            st.error(f"No organizations found under:\n`{ORG_BASE}`")
            return None

        org = st.selectbox("Organization", orgs, key="sidebar_org")
        repos = list_repos_for(org)
        if not repos:
            st.warning("No repos found for this organization.")
            return None

        repo = st.selectbox("Repository", repos, key="sidebar_repo")
        repo_full = f"{org}/{repo}"

        # Reset simulation state when repo changes
        if st.session_state.get("_last_repo") != repo_full:
            st.session_state["_last_repo"] = repo_full
            st.session_state["page"] = "input"

        if st.button("Clear cache", help="Reload all files from disk after running the pipeline"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.caption("Data status:")
        status = repo_data_status(repo_full)
        if status.get("distrac"):
            st.markdown("✅ **distrac/ outputs ready**")
            generated_at = status.get("generated_at", "")
            if generated_at:
                try:
                    ts = datetime.datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
                    age_h = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds() / 3600
                    age_str = f"{age_h:.0f}h ago" if age_h < 48 else f"{age_h / 24:.0f}d ago"
                    st.caption(f"Generated: {age_str}")
                except Exception:
                    st.caption(f"Generated: {generated_at[:10]}")
        else:
            st.warning("distrac/ not built — run pipeline first")

        icons = {True: "✅", False: "❌"}
        st.markdown(f"{icons[status['timeline']]} Labeled timeline")
        st.markdown(f"{icons[status['model']]} Model predictions")
        st.markdown(f"{icons[status['truck_factor']]} Truck Factor")
        st.markdown(f"{icons[status['knowledge_dist']]} Knowledge distribution")

    return repo_full


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Departure Simulation · DISTRAC",
        layout="wide",
    )

    if "page" not in st.session_state:
        st.session_state.page = "input"

    repo_full = render_sidebar()
    if not repo_full:
        return

    if st.session_state.page == "simulation":
        render_simulation_page(repo_full)
    else:
        render_input_page(repo_full)


if __name__ == "__main__":
    main()
