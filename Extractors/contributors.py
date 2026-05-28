"""
Contributor Dashboard — Page 1
Mimics the GitHub /graphs/contributors page.

Install: pip install streamlit streamlit-echarts pandas
Run:     streamlit run contributors.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from streamlit_echarts import st_echarts

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Contributors",
    layout="wide",
    page_icon="📊",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STATE_COLORS = {
    "active":     "#3fb950",   # green
    "non_coding": "#d29922",   # yellow
    "inactive":   "#f85149",   # red
}

# Semi-transparent versions for markArea backgrounds
STATE_COLORS_BG = {
    "active":     "rgba(63,185,80,0.18)",
    "non_coding": "rgba(210,153,34,0.22)",
    "inactive":   "rgba(248,81,73,0.20)",
}


def build_mark_areas(dates: list, states: list) -> list:
    """
    Group consecutive same-state weeks into markArea band pairs for echarts.
    Returns a list of [{xAxis, itemStyle}, {xAxis}] pairs.
    """
    if not dates:
        return []
    areas = []
    run_start = 0
    for i in range(1, len(dates)):
        if states[i] != states[run_start]:
            areas.append([
                {"xAxis": dates[run_start],
                 "itemStyle": {"color": STATE_COLORS_BG[states[run_start]]}},
                {"xAxis": dates[i - 1]},
            ])
            run_start = i
    # flush the final run
    areas.append([
        {"xAxis": dates[run_start],
         "itemStyle": {"color": STATE_COLORS_BG[states[run_start]]}},
        {"xAxis": dates[-1]},
    ])
    return areas

# ─────────────────────────────────────────────
# MOCK DATA  (replace with your real data)
# ─────────────────────────────────────────────
@st.cache_data
def make_mock_data():
    """
    Returns (df_commits, df_meta).
    df_commits columns: date, contributor, commits, color, state
    df_meta    columns: name, total_commits, additions, deletions, avatar, rank, color
    """
    np.random.seed(42)
    contributors = [
        {"name": "mattdowle",      "avatar": "🧑‍💻", "rank": 1, "color": "#1f77b4"},
        {"name": "MichaelChirico", "avatar": "👨‍💻", "rank": 2, "color": "#ff7f0e"},
        {"name": "jangorecki",     "avatar": "🧑‍🔬", "rank": 3, "color": "#2ca02c"},
        {"name": "tdhock",         "avatar": "👩‍💻", "rank": 4, "color": "#d62728"},
        {"name": "Rdatatable",     "avatar": "🤖",   "rank": 5, "color": "#9467bd"},
    ]

    start = datetime(2008, 9, 1)
    end   = datetime(2026, 3, 28)
    weeks = pd.date_range(start, end, freq="W")

    rows = []
    for c in contributors:
        base = np.random.randint(5, 40)
        for i, week in enumerate(weeks):
            activity = max(0, int(
                base
                * np.random.exponential(1.0)
                * (1 + 2 * np.sin(i / 52 * np.pi + np.random.uniform(0, np.pi)))
            ))
            if np.random.random() < 0.35:
                activity = 0

            # Assign activity state — biased toward inactive when no commits
            if activity == 0:
                state = np.random.choice(
                    ["inactive", "non_coding"], p=[0.75, 0.25]
                )
            else:
                state = np.random.choice(
                    ["active", "non_coding", "inactive"], p=[0.70, 0.20, 0.10]
                )

            rows.append({
                "date":        week,
                "contributor": c["name"],
                "commits":     activity,
                "color":       c["color"],
                "state":       state,
            })

    df = pd.DataFrame(rows)

    # Compute all-time totals for cards
    totals = df.groupby("contributor")["commits"].sum().reset_index()
    totals.columns = ["name", "total_commits"]

    rng = np.random.default_rng(0)
    totals["additions"] = (totals["total_commits"] * rng.uniform(100, 200, len(totals))).astype(int)
    totals["deletions"] = (totals["total_commits"] * rng.uniform(30,  80,  len(totals))).astype(int)

    meta = pd.DataFrame(contributors)
    totals = totals.merge(meta, on="name").sort_values("rank")

    return df, totals

df_all, df_meta = make_mock_data()

# ─────────────────────────────────────────────
# SESSION STATE  — single source of truth
# ─────────────────────────────────────────────
if "window_start_pct" not in st.session_state:
    st.session_state.window_start_pct = 0.0
if "window_end_pct" not in st.session_state:
    st.session_state.window_end_pct   = 100.0
if "selected_user" not in st.session_state:
    st.session_state.selected_user    = df_meta.iloc[0]["name"]
if "period_preset" not in st.session_state:
    st.session_state.period_preset    = "All"

_PRESET_OPTIONS = [
    "All", "Last month", "Last 3 months", "Last 6 months",
    "Last 12 months", "Last 24 months", "Custom range",
]

# ─────────────────────────────────────────────
# DATE / PCT HELPERS
# ─────────────────────────────────────────────
_min_ts = df_all["date"].min().timestamp()
_max_ts = df_all["date"].max().timestamp()

def pct_to_date(pct: float) -> datetime:
    ts = _min_ts + (pct / 100.0) * (_max_ts - _min_ts)
    return datetime.fromtimestamp(ts)

def date_to_pct(dt: datetime) -> float:
    return max(0.0, min(100.0, (dt.timestamp() - _min_ts) / (_max_ts - _min_ts) * 100))

def pct_for_preset(preset: str):
    """Return (start_pct, end_pct) for a named preset."""
    last_date = df_all["date"].max().to_pydatetime()
    if preset == "All":
        return 0.0, 100.0
    offsets = {
        "Last month":     timedelta(days=30),
        "Last 3 months":  timedelta(days=90),
        "Last 6 months":  timedelta(days=180),
        "Last 12 months": timedelta(days=365),
        "Last 24 months": timedelta(days=730),
    }
    if preset in offsets:
        return date_to_pct(last_date - offsets[preset]), 100.0
    # "Custom range" — keep current
    return st.session_state.window_start_pct, st.session_state.window_end_pct

# ─────────────────────────────────────────────
# COMPUTE WINDOW DATES  (derived from session state)
# ─────────────────────────────────────────────
window_start = pct_to_date(st.session_state.window_start_pct)
window_end   = pct_to_date(st.session_state.window_end_pct)

df_window = df_all[
    (df_all["date"] >= window_start) &
    (df_all["date"] <= window_end)
].copy()

# ─────────────────────────────────────────────
# WINDOW-SCOPED CARD STATS
# ─────────────────────────────────────────────
_window_commits = df_window.groupby("contributor")["commits"].sum().rename("window_commits")
_all_commits    = df_all.groupby("contributor")["commits"].sum().rename("all_commits")

_meta_idx = df_meta.set_index("name")
_meta_joined = (
    _meta_idx
    .join(_all_commits,    how="left")
    .join(_window_commits, how="left")
    .fillna({"window_commits": 0, "all_commits": 1})
)
_meta_joined["window_commits"] = _meta_joined["window_commits"].astype(int)
_ratio = _meta_joined["window_commits"] / _meta_joined["all_commits"].clip(lower=1)
_meta_joined["window_additions"] = (_meta_joined["additions"] * _ratio).astype(int)
_meta_joined["window_deletions"] = (_meta_joined["deletions"] * _ratio).astype(int)
df_meta_window = _meta_joined.reset_index()   # "name" column comes from the named index

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; color: #e6edf3; }
[data-testid="stHeader"] { background: #161b22; }
h1, h2, h3, p, label { color: #e6edf3 !important; }

.contrib-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
    cursor: pointer;
    transition: border-color 0.15s, box-shadow 0.15s;
    margin-bottom: 4px;
}
.contrib-card:hover  { border-color: #58a6ff; box-shadow: 0 0 0 3px rgba(88,166,255,0.15); }
.contrib-card.selected { border-color: #58a6ff; box-shadow: 0 0 0 3px rgba(88,166,255,0.25); }
.card-header  { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.card-username { font-weight: 600; font-size: 15px; color: #58a6ff; }
.card-rank    { margin-left: auto; background: #21262d; border: 1px solid #30363d;
                border-radius: 4px; padding: 2px 8px; font-size: 12px; color: #8b949e; }
.card-stats   { font-size: 13px; color: #8b949e; margin-bottom: 4px; }
.additions    { color: #3fb950; }
.deletions    { color: #f85149; }

.page-title   { font-size: 28px; font-weight: 600; color: #e6edf3; margin-bottom: 2px; }
.page-subtitle { font-size: 13px; color: #8b949e; margin-bottom: 24px; }
.window-badge { display: inline-block; background: #21262d; border: 1px solid #30363d;
                border-radius: 6px; padding: 4px 12px; font-size: 12px;
                color: #8b949e; margin-bottom: 16px; }

/* State legend */
.legend-row   { display: flex; gap: 16px; align-items: center; margin-bottom: 8px; }
.legend-dot   { display: inline-block; width: 10px; height: 10px;
                border-radius: 50%; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="page-title">Contributors</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Contributions per week to master, excluding merge commits</div>',
    unsafe_allow_html=True,
)

# ── Period preset selector ───────────────────────────────────────────────
_preset_col, _badge_col = st.columns([2, 5])
with _preset_col:
    chosen_preset = st.selectbox(
        "Period",
        options=_PRESET_OPTIONS,
        index=(_PRESET_OPTIONS.index(st.session_state.period_preset)
               if st.session_state.period_preset in _PRESET_OPTIONS else 0),
        key="_period_selectbox",
        label_visibility="collapsed",
    )
    # When user picks a non-custom preset that differs from current → apply it
    if chosen_preset != st.session_state.period_preset and chosen_preset != "Custom range":
        new_s, new_e = pct_for_preset(chosen_preset)
        st.session_state.period_preset    = chosen_preset
        st.session_state.window_start_pct = new_s
        st.session_state.window_end_pct   = new_e
        st.rerun()

with _badge_col:
    st.markdown(
        f'<div class="window-badge">📅 {window_start.strftime("%b %d, %Y")} → '
        f'{window_end.strftime("%b %d, %Y")} &nbsp;·&nbsp; '
        f'Selected: <strong style="color:#58a6ff">{st.session_state.selected_user}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

# State legend
st.markdown(
    '<div class="legend-row">'
    '<span><span class="legend-dot" style="background:#3fb950"></span>Active</span>'
    '<span><span class="legend-dot" style="background:#d29922"></span>Non-coding</span>'
    '<span><span class="legend-dot" style="background:#f85149"></span>Inactive</span>'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MAIN CHART  (all contributors combined)
# ─────────────────────────────────────────────
df_total = (
    df_all.groupby("date")["commits"]
    .sum()
    .reset_index()
    .sort_values("date")
)
dates_str   = df_total["date"].dt.strftime("%Y-%m-%d").tolist()
commit_vals = df_total["commits"].tolist()

main_chart_options = {
    "backgroundColor": "#161b22",
    "title": {
        "text": "Commits over time",
        "subtext": f"Weekly from {dates_str[0]} to {dates_str[-1]}",
        "textStyle":    {"color": "#e6edf3", "fontSize": 16},
        "subtextStyle": {"color": "#8b949e", "fontSize": 12},
        "left": "16px", "top": "12px",
    },
    "tooltip": {
        "trigger": "axis",
        "backgroundColor": "#1c2128",
        "borderColor": "#444c56",
        "textStyle": {"color": "#e6edf3"},
        "formatter": "{b}<br/>Commits: <strong>{c}</strong>",
    },
    "grid": {"left": "60px", "right": "20px", "top": "80px", "bottom": "80px"},
    "xAxis": {
        "type": "category",
        "data": dates_str,
        "axisLine":  {"lineStyle": {"color": "#30363d"}},
        "axisLabel": {"color": "#8b949e"},
        "splitLine": {"lineStyle": {"color": "#21262d"}},
    },
    "yAxis": {
        "type": "value",
        "axisLine":  {"lineStyle": {"color": "#30363d"}},
        "axisLabel": {"color": "#8b949e"},
        "splitLine": {"lineStyle": {"color": "#21262d"}},
    },
    "dataZoom": [
        {
            "type": "slider",
            "xAxisIndex": 0,
            "start": st.session_state.window_start_pct,
            "end":   st.session_state.window_end_pct,
            "height": 40, "bottom": 10,
            "fillerColor": "rgba(31,119,180,0.15)",
            "borderColor": "#30363d",
            "handleStyle": {"color": "#58a6ff"},
            "textStyle":   {"color": "#8b949e"},
            "dataBackground": {
                "lineStyle": {"color": "#58a6ff", "opacity": 0.4},
                "areaStyle": {"color": "#58a6ff", "opacity": 0.08},
            },
            "selectedDataBackground": {
                "lineStyle": {"color": "#58a6ff", "opacity": 0.7},
                "areaStyle": {"color": "#58a6ff", "opacity": 0.2},
            },
        },
        {"type": "inside", "xAxisIndex": 0},
    ],
    "series": [{
        "name": "Commits",
        "type": "bar",
        "data": commit_vals,
        "itemStyle": {"color": "#1f6feb"},
        "emphasis":  {"itemStyle": {"color": "#58a6ff"}},
        "barMaxWidth": 12,
    }],
}

# Zoom event — returns {start, end} or [start, end]
zoom_result = st_echarts(
    options=main_chart_options,
    events={"dataZoom": "function(p){var s=p.start!=null?p.start:(p.batch?p.batch[0].start:null);var e=p.end!=null?p.end:(p.batch?p.batch[0].end:null);if(s==null)return null;return {start:s,end:e};}"},
    height="380px",
    key="main_chart",
)

if zoom_result is not None:
    if isinstance(zoom_result, dict):
        new_start = float(zoom_result.get("start", st.session_state.window_start_pct))
        new_end   = float(zoom_result.get("end",   st.session_state.window_end_pct))
    else:
        new_start, new_end = float(zoom_result[0]), float(zoom_result[1])

    if (
        abs(new_start - st.session_state.window_start_pct) > 0.1
        or abs(new_end - st.session_state.window_end_pct) > 0.1
    ):
        st.session_state.window_start_pct = new_start
        st.session_state.window_end_pct   = new_end
        st.session_state.period_preset    = "Custom range"
        st.rerun()

# ─────────────────────────────────────────────
# CONTRIBUTOR CARDS
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Contributors")

# Stable key suffix that changes whenever window changes → forces mini chart re-mount
_window_key = f"{int(st.session_state.window_start_pct*10)}_{int(st.session_state.window_end_pct*10)}"

cols = st.columns(2)

for i, row in df_meta.iterrows():
    name   = row["name"]
    is_sel = (name == st.session_state.selected_user)
    col    = cols[i % 2]

    # Window-scoped stats
    _w_rows = df_meta_window[df_meta_window["name"] == name]
    if not _w_rows.empty:
        _w = _w_rows.iloc[0]
        w_commits   = int(_w["window_commits"])
        w_additions = int(_w["window_additions"])
        w_deletions = int(_w["window_deletions"])
    else:
        w_commits = w_additions = w_deletions = 0

    with col:
        selected_class = "selected" if is_sel else ""
        st.markdown(f"""
        <div class="contrib-card {selected_class}">
            <div class="card-header">
                <span style="font-size:22px">{row['avatar']}</span>
                <span class="card-username">{name}</span>
                <span class="card-rank">#{row['rank']}</span>
            </div>
            <div class="card-stats">
                {w_commits:,} commits &nbsp;
                <span class="additions">+{w_additions:,}</span> &nbsp;
                <span class="deletions">-{w_deletions:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mini chart: bars colored by activity state ───────────────
        df_person = (
            df_window[df_window["contributor"] == name]
            .sort_values("date")
        )
        person_dates  = df_person["date"].dt.strftime("%Y-%m-%d").tolist()
        person_values = df_person["commits"].tolist()
        person_states = df_person["state"].tolist()

        # Build colored background bands from consecutive state runs
        mark_areas = build_mark_areas(person_dates, person_states)

        mini_options = {
            "backgroundColor": "#161b22",
            "grid": {"left": "50px", "right": "10px", "top": "10px", "bottom": "30px"},
            "tooltip": {
                "trigger": "axis",
                "backgroundColor": "#1c2128",
                "borderColor": "#444c56",
                "textStyle": {"color": "#e6edf3"},
            },
            "xAxis": {
                "type": "category",
                "data": person_dates,
                "axisLabel": {"color": "#8b949e", "fontSize": 10},
                "axisLine":  {"lineStyle": {"color": "#30363d"}},
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#8b949e", "fontSize": 10},
                "splitLine": {"lineStyle": {"color": "#21262d"}},
            },
            "series": [{
                "type": "bar",
                "data": person_values,
                "itemStyle": {"color": row["color"]},   # contributor's own color
                "barMaxWidth": 8,
                "markArea": {
                    "silent": True,
                    "data": mark_areas,
                },
            }],
        }

        click_result = st_echarts(
            options=mini_options,
            events={"click": f"function(params) {{ return '{name}'; }}"},
            height="180px",
            key=f"card_{name}_{_window_key}",   # re-mount when window changes
        )

        if click_result is not None and click_result != st.session_state.selected_user:
            st.session_state.selected_user = click_result
            st.rerun()

        btn_label = f"✓ Selected: {name}" if is_sel else f"Select {name}"
        btn_type  = "primary" if is_sel else "secondary"
        if st.button(btn_label, key=f"btn_{name}", type=btn_type, use_container_width=True):
            st.session_state.selected_user = name
            st.rerun()

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### Current Selections")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Selected Developer", st.session_state.selected_user)
with c2:
    st.metric("Window Start", window_start.strftime("%b %d, %Y"))
with c3:
    st.metric("Window End", window_end.strftime("%b %d, %Y"))
