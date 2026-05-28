# DISTRAC — Project Health Panel: Build Plan

## Overview

This document is an agent-facing build plan for the **Project Health** panel of the DISTRAC dashboard. DISTRAC is a Streamlit-based tool for OSS developers that analyzes the impact of a contributor's inactivity or departure. Two panels are already complete:

- **Social-Technical Network** — visualizes developer-to-developer interactions
- **Knowledge Distribution** — visualizes developer-to-file interactions

The final panel, **Project Health**, is the subject of this plan.

---

## Project Goal

Give OSS project maintainers and researchers a **at-a-glance, data-driven view** of how a specific contributor's predicted absence will affect the project's activity, velocity, and health over time.

The panel answers: *"If this developer is gone for N weeks, what concretely changes?"*
this is a **before/after simulation** tied to a real contributor's historical activity patterns. the main idea is not about diversity metrics

---

## Audience

- OSS project maintainers investigating succession risk
- Researchers studying inactivity patterns in open source

---

## Key Differences from Related Work (Community Tapestry)

Community Tapestry (2025) focuses on turnover rate and diversity makeup as signals of project health. DISTRAC's goals are:

1. Detect individual developer inactivity patterns
2. Predict future inactivity windows
3. Predict and visualize the **impact** of that inactivity on the project's health

The Project Health panel serves goal 3. It may optionally include some Community Tapestry-style context metrics (turnover, diversity) as secondary signals.

---

## Inputs

All inputs are assumed to be available via Streamlit session state or passed from the parent app.

| Input | Type | Source | Description |
|---|---|---|---|
| `repo_full` | `str` | Sidebar selector | Full repo path, e.g. `"org/repo"` |
| `selected_dev` | `str` | Contributor selector | The developer being simulated |
| `break_length` | `str` or `int` | Model prediction | Predicted absence length (e.g. `"2 weeks"` or `14` days) |
| `selected_date` | `datetime.date` | Date toggle/slider | Snapshot date for the simulation |

### Derived Inputs (computed from data)

| Derived Input | Description |
|---|---|
| Per-developer weekly commit count | Rolling 4–12 week average prior to `selected_date` |
| Per-developer weekly PR count | Same window |
| Per-developer weekly issue count | Same window |
| Per-developer weekly review count | Reviews left on others' PRs |
| Repo-wide weekly commit count | Total across all contributors |
| Repo-wide weekly PR count | Total across all contributors |
| Repo-wide weekly issue close rate | Issues closed per week |

---

## Outputs

The panel renders an HTML component embedded in Streamlit via `st.components.v1.html(...)`.

| Output | Description |
|---|---|
| Activity baseline cards | Developer's avg weekly commits/PRs/issues before departure |
| Projected gap chart | Side-by-side or overlay bar/line chart: baseline vs projected during absence |
| Repo impact summary | How much of the repo's total activity this developer represents |
| Impact severity badges | Low / Medium / High / Critical labels per metric |
| (Future) Response time delta | Estimated change in median PR review time |
| (Future) Version delay estimate | Rough estimate of release schedule impact |
| (Future) Diversity/turnover snapshot | Secondary context metric |

---

## Technical Architecture

### Rendering Approach

The Project Health panel is implemented as a **self-contained HTML file** (no external framework dependencies) injected into Streamlit via:

```python
import streamlit.components.v1 as components
components.html(html_string, height=800, scrolling=True)
```

Data is passed from Python to HTML via string interpolation or `json.dumps()` embedded in a `<script>` tag as a JS variable.

### File Structure

```
distrac/
├── dashboard/
│   ├── project_health.py       # Python wrapper: loads data, calls render
│   └── project_health.html     # HTML/JS/CSS template (Jinja2 or f-string)
```

### Data Contract (Python → HTML)

```python
payload = {
    "dev_name": "Alice",
    "break_weeks": 2,
    "snapshot_date": "2024-11-01",
    "baseline": {
        "commits_per_week": 8.3,
        "prs_per_week": 2.1,
        "issues_per_week": 1.4,
        "reviews_per_week": 3.0
    },
    "repo_totals": {
        "commits_per_week": 42.0,
        "prs_per_week": 9.5,
        "issues_per_week": 6.2,
        "reviews_per_week": 14.0
    },
    "weekly_history": [
        {"week": "Oct 7",  "commits": 9, "prs": 2, "issues": 1},
        {"week": "Oct 14", "commits": 7, "prs": 3, "issues": 2},
        ...
    ]
}
```

---

## Iterative Build Plan

Each iteration builds on the previous. Start with mock data. Connect to real data only after the UI is approved.

---

### Iteration 1 — Activity Baseline + Gap Projection (START HERE)

**Goal:** Show what a developer normally contributes per week and project the gap during their absence.

**Metrics:**
- Commits per week (avg over last N weeks)
- Pull requests opened per week
- Issues opened/closed per week

**Components:**
1. Three metric cards: "Avg weekly commits", "Avg weekly PRs", "Avg weekly issues"
2. A bar chart per metric: historical weekly bars (last 8 weeks) + shaded "absence window" projection showing zero or reduced activity
3. A simple "% of repo total" label under each card

**Mock data:** Generate plausible weekly counts for a fictional dev named "Alex Chen" on a fictional repo. Show 8 weeks of history + 2 projected absence weeks.

**Implementation notes:**
- Use Chart.js (loaded from cdnjs) for the bar charts
- Use CSS variables for theming (light/dark safe)
- All data embedded in a `<script>` tag as a JS object
- No external API calls in this iteration

**Acceptance criteria:**
- Renders correctly in Streamlit via `components.html()`
- Looks polished enough to show to developers for feedback
- All charts readable on light and dark backgrounds

---

### Iteration 2 — Repo Impact Severity

**Goal:** Contextualize the developer's absence relative to the whole project.

**New components:**
1. "Repo share" donut charts: this dev's contribution vs rest of team (per metric)
2. Impact severity badge per metric: Low / Medium / High / Critical
   - Low: dev contributes < 10% of that metric
   - Medium: 10–25%
   - High: 25–40%
   - Critical: > 40%
3. A natural language summary sentence: *"Alex is responsible for 38% of all commits. Their 2-week absence is projected to reduce weekly commits by ~3.2."*

**Mock data additions:** Add `repo_totals` to the payload.

---

### Iteration 3 — Historical Trend + Inactivity Patterns

**Goal:** Show the developer's activity pattern over a longer window and flag prior inactivity events.

**New components:**
1. Extend history to 16–24 weeks
2. Mark prior inactivity windows on the chart (gray bands)
3. Show rolling average trendline
4. "Recent trend" indicator: up/down arrow with % change vs 4-week average

**Mock data additions:** Add flagged prior absence periods to the payload.

---

### Iteration 4 — Response Time & Review Impact

**Goal:** Show how the developer's absence affects code review latency.

**New metrics:**
- Median PR review time (before)
- Projected median PR review time (during absence)
- Number of PRs this dev typically reviews per week (from `reviews_per_week`)

**New components:**
1. "Review latency" metric card with before/after delta
2. A timeline-style chart showing projected review backlog growth during absence

**Data needed:** PR review timestamps (first review after PR opened).

---

### Iteration 5 — Release & Deployment Impact (Advanced)

**Goal:** Estimate whether the absence delays the next release.

**New metrics:**
- Commit velocity to default branch (proxy for release readiness)
- Whether the dev is on the "critical path" for recent milestones

**New components:**
1. "Release readiness" bar: days to next estimated release, with and without developer
2. A simple text annotation: *"At current pace, next release was projected for Dec 3. With Alex absent for 2 weeks, estimated delay: +4–6 days."*

**Data needed:** Tagged releases, recent commit velocity, milestone assignments.

---

### Iteration 6 — Diversity & Turnover Context (Optional / Secondary)

**Goal:** Add Community Tapestry-style secondary context to the panel.

**New metrics:**
- New contributor rate (last 90 days)
- Contributor churn rate (left in last 90 days)
- Gini coefficient of commit distribution (measure of concentration risk)

**New components:**
1. Small secondary stats row below the main panel
2. Sparkline trend for each metric

---

## Metrics Backlog

These are candidate metrics identified during research — not all will be implemented. Prioritize based on developer feedback.

| Metric | Category | Priority | Notes |
|---|---|---|---|
| Commits per week | Activity | P0 | Core metric for Iter 1 |
| PRs opened per week | Activity | P0 | Core metric for Iter 1 |
| Issues opened/closed per week | Activity | P0 | Core metric for Iter 1 |
| % of repo total (per metric) | Impact | P1 | Iter 2 |
| Code review frequency | Review | P1 | Iter 4 |
| Median review latency impact | Review | P1 | Iter 4 |
| Rolling activity trend (up/down) | Trend | P1 | Iter 3 |
| Prior inactivity events | Pattern | P2 | Iter 3 |
| Release delay estimate | Velocity | P2 | Iter 5 |
| Gini coefficient of commits | Diversity | P3 | Iter 6 |
| New contributor rate | Diversity | P3 | Iter 6 |
| Contributor churn rate | Diversity | P3 | Iter 6 |
| Bus factor delta | Risk | P2 | After truck factor integration |
| Issue response time delta | Responsiveness | P2 | Iter 4+ |
| Milestone/project board impact | Planning | P3 | Iter 5 |

---

## Coding Agent Instructions

When implementing any iteration:

1. **Start with mock data.** Do not connect to real data until the iteration's UI is approved. Mock data should be realistic (right order of magnitude, plausible variance).

2. **All output is a single self-contained HTML string.** No external files. CSS and JS are inline. Chart.js loads from `cdnjs.cloudflare.com`.

3. **Use CSS variables for all colors.** Never hardcode hex values for text or backgrounds — use `var(--color-text-primary)`, `var(--color-background-secondary)`, etc. so the component is light/dark safe.

4. **Pass data via a JS variable at the top of the script block.** Example:
   ```html
   <script>
   const DATA = {{ payload_json }};
   // ... rest of rendering code
   </script>
   ```

5. **Do not use `<form>` tags.** Use `onclick` and `onchange` handlers instead.

6. **Each iteration should be a complete replacement of the previous HTML file**, not a patch. Keep the file clean.

7. **After building each iteration, present the HTML inline in the chat** (using the visualizer tool) so the human can review it before moving to the next iteration.

8. **Name the output file** `project_health_v{N}.html` where N is the iteration number.

---

## Current Status

- [x] Plan written
- [ ] Iteration 1 — mock data baseline dashboard
- [ ] Iteration 2 — repo impact severity
- [ ] Iteration 3 — historical trends
- [ ] Iteration 4 — review latency
- [ ] Iteration 5 — release impact
- [ ] Iteration 6 — diversity context

**Start with Iteration 1.**
