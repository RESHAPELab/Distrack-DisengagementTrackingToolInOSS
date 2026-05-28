#   conda activate osslab
#   cd Extractors
#   python SocialTechnicalNetwork.py
#   python SocialTechnicalNetwork.py rails/rails MichaelChirico 365
#
#   Or from the dashboard (DemoAppV2.2.py):
#   stn.main(repo_full_name, tf_devs, tables=raw_data_tables)

### IMPORTS
import os, sys, csv, logging, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

sys.path.append('../')
import Settings as cfg
try:
    import KnowledgeDistribution as _kd
    _create_dev_id = _kd.create_developer_id
except ImportError:
    _create_dev_id = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: resolve the best login string for a DataFrame row
# Priority: author_login > author_id > author_name > author_email
# Returns None if nothing usable is found.
# ---------------------------------------------------------------------------
def _resolve_login(row) -> str | None:
    for field in ("author_login", "author_id", "author_name", "author_email"):
        val = row.get(field)
        if val and str(val).strip() and str(val).strip().lower() not in ("nan", "na", "none", ""):
            return str(val).strip()
    return None

def _is_bot(login: str) -> bool:
    """
    Filter out GitHub bot accounts.
    Catches:
      - Official GitHub bots: ending with [bot]
      - Hyphenated bots: dependabot, rails-bot, stale-bot
      - Known bot names: renovate, coveralls, codecov, etc.
    """
    login_lower = login.lower()

    if login_lower.endswith("[bot]"):
        return True

    bot_patterns = ["-bot", "_bot", "bot-", "bot_"]
    if any(pattern in login_lower for pattern in bot_patterns):
        return True

    if login_lower == "bot":
        return True

    known_bots = {
        "dependabot", "renovate", "renovate-bot", "greenkeeper",
        "coveralls", "codecov", "snyk-bot", "stale",
        "imgbot", "allcontributors", "netlify", "vercel",
        "github-actions", "pull", "restyled-io"
    }
    if login_lower in known_bots:
        return True

    return False


# ---------------------------------------------------------------------------
# STEP 1 — Load & clean the four input DataFrames
# ---------------------------------------------------------------------------
def load_data(tables: dict) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Step 1: Validate and clean the four input DataFrames from the `tables` dict.

    Adds a `user` column to each DataFrame — the resolved login we will use
    as the canonical developer identifier throughout the rest of the pipeline.

    Returns (issues, issue_activity, prs_repo, prs_comments).
    """
    issues         = tables.get("issues",         pandas.DataFrame()).copy()
    issue_activity = tables.get("issue_activity", pandas.DataFrame()).copy()
    prs_repo       = tables.get("prs_repo",       pandas.DataFrame()).copy()
    prs_comments   = tables.get("prs_comments",   pandas.DataFrame()).copy()

    for name, df in [("issues", issues), ("issue_activity", issue_activity),
                     ("prs_repo", prs_repo), ("prs_comments", prs_comments)]:
        if df.empty:
            logger.warning("Table '%s' is empty — no data to process.", name)

    # Resolve canonical login for each row in every table
    for df in (issues, issue_activity, prs_repo, prs_comments):
        if not df.empty:
            df["user"] = df.apply(_resolve_login, axis=1)

    # issue_activity: the `author_*` columns store the ISSUE OPENER's identity,
    # not the commenter's. The actual performer of each activity is in `created_by`.
    # Override `user` with `created_by` wherever it is present and non-null.
    if not issue_activity.empty and "created_by" in issue_activity.columns:
        cb = issue_activity["created_by"].astype(str).str.strip()
        valid = cb.str.len() > 0
        valid &= ~cb.str.lower().isin(["nan", "na", "none", ""])
        issue_activity.loc[valid, "user"] = cb[valid]

    print(f"  issues:         {len(issues):>6} rows")
    print(f"  issue_activity: {len(issue_activity):>6} rows")
    print(f"  prs_repo:       {len(prs_repo):>6} rows")
    print(f"  prs_comments:   {len(prs_comments):>6} rows")

    return issues, issue_activity, prs_repo, prs_comments


# ---------------------------------------------------------------------------
# STEP 2 — Build per-thread participant sets (with timestamps)
# ---------------------------------------------------------------------------
def build_participation(
    issues: pandas.DataFrame,
    issue_activity: pandas.DataFrame,
    prs_repo: pandas.DataFrame,
    prs_comments: pandas.DataFrame,) -> tuple[dict, dict]:
    """
    Step 2: For each issue/PR thread, collect the set of all participants
    and the thread's earliest timestamp.

    An issue thread includes:
        - the user who opened the issue (from `issues`)
        - every user who commented or reacted (from `issue_activity`)

    A PR thread includes:
        - the user who opened the PR (from `prs_repo`)
        - every user who commented or reviewed (from `prs_comments`)

    Bot accounts are excluded.

    Returns:
        issue_threads — dict { issue_number -> {"users": set, "created_at": datetime|None} }
        pr_threads    — dict { PR_id        -> {"users": set, "created_at": datetime|None} }

    The `created_at` field is the earliest activity timestamp seen for that thread.
    It is used by build_edge_list() for temporal windowing.
    """
    # Each thread stores users + the earliest timestamp
    issue_threads: dict = defaultdict(lambda: {"users": set(), "created_at": None})
    pr_threads:    dict = defaultdict(lambda: {"users": set(), "created_at": None})

    def _parse_ts(val):
        """Parse a created_at string to a UTC-aware datetime, or return None."""
        if not val or str(val).strip().lower() in ("nan", "na", "none", ""):
            return None
        try:
            ts = pandas.to_datetime(val, utc=True, errors="coerce")
            if pandas.isna(ts):
                return None
            return ts.to_pydatetime()
        except Exception:
            return None

    def _update_ts(thread_dict, new_ts):
        """Keep the earliest timestamp seen for this thread."""
        if new_ts and (thread_dict["created_at"] is None or new_ts < thread_dict["created_at"]):
            thread_dict["created_at"] = new_ts

    # --- Issue openers ---
    if not issues.empty and "issue_number" in issues.columns:
        for _, row in issues.iterrows():
            user = row.get("user")
            thread_id = row.get("issue_number")
            if user and thread_id and not _is_bot(user):
                key = str(thread_id)
                issue_threads[key]["users"].add(user)
                _update_ts(issue_threads[key], _parse_ts(row.get("created_at")))

    # --- Issue commenters ---
    # Only count rows that are actual human comments (IssueComment item_type).
    # The issue_activity table also contains automated events like ReferencedEvent,
    # ClosedEvent, LabeledEvent, etc. — these are NOT social interactions and using
    # them would make every thread appear to have only 1 participant (the opener).
    if not issue_activity.empty and "issue_number" in issue_activity.columns:
        # Filter to actual human comments only — skip automated events
        if "item_type" in issue_activity.columns:
            comment_rows = issue_activity[issue_activity["item_type"] == "IssueComment"]
        else:
            comment_rows = issue_activity   # no item_type column: use all rows
        print(f"  issue activity comment rows (IssueComment): {len(comment_rows)}")
        for _, row in comment_rows.iterrows():
            user = row.get("user")
            thread_id = row.get("issue_number")
            if user and thread_id and not _is_bot(user):
                key = str(thread_id)
                issue_threads[key]["users"].add(user)
                _update_ts(issue_threads[key], _parse_ts(row.get("created_at")))

    # --- PR openers ---
    if not prs_repo.empty and "PR_id" in prs_repo.columns:
        for _, row in prs_repo.iterrows():
            user = row.get("user")
            thread_id = row.get("PR_id")
            if user and thread_id and not _is_bot(user):
                key = str(thread_id)
                pr_threads[key]["users"].add(user)
                _update_ts(pr_threads[key], _parse_ts(row.get("created_at")))

    # --- PR commenters / reviewers ---
    if not prs_comments.empty and "PR_id" in prs_comments.columns:
        for _, row in prs_comments.iterrows():
            user = row.get("user")
            thread_id = row.get("PR_id")
            if user and thread_id and not _is_bot(user):
                key = str(thread_id)
                pr_threads[key]["users"].add(user)
                _update_ts(pr_threads[key], _parse_ts(row.get("created_at")))

    # Filter out threads with fewer than 2 participants (no pairs possible)
    issue_threads = {k: v for k, v in issue_threads.items() if len(v["users"]) >= 2}
    pr_threads    = {k: v for k, v in pr_threads.items()    if len(v["users"]) >= 2}

    print(f"  issue threads with 2+ participants: {len(issue_threads)}")
    if len(issue_threads) == 0 and not issue_activity.empty:
        print(f"  [WARNING] 0 issue threads found despite {len(issue_activity)} issue_activity rows.")
        print(f"            This usually means issue_activity.csv only captured the issue opener's")
        print(f"            own actions — other participants' author_login/author_id are NULL.")
        print(f"            Issue-based edges will be 0; PR threads are used as the primary signal.")
    print(f"  PR threads with 2+ participants:    {len(pr_threads)}")

    return dict(issue_threads), dict(pr_threads)


# ---------------------------------------------------------------------------
# STEP 3 — Build the edge list (thread co-participation model)
# ---------------------------------------------------------------------------
def build_edge_list(
    issue_threads: dict,
    pr_threads: dict,
    since_date: datetime | None = None,
    prs_repo: pandas.DataFrame | None = None,) -> pandas.DataFrame:
    """
    Step 3: Build a weighted, undirected edge list using the thread
    co-participation model, plus a direct PR-review signal.

    Two edge weights are computed:
      - weight_issues : co-presence in the same issue thread (once per thread)
      - weight_prs    : co-presence in the same PR thread (once per thread)
      - weight_reviews: direct reviewer → PR-author interactions
                        (reviewer and PR author get +1 for each PR reviewed)
      - weight_total  : sum of all three

    Parameters
    ----------
    issue_threads, pr_threads : dicts from build_participation()
    since_date : optional datetime (UTC-aware)
        If set, only threads with created_at >= since_date are counted.
    prs_repo : optional DataFrame (from load_data) with PR_id and user columns
        Used to identify the PR author for review-edge computation.

    Returns a DataFrame with columns:
        developer_a, developer_b, weight_issues, weight_prs, weight_reviews, weight_total

    Edges are stored once per pair (developer_a < developer_b alphabetically).
    Self-loops are excluded.
    """
    # slot 0 = issues, slot 1 = prs, slot 2 = reviews
    edge_weights: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])

    for thread in issue_threads.values():
        # Apply temporal filter if requested
        if since_date and thread["created_at"] and thread["created_at"] < since_date:
            continue
        for a, b in combinations(sorted(thread["users"]), 2):
            edge_weights[(a, b)][0] += 1   # issues slot

    # Build a PR_id → author lookup from prs_repo if available
    pr_author_map: dict[str, str] = {}
    if prs_repo is not None and not prs_repo.empty and "PR_id" in prs_repo.columns:
        for _, row in prs_repo.iterrows():
            pr_id = str(row.get("PR_id", ""))
            author = row.get("user")
            if pr_id and author and not _is_bot(str(author)):
                pr_author_map[pr_id] = str(author)

    for pr_id, thread in pr_threads.items():
        if since_date and thread["created_at"] and thread["created_at"] < since_date:
            continue
        users = thread["users"]
        # co-participation weight (same as before)
        for a, b in combinations(sorted(users), 2):
            edge_weights[(a, b)][1] += 1   # prs slot
        # direct review weight: each non-author participant → PR author
        pr_author = pr_author_map.get(str(pr_id))
        if pr_author and pr_author in users:
            for participant in users:
                if participant != pr_author:
                    key = tuple(sorted([pr_author, participant]))
                    edge_weights[key][2] += 1   # reviews slot

    rows = []
    for (a, b), (w_issues, w_prs, w_reviews) in edge_weights.items():
        rows.append({
            "developer_a":    a,
            "developer_b":    b,
            "weight_issues":  w_issues,
            "weight_prs":     w_prs,
            "weight_reviews": w_reviews,
            "weight_total":   w_issues + w_prs + w_reviews,
        })

    edge_df = pandas.DataFrame(rows, columns=["developer_a", "developer_b",
                                               "weight_issues", "weight_prs",
                                               "weight_reviews", "weight_total"])
    edge_df = edge_df.sort_values("weight_total", ascending=False).reset_index(drop=True)

    if since_date:
        print(f"  (temporal window: >= {since_date.date()})")
    print(f"  unique developer pairs (edges): {len(edge_df)}")
    print(f"  unique developers (nodes):      "
          f"{len(set(edge_df['developer_a']) | set(edge_df['developer_b'])) if not edge_df.empty else 0}")

    return edge_df


# ---------------------------------------------------------------------------
# STEP 4 — Per-user metrics (foundation for departure simulation)
# ---------------------------------------------------------------------------
def calculate_metrics(edge_df: pandas.DataFrame) -> pandas.DataFrame:
    import numpy as np
    if edge_df.empty:
        return pandas.DataFrame(columns=[
            "user", "degree", "weighted_degree", "issue_degree", "pr_degree",
            "top_collaborator", "top_collaborator_weight",
        ])

    # ── Per-user metrics via groupby (replaces the iterrows + dict build) ───
    # "Double" the edge list so each edge appears once for each endpoint
    cols = ["weight_issues", "weight_prs", "weight_total"]
    left = edge_df.rename(columns={"developer_a": "user", "developer_b": "partner"})[
        ["user", "partner"] + cols
    ]
    right = edge_df.rename(columns={"developer_b": "user", "developer_a": "partner"})[
        ["user", "partner"] + cols
    ]
    doubled = pandas.concat([left, right], ignore_index=True)

    agg = doubled.groupby("user", sort=False).agg(
        degree=("partner", "count"),
        weighted_degree=("weight_total", "sum"),
        issue_degree=("weight_issues", "sum"),
        pr_degree=("weight_prs", "sum"),
    )

    # Top collaborator per user — idxmax on weight_total
    top_idx = doubled.groupby("user", sort=False)["weight_total"].idxmax()
    top = (
        doubled.loc[top_idx, ["user", "partner", "weight_total"]]
        .rename(columns={"partner": "top_collaborator",
                         "weight_total": "top_collaborator_weight"})
        .set_index("user")
    )

    metrics_df = agg.join(top).reset_index()
    metrics_df = metrics_df.sort_values("weighted_degree", ascending=False).reset_index(drop=True)

    # ── Graph-level metrics (replaces the second iterrows loop) ─────────────
    G = nx.from_pandas_edgelist(
        edge_df, source="developer_a", target="developer_b", edge_attr="weight_total"
    )

    # Betweenness — approximate if the graph is large (>500 nodes)
    n_nodes = G.number_of_nodes()
    if n_nodes > 500:
        k = min(200, n_nodes)
        bc = nx.betweenness_centrality(G, k=k, normalized=True, weight="weight_total", seed=42)
    else:
        bc = nx.betweenness_centrality(G, normalized=True, weight="weight_total")

    metrics_df["betweenness_centrality"] = (
        metrics_df["user"].map(bc).fillna(0.0).round(4)
    )

    art_points = set(nx.articulation_points(G))
    metrics_df["is_articulation_point"] = metrics_df["user"].isin(art_points)

    # ── Role classification — vectorized via np.select ──────────────────────
    deg_p75 = metrics_df["degree"].quantile(0.75)
    bc_p75  = metrics_df["betweenness_centrality"].quantile(0.75)

    hi_deg = metrics_df["degree"] >= deg_p75
    hi_bc  = metrics_df["betweenness_centrality"] >= bc_p75

    metrics_df["role"] = np.select(
        condlist=[hi_deg & hi_bc, hi_deg & ~hi_bc, ~hi_deg & hi_bc],
        choicelist=["Maintainer", "Collaborator", "Bridge"],
        default="Peripheral",
    )

    return metrics_df


# ---------------------------------------------------------------------------
# SIMULATION — Developer departure impact analysis
# ---------------------------------------------------------------------------
def simulate_departure(
    departing_developer: str,
    edge_df: pandas.DataFrame,
    metrics_df: pandas.DataFrame,
    issue_threads: dict,
    pr_threads: dict,
    lookback_days: int | None = 365,) -> dict:
    """
    Simulate the social impact of a developer leaving the project.

    This answers the proposal question: "estimate the social capital at risk
    and identify where additional community support may be needed."

    Parameters
    ----------
    departing_developer : str
        The login of the developer who is leaving.
    edge_df : DataFrame
        Full (all-time) edge list from build_edge_list().
    metrics_df : DataFrame
        Per-user metrics from calculate_metrics().
    issue_threads, pr_threads : dicts
        Thread participation data from build_participation() — needed so we
        can rebuild a temporally-filtered edge list for the simulation.
    lookback_days : int or None
        Only consider interactions from the last N days.
        None = use all available history.
        Default: 365 days (1 year).

    Returns
    -------
    dict with keys:
        departing_developer, lookback_days, window_start,

        # Pre-departure role
        degree, weighted_degree, betweenness_centrality,
        is_articulation_point, is_community_bridge,
        issue_focus_pct, pr_focus_pct, communities_spanned,

        # Post-departure network impact
        components_before, components_after, new_splits,
        isolated_users, disconnected_groups,

        # Ranked impact on direct collaborators
        most_affected  (list of dicts, sorted by lost_weight desc)
    """
    dev = departing_developer

    # --- Temporal filter ---
    window_start = None
    if lookback_days is not None:
        window_start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        sim_edge_df = build_edge_list(issue_threads, pr_threads, since_date=window_start)
    else:
        sim_edge_df = edge_df.copy()
        # Ensure weight_reviews column exists even for older CSVs
        if "weight_reviews" not in sim_edge_df.columns:
            sim_edge_df["weight_reviews"] = 0

    # If the developer has no edges in the window, return a minimal result
    dev_in_graph = (
        (sim_edge_df["developer_a"] == dev) | (sim_edge_df["developer_b"] == dev)
    ).any() if not sim_edge_df.empty else False

    if not dev_in_graph:
        return {
            "departing_developer":    dev,
            "lookback_days":          lookback_days,
            "window_start":           window_start.date().isoformat() if window_start else None,
            "error":                  f"'{dev}' has no interactions in the selected time window.",
        }

    # --- Build undirected weighted graph ---
    G = nx.Graph()
    for _, row in sim_edge_df.iterrows():
        G.add_edge(row["developer_a"], row["developer_b"], weight=int(row["weight_total"]))

    # --- Pre-departure metrics from metrics_df ---
    # Use the full (all-time) metrics_df if available; fall back to sim
    dev_row = metrics_df[metrics_df["user"] == dev]
    if dev_row.empty:
        sim_metrics = calculate_metrics(sim_edge_df)
        dev_row = sim_metrics[sim_metrics["user"] == dev]

    if not dev_row.empty:
        degree          = int(dev_row.iloc[0]["degree"])
        weighted_degree = int(dev_row.iloc[0]["weighted_degree"])
        issue_degree    = int(dev_row.iloc[0]["issue_degree"])
        pr_degree       = int(dev_row.iloc[0]["pr_degree"])
    else:
        degree = weighted_degree = issue_degree = pr_degree = 0

    issue_focus_pct = round(issue_degree / weighted_degree, 3) if weighted_degree else 0.0
    pr_focus_pct    = round(pr_degree    / weighted_degree, 3) if weighted_degree else 0.0

    # --- Betweenness centrality ---
    # Normalized so 1.0 = every shortest path passes through this node.
    bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
    betweenness = round(bc.get(dev, 0.0), 4)

    # --- Articulation point check ---
    # An articulation point is a node whose removal disconnects the graph.
    art_points = set(nx.articulation_points(G))
    is_art = dev in art_points

    # --- Community detection + bridge check ---
    communities = list(greedy_modularity_communities(G, weight="weight"))
    # Map each developer to their community index
    dev_community_idx = None
    neighbor_community_idxs = set()
    for idx, comm in enumerate(communities):
        if dev in comm:
            dev_community_idx = idx
        for neighbor in G.neighbors(dev):
            if neighbor in comm:
                neighbor_community_idxs.add(idx)

    # Remove the developer's own community from the neighbor set for bridge check
    if dev_community_idx is not None:
        neighbor_community_idxs.discard(dev_community_idx)

    communities_spanned = len(neighbor_community_idxs) + (1 if dev_community_idx is not None else 0)
    is_bridge = len(neighbor_community_idxs) >= 1  # connects own community to at least 1 other

    # --- Remove node and measure network fragmentation ---
    components_before = nx.number_connected_components(G)

    G_after = G.copy()
    G_after.remove_node(dev)

    components_after = nx.number_connected_components(G_after)
    new_splits = components_after - components_before

    # Find isolated users (left alone) and disconnected groups (small islands)
    isolated_users = []
    disconnected_groups = []
    for comp in nx.connected_components(G_after):
        comp_list = sorted(comp)
        if len(comp_list) == 1:
            isolated_users.append(comp_list[0])
        else:
            disconnected_groups.append(comp_list)

    # Sort disconnected groups smallest first so the most impactful isolation shows up
    disconnected_groups.sort(key=len)

    # --- Most affected direct collaborators ---
    # For each neighbor, compute how much weighted_degree they lose
    sim_metrics_all = calculate_metrics(sim_edge_df)
    user_wd = dict(zip(sim_metrics_all["user"], sim_metrics_all["weighted_degree"]))

    most_affected = []
    for neighbor in G.neighbors(dev):
        lost_weight = G[dev][neighbor]["weight"]
        neighbor_wd = user_wd.get(neighbor, 0)
        pct_lost = round(lost_weight / neighbor_wd, 3) if neighbor_wd else 0.0
        most_affected.append({
            "user":            neighbor,
            "lost_weight":     lost_weight,       # shared threads that disappear
            "their_total_wd":  neighbor_wd,        # their total before departure
            "pct_wd_lost":     pct_lost,           # fraction of their collaboration lost
        })

    most_affected.sort(key=lambda x: x["lost_weight"], reverse=True)

    return {
        "departing_developer":    dev,
        "lookback_days":          lookback_days,
        "window_start":           window_start.date().isoformat() if window_start else None,

        # Pre-departure role
        "degree":                 degree,
        "weighted_degree":        weighted_degree,
        "betweenness_centrality": betweenness,
        "is_articulation_point":  is_art,
        "is_community_bridge":    is_bridge,
        "issue_focus_pct":        issue_focus_pct,
        "pr_focus_pct":           pr_focus_pct,
        "communities_spanned":    communities_spanned,

        # Post-departure network impact
        "components_before":      components_before,
        "components_after":       components_after,
        "new_splits":             new_splits,
        "isolated_users":         isolated_users,
        "disconnected_groups":    disconnected_groups,

        # Ranked impact on direct collaborators (top 10)
        "most_affected":          most_affected[:10],
    }


# ---------------------------------------------------------------------------
# MAIN — 4-step orchestration
# ---------------------------------------------------------------------------
def build_daily_interaction_features(
    issues: pandas.DataFrame,
    issue_activity: pandas.DataFrame,
    prs_repo: pandas.DataFrame,
    prs_comments: pandas.DataFrame,) -> pandas.DataFrame:
    """
    Build per-developer per-day social interaction feature columns.

    Returns a DataFrame with one row per (dev, date) keyed by the same
    create_developer_id() value used in the labeled timeline, with columns:

        issue_interactions_today    — total co-participant slots in issue threads today
        issue_unique_partners_today — distinct people in today's issue threads
        issue_new_partners_today    — partners never seen before today (issue)
        issue_threads_today         — distinct issue threads touched today
        pr_interactions_today       — total co-participant slots in PR threads today
        pr_unique_partners_today    — distinct people in today's PR threads
        pr_new_partners_today       — partners never seen before today (PR)
        pr_threads_today            — distinct PR threads touched today
        total_unique_partners_today — union of issue + PR unique partners
        mention_out_today           — @mentions written by this dev today
        mention_in_today            — times this dev was @mentioned by others today
        solo_commit_day             — 1 if no social interaction today, else 0
        all_new_partners_today      — union of new issue + new PR partners (new to this dev across both channels)
        new_to_community_today      — partners whose global first appearance in the repo is today
        regulars_today              — partners this dev has interacted with ≥ 3 times historically
    """
    MENTION_RE = re.compile(r"@([\w-]+)")

    # ── helper: normalize created_at → date ─────────────────────────────────
    def _to_date(series: pandas.Series) -> pandas.Series:
        return (
            pandas.to_datetime(series, utc=True, errors="coerce")
            .dt.tz_localize(None)
            .dt.normalize()
        )

    # ── Step 1: build all-time thread → participant-set mappings ─────────────
    # Uses the `user` (login) column already added by load_data().
    thread_all: dict[str, set] = defaultdict(set)   # issue_number → logins
    pr_all:     dict[str, set] = defaultdict(set)   # PR_id        → logins

    for df_src, col, dest in [
        (issues,         "issue_number", thread_all),
        (issue_activity, "issue_number", thread_all),
        (prs_repo,       "PR_id",        pr_all),
        (prs_comments,   "PR_id",        pr_all),
    ]:
        if df_src.empty or col not in df_src.columns:
            continue
        mask = df_src["user"].notna() & df_src[col].notna()
        mask &= ~df_src["user"].map(lambda u: _is_bot(str(u)))
        for u, t in df_src.loc[mask, ["user", col]].itertuples(index=False):
            dest[str(t)].add(str(u))

    thread_all = {k: frozenset(v) for k, v in thread_all.items()}
    pr_all     = {k: frozenset(v) for k, v in pr_all.items()}

    # ── Step 2: build event records per channel ──────────────────────────────
    # Each event: dev_id (create_developer_id key), login (_resolve_login),
    #             date (normalized), thread_id
    print(f"[STN] Step 5: Building event records (issues: {len(issue_activity)}, PRs: {len(prs_comments)} rows)...")
    id_cols = ["author_id", "author_name", "author_login", "author_email"]

    def _event_df(df_src: pandas.DataFrame, thread_col: str) -> pandas.DataFrame:
        if df_src.empty or thread_col not in df_src.columns:
            return pandas.DataFrame()
        keep = ["user", "created_at", thread_col] + [c for c in id_cols if c in df_src.columns]
        sub  = df_src[keep].copy()
        sub  = sub.dropna(subset=["user"])
        sub  = sub[~sub["user"].map(lambda u: _is_bot(str(u)))]
        sub["date"] = _to_date(sub["created_at"])
        sub  = sub.dropna(subset=["date"])
        sub["thread"] = sub[thread_col].astype(str)
        # Compute dev_id — use create_developer_id if available, else login
        if _create_dev_id is not None:
            sub["dev_id"] = sub.apply(_create_dev_id, axis=1)
        else:
            sub["dev_id"] = sub["user"]
        sub = sub.dropna(subset=["dev_id"])
        return sub[["dev_id", "user", "date", "thread"]].copy()

    issue_ev = pandas.concat(
        [_event_df(issues, "issue_number"), _event_df(issue_activity, "issue_number")],
        ignore_index=True,
    )
    pr_ev = pandas.concat(
        [_event_df(prs_repo, "PR_id"), _event_df(prs_comments, "PR_id")],
        ignore_index=True,
    )

    if issue_ev.empty and pr_ev.empty:
        print("[STN] build_daily_interaction_features: no events found")
        return pandas.DataFrame()

    print(f"[STN] Step 5: Events built — {len(issue_ev)} issue, {len(pr_ev)} PR rows.  "
          f"Building dev-date aggregations...")

    # ── Step 3: dev_id ↔ login mapping ───────────────────────────────────────
    dev_to_login: dict[str, str] = {}
    for ev_df in [issue_ev, pr_ev]:
        if ev_df.empty:
            continue
        for dev_id, user in ev_df[["dev_id", "user"]].dropna().itertuples(index=False):
            dev_to_login.setdefault(str(dev_id), str(user))

    login_to_dev: dict[str, str] = {v: k for k, v in dev_to_login.items()}

    # ── Step 4: per-(dev_id, date) thread sets ───────────────────────────────
    def _group_threads(ev_df: pandas.DataFrame) -> dict[tuple, set]:
        result: dict[tuple, set] = defaultdict(set)
        if ev_df.empty:
            return result
        for dev_id, date, thread in ev_df[["dev_id", "date", "thread"]].itertuples(index=False):
            result[(dev_id, date)].add(thread)
        return result

    issue_by = _group_threads(issue_ev)
    pr_by    = _group_threads(pr_ev)

    # ── Step 5: @ mention counts ─────────────────────────────────────────────
    mention_out: dict[tuple, int] = defaultdict(int)  # (dev_id, date) → count
    mention_in:  dict[tuple, int] = defaultdict(int)  # (login.lower(), date) → count

    if not issue_activity.empty and "body" in issue_activity.columns:
        body_ev = _event_df(issue_activity, "issue_number")
        # Merge body back in
        body_col = issue_activity["body"].fillna("").astype(str).reset_index(drop=True)
        if len(body_ev) == len(body_col):
            body_ev = body_ev.copy()
            body_ev["body"] = body_col.values
        else:
            # Safer: recompute with body column included
            sub2 = issue_activity[["user", "created_at", "issue_number", "body"] +
                                   [c for c in id_cols if c in issue_activity.columns]].copy()
            sub2 = sub2.dropna(subset=["user"])
            sub2 = sub2[~sub2["user"].map(lambda u: _is_bot(str(u)))]
            sub2["date"] = _to_date(sub2["created_at"])
            sub2 = sub2.dropna(subset=["date"])
            if _create_dev_id is not None:
                sub2["dev_id"] = sub2.apply(_create_dev_id, axis=1)
            else:
                sub2["dev_id"] = sub2["user"]
            body_ev = sub2[["dev_id", "user", "date", "body"]].dropna(subset=["dev_id"])

        for dev_id, date, body in body_ev[["dev_id", "date", "body"]].itertuples(index=False):
            body = str(body) if body else ""
            if not body:
                continue
            mentioned = MENTION_RE.findall(body)
            if not mentioned:
                continue
            mention_out[(dev_id, date)] += len(mentioned)
            for m_login in mentioned:
                if not _is_bot(m_login):
                    mention_in[(m_login.lower(), date)] += 1

    # ── Step 6: aggregate rows ───────────────────────────────────────────────
    all_keys = set(issue_by.keys()) | set(pr_by.keys())
    all_devs = {k[0] for k in all_keys}

    # Pre-compute each login's first appearance date across ALL event types.
    # Used to flag partners who are completely new to the repo community today.
    _fa_frames = []
    for ev_df in [issue_ev, pr_ev]:
        if not ev_df.empty:
            _fa_frames.append(ev_df[["user", "date"]].dropna())
    if _fa_frames:
        _fa_combined = pandas.concat(_fa_frames, ignore_index=True)
        first_appearance: dict = _fa_combined.groupby("user")["date"].min().to_dict()
    else:
        first_appearance: dict = {}

    # Pre-group all_keys by dev_id once — avoids O(n²) scan inside the loop.
    keys_by_dev: dict = defaultdict(set)
    for dev_id, date in all_keys:
        keys_by_dev[dev_id].add(date)

    total_devs = len(all_devs)
    print(f"[STN] Step 5: Aggregating daily features for {total_devs} developers...")

    # Worker for one developer — all inputs are read-only dicts captured by closure.
    def _process_dev(dev_id: str) -> list:
        login = dev_to_login.get(dev_id, "")
        login_lower = login.lower() if login else ""
        cumulative_issue_partners: set = set()
        cumulative_pr_partners:    set = set()
        cumulative_partner_counts: dict = defaultdict(int)
        dev_rows = []

        for date in sorted(keys_by_dev[dev_id]):
            # --- issue channel ---
            i_threads = issue_by.get((dev_id, date), set())
            i_partners: set = set()
            i_total = 0
            for t in i_threads:
                others = {p for p in thread_all.get(t, frozenset()) if p != login}
                i_partners.update(others)
                i_total += len(others)
            i_new = i_partners - cumulative_issue_partners
            cumulative_issue_partners.update(i_partners)

            # --- PR channel ---
            p_threads = pr_by.get((dev_id, date), set())
            p_partners: set = set()
            p_total = 0
            for t in p_threads:
                others = {p for p in pr_all.get(t, frozenset()) if p != login}
                p_partners.update(others)
                p_total += len(others)
            p_new = p_partners - cumulative_pr_partners
            cumulative_pr_partners.update(p_partners)

            total_partners = i_partners | p_partners
            m_out = mention_out.get((dev_id, date), 0)
            m_in  = mention_in.get((login_lower, date), 0) if login_lower else 0

            all_new_today = i_new | p_new
            regulars_today = sum(
                1 for p in total_partners if cumulative_partner_counts.get(p, 0) >= 3
            )
            new_to_community_today = sum(
                1 for p in total_partners if first_appearance.get(p) == date
            )
            for p in total_partners:
                cumulative_partner_counts[p] += 1

            dev_rows.append({
                "dev":                         dev_id,
                "date":                        pandas.Timestamp(date),
                "issue_interactions_today":    i_total,
                "issue_unique_partners_today": len(i_partners),
                "issue_new_partners_today":    len(i_new),
                "issue_threads_today":         len(i_threads),
                "pr_interactions_today":       p_total,
                "pr_unique_partners_today":    len(p_partners),
                "pr_new_partners_today":       len(p_new),
                "pr_threads_today":            len(p_threads),
                "total_unique_partners_today": len(total_partners),
                "mention_out_today":           m_out,
                "mention_in_today":            m_in,
                "solo_commit_day":             1 if len(total_partners) == 0 else 0,
                "all_new_partners_today":      len(all_new_today),
                "new_to_community_today":      new_to_community_today,
                "regulars_today":              regulars_today,
            })
        return dev_rows

    workers = min(8, os.cpu_count() or 4)
    rows = []
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_dev = {ex.submit(_process_dev, dev_id): dev_id for dev_id in all_devs}
        for future in as_completed(future_to_dev):
            rows.extend(future.result())
            completed += 1
            if completed % 100 == 0 or completed == total_devs:
                print(f"[STN]   {completed}/{total_devs} developers processed...", flush=True)

    if not rows:
        return pandas.DataFrame()

    daily_df = pandas.DataFrame(rows).sort_values(["dev", "date"]).reset_index(drop=True)
    print(f"[STN] Daily interaction features: {len(daily_df)} rows, "
          f"{daily_df['dev'].nunique()} developers")
    return daily_df


def main(
    repo_full_name: str = None,
    tf_devs=None,
    tables: dict = None,
    departing_developer: str | None = None,
    lookback_days: int | None = 365,):
    """
    Build the Social-Technical interaction network for a repository.

    Parameters
    ----------
    repo_full_name : str
        "org/repo" — used to locate the output folder.
    tf_devs : reserved for future use (truck-factor developers list).
    tables : dict
        DataFrames loaded by the dashboard.  Keys:
        "issues", "issue_activity", "prs_repo", "prs_comments"
    departing_developer : str or None
        If set, run the departure simulation for this developer and return
        the result as a 4th return value.
    lookback_days : int or None
        Temporal window for the departure simulation (default: 1 year).

    Returns
    -------
    edge_df, metrics_df, edge_df, simulation_result
        The 3rd value duplicates edge_df for backward compatibility with
        the dashboard's existing 3-value unpack.
        simulation_result is None if departing_developer was not specified.
    """
    if tables is None:
        tables = {}

    # --- Output folder setup ---
    org, repo = repo_full_name.split('/')
    out_folder = Path(cfg.main_folder, org, repo, cfg.social_technical_metrics_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    edge_file    = out_folder / cfg.social_technical_edge_list_file
    metrics_file = out_folder / cfg.social_technical_metrics_file

    # -----------------------------------------------------------------------
    # Step 1: Load & clean data
    # -----------------------------------------------------------------------
    print("\n[STN] Step 1: Loading data...")
    issues, issue_activity, prs_repo, prs_comments = load_data(tables)

    # -----------------------------------------------------------------------
    # Step 2: Build per-thread participant sets (with timestamps)
    # -----------------------------------------------------------------------
    print("[STN] Step 2: Building thread participation sets...")
    issue_threads, pr_threads = build_participation(issues, issue_activity, prs_repo, prs_comments)

    # -----------------------------------------------------------------------
    # Step 3: Build the full (all-time) edge list
    # -----------------------------------------------------------------------
    print("[STN] Step 3: Building edge list (thread co-participation model + PR reviews)...")
    edge_df = build_edge_list(issue_threads, pr_threads, prs_repo=prs_repo)

    # -----------------------------------------------------------------------
    # Step 4: Calculate per-user metrics
    # -----------------------------------------------------------------------
    print("[STN] Step 4: Calculating per-user metrics...")
    metrics_df = calculate_metrics(edge_df)

    # -----------------------------------------------------------------------
    # Step 5: Build daily per-user interaction features
    # -----------------------------------------------------------------------
    print("[STN] Step 5: Building daily interaction features...")
    daily_df = build_daily_interaction_features(issues, issue_activity, prs_repo, prs_comments)

    # --- Save outputs ---
    edge_df.to_csv(edge_file, index=False)
    metrics_df.to_csv(metrics_file, index=False)
    print(f"[STN] Edge list saved    -> {edge_file}")
    print(f"[STN] User metrics saved -> {metrics_file}")

    if not daily_df.empty:
        daily_file = out_folder / cfg.social_technical_daily_interactions_file
        daily_df.to_csv(daily_file, index=False)
        print(f"[STN] Daily features saved -> {daily_file}")

    # --- Optional: departure simulation ---
    simulation_result = None
    if departing_developer:
        print(f"\n[STN] Running departure simulation for: {departing_developer}")
        simulation_result = simulate_departure(
            departing_developer,
            edge_df,
            metrics_df,
            issue_threads,
            pr_threads,
            lookback_days=lookback_days,
        )

    # 3rd return is now daily_df (was duplicate edge_df)
    return edge_df, metrics_df, daily_df, simulation_result


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

# ---------------------------------------------------------------------------
# Standalone test runner
# Run from Extractors/:
#   python SocialTechnicalNetwork.py
#   python SocialTechnicalNetwork.py Rdatatable/data.table MichaelChirico 365
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    repo_full_name     = sys.argv[1] if len(sys.argv) > 1 else "Rdatatable/data.table"
    departing_dev      = sys.argv[2] if len(sys.argv) > 2 else None
    lookback           = int(sys.argv[3]) if len(sys.argv) > 3 else 365

    org, repo = repo_full_name.split('/')
    base = Path(__file__).resolve().parents[1] / "Organizations" / org / repo

    def _read(filename):
        p = base / filename
        if not p.exists():
            print(f"  [WARN] {p} not found — using empty DataFrame")
            return pandas.DataFrame()
        df = pandas.read_csv(p)
        print(f"  Loaded {filename}: {len(df)} rows")
        return df

    print(f"\n=== Social-Technical Network: {repo_full_name} ===")
    tables = {
        "issues":         _read("issues.csv"),
        "issue_activity": _read("issue_activity.csv"),
        "prs_repo":       _read("prs_repo.csv"),
        "prs_comments":   _read("prs_comments.csv"),
    }

    # If no specific developer given, pick the most connected one automatically
    if departing_dev is None:
        # Run without simulation first to find the top developer
        edge_df, metrics_df, _, _ = main(
            repo_full_name=repo_full_name, tables=tables, lookback_days=lookback
        )
        if not metrics_df.empty:
            departing_dev = metrics_df.iloc[0]["user"]
            print(f"\n  Auto-selected top developer for simulation: {departing_dev}")
    else:
        edge_df, metrics_df, _, _ = main(
            repo_full_name=repo_full_name, tables=tables, lookback_days=lookback
        )

    print("\n--- Edge list (top 10 by weight) ---")
    print(edge_df.head(10).to_string(index=False))

    print("\n--- User metrics (top 10 by weighted degree) ---")
    print(metrics_df.head(10).to_string(index=False))

    # Now run the simulation
    if departing_dev and not edge_df.empty:
        print(f"\n=== Departure Simulation: {departing_dev} (lookback={lookback} days) ===")

        # Re-load threads from the already-run main() — re-run participation for simulation
        issues         = tables["issues"].copy()
        issue_activity = tables["issue_activity"].copy()
        prs_repo       = tables["prs_repo"].copy()
        prs_comments   = tables["prs_comments"].copy()
        for df in (issues, issue_activity, prs_repo, prs_comments):
            if not df.empty:
                df["user"] = df.apply(_resolve_login, axis=1)

        from collections import defaultdict
        i_threads, p_threads = build_participation(issues, issue_activity, prs_repo, prs_comments)

        result = simulate_departure(
            departing_dev, edge_df, metrics_df,
            i_threads, p_threads,
            lookback_days=lookback,
        )

        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Role in network:")
            print(f"    Degree (collaborators):   {result['degree']}")
            print(f"    Weighted degree:           {result['weighted_degree']}")
            print(f"    Betweenness centrality:    {result['betweenness_centrality']}")
            print(f"    Articulation point:        {result['is_articulation_point']}")
            print(f"    Community bridge:          {result['is_community_bridge']}")
            print(f"    Communities spanned:       {result['communities_spanned']}")
            print(f"    Issue focus:               {result['issue_focus_pct']*100:.0f}%  |  PR focus: {result['pr_focus_pct']*100:.0f}%")
            print(f"\n  Network fragmentation after departure:")
            print(f"    Components before:  {result['components_before']}")
            print(f"    Components after:   {result['components_after']}")
            print(f"    New splits:         {result['new_splits']}")
            print(f"    Isolated users:     {result['isolated_users'][:5]}")
            if result['disconnected_groups']:
                print(f"    Disconnected groups (smallest): {result['disconnected_groups'][:3]}")
            print(f"\n  Top 10 most affected collaborators:")
            for i, row in enumerate(result['most_affected'], 1):
                print(f"    {i:2}. {row['user']:<20}  lost {row['lost_weight']:>4} threads  "
                      f"({row['pct_wd_lost']*100:.0f}% of their collaboration)")
