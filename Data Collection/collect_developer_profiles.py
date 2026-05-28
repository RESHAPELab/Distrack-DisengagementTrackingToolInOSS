"""
collect_developer_profiles.py
──────────────────────────────
Collects GitHub user profile data for all developers found in a repo's
commit history and saves it to:

    Organizations/{org}/{repo}/Developers/
        profiles_summary.csv        ← one row per developer
        {login}/
            profile.json            ← REST response + contributions calendar
            avatar.png              ← downloaded profile picture

Usage (standalone):
    python Extractors/collect_developer_profiles.py org/repo

Usage (from DemoApp or other Python code):
    from collect_developer_profiles import collect_developer_profiles
    collect_developer_profiles("angular/angular", token="ghp_...")
"""

import json
import time
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Settings as cfg
import Utilities as util

ORG_BASE = Path(cfg.main_folder)

log = logging.getLogger(__name__)

# ── GitHub API constants ──────────────────────────────────────────────────────

_REST_BASE   = "https://api.github.com"
_GRAPHQL_URL = "https://api.github.com/graphql"

_CONTRIBUTIONS_QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      contributionCalendar {
        totalContributions
        weeks {
          firstDay
          contributionDays {
            date
            contributionCount
          }
        }
      }
    }
  }
}
"""

# Fields we keep from the REST /users/{login} response
_REST_KEEP = [
    "login", "name", "bio", "company", "location", "email",
    "avatar_url", "html_url", "public_repos", "public_gists",
    "followers", "following", "created_at", "updated_at", "type",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _fetch_rest_profile(login: str, token: str) -> dict | None:
    """Call GET /users/{login}. Returns trimmed dict or None on error."""
    url = f"{_REST_BASE}/users/{login}"
    try:
        resp = requests.get(url, headers=_auth_headers(token), timeout=15)
        if resp.status_code == 404:
            log.warning("[profiles] 404 for user '%s' — account may be deleted", login)
            return None
        if resp.status_code == 403:
            log.warning("[profiles] 403 for user '%s' — rate limited", login)
            return None
        resp.raise_for_status()
        data = resp.json()
        return {k: data.get(k) for k in _REST_KEEP}
    except Exception as exc:
        log.warning("[profiles] REST fetch failed for '%s': %s", login, exc)
        return None


def _fetch_contributions(login: str, token: str) -> dict | None:
    """
    Fetch the contribution calendar for the last 365 days via GraphQL.
    Returns dict with keys 'total' and 'weeks', or None on error.
    """
    now = datetime.now(timezone.utc)
    year_ago = now - timedelta(days=365)
    variables = {
        "login": login,
        "from":  year_ago.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to":    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    try:
        resp = requests.post(
            _GRAPHQL_URL,
            headers=_auth_headers(token),
            json={"query": _CONTRIBUTIONS_QUERY, "variables": variables},
            timeout=20,
        )
        if resp.status_code == 403:
            log.warning("[profiles] GraphQL 403 for '%s' — rate limited", login)
            return None
        resp.raise_for_status()
        body = resp.json()
        if "errors" in body:
            log.warning("[profiles] GraphQL errors for '%s': %s", login, body["errors"])
            return None
        cal = (
            body.get("data", {})
                .get("user", {})
                .get("contributionsCollection", {})
                .get("contributionCalendar", {})
        )
        if not cal:
            return None

        weeks = []
        for week in cal.get("weeks", []):
            weeks.append({
                "week": week["firstDay"],
                "days": [d["contributionCount"] for d in week.get("contributionDays", [])],
            })
        return {"total": cal.get("totalContributions", 0), "weeks": weeks}
    except Exception as exc:
        log.warning("[profiles] GraphQL fetch failed for '%s': %s", login, exc)
        return None


def _download_avatar(avatar_url: str, dest: Path, token: str) -> bool:
    """Download avatar image to dest. Returns True on success."""
    try:
        resp = requests.get(avatar_url, headers=_auth_headers(token), timeout=20)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        log.warning("[profiles] Avatar download failed (%s): %s", avatar_url, exc)
        return False


def _tf_logins(repo_full_name: str) -> list[str]:
    """
    Return the GitHub logins for the Truck Factor developers only.

    Reads:
      KnowledgeDistribution/truck_factor.json  → list of dev_ids
      Results/dev_names.csv                    → dev_id → login mapping
    """
    tf_path    = ORG_BASE / repo_full_name / "KnowledgeDistribution" / cfg.truck_factor_file
    names_path = ORG_BASE / repo_full_name / "Results" / "dev_names.csv"

    if not tf_path.exists():
        log.warning("[profiles] truck_factor.json not found at %s — run KD step first.", tf_path)
        return []
    if not names_path.exists():
        log.warning("[profiles] dev_names.csv not found at %s — run pipeline first.", names_path)
        return []

    try:
        with open(tf_path) as f:
            tf_data = json.load(f)
        tf_dev_ids = set(tf_data.get("tf_list", []))

        names_df = pd.read_csv(names_path, dtype=str).fillna("")
        # Keep only TF developers, then pull their login
        tf_rows  = names_df[names_df["dev_id"].isin(tf_dev_ids)]
        logins   = (
            tf_rows["login"]
            .str.strip()
            .loc[lambda s: s != ""]
            .loc[lambda s: ~s.str.contains(r"\[bot\]", case=False, na=False)]
            .unique()
            .tolist()
        )
        log.info("[profiles] %d TF developers → %d logins to collect for %s",
                 len(tf_dev_ids), len(logins), repo_full_name)
        return sorted(logins)
    except Exception as exc:
        log.warning("[profiles] Could not load TF logins for %s: %s", repo_full_name, exc)
        return []


# ── Main public function ──────────────────────────────────────────────────────

def collect_developer_profiles(
    repo_full_name: str,
    token: str,
    logins: list[str] | None = None,
    overwrite: bool = False,
    sleep_between: float = 0.75,
    progress_callback=None,) -> dict:
    """
    Collect GitHub profile data for all developers in a repo.

    Parameters
    ----------
    repo_full_name : str
        Repository in "org/repo" format.
    token : str
        GitHub Personal Access Token.
    logins : list[str] | None
        Explicit list of logins to collect. If None, loads logins from
        KnowledgeDistribution/truck_factor.json joined with Results/dev_names.csv
        (Truck Factor developers only — much shorter list than all committers).
    overwrite : bool
        If False (default), skip developers whose profile.json already exists.
    sleep_between : float
        Seconds to sleep between API calls (avoids secondary rate limits).
    progress_callback : callable | None
        Optional callback(current, total, login) for progress reporting.

    Returns
    -------
    dict with keys:
        collected   — list of logins successfully collected
        skipped     — list of logins skipped (cache hit)
        failed      — list of logins that errored
    """
    if logins is None:
        logins = _tf_logins(repo_full_name)
    if not logins:
        log.warning("[profiles] No logins found for %s — nothing to collect.", repo_full_name)
        return {"collected": [], "skipped": [], "failed": []}

    dev_root = ORG_BASE / repo_full_name / "Developers"
    dev_root.mkdir(parents=True, exist_ok=True)

    collected, skipped, failed = [], [], []
    total = len(logins)

    for idx, login in enumerate(logins, 1):
        print("login", login)
        if progress_callback:
            progress_callback(idx, total, login)

        login_dir    = dev_root / login
        profile_path = login_dir / "profile.json"

        # Cache check
        if profile_path.exists() and not overwrite:
            skipped.append(login)
            continue

        login_dir.mkdir(exist_ok=True)

        # ── REST profile ──────────────────────────────────────────────────
        profile = _fetch_rest_profile(login, token)
        if profile is None:
            failed.append(login)
            time.sleep(sleep_between)
            continue

        # ── Contribution calendar ─────────────────────────────────────────
        contributions = _fetch_contributions(login, token)
        profile["contributions"] = contributions  # None if unavailable

        # ── Avatar ───────────────────────────────────────────────────────
        avatar_url = profile.get("avatar_url")
        avatar_file = None
        if avatar_url:
            avatar_dest = login_dir / "avatar.png"
            if _download_avatar(avatar_url, avatar_dest, token):
                avatar_file = str(avatar_dest.relative_to(dev_root))

        profile["avatar_file"] = avatar_file

        # ── Save profile.json ─────────────────────────────────────────────
        profile_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        collected.append(login)
        log.info("[profiles] ✓ %s (%d/%d)", login, idx, total)
        time.sleep(sleep_between)

    # ── Write/refresh profiles_summary.csv ───────────────────────────────────
    _write_summary(dev_root)

    result = {"collected": collected, "skipped": skipped, "failed": failed}
    log.info(
        "[profiles] Done — collected:%d  skipped:%d  failed:%d",
        len(collected), len(skipped), len(failed),
    )
    return result


def _write_summary(dev_root: Path) -> None:
    """Rebuild profiles_summary.csv from all profile.json files in dev_root."""
    rows = []
    for profile_path in sorted(dev_root.glob("*/profile.json")):
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            rows.append({
                "login":        data.get("login", ""),
                "name":         data.get("name", ""),
                "bio":          data.get("bio", ""),
                "company":      data.get("company", ""),
                "location":     data.get("location", ""),
                "email":        data.get("email", ""),
                "public_repos": data.get("public_repos", ""),
                "followers":    data.get("followers", ""),
                "following":    data.get("following", ""),
                "created_at":   data.get("created_at", ""),
                "type":         data.get("type", ""),
                "html_url":     data.get("html_url", ""),
                "avatar_file":  data.get("avatar_file", ""),
                "contributions_total": (data.get("contributions") or {}).get("total", ""),
            })
        except Exception:
            pass
    if rows:
        summary_path = dev_root / "profiles_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        log.info("[profiles] Summary written: %s (%d rows)", summary_path, len(rows))


# ── Convenience loader (used by Dashboard) ───────────────────────────────────

def load_developer_profile(repo_full_name: str, login: str) -> dict | None:
    """Load a single developer's profile.json. Returns None if not found."""
    path = ORG_BASE / repo_full_name / "Developers" / login / "profile.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_profiles_summary(repo_full_name: str) -> pd.DataFrame:
    """Load profiles_summary.csv for a repo. Returns empty DataFrame if missing."""
    path = ORG_BASE / repo_full_name / "Developers" / "profiles_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def get_avatar_path(repo_full_name: str, login: str) -> Path | None:
    """Return the Path to avatar.png for a login, or None if not downloaded."""
    p = ORG_BASE / repo_full_name / "Developers" / login / "avatar.png"
    return p if p.exists() else None


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if len(sys.argv) < 2:
        print("Usage: python collect_developer_profiles.py org/repo [--overwrite]")
        sys.exit(1)

    repo_arg     = sys.argv[1]
    do_overwrite = "--overwrite" in sys.argv

    try:
        token_str = util.getSpisificToken(0)
    except Exception as e:
        print(f"Could not load token: {e}")
        sys.exit(1)

    def _progress(current, total, login):
        print(f"  [{current}/{total}] {login}")

    result = collect_developer_profiles(
        repo_full_name=repo_arg,
        token=token_str,
        overwrite=do_overwrite,
        progress_callback=_progress,
    )
    print(f"\nDone.  collected={len(result['collected'])}  "
          f"skipped={len(result['skipped'])}  failed={len(result['failed'])}")
    if result["failed"]:
        print("Failed logins:", result["failed"])
