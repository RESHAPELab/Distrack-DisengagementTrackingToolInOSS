"""
reset_issue_collection.py
=========================
Resets the issue data collection for all repos so that CommitExtractorV3.py
will re-collect issue timelines with the corrected actor/author attribution.

What it does:
  1. Deletes issue_activity.csv (corrupted — had issue opener's login on every row)
  2. Deletes issues.csv (must delete because the extractor APPENDS; re-collection
     would create duplicate issue rows otherwise)
  3. Resets the "issues_with_timeline" stream in data_cursor.json to uncollected
     (sets after=null, processed=0, total=null, complete=false)
     -- leaves "commits" and "prs_with_comments" streams untouched --

Usage:
  python reset_issue_collection.py           # dry run, prints what would happen
  python reset_issue_collection.py --execute  # actually deletes and resets
"""

import json
import sys
from pathlib import Path

DRY_RUN = "--execute" not in sys.argv
BASE = Path(__file__).resolve().parent / "Organizations"

STREAM_RESET = {
    "after": None,
    "processed": 0,
    "total": None,
    "complete": False,
}

deleted_files = []
reset_cursors = []
skipped = []

for cursor_path in sorted(BASE.rglob("data_cursor.json")):
    repo_dir = cursor_path.parent

    # ── 1 & 2: delete issue_activity.csv and issues.csv ──────────────────────
    for fname in ("issue_activity.csv", "issues.csv"):
        fp = repo_dir / fname
        if fp.exists():
            deleted_files.append(fp)
            if not DRY_RUN:
                fp.unlink()

    # ── 3: reset issues_with_timeline stream in cursor ───────────────────────
    try:
        data = json.loads(cursor_path.read_text(encoding="utf-8"))
    except Exception as e:
        skipped.append(f"{cursor_path}: {e}")
        continue

    streams = data.get("streams", {})
    if "issues_with_timeline" not in streams:
        skipped.append(f"{cursor_path}: no issues_with_timeline stream")
        continue

    already_reset = (
        streams["issues_with_timeline"].get("complete") == False
        and streams["issues_with_timeline"].get("after") is None
        and streams["issues_with_timeline"].get("processed") == 0
    )
    if already_reset:
        skipped.append(f"{cursor_path}: already uncollected, skipping")
        continue

    streams["issues_with_timeline"] = STREAM_RESET
    reset_cursors.append(cursor_path)
    if not DRY_RUN:
        cursor_path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

# ── Report ────────────────────────────────────────────────────────────────────
mode = "DRY RUN" if DRY_RUN else "EXECUTED"
print(f"\n{'='*60}")
print(f"  reset_issue_collection.py — {mode}")
print(f"{'='*60}")

print(f"\nFiles that will be deleted ({len(deleted_files)}):")
for f in deleted_files:
    rel = f.relative_to(BASE)
    print(f"  DELETE  {rel}")

print(f"\nCursors that will be reset ({len(reset_cursors)}):")
for c in reset_cursors:
    rel = c.relative_to(BASE)
    print(f"  RESET   {rel}")

if skipped:
    print(f"\nSkipped ({len(skipped)}):")
    for s in skipped:
        print(f"  SKIP    {s}")

print(f"\nSummary: {len(deleted_files)} files deleted, {len(reset_cursors)} cursors reset")

if DRY_RUN:
    print("\n*** DRY RUN — nothing changed. Run with --execute to apply. ***\n")
else:
    print("\n*** Done. Re-run CommitExtractorV3.py to re-collect issue data. ***")
    print("*** After re-collection, re-run the pipeline (DemoAppV2.2.py) to  ***")
    print("*** regenerate test_df.csv, activity_weekly.parquet, and STN files. ***\n")
