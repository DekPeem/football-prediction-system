"""
Build web/index.html by embedding every web/leagues/*.json (one per league —
run src/export_all_leagues.py first) into web/template.html, replacing the
__ALL_LEAGUES_JSON__ placeholder with one {slug: leagueData} object. The
result is a single self-contained HTML file — open it directly in a browser,
no server needed.

Usage:
    python src/export_all_leagues.py   # (re)exports web/leagues/*.json
    python src/build_web_page.py
"""
from __future__ import annotations

import glob
import json
import os

TEMPLATE_PATH = "web/template.html"
LEAGUES_DIR = "web/leagues"
OUT_PATH = "web/index.html"

# First entry is the switcher's default. Keep in sync with export_all_leagues.py's LEAGUES.
SLUG_ORDER = [
    "premier-league", "la-liga", "serie-a", "ligue-1",
    "championship", "segunda", "serie-b", "ligue-2",
]


def main() -> None:
    files = glob.glob(f"{LEAGUES_DIR}/*.json")
    if not files:
        raise SystemExit(f"No league files in {LEAGUES_DIR}/ — run src/export_all_leagues.py first")

    all_leagues = {}
    for path in files:
        slug = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            all_leagues[slug] = json.load(f)

    # Known slugs first (switcher order), then anything else alphabetically.
    ordered = {s: all_leagues[s] for s in SLUG_ORDER if s in all_leagues}
    ordered.update({s: all_leagues[s] for s in sorted(all_leagues) if s not in ordered})

    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = f.read()

    if "__ALL_LEAGUES_JSON__" not in template:
        raise SystemExit(f"{TEMPLATE_PATH} is missing the __ALL_LEAGUES_JSON__ placeholder")

    out = template.replace("__ALL_LEAGUES_JSON__", json.dumps(ordered))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Wrote {OUT_PATH} ({len(out):,} bytes, {len(ordered)} leagues: {', '.join(ordered)})")


if __name__ == "__main__":
    main()
