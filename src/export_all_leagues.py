"""
Run export_web_demo_data.py for every league the web demo's switcher covers,
writing web/leagues/<slug>.json for each. build_web_page.py then combines
them into the page.

Cup competitions aren't included here — see README.md's "Cup competitions"
section for why (three of the five score at or below their own baseline;
showing them next to leagues that clearly beat baseline would overstate how
much to trust them).

Usage:
    python src/export_all_leagues.py
"""
from __future__ import annotations

import subprocess
import sys

LEAGUES = [
    # (slug, display label, data dir or None for the Premier League's root-level files)
    ("premier-league", "Premier League", None),
    ("la-liga", "La Liga", "la-liga"),
    ("serie-a", "Serie A", "serie-a"),
    ("ligue-1", "Ligue 1", "ligue-1"),
    ("championship", "Championship", "championship"),
    ("segunda", "Segunda División", "segunda"),
    ("serie-b", "Serie B", "serie-b"),
    ("ligue-2", "Ligue 2", "ligue-2"),
]


def main() -> None:
    for slug, label, subdir in LEAGUES:
        data = f"data/{subdir}/matches.csv" if subdir else "data/matches.csv"
        fixtures = f"data/{subdir}/fixtures.csv" if subdir else "data/fixtures.csv"
        model = f"models/{subdir}/model.joblib" if subdir else "models/model.joblib"
        out = f"web/leagues/{slug}.json"
        result = subprocess.run([
            sys.executable, "src/export_web_demo_data.py",
            "--data", data, "--fixtures", fixtures, "--model", model,
            "--out", out, "--label", label,
        ])
        if result.returncode != 0:
            raise SystemExit(f"export failed for {label} ({slug})")


if __name__ == "__main__":
    main()
