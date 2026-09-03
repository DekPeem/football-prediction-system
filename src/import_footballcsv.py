"""
Convert English Premier League season files from the open-source footballcsv
archive (https://github.com/footballcsv/england, `Round,Date,Team 1,FT,Team 2`
format) into this project's matches.csv layout (`Date,HomeTeam,AwayTeam,FTHG,
FTAG,FTR`).

Team names vary release to release ("Arsenal" vs "Arsenal FC", "Manchester
Utd" vs "Manchester United FC", ...) — TEAM_ALIASES below normalizes them to
one canonical name per club so Elo/form state carries correctly across
seasons.

Usage:
    python src/import_footballcsv.py <path-to-eng.1.csv> [<more files>...] --out data/matches.csv

Example, from a local clone of footballcsv/england:
    python src/import_footballcsv.py \
        ../footballcsv/england/2010s/2017-18/eng.1.csv \
        ../footballcsv/england/2010s/2018-19/eng.1.csv \
        ../footballcsv/england/2010s/2019-20/eng.1.csv \
        ../footballcsv/england/2020s/2020-21/eng.1.csv \
        --out data/matches.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime

TEAM_ALIASES = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Cardiff City FC": "Cardiff",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Huddersfield Town AFC": "Huddersfield",
    "Hull City AFC": "Hull City",
    "Leeds United": "Leeds",
    "Leicester City FC": "Leicester",
    "Leicester City": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City": "Man City",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Manchester Utd": "Man United",
    "Middlesbrough FC": "Middlesbrough",
    "Newcastle United FC": "Newcastle",
    "Newcastle Utd": "Newcastle",
    "Norwich City FC": "Norwich",
    "Sheffield United FC": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Southampton FC": "Southampton",
    "Stoke City FC": "Stoke",
    "Sunderland AFC": "Sunderland",
    "Swansea City FC": "Swansea",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Brom": "West Brom",
    "West Bromwich Albion FC": "West Brom",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def parse_date(raw: str) -> str:
    # e.g. "Sat Sep 12 2020" or "Tue Jan 12 2021(P)" (P = postponed, later replayed)
    cleaned = re.sub(r"\([A-Za-z]\)\s*$", "", raw).strip()
    dt = datetime.strptime(cleaned, "%a %b %d %Y")
    return dt.strftime("%d/%m/%Y")


def convert_file(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = row["FT"].strip()
            if not re.match(r"^\d+[–-]\d+$", score):
                raise ValueError(f"Unexpected score format {score!r} in {path}")
            home_goals, away_goals = re.split(r"[–-]", score)
            home_goals, away_goals = int(home_goals), int(away_goals)
            ftr = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")
            rows.append({
                "Date": parse_date(row["Date"]),
                "HomeTeam": normalize_team(row["Team 1"].strip()),
                "AwayTeam": normalize_team(row["Team 2"].strip()),
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": ftr,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="footballcsv eng.N.csv season files, oldest first")
    parser.add_argument("--out", default="data/matches.csv")
    args = parser.parse_args()

    all_rows = []
    for path in args.files:
        rows = convert_file(path)
        all_rows.extend(rows)
        print(f"{path}: {len(rows)} matches")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} matches -> {args.out}")


if __name__ == "__main__":
    main()
