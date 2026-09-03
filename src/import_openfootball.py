"""
Convert English top-flight season files from the open-source openfootball
archive (https://github.com/openfootball/england) into this project's CSV
layout. Two things make this format trickier than footballcsv/england:

1. The per-line layout changed across seasons — some use
   "Team A   2-1 (1-0)   Team B", others "Team A   v   Team B   2-1 (1-0)".
   parse_season_file() tries both.
2. A season file mixes **played** matches (have a score) and **future
   fixtures** (no score yet, e.g. a season in progress). Both are parsed;
   train.py only ever sees the played ones (features.py drops rows with no
   FTR), while the unplayed ones become data/fixtures.csv — the input to
   predict_fixtures.py for "what's the model's call on next weekend's games".

Usage:
    python src/import_openfootball.py \
        ../openfootball/england/2022-23/1-premierleague.txt \
        ../openfootball/england/2023-24/1-premierleague.txt \
        ../openfootball/england/2024-25/1-premierleague.txt \
        ../openfootball/england/2025-26/1-premierleague-full.txt \
        ../openfootball/england/2026-27/1-premierleague.txt \
        --out data/matches.csv --fixtures-out data/fixtures.csv
"""
from __future__ import annotations

import argparse
import csv
import re

MONTHS = {m: i + 1 for i, m in enumerate([
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
])}

DATE_HEADER_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
SEASON_START_RE = re.compile(r"^#\s*Dates?\s+\S+\s+([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{4})\s*-")
SCORE_RE = r"(\d+)-(\d+)(?:\s*\([\d]+-[\d]+\))?"
# "Team A ... v ... Team B ... [score]" (score optional -> future fixture)
V_FORMAT_RE = re.compile(rf"^(?:\d{{2}}:\d{{2}}\s+)?(.+?)\s+v\s+(.+?)(?:\s{{2,}}{SCORE_RE})?\s*$")
# "Team A ... [score] ... Team B" (score always present in this layout, so it's how we tell them apart)
SCORE_MID_FORMAT_RE = re.compile(rf"^(?:\d{{2}}:\d{{2}}\s+)?(.+?)\s{{2,}}{SCORE_RE}\s+(.+?)\s*$")

TEAM_ALIASES = {
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Arsenal FC": "Arsenal", "Arsenal": "Arsenal",
    "Aston Villa FC": "Aston Villa", "Aston Villa": "Aston Villa",
    "Brentford FC": "Brentford", "Brentford": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Brighton & Hove Albion": "Brighton", "Brighton": "Brighton",
    "Burnley FC": "Burnley", "Burnley": "Burnley",
    "Cardiff City FC": "Cardiff", "Cardiff City": "Cardiff",
    "Chelsea FC": "Chelsea", "Chelsea": "Chelsea",
    "Coventry City FC": "Coventry City", "Coventry City": "Coventry City",
    "Crystal Palace FC": "Crystal Palace", "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton", "Everton": "Everton",
    "Fulham FC": "Fulham", "Fulham": "Fulham",
    "Huddersfield Town AFC": "Huddersfield", "Huddersfield Town": "Huddersfield",
    "Hull City AFC": "Hull City", "Hull City": "Hull City",
    "Ipswich Town FC": "Ipswich Town", "Ipswich Town": "Ipswich Town",
    "Leeds United FC": "Leeds", "Leeds United": "Leeds",
    "Leicester City FC": "Leicester", "Leicester City": "Leicester",
    "Liverpool FC": "Liverpool", "Liverpool": "Liverpool",
    "Luton Town FC": "Luton Town", "Luton Town": "Luton Town",
    "Manchester City FC": "Man City", "Manchester City": "Man City",
    "Manchester United FC": "Man United", "Manchester United": "Man United",
    "Middlesbrough FC": "Middlesbrough", "Middlesbrough": "Middlesbrough",
    "Newcastle United FC": "Newcastle", "Newcastle United": "Newcastle",
    "Norwich City FC": "Norwich", "Norwich City": "Norwich",
    "Nottingham Forest FC": "Nottingham Forest", "Nottingham Forest": "Nottingham Forest",
    "Sheffield United FC": "Sheffield United", "Sheffield Utd": "Sheffield United",
    "Southampton FC": "Southampton", "Southampton": "Southampton",
    "Stoke City FC": "Stoke", "Stoke City": "Stoke",
    "Sunderland AFC": "Sunderland", "Sunderland": "Sunderland",
    "Swansea City FC": "Swansea", "Swansea City": "Swansea",
    "Tottenham Hotspur FC": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Watford FC": "Watford", "Watford": "Watford",
    "West Bromwich Albion FC": "West Brom", "West Bromwich Albion": "West Brom", "West Brom": "West Brom",
    "West Ham United FC": "West Ham", "West Ham United": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves", "Wolverhampton Wanderers": "Wolves", "Wolves": "Wolves",
}


def normalize_team(name: str) -> str:
    name = name.strip()
    return TEAM_ALIASES.get(name, name)


def parse_season_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    start_year = None
    for line in lines[:10]:
        m = SEASON_START_RE.match(line)
        if m:
            start_year = int(m.group(3))
            break
    if start_year is None:
        raise ValueError(f"Couldn't find a '# Date ...' header in {path}")

    rows = []
    current_date = None
    current_year = start_year
    prev_month = None

    for line in lines:
        content = line.strip()
        if not content:
            continue

        m = DATE_HEADER_RE.match(content)
        if m:
            month = MONTHS[m.group(1)]
            day = int(m.group(2))
            if m.group(3):
                current_year = int(m.group(3))
            elif prev_month is not None and month < prev_month:
                current_year += 1  # crossed into a new year without the source restating it
            prev_month = month
            current_date = f"{day:02d}/{month:02d}/{current_year}"
            continue

        if content.startswith(("=", "#", "▪", "(")):
            continue
        if current_date is None:
            continue  # header/blank-line noise before the first date

        home = away = home_goals = away_goals = None
        m = SCORE_MID_FORMAT_RE.match(content)
        if m and " v " not in m.group(1):
            home, hg, ag, away = m.group(1), m.group(2), m.group(3), m.group(4)
            home_goals, away_goals = int(hg), int(ag)
        else:
            m = V_FORMAT_RE.match(content)
            if m:
                home, away = m.group(1), m.group(2)
                if m.group(3) is not None:
                    home_goals, away_goals = int(m.group(3)), int(m.group(4))

        if home is None or away is None:
            continue  # not a match line (blank/annotation we didn't already skip)

        home, away = normalize_team(home), normalize_team(away)
        ftr = None
        if home_goals is not None:
            ftr = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

        rows.append({
            "Date": current_date, "HomeTeam": home, "AwayTeam": away,
            "FTHG": home_goals, "FTAG": away_goals, "FTR": ftr,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="openfootball season .txt files, oldest first")
    parser.add_argument("--out", default="data/matches.csv", help="where played matches go")
    parser.add_argument("--fixtures-out", default="data/fixtures.csv", help="where not-yet-played matches go")
    args = parser.parse_args()

    played, upcoming = [], []
    for path in args.files:
        rows = parse_season_file(path)
        file_played = [r for r in rows if r["FTR"] is not None]
        file_upcoming = [r for r in rows if r["FTR"] is None]
        played.extend(file_played)
        upcoming.extend(file_upcoming)
        print(f"{path}: {len(file_played)} played, {len(file_upcoming)} upcoming")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        writer.writeheader()
        writer.writerows(played)
    print(f"\nWrote {len(played)} played matches -> {args.out}")

    with open(args.fixtures_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "HomeTeam", "AwayTeam"])
        writer.writeheader()
        writer.writerows({"Date": r["Date"], "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"]} for r in upcoming)
    print(f"Wrote {len(upcoming)} upcoming fixtures -> {args.fixtures_out}")


if __name__ == "__main__":
    main()
