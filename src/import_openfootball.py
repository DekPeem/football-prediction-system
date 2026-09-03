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

    # -- La Liga (openfootball/espana) --
    "Athletic Club": "Athletic Bilbao",
    "Atlético Madrid": "Atletico Madrid", "Atlético de Madrid": "Atletico Madrid", "Club Atlético de Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona", "FC Barcelona": "Barcelona",
    "CA Osasuna": "Osasuna",
    "CD Alavés": "Alaves", "Deportivo Alavés": "Alaves",
    "CD Leganés": "Leganes",
    "Cádiz CF": "Cadiz",
    "Deportivo La Coruña": "Deportivo La Coruna", "RC Deportivo La Coruña": "Deportivo La Coruna",
    "Elche CF": "Elche",
    "Espanyol Barcelona": "Espanyol", "RCD Espanyol": "Espanyol", "RCD Espanyol de Barcelona": "Espanyol",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Granada CF": "Granada",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RC Celta": "Celta Vigo", "RC Celta de Vigo": "Celta Vigo",
    "RCD Mallorca": "Mallorca",
    "Rayo Vallecano": "Rayo Vallecano", "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Real Betis": "Real Betis", "Real Betis Balompié": "Real Betis",
    "Real Madrid": "Real Madrid", "Real Madrid C.F.": "Real Madrid", "Real Madrid CF": "Real Madrid",
    "Real Racing Club de Santander": "Racing Santander",
    "Real Sociedad": "Real Sociedad", "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Valladolid": "Valladolid", "Real Valladolid CF": "Valladolid",
    "SD Eibar": "Eibar",
    "SD Huesca": "Huesca",
    "Sevilla": "Sevilla", "Sevilla FC": "Sevilla",
    "Sporting Gijón": "Sporting Gijon",
    "UD Almería": "Almeria",
    "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia",
    "Villarreal": "Villarreal", "Villarreal CF": "Villarreal",

    # -- Serie A (openfootball/italy) --
    "AC Milan": "AC Milan", "Milan": "AC Milan",
    "AC Monza": "Monza",
    "ACF Fiorentina": "Fiorentina", "Fiorentina": "Fiorentina",
    "AS Roma": "AS Roma",
    "Atalanta": "Atalanta", "Atalanta BC": "Atalanta",
    "Benevento Calcio": "Benevento",
    "Bologna": "Bologna", "Bologna FC": "Bologna", "Bologna FC 1909": "Bologna",
    "Brescia Calcio": "Brescia",
    "Cagliari": "Cagliari", "Cagliari Calcio": "Cagliari",
    "Carpi FC": "Carpi",
    "Chievo Verona": "Chievo",
    "Como": "Como", "Como 1907": "Como",
    "Cremonese": "Cremonese", "US Cremonese": "Cremonese",
    "Delfino Pescara": "Pescara",
    "Empoli FC": "Empoli",
    "FC Crotone": "Crotone",
    "FC Internazionale Milano": "Inter", "Inter": "Inter",
    "Frosinone Calcio": "Frosinone",
    "Genoa": "Genoa", "Genoa CFC": "Genoa",
    "Hellas Verona": "Hellas Verona", "Hellas Verona FC": "Hellas Verona",
    "Juventus": "Juventus", "Juventus FC": "Juventus",
    "Lazio": "Lazio", "Lazio Roma": "Lazio", "SS Lazio": "Lazio",
    "Lecce": "Lecce", "US Lecce": "Lecce",
    "Napoli": "Napoli", "SSC Napoli": "Napoli",
    "Parma": "Parma", "Parma Calcio 1913": "Parma",
    "Pisa": "Pisa",
    "SPAL 2013 Ferrara": "SPAL",
    "Sampdoria": "Sampdoria", "UC Sampdoria": "Sampdoria",
    "Sassuolo": "Sassuolo", "Sassuolo Calcio": "Sassuolo", "US Sassuolo Calcio": "Sassuolo",
    "Spezia Calcio": "Spezia",
    "Torino": "Torino", "Torino FC": "Torino",
    "US Palermo": "Palermo",
    "US Salernitana 1919": "Salernitana",
    "Udinese": "Udinese", "Udinese Calcio": "Udinese",
    "Venezia FC": "Venezia",

    # -- Ligue 1 (openfootball/france, france/ subfolder) --
    "AC Ajaccio": "Ajaccio",
    "AJ Auxerre": "Auxerre", "Auxerre": "Auxerre",
    "AS Monaco": "Monaco", "AS Monaco FC": "Monaco",
    "AS Nancy Lorraine": "Nancy",
    "AS Saint-Étienne": "Saint-Etienne",
    "Amiens SC": "Amiens",
    "Angers SCO": "Angers",
    "Clermont Foot 63": "Clermont",
    "Dijon FCO": "Dijon",
    "EA Guingamp": "Guingamp",
    "ES Troyes AC": "Troyes", "ESTAC Troyes": "Troyes",
    "FC Lorient": "Lorient",
    "FC Metz": "Metz",
    "FC Nantes": "Nantes",
    "Gazélec FC Ajaccio": "Gazelec Ajaccio",
    "Girondins Bordeaux": "Bordeaux",
    "Le Havre": "Le Havre", "Le Havre AC": "Le Havre",
    "Le Mans FC": "Le Mans",
    "Lens": "Lens", "RC Lens": "Lens", "Racing Club de Lens": "Lens",
    "Lille": "Lille", "Lille OSC": "Lille",
    "Montpellier HSC": "Montpellier",
    "Nîmes Olympique": "Nimes",
    "OGC Nice": "Nice",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille", "Olympique de Marseille": "Marseille",
    "Paris FC": "Paris FC",
    "Paris Saint-Germain": "PSG", "Paris Saint-Germain FC": "PSG",
    "RC Strasbourg": "Strasbourg", "RC Strasbourg Alsace": "Strasbourg",
    "Rennes": "Rennes", "Stade Rennais": "Rennes", "Stade Rennais FC 1901": "Rennes",
    "SC Bastia": "Bastia",
    "SM Caen": "Caen",
    "Stade Brestois": "Brest", "Stade Brestois 29": "Brest",
    "Stade de Reims": "Reims",
    "Toulouse FC": "Toulouse",
}

# Some seasons (e.g. Ligue 1 2019-20, cut short by COVID) mark a team's
# remaining fixtures with a trailing "<score> [awarded]" or "[cancelled]"
# annotation instead of just omitting them. Strip that before normalizing,
# or it reads as a distinct "team".
ANNOTATION_RE = re.compile(r"\s+(?:\d+-\d+\s+)?\[(?:awarded|cancelled)\]\s*$")


def normalize_team(name: str) -> str:
    name = ANNOTATION_RE.sub("", name.strip()).strip()
    return TEAM_ALIASES.get(name, name)


def parse_season_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    start_year = start_month = None
    for line in lines[:10]:
        m = SEASON_START_RE.match(line)
        if m:
            start_month, start_year = MONTHS[m.group(1)], int(m.group(3))
            break
    if start_year is None:
        raise ValueError(f"Couldn't find a '# Date ...' header in {path}")

    rows = []
    current_date = None

    for line in lines:
        content = line.strip()
        if not content:
            continue

        m = DATE_HEADER_RE.match(content)
        if m:
            month = MONTHS[m.group(1)]
            day = int(m.group(2))
            if m.group(3):
                year = int(m.group(3))
            else:
                # A season runs, e.g., Aug year0 - May/Jul year1. Deriving the
                # year from the season's calendar (rather than tracking month
                # increases through the file) also copes with a source that
                # lists rescheduled/postponed matches out of chronological
                # order — real for COVID-affected seasons.
                year = start_year if month >= start_month else start_year + 1
            current_date = f"{day:02d}/{month:02d}/{year}"
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
    last_file_index = len(args.files) - 1
    dropped = 0
    for i, path in enumerate(args.files):
        rows = parse_season_file(path)
        file_played = [r for r in rows if r["FTR"] is not None]
        file_upcoming = [r for r in rows if r["FTR"] is None]
        played.extend(file_played)
        if i == last_file_index:
            # Only the last (current-season) file's unplayed rows are real
            # upcoming fixtures. An earlier file's unplayed rows are stale —
            # a postponed/COVID-cancelled match the source never resolved —
            # and would corrupt "next matchday" if mixed in, so they're
            # dropped rather than written anywhere.
            upcoming.extend(file_upcoming)
        else:
            dropped += len(file_upcoming)
        print(f"{path}: {len(file_played)} played, {len(file_upcoming)} upcoming"
              f"{'' if i == last_file_index else ' (dropped, not the current season)' if file_upcoming else ''}")
    if dropped:
        print(f"\n({dropped} stale unplayed row(s) from non-current seasons dropped entirely)")

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
