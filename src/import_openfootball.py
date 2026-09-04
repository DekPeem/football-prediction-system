"""
Convert season files from the open-source openfootball family of archives
(https://github.com/openfootball — england/espana/italy/france) into this
project's CSV layout. Covers 13 competitions across those 4 countries: 4 top
flights, 4 second divisions, 5 domestic cups (see README.md's "Other leagues"
and "Cup competitions" sections for the exact repo/file per competition).
Three things make this format trickier than a single clean CSV:

1. **The per-line layout changed across seasons and competitions** — some
   use "Team A   2-1 (1-0)   Team B", others "Team A   v   Team B  2-1
   (1-0)". split_match_line() handles both by splitting on runs of 2+ spaces
   (what actually separates the columns) rather than anchoring a format-
   specific regex.
2. **Cup scorelines carry extra-time/penalty-shootout annotations** —
   "2-1 a.e.t. (1-1, 0-1)", "9-8 pen. (0-0)". extract_score() always resolves
   to the actual 90+30 minute result, never the shootout score (a shootout
   only happens after a draw, so that's correctly "D" for training).
3. **A season file mixes played matches and future fixtures** (no score yet,
   for a season in progress). Both are parsed; train.py only ever sees the
   played ones (features.py drops rows with no FTR), while the unplayed ones
   become data/fixtures.csv — the input to predict_fixtures.py for "what's
   the model's call on next weekend's games".

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

FIELD_SPLIT_RE = re.compile(r"\s{2,}")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
# The real (90+30 min) result, allowing for a penalty-shootout prefix that's
# discarded (a shootout only happens after a draw, so the match itself is
# correctly a "D" for training either way) and/or an extra-time marker. The
# score sometimes isn't restated outside the parenthetical at all ("9-8 pen.
# (0-0)"), so that's the fallback source if nothing was captured directly.
CUP_SCORE_RE = re.compile(
    r"^(?:\d+-\d+\s+pen\.\s+)?"
    r"(?:(\d+)-(\d+))?"
    r"(?:\s*a\.e\.t\.)?"
    r"(?:\s*\((\d+)-(\d+)[^)]*\))?"
)


def extract_score(text: str) -> tuple[int, int] | None:
    m = CUP_SCORE_RE.match(text.strip())
    if not m:
        return None
    real_h, real_a, paren_h, paren_a = m.groups()
    if real_h is not None:
        return int(real_h), int(real_a)
    if paren_h is not None:
        return int(paren_h), int(paren_a)
    return None


def split_match_line(content: str) -> tuple[str, str, str | None, str | None] | None:
    """Split one non-header line into (home, away, score_text_or_None, kickoff_time_or_None).

    Splitting on runs of 2+ spaces is what actually separates the columns —
    team names and the scoreline can each contain single spaces internally,
    but the source always pads between columns with 2+. Handles both known
    row layouts ("Team A v Team B  <score>" and "Team A  <score>  Team B"),
    plus the case where a long team name overflows its column padding down to
    a single space before "v" (so "Team A v Team B" collapses into one field
    once split), and a multi-token scoreline ("0-3    [awarded]",
    "5-4 pen. 0-0 a.e.t. (0-0)") that itself contains 2+-space gaps. A leading
    kickoff-time field ("21:00  Team A v Team B") is captured, not just
    stripped, so callers can surface it (a future fixture especially benefits
    from knowing when it kicks off, not just what day).
    """
    fields = FIELD_SPLIT_RE.split(content)
    time_text = None
    if fields and TIME_RE.match(fields[0]):
        time_text = fields[0]
        fields = fields[1:]
    if not fields:
        return None

    if " v " in fields[0]:
        home, away = fields[0].split(" v ", 1)
        score_text = " ".join(fields[1:]) if len(fields) > 1 else None
    elif len(fields) >= 2 and fields[1].startswith("v "):
        home, away = fields[0], fields[1][2:]
        score_text = " ".join(fields[2:]) if len(fields) > 2 else None
    elif len(fields) >= 3:
        home = fields[0]
        if fields[-1].startswith("[") and len(fields) >= 4:
            # A trailing "[awarded]"/"[cancelled]" annotation is its own
            # field after the away team, not part of it.
            away, score_text = fields[-2], " ".join(fields[1:-2] + fields[-1:])
        else:
            away, score_text = fields[-1], " ".join(fields[1:-1])
    else:
        return None
    return home.strip(), away.strip(), (score_text or None), time_text

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

    # -- Championship (openfootball/england, 2-championship.txt) --
    "Barnsley FC": "Barnsley",
    "Birmingham City FC": "Birmingham City",
    "Blackburn Rovers FC": "Blackburn Rovers",
    "Blackpool FC": "Blackpool",
    "Bolton Wanderers FC": "Bolton Wanderers",
    "Bristol City FC": "Bristol City",
    "Charlton Athletic FC": "Charlton Athletic",
    "Derby County FC": "Derby County",
    "Lincoln City FC": "Lincoln City",
    "Millwall FC": "Millwall",
    "Oxford United FC": "Oxford United",
    "Peterborough United FC": "Peterborough United",
    "Plymouth Argyle FC": "Plymouth Argyle",
    "Portsmouth FC": "Portsmouth",
    "Preston North End FC": "Preston North End",
    "Queens Park Rangers FC": "QPR",
    "Reading FC": "Reading",
    "Rotherham United FC": "Rotherham United",
    "Sheffield Wednesday FC": "Sheffield Wednesday",
    "Swansea City AFC": "Swansea",
    "Wigan Athletic FC": "Wigan Athletic",
    "Wrexham AFC": "Wrexham",

    # -- Segunda División (openfootball/espana, 2-liga2.txt) --
    "AD Alcorcón": "Alcorcon",
    "AD Ceuta FC": "Ceuta",
    "Burgos CF": "Burgos",
    "CD Castellón": "Castellon",
    "CD Eldense": "Eldense",
    "CD Lugo": "Lugo",
    "CD Mirandés": "Mirandes",
    "CD Tenerife": "Tenerife",
    "CF Fuenlabrada": "Fuenlabrada",
    "Córdoba CF": "Cordoba",
    "FC Andorra": "Andorra",
    "FC Cartagena": "Cartagena",
    "SD Amorebieta": "Amorebieta",
    "SD Ponferradina": "Ponferradina",
    "UD Ibiza": "Ibiza",
    "Villarreal CF B": "Villarreal B",

    # -- Serie B (openfootball/italy, 2-serieb.txt) --
    "AC Perugia Calcio": "Perugia",
    "AC Reggiana 1919": "Reggiana",
    "AS Cittadella": "Cittadella",
    "Ascoli Calcio": "Ascoli",
    "Calcio Lecco 1912": "Lecco",
    "Calcio Padova": "Padova",
    "Carrarese Calcio": "Carrarese",
    "Cesena FC": "Cesena",
    "Cosenza Calcio": "Cosenza",
    "FC Südtirol": "Sudtirol",
    "Feralpisalò": "Feralpisalo",
    "L.R. Vicenza": "Vicenza",
    "Mantova 1911 SSD": "Mantova",
    "Modena FC": "Modena",
    "Palermo FC": "Palermo",
    "Pisa SC": "Pisa",
    "Pordenone Calcio": "Pordenone",
    "Reggina 1914": "Reggina",
    "SSC Bari": "Bari",
    "Ternana Calcio": "Ternana",
    "US Alessandria 1912": "Alessandria",
    "US Avellino": "Avellino",
    "US Catanzaro": "Catanzaro",

    # -- Ligue 2 (openfootball/france, france/*_fr2.txt) --
    "Chamois Niortais": "Niort",
    "FC Annecy": "Annecy",
    "FC Martigues": "Martigues",
    "FC Sochaux": "Sochaux",
    "Grenoble Foot 38": "Grenoble",
    "Havre AC": "Le Havre",
    "Pau FC": "Pau",
    "Red Star FC": "Red Star",
    "Rodez AF": "Rodez",
    "Stade Lavallois": "Laval",
    "US Boulogne": "Boulogne",
    "US Concarneau": "Concarneau",
    "US Quevilly-Rouen": "Quevilly-Rouen",
    "USL Dunkerque": "Dunkerque",
    "Valenciennes FC": "Valenciennes",
}

# Safety net, not the primary mechanism: split_match_line()/extract_score()
# keep a "<score> [awarded]"/"[cancelled]"/"[postponed]" annotation or a
# penalty-shootout scoreline out of the team-name fields in every layout this
# project has actually seen. This catches whatever format quirk shows up
# next instead of quietly training on a mis-split row.
SUSPECT_TEAM_RE = re.compile(r"a\.e\.t|pen\.|postponed|cancelled|awarded|\[|\]", re.IGNORECASE)


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


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

        parsed = split_match_line(content)
        if parsed is None:
            continue  # not a match line (blank/annotation we didn't already skip)
        home, away, score_text, time_text = parsed

        home_goals = away_goals = None
        if score_text:
            score = extract_score(score_text)
            if score is not None:
                home_goals, away_goals = score

        home, away = normalize_team(home), normalize_team(away)
        ftr = None
        if home_goals is not None:
            ftr = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

        rows.append({
            "Date": current_date, "Time": time_text or "", "HomeTeam": home, "AwayTeam": away,
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
    dropped = malformed = 0
    for i, path in enumerate(args.files):
        rows = parse_season_file(path)
        clean_rows = [r for r in rows if not (SUSPECT_TEAM_RE.search(r["HomeTeam"]) or SUSPECT_TEAM_RE.search(r["AwayTeam"]))]
        malformed += len(rows) - len(clean_rows)
        rows = clean_rows
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
    if malformed:
        print(f"({malformed} row(s) with an unparsed playoff/extra-time/penalty scoreline dropped entirely)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        writer.writeheader()
        writer.writerows(played)
    print(f"\nWrote {len(played)} played matches -> {args.out}")

    with open(args.fixtures_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Time", "HomeTeam", "AwayTeam"])
        writer.writeheader()
        writer.writerows({"Date": r["Date"], "Time": r["Time"], "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"]} for r in upcoming)
    print(f"Wrote {len(upcoming)} upcoming fixtures -> {args.fixtures_out}")


if __name__ == "__main__":
    main()
