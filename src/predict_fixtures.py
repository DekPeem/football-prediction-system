"""
Predict every match in data/fixtures.csv (not-yet-played fixtures, produced by
src/import_openfootball.py) using team state replayed from data/matches.csv.

By default, prints just the next matchday (the earliest date in the fixtures
file) — use --all to predict everything in the file.

Usage:
    python src/predict_fixtures.py                  # next matchday only
    python src/predict_fixtures.py --all             # every upcoming fixture
    python src/predict_fixtures.py --date 2026-09-04 # one specific date
"""
from __future__ import annotations

import argparse

import pandas as pd

from features import FEATURE_COLUMNS, TeamState, build_features, load_matches
from predict import LABELS
import joblib

FIXTURES_DATE_FORMAT = "%d/%m/%Y"


def load_fixtures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format=FIXTURES_DATE_FORMAT, dayfirst=True)
    return df.sort_values("Date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/matches.csv")
    parser.add_argument("--fixtures", default="data/fixtures.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--all", action="store_true", help="predict every upcoming fixture, not just the next matchday")
    parser.add_argument("--date", default=None, help="only predict fixtures on this date (YYYY-MM-DD)")
    args = parser.parse_args()

    matches = load_matches(args.data)
    fixtures = load_fixtures(args.fixtures)
    if fixtures.empty:
        raise SystemExit(f"No upcoming fixtures in {args.fixtures}")

    if args.date:
        fixtures = fixtures[fixtures["Date"] == pd.Timestamp(args.date)]
    elif not args.all:
        # A round of fixtures spans a few days (Fri-Mon), not just one date.
        next_round_start = fixtures["Date"].min()
        fixtures = fixtures[fixtures["Date"] <= next_round_start + pd.Timedelta(days=3)]

    if fixtures.empty:
        raise SystemExit("No fixtures match that filter.")

    state = TeamState()
    build_features(matches, state=state)  # replay full history -> current Elo/form/h2h

    bundle = joblib.load(args.model)
    model = bundle["model"]

    print(f"{len(fixtures)} fixture(s), {fixtures['Date'].min().strftime('%d %b %Y')}"
          f"{'' if fixtures['Date'].min() == fixtures['Date'].max() else ' - ' + fixtures['Date'].max().strftime('%d %b %Y')}\n")

    for row in fixtures.itertuples(index=False):
        feats = state.features_for(row.HomeTeam, row.AwayTeam, row.Date)
        X = pd.DataFrame([feats])[FEATURE_COLUMNS]
        proba = model.predict_proba(X)[0]
        probs = dict(zip(model.classes_, proba))
        top = max(probs, key=probs.get)

        date_str = row.Date.strftime("%a %d %b")
        print(f"{date_str}  {row.HomeTeam} vs {row.AwayTeam}")
        print(f"    H {probs['H']:.0%}  D {probs['D']:.0%}  A {probs['A']:.0%}"
              f"   -> {LABELS[top]} ({probs[top]:.0%})")


if __name__ == "__main__":
    main()
