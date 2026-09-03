"""
Predict the outcome (Home win / Draw / Away win) of a single upcoming fixture,
using team state (Elo, form, head-to-head) rebuilt from full match history.

Usage:
    python src/predict.py "Chonburi Sharks" "Bangkok Rovers"
    python src/predict.py "Chonburi Sharks" "Bangkok Rovers" --date 2025-09-20
"""
from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, TeamState, build_features, load_matches

LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}


def predict_fixture(home: str, away: str, data_path: str, model_path: str, as_of: str | None = None) -> dict:
    matches = load_matches(data_path)
    known_teams = set(matches["HomeTeam"]) | set(matches["AwayTeam"])
    for team in (home, away):
        if team not in known_teams:
            raise SystemExit(f"Unknown team '{team}'. Known teams:\n  " + "\n  ".join(sorted(known_teams)))

    state = TeamState()
    build_features(matches, state=state)  # replays full history into state

    as_of_date = pd.Timestamp(as_of) if as_of else matches["Date"].max() + pd.Timedelta(days=7)
    feats = state.features_for(home, away, as_of_date)
    X = pd.DataFrame([feats])[[c for c in FEATURE_COLUMNS]]  # reorder/select to match training columns
    # features_for returns extra keys (elo_home/elo_away) not in FEATURE_COLUMNS; that's fine, we select above.

    bundle = joblib.load(model_path)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    probs = dict(zip(model.classes_, proba))

    return {
        "home": home, "away": away,
        "probabilities": {LABELS[k]: round(float(v), 3) for k, v in probs.items()},
        "predicted": LABELS[max(probs, key=probs.get)],
        "elo_home": round(feats["elo_home"], 1), "elo_away": round(feats["elo_away"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team")
    parser.add_argument("away_team")
    parser.add_argument("--data", default="data/matches.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--date", default=None, help="Fixture date (YYYY-MM-DD), defaults to next week")
    args = parser.parse_args()

    try:
        result = predict_fixture(args.home_team, args.away_team, args.data, args.model, args.date)
    except FileNotFoundError:
        print("Model not found — run `python src/train.py` first.", file=sys.stderr)
        raise SystemExit(1)

    print(f"{result['home']} (Elo {result['elo_home']})  vs  {result['away']} (Elo {result['elo_away']})")
    for outcome, p in result["probabilities"].items():
        print(f"  {outcome:10s} {p:.1%}")
    print(f"\nPredicted: {result['predicted']}")


if __name__ == "__main__":
    main()
