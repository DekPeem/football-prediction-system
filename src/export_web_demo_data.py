"""
Export the trained model's coefficients + each team's current state (Elo, form,
head-to-head, rest days) as a single JSON file, so a static web page can
reproduce the exact same prediction client-side (no server needed) — the page
runs the identical scaler + logistic-regression math in JavaScript. Also
exports the next matchday from data/fixtures.csv, for the page's "predict
today's matches" panel.

Usage:
    python src/export_web_demo_data.py
"""
from __future__ import annotations

import itertools
import json
import os

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, TeamState, build_features, load_matches


def main() -> None:
    matches = load_matches("data/matches.csv")
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))

    state = TeamState()
    build_features(matches, state=state)  # replay full history -> final state

    as_of = matches["Date"].max() + pd.Timedelta(days=7)

    team_info = {}
    for team in teams:
        pts, gd = state._form_stats(team)
        rest = (as_of - state.last_match_date[team]).days if team in state.last_match_date else 14
        team_info[team] = {
            "elo": round(state.elo.get(team, 1500), 1),
            "form_pts": round(pts, 3),
            "form_gd": round(gd, 3),
            "rest_days": rest,
        }

    # h2h_home_edge, keyed "TeamA|TeamB" for every ordered pair with history,
    # from TeamA's perspective as the home side (same convention as TeamState.features_for)
    h2h_pairs = {}
    for a, b in itertools.permutations(teams, 2):
        key = tuple(sorted([a, b]))
        diffs = state.h2h.get(key, [])
        if not diffs:
            continue
        edge = sum(diffs) / len(diffs)
        if key[0] != a:
            edge = -edge
        h2h_pairs[f"{a}|{b}"] = round(edge, 3)

    bundle = joblib.load("models/model.joblib")
    model = bundle["model"]
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    next_matchday = []
    if os.path.exists("data/fixtures.csv"):
        fixtures = pd.read_csv("data/fixtures.csv")
        if not fixtures.empty:
            fixtures["Date"] = pd.to_datetime(fixtures["Date"], format="%d/%m/%Y")
            fixtures = fixtures.sort_values("Date")
            # A round spans a few days (Fri-Mon), not just one date - same
            # window predict_fixtures.py uses for "next matchday".
            window_start = fixtures["Date"].min()
            round_fixtures = fixtures[fixtures["Date"] <= window_start + pd.Timedelta(days=3)]
            next_matchday = [
                {"date": row.Date.strftime("%Y-%m-%d"), "home": row.HomeTeam, "away": row.AwayTeam}
                for row in round_fixtures.itertuples(index=False)
            ]

    export = {
        "feature_columns": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": clf.classes_.tolist(),
        "teams": team_info,
        "h2h": h2h_pairs,
        "next_matchday": next_matchday,
        "n_train_matches": int(len(matches)),
        "date_range": [matches["Date"].min().strftime("%Y-%m-%d"), matches["Date"].max().strftime("%Y-%m-%d")],
    }

    out_path = "web/model_export.json"
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Wrote {out_path} ({len(teams)} teams, {len(h2h_pairs)} h2h pairs)")


if __name__ == "__main__":
    main()
