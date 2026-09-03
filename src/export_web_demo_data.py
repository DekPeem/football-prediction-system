"""
Export one league's trained model coefficients + every team's current state
(Elo, form, head-to-head, rest days) as a JSON file, so a static web page can
reproduce the exact same prediction client-side (no server needed) — the page
runs the identical scaler + logistic-regression math in JavaScript. Also
exports the next matchday (from fixtures.csv) for the page's "today's
matchday" panel, and recent results (from matches.csv) so the page can check
a previously-saved prediction against what actually happened.

Usage:
    python src/export_web_demo_data.py \
        --data data/la-liga/matches.csv --fixtures data/la-liga/fixtures.csv \
        --model models/la-liga/model.joblib --out web/leagues/la-liga.json \
        --label "La Liga"

Run once per league you want the web demo to cover — build_web_page.py then
combines every web/leagues/*.json into the page.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import joblib
import pandas as pd

from features import FEATURE_COLUMNS, TeamState, build_features, load_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/matches.csv")
    parser.add_argument("--fixtures", default="data/fixtures.csv")
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--out", default="web/model_export.json")
    parser.add_argument("--label", default="Premier League", help="display name shown in the league switcher")
    parser.add_argument("--recent-days", type=int, default=60, help="how much match history to embed for result-checking")
    args = parser.parse_args()

    matches = load_matches(args.data)
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

    bundle = joblib.load(args.model)
    model = bundle["model"]
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    metrics_path = os.path.join(os.path.dirname(args.model) or ".", "metrics.json")
    accuracy = baseline_accuracy = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        accuracy, baseline_accuracy = m["accuracy"], m["baseline_accuracy"]

    next_matchday = []
    if os.path.exists(args.fixtures):
        fixtures = pd.read_csv(args.fixtures)
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

    # Recent played results, so the page can check a previously-saved
    # prediction against what actually happened (a saved prediction's date+
    # teams match a row here once that round has been played and the data
    # re-imported).
    cutoff = matches["Date"].max() - pd.Timedelta(days=args.recent_days)
    recent = matches[matches["Date"] >= cutoff]
    recent_results = [
        {
            "date": row.Date.strftime("%Y-%m-%d"), "home": row.HomeTeam, "away": row.AwayTeam,
            "home_goals": int(row.FTHG), "away_goals": int(row.FTAG), "result": row.FTR,
        }
        for row in recent.itertuples(index=False)
    ]

    export = {
        "label": args.label,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "feature_columns": FEATURE_COLUMNS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "classes": clf.classes_.tolist(),
        "teams": team_info,
        "h2h": h2h_pairs,
        "next_matchday": next_matchday,
        "recent_results": recent_results,
        "n_train_matches": int(len(matches)),
        "date_range": [matches["Date"].min().strftime("%Y-%m-%d"), matches["Date"].max().strftime("%Y-%m-%d")],
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Wrote {args.out} ({len(teams)} teams, {len(h2h_pairs)} h2h pairs, {len(recent_results)} recent results)")


if __name__ == "__main__":
    main()
