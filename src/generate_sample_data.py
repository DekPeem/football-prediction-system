"""
Generate a synthetic football results dataset in the same column layout used by
football-data.co.uk (Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, shots, cards, odds, ...).

This is SYNTHETIC data (no real match results) meant as a drop-in example so the
rest of the pipeline (features -> train -> predict) can be built and tested end
to end. Swap data/matches.csv for a real football-data.co.uk CSV later — the
loader only requires the columns listed in README.md.

Usage:
    python src/generate_sample_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

TEAMS = [
    "Ayutthaya United", "Bangkok Rovers", "Chonburi Sharks", "Dao Khanong FC",
    "Eastern Tigers", "Fah Sai City", "Grand Andaman", "Hat Yai Warriors",
    "Isan Dynamo", "Jomtien Athletic",
]

# Fixed underlying "true" strength per team (log-scale attack/defense), so results
# are internally consistent (better teams win more often) rather than pure noise.
TEAM_STRENGTH = {team: RNG.normal(0, 0.35) for team in TEAMS}
HOME_ADVANTAGE = 0.30


def _simulate_score(home: str, away: str) -> tuple[int, int]:
    home_rate = np.exp(0.9 + TEAM_STRENGTH[home] - TEAM_STRENGTH[away] + HOME_ADVANTAGE)
    away_rate = np.exp(0.7 + TEAM_STRENGTH[away] - TEAM_STRENGTH[home])
    home_goals = RNG.poisson(home_rate)
    away_goals = RNG.poisson(away_rate)
    return int(home_goals), int(away_goals)


def _shots_and_cards(goals: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    shots = max(goals, rng.poisson(10 + goals * 2))
    shots_on_target = min(shots, goals + rng.poisson(3))
    fouls = rng.poisson(11)
    yellows = rng.binomial(fouls, 0.18)
    return shots, shots_on_target, fouls, yellows


def generate_season(season_label: str, start_date: pd.Timestamp) -> pd.DataFrame:
    """Round-robin (home & away) fixtures for one season, played weekly."""
    fixtures = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    RNG.shuffle(fixtures)

    rows = []
    date = start_date
    matches_per_week = 5
    for i, (home, away) in enumerate(fixtures):
        if i > 0 and i % matches_per_week == 0:
            date += pd.Timedelta(days=7)

        hg, ag = _simulate_score(home, away)
        ftr = "H" if hg > ag else ("A" if ag > hg else "D")
        hs, hst, hf, hy = _shots_and_cards(hg, RNG)
        aws, awst, af, ay = _shots_and_cards(ag, RNG)

        # Rough bookmaker-style odds derived from relative strength (illustrative only).
        strength_gap = TEAM_STRENGTH[home] - TEAM_STRENGTH[away] + HOME_ADVANTAGE
        p_home = 1 / (1 + np.exp(-2.0 * strength_gap))
        p_away = (1 - p_home) * 0.62
        p_draw = 1 - p_home - p_away
        b365h = round(1 / max(p_home, 0.05) * RNG.uniform(0.95, 1.05), 2)
        b365d = round(1 / max(p_draw, 0.08) * RNG.uniform(0.95, 1.05), 2)
        b365a = round(1 / max(p_away, 0.05) * RNG.uniform(0.95, 1.05), 2)

        rows.append({
            "Date": date.strftime("%d/%m/%Y"),
            "Season": season_label,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": hg, "FTAG": ag, "FTR": ftr,
            "HS": hs, "AS": aws, "HST": hst, "AST": awst,
            "HF": hf, "AF": af, "HY": hy, "AY": ay,
            "HR": int(RNG.random() < 0.03), "AR": int(RNG.random() < 0.03),
            "B365H": b365h, "B365D": b365d, "B365A": b365a,
        })
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def main() -> None:
    seasons = [
        generate_season("2023-24", pd.Timestamp("2023-08-12")),
        generate_season("2024-25", pd.Timestamp("2024-08-11")),
    ]
    df = pd.concat(seasons, ignore_index=True)
    out_path = "data/matches.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic matches to {out_path}")
    print(df["FTR"].value_counts(normalize=True).rename("share").round(3))


if __name__ == "__main__":
    main()
