"""
Turn a match-results table into per-match features usable *before kickoff*
(no leakage: every feature for a given match only uses information available
up to the day before that match).

Features built:
  - Elo-style team ratings (updated after every match)
  - Rolling form: points per game and goal difference over the last N matches
  - Head-to-head record between the two teams
  - Rest days since each team's previous match
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FORM_WINDOW = 5
ELO_K = 20
ELO_HOME_ADV = 60
ELO_BASE = 1500


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def _points(gf: int, ga: int) -> int:
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


class TeamState:
    """Running per-team state (Elo, recent form, head-to-head, rest days) built up
    match by match. Reused by both training (build_features) and single-fixture
    prediction (predict.py) so the exact same feature logic applies to both."""

    def __init__(self) -> None:
        self.elo: dict[str, float] = {}
        self.last_match_date: dict[str, pd.Timestamp] = {}
        # team -> list of (date, points, goals_for, goals_against), most recent first
        self.recent_form: dict[str, list[tuple[pd.Timestamp, int, int, int]]] = {}
        self.h2h: dict[tuple[str, str], list[int]] = {}  # sorted (teamA, teamB) -> point diffs from A's view

    def _form_stats(self, team: str) -> tuple[float, float]:
        games = self.recent_form.get(team, [])[:FORM_WINDOW]
        if not games:
            return 1.0, 0.0  # neutral prior: ~1 pt/game, level goal diff
        pts = np.mean([g[1] for g in games])
        gd = np.mean([g[2] - g[3] for g in games])
        return pts, gd

    def features_for(self, home: str, away: str, as_of: pd.Timestamp) -> dict:
        elo_home = self.elo.get(home, ELO_BASE)
        elo_away = self.elo.get(away, ELO_BASE)
        home_pts, home_gd = self._form_stats(home)
        away_pts, away_gd = self._form_stats(away)

        key = tuple(sorted([home, away]))
        h2h_diffs = self.h2h.get(key, [])
        h2h_home_edge = np.mean(h2h_diffs) if h2h_diffs else 0.0
        if key[0] != home:
            h2h_home_edge = -h2h_home_edge

        rest_home = (as_of - self.last_match_date[home]).days if home in self.last_match_date else 14
        rest_away = (as_of - self.last_match_date[away]).days if away in self.last_match_date else 14

        return {
            "elo_home": elo_home, "elo_away": elo_away, "elo_diff": elo_home - elo_away,
            "home_form_pts": home_pts, "away_form_pts": away_pts, "form_pts_diff": home_pts - away_pts,
            "home_form_gd": home_gd, "away_form_gd": away_gd, "form_gd_diff": home_gd - away_gd,
            "h2h_home_edge": h2h_home_edge,
            "rest_days_home": rest_home, "rest_days_away": rest_away,
        }

    def apply_result(self, home: str, away: str, ftr: str, fthg: int, ftag: int, date: pd.Timestamp) -> None:
        elo_home = self.elo.get(home, ELO_BASE)
        elo_away = self.elo.get(away, ELO_BASE)
        exp_home = _expected_score(elo_home + ELO_HOME_ADV, elo_away)
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[ftr]
        self.elo[home] = elo_home + ELO_K * (actual_home - exp_home)
        self.elo[away] = elo_away + ELO_K * ((1 - actual_home) - (1 - exp_home))

        home_points, away_points = _points(fthg, ftag), _points(ftag, fthg)
        self.recent_form.setdefault(home, []).insert(0, (date, home_points, fthg, ftag))
        self.recent_form.setdefault(away, []).insert(0, (date, away_points, ftag, fthg))

        key = tuple(sorted([home, away]))
        point_diff_from_home = home_points - away_points
        diff_from_key0 = point_diff_from_home if key[0] == home else -point_diff_from_home
        self.h2h.setdefault(key, []).insert(0, diff_from_key0)

        self.last_match_date[home] = date
        self.last_match_date[away] = date


def build_features(df: pd.DataFrame, state: "TeamState | None" = None) -> pd.DataFrame:
    """Compute pre-match features for every row of df, in chronological order.
    Pass in a TeamState to seed with history (e.g. continue after a training set);
    the state is mutated in place so callers can inspect it afterwards."""
    state = state if state is not None else TeamState()

    records = []
    for row in df.itertuples(index=False):
        home, away = row.HomeTeam, row.AwayTeam
        feats = state.features_for(home, away, row.Date)
        records.append({
            "Date": row.Date, "HomeTeam": home, "AwayTeam": away,
            **feats,
            "FTR": getattr(row, "FTR", None),
            "FTHG": getattr(row, "FTHG", None),
            "FTAG": getattr(row, "FTAG", None),
        })
        if pd.notna(row.FTR):
            state.apply_result(home, away, row.FTR, row.FTHG, row.FTAG, row.Date)

    return pd.DataFrame(records)


FEATURE_COLUMNS = [
    "elo_diff", "form_pts_diff", "form_gd_diff", "h2h_home_edge",
    "home_form_pts", "away_form_pts", "home_form_gd", "away_form_gd",
    "rest_days_home", "rest_days_away",
]
