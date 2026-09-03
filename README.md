# Football Prediction System

Predicts football match outcomes — **Home win / Draw / Away win** — from historical
match statistics (goals, shots, cards, recent form, head-to-head record).

> The bundled `data/matches.csv` is **synthetic** sample data (generated, not real
> results) so the pipeline runs out of the box. Swap in a real CSV from
> [football-data.co.uk](https://www.football-data.co.uk/data.php) (same column
> layout) to get real predictions — see [Using real data](#using-real-data) below.

## How it works

1. **`src/generate_sample_data.py`** — simulates two seasons of results for a
   10-team league with fixed underlying team strengths, in the football-data.co.uk
   column layout (`Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HS, AS, HST, AST, ...`).
2. **`src/features.py`** — turns raw results into pre-match features with no
   lookahead leakage: an Elo-style rating per team, rolling form (points per game,
   goal difference) over the last 5 matches, head-to-head record, and rest days
   since each team's last match.
3. **`src/train.py`** — trains a multinomial logistic regression on those
   features, evaluated with a **chronological** train/test split (matches are
   time-ordered, so a random split would leak future form into training).
4. **`src/predict.py`** — replays full match history to get each team's current
   Elo/form, then predicts the outcome of a new fixture.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python src/generate_sample_data.py     # writes data/matches.csv
python src/train.py                    # trains models/model.joblib, prints accuracy
python src/predict.py "Chonburi Sharks" "Bangkok Rovers"
```

Example output:

```
Chonburi Sharks (Elo 1523.4)  vs  Bangkok Rovers (Elo 1487.1)
  Home win   48.2%
  Draw       26.1%
  Away win   25.7%

Predicted: Home win
```

## Using real data

Download a league CSV from football-data.co.uk (e.g. English Premier League,
`E0.csv`) and save it as `data/matches.csv`. The loader needs at minimum:

| Column | Meaning |
| --- | --- |
| `Date` | match date, `DD/MM/YYYY` |
| `HomeTeam`, `AwayTeam` | team names |
| `FTHG`, `FTAG` | full-time home/away goals |
| `FTR` | full-time result: `H` / `D` / `A` |

Then just re-run `python src/train.py`. Multiple seasons/leagues can be
concatenated into one CSV as long as column names match and rows stay sorted
(or at least sortable) by `Date`.

## Model quality — read this before trusting a prediction

- **Baseline matters more than accuracy.** Football has ~45% home wins, ~25%
  draws, ~30% away wins in most leagues, so "always predict home win" already
  scores ~45%. `train.py` prints this baseline next to the model's accuracy —
  only the gap over baseline is real signal.
- Draws are structurally the hardest class to call; expect low recall there.
- This is a starting point (Elo + form + head-to-head + logistic regression),
  not a betting-grade model. Realistic next steps: add bookmaker odds as a
  feature (they're already the best public predictor), try gradient boosting
  (`GradientBoostingClassifier` / XGBoost), add player-availability/injury
  data, and validate with a rolling-season backtest instead of one train/test
  split.

## Project layout

```
data/matches.csv          match results (synthetic sample, or your real CSV)
src/generate_sample_data.py  synthetic data generator
src/features.py            feature engineering (Elo, form, head-to-head)
src/train.py                training + evaluation
src/predict.py              CLI to predict a single fixture
models/                     trained model + metrics (git-ignored)
```
