# Football Prediction System

Predicts football match outcomes — **Home win / Draw / Away win** — from historical
match statistics (goals, shots, cards, recent form, head-to-head record).

> The bundled `data/matches.csv` is **real English Premier League results** —
> 2,280 matches across the 2015–16 through 2020–21 seasons, converted from the
> open-source [footballcsv/england](https://github.com/footballcsv/england)
> archive (see [`src/import_footballcsv.py`](src/import_footballcsv.py)).
> It stops at 2020–21 because that's as far as that archive goes; see
> [Getting more recent data](#getting-more-recent-data) to bring it up to date.
> `src/generate_sample_data.py` can still generate a synthetic league if you'd
> rather not use real club names while testing.

## How it works

1. **`src/import_footballcsv.py`** — converts footballcsv/england season files
   into this project's `Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR` layout, normalizing
   team names that vary release to release ("Arsenal FC" vs "Arsenal"). This is
   what built the bundled `data/matches.csv`.
   **`src/generate_sample_data.py`** does the same job with a synthetic
   10-team league instead, for testing without real club names.
2. **`src/features.py`** — turns raw results into pre-match features with no
   lookahead leakage: an Elo-style rating per team, rolling form (points per game,
   goal difference) over the last 5 matches, head-to-head record, and rest days
   since each team's last match.
3. **`src/train.py`** — trains a multinomial logistic regression on those
   features, evaluated with a **chronological** train/test split (matches are
   time-ordered, so a random split would leak future form into training).
4. **`src/predict.py`** — replays full match history to get each team's current
   Elo/form, then predicts the outcome of a new fixture.
5. **`src/export_web_demo_data.py`** — exports the trained model's coefficients
   and every team's current Elo/form/head-to-head as `web/model_export.json`,
   so a static page can reproduce the exact same prediction in the browser
   (see [Web demo](#web-demo)).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python src/train.py                    # trains on data/matches.csv, prints accuracy
python src/predict.py "Liverpool" "Man United"
```

Example output:

```
Liverpool (Elo 1724.1)  vs  Man United (Elo 1679.2)
  Home win   49.9%
  Draw       25.8%
  Away win   24.2%

Predicted: Home win
```

## Web demo

A pick-two-teams-and-see-the-probabilities page, running the trained model
client-side — no server, no backend, works offline once loaded.

- **Open it locally:** `web/index.html` is a single self-contained file —
  double-click it (or in VS Code, right-click → Open with Live Server) to
  open it in a browser. It has the model's weights baked in, so it works
  straight from disk.
- **Or view it hosted:** https://claude.ai/code/artifact/ceec7cc0-6284-4ca7-b495-718b45ad84a7

Both read from the same scaler + logistic-regression weights `predict.py`
uses, so all three always agree. After retraining on new data, refresh both:

```bash
python src/export_web_demo_data.py   # -> web/model_export.json
python src/build_web_page.py         # embeds it into web/index.html
```

(`web/template.html` is the page shell with a `__MODEL_DATA_JSON__`
placeholder — edit it to change the page's look; `build_web_page.py` fills
in the placeholder.) To update the hosted version too, paste `web/index.html`'s
content into the Artifact and republish it.

## Getting more recent data

`data/matches.csv` stops at the 2020–21 season because that's as far as the
open-source footballcsv/england archive goes. To bring it up to date, get a
newer season CSV from [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
(e.g. `E0.csv` for the Premier League — free, no sign-up) and save it as
`data/matches.csv`. The loader needs at minimum:

| Column | Meaning |
| --- | --- |
| `Date` | match date, `DD/MM/YYYY` |
| `HomeTeam`, `AwayTeam` | team names |
| `FTHG`, `FTAG` | full-time home/away goals |
| `FTR` | full-time result: `H` / `D` / `A` |

Then just re-run `python src/train.py`. Multiple seasons/leagues can be
concatenated into one CSV as long as column names match and rows stay sorted
(or at least sortable) by `Date` — keeping the older footballcsv-derived rows
and appending newer football-data.co.uk seasons works fine as long as team
names match (check against `TEAM_ALIASES` in `src/import_footballcsv.py`;
football-data.co.uk tends to use short names like "Man United" already).

## Model quality — read this before trusting a prediction

- **Baseline matters more than accuracy.** On the bundled data, "always
  predict home win" scores 39.7% (the test season, 2020–21, was played
  behind closed doors — home advantage was unusually weak league-wide that
  year, which is also why the model's own accuracy, 51.1%, is lower than
  you'd see on a normal season). `train.py` prints this baseline next to the
  model's accuracy — only the gap over baseline is real signal.
- Draws are structurally the hardest class to call; expect low recall there.
- This is a starting point (Elo + form + head-to-head + logistic regression),
  not a betting-grade model. Realistic next steps: add bookmaker odds as a
  feature (they're already the best public predictor), try gradient boosting
  (`GradientBoostingClassifier` / XGBoost), add player-availability/injury
  data, and validate with a rolling-season backtest instead of one train/test
  split.

## Project layout

```
data/matches.csv            match results (real EPL 2015-21 by default)
src/import_footballcsv.py   converts footballcsv/england season files
src/generate_sample_data.py synthetic data generator (alternative to real data)
src/features.py             feature engineering (Elo, form, head-to-head)
src/train.py                 training + evaluation
src/predict.py               CLI to predict a single fixture
src/export_web_demo_data.py  exports model + team state -> web/model_export.json
src/build_web_page.py        embeds web/model_export.json into web/index.html
web/template.html            web demo page shell (has the JSON placeholder)
web/model_export.json        exported model + team state
web/index.html                the runnable web demo (open directly in a browser)
models/                      trained model + metrics (git-ignored)
```
