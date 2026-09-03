# Football Prediction System

Predicts football match outcomes — **Home win / Draw / Away win** — from historical
match statistics (goals, shots, cards, recent form, head-to-head record).

> The bundled `data/matches.csv` is **real English Premier League results** —
> 4,200 matches, 2015–16 through the current (2026–27) season, converted from
> the open-source [openfootball](https://github.com/openfootball) family of
> archives (see [`src/import_openfootball.py`](src/import_openfootball.py)).
> `data/fixtures.csv` holds every match from the current season that hasn't
> been played yet — see [Predicting the next matchday](#predicting-the-next-matchday).
> Three more leagues (La Liga, Serie A, Ligue 1) ship the same way under
> `data/<league>/` — see [Other leagues](#other-leagues).
> `src/generate_sample_data.py` can still generate a synthetic league if you'd
> rather not use real club names while testing.

## How it works

1. **`src/import_openfootball.py`** — converts openfootball season files
   (England, Spain, Italy or France — see [Other leagues](#other-leagues))
   into this project's `Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR` layout,
   normalizing team names that vary release to release ("Arsenal FC" vs
   "Arsenal") and file to file (some seasons use a `Team A v Team B  2-1`
   layout, others `Team A  2-1  Team B`). Played matches go to
   `data/matches.csv`; not-yet-played matches in the current season go to
   `data/fixtures.csv`. This is what built the bundled Premier League files
   (and the other three leagues' equivalents under `data/<league>/`).
   **`src/generate_sample_data.py`** does the same training-data job with a
   synthetic 10-team league instead, for testing without real club names.
2. **`src/features.py`** — turns raw results into pre-match features with no
   lookahead leakage: an Elo-style rating per team, rolling form (points per game,
   goal difference) over the last 5 matches, head-to-head record, and rest days
   since each team's last match.
3. **`src/train.py`** — trains a multinomial logistic regression on those
   features, evaluated with a **chronological** train/test split (matches are
   time-ordered, so a random split would leak future form into training).
4. **`src/predict.py`** — replays full match history to get each team's current
   Elo/form, then predicts the outcome of a single fixture you name.
5. **`src/predict_fixtures.py`** — same idea, but predicts every match in
   `data/fixtures.csv` — by default just the next matchday
   (see [Predicting the next matchday](#predicting-the-next-matchday)).
6. **`src/export_web_demo_data.py`** — exports the trained model's coefficients
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
Liverpool (Elo 1658.8)  vs  Man United (Elo 1646.1)
  Home win   45.7%
  Draw       26.0%
  Away win   28.4%

Predicted: Home win
```

## Predicting the next matchday

`data/fixtures.csv` holds the current season's not-yet-played matches (built
by `import_openfootball.py` alongside `data/matches.csv`). To see the model's
call on every game in the next round:

```bash
python src/predict_fixtures.py
```

```
10 fixture(s), 04 Sep 2026 - 06 Sep 2026

Fri 04 Sep  Ipswich Town vs Liverpool
    H 14%  D 22%  A 65%   -> Away win (65%)
Sat 05 Sep  Man City vs Coventry City
    H 82%  D 14%  A 5%   -> Home win (82%)
...
```

`--all` predicts every remaining fixture in the season instead of just the
next round; `--date 2026-09-05` predicts one specific date. Once a matchday
is actually played, re-run the import to move it from `data/fixtures.csv`
into `data/matches.csv` (see [Keeping the data current](#keeping-the-data-current)).

## Other leagues

La Liga, Serie A and Ligue 1 ship the same way, each in its own subfolder
with its own model (a Ligue 1 team's Elo has nothing to do with a Premier
League team's, so the leagues are trained separately, not merged):

```
data/la-liga/matches.csv    data/la-liga/fixtures.csv    models/la-liga/model.joblib
data/serie-a/matches.csv    data/serie-a/fixtures.csv    models/serie-a/model.joblib
data/ligue-1/matches.csv    data/ligue-1/fixtures.csv    models/ligue-1/model.joblib
```

Every script takes `--data` / `--fixtures` / `--model` (and `train.py` takes
`--model-out`), so point them at a league's files instead of the
Premier League default:

```bash
python src/predict_fixtures.py --data data/la-liga/matches.csv \
    --fixtures data/la-liga/fixtures.csv --model models/la-liga/model.joblib

python src/predict.py "Real Madrid" "Barcelona" --data data/la-liga/matches.csv --model models/la-liga/model.joblib
```

Built from `openfootball/espana` (`1-liga.txt`), `openfootball/italy`
(`1-seriea.txt`), and the `france/` folder of `openfootball/france`
(`*_fr1.txt`) — same `import_openfootball.py`, same date/team-name handling,
just a different set of season files and alias entries in `TEAM_ALIASES`.
The web demo (below) only covers the Premier League for now.

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

## Keeping the data current

openfootball/england updates its current-season file after real matchdays
are played. To pull in newer results (and refresh `data/fixtures.csv` with
what's left to play):

```bash
git clone --depth 1 https://github.com/openfootball/england /tmp/openfootball-england
python src/import_openfootball.py \
    /tmp/openfootball-england/2015-16/1-premierleague.txt \
    /tmp/openfootball-england/2016-17/1-premierleague.txt \
    ... \
    /tmp/openfootball-england/2026-27/1-premierleague.txt \
    --out data/matches.csv --fixtures-out data/fixtures.csv
python src/train.py
python src/export_web_demo_data.py && python src/build_web_page.py
```

Pass every season file oldest-first (see the `git log` on
[`import_openfootball.py`](src/import_openfootball.py) for the exact list
this repo was built from). A file the parser can't make sense of raises
`ValueError` rather than silently mis-parsing it — if openfootball changes
its layout again, that's your signal to check `parse_season_file()`.

For the other leagues, swap in the matching repo/file and `--out`/`--fixtures-out`:
`openfootball/espana` + `1-liga.txt` → `data/la-liga/`, `openfootball/italy` +
`1-seriea.txt` → `data/serie-a/`, the `france/` folder of `openfootball/france`
+ `*_fr1.txt` → `data/ligue-1/` (then `--model-out models/<league>/model.joblib`
on the `train.py` step).

Alternatively, a manually downloaded CSV from
[football-data.co.uk](https://www.football-data.co.uk/englandm.php) also
works directly as `data/matches.csv` (no import script needed) as long as it
has `Date` (`DD/MM/YYYY`), `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`
columns — but it won't give you `data/fixtures.csv` (unplayed matches),
since football-data.co.uk only publishes results.

## Model quality — read this before trusting a prediction

- **Baseline matters more than accuracy.** On the bundled data, "always
  predict home win" scores 42.2% on the (most recent, chronological) test
  split; the model itself scores 49.8%. `train.py` prints this baseline next
  to the model's accuracy every time you retrain — only the gap over
  baseline is real signal.
- Draws are structurally the hardest class to call; expect low recall there.
- This is a starting point (Elo + form + head-to-head + logistic regression),
  not a betting-grade model. Realistic next steps: add bookmaker odds as a
  feature (they're already the best public predictor), try gradient boosting
  (`GradientBoostingClassifier` / XGBoost), add player-availability/injury
  data, and validate with a rolling-season backtest instead of one train/test
  split.

## Project layout

```
data/matches.csv            played matches, Premier League (real, 2015-16 to current season)
data/fixtures.csv           Premier League's current-season not-yet-played matches
data/la-liga/                same pair of files, La Liga
data/serie-a/                same pair of files, Serie A
data/ligue-1/                same pair of files, Ligue 1
src/import_openfootball.py  converts openfootball season files (any of the 4 leagues)
src/generate_sample_data.py synthetic data generator (alternative to real data)
src/features.py             feature engineering (Elo, form, head-to-head)
src/train.py                 training + evaluation
src/predict.py               CLI to predict a single fixture
src/predict_fixtures.py      CLI to predict a fixtures.csv (next matchday by default)
src/export_web_demo_data.py  exports model + team state -> web/model_export.json
src/build_web_page.py        embeds web/model_export.json into web/index.html
web/template.html            web demo page shell (has the JSON placeholder)
web/model_export.json        exported model + team state (Premier League)
web/index.html                the runnable web demo (open directly in a browser)
models/                      trained Premier League model + metrics (git-ignored)
models/la-liga/, models/serie-a/, models/ligue-1/   same, per league (git-ignored)
```
