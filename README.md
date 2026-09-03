# Football Prediction System

Predicts football match outcomes — **Home win / Draw / Away win** — from historical
match statistics (goals, shots, cards, recent form, head-to-head record).

> The bundled `data/matches.csv` is **real English Premier League results** —
> 4,200 matches, 2015–16 through the current (2026–27) season, converted from
> the open-source [openfootball](https://github.com/openfootball) family of
> archives (see [`src/import_openfootball.py`](src/import_openfootball.py)).
> `data/fixtures.csv` holds every match from the current season that hasn't
> been played yet — see [Predicting the next matchday](#predicting-the-next-matchday).
> Twelve more competitions ship the same way under `data/<league>/` — three
> more top flights (La Liga, Serie A, Ligue 1), four second divisions
> (Championship, Segunda, Serie B, Ligue 2), and five cups (FA Cup, EFL Cup,
> Coppa Italia, Copa del Rey, Coupe de France) — see
> [Other leagues](#other-leagues) and
> [Cup competitions](#cup-competitions--included-but-read-this-before-trusting-one).
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

Seven more leagues/divisions ship the same way, each in its own subfolder with its
own model (a Ligue 2 team's Elo has nothing to do with a Premier League
team's, so every league/division is trained separately, never merged):

| League | Folder | Source file | Accuracy vs baseline |
| --- | --- | --- | --- |
| La Liga | `data/la-liga/` | `openfootball/espana` `1-liga.txt` | 53.5% vs 46.6% |
| Serie A | `data/serie-a/` | `openfootball/italy` `1-seriea.txt` | 52.0% vs 39.0% |
| Ligue 1 | `data/ligue-1/` | `openfootball/france` `france/*_fr1.txt` | 52.1% vs 44.2% |
| Championship | `data/championship/` | `openfootball/england` `2-championship.txt` | 42.9% vs 41.4% |
| Segunda División | `data/segunda/` | `openfootball/espana` `2-liga2.txt` | 47.1% vs 45.8% |
| Serie B | `data/serie-b/` | `openfootball/italy` `2-serieb.txt` | 44.5% vs 43.7% |
| Ligue 2 | `data/ligue-2/` | `openfootball/france` `france/*_fr2.txt` | 47.9% vs 43.2% |

Second divisions are close to their baseline — expect less signal than the
top flights above (more competitive parity, fewer big-Elo-gap fixtures).

Every script takes `--data` / `--fixtures` / `--model` (and `train.py` takes
`--model-out`), so point them at a league's files instead of the
Premier League default:

```bash
python src/predict_fixtures.py --data data/la-liga/matches.csv \
    --fixtures data/la-liga/fixtures.csv --model models/la-liga/model.joblib

python src/predict.py "Palermo" "Modena" --data data/serie-b/matches.csv --model models/serie-b/model.joblib
```

**`predict_fixtures.py`'s "next matchday" is only reliable for leagues whose
source has a current-season file** (Premier League, La Liga, Serie A, Ligue 1,
Championship — all track 2026-27). Segunda, Serie B and Ligue 2 stop at
2025-26 in this archive, so their `fixtures.csv` holds whatever was unplayed
at the end of that season, not real next-week fixtures — `predict.py` (name
two teams yourself) still works fine, it just won't be "this weekend."

Built with the same `import_openfootball.py`, same date/team-name handling —
just a different season-file source and `TEAM_ALIASES` entries per
league/division. The web demo (below) only covers the Premier League for now.

### Cup competitions — included, but read this before trusting one

| Cup | Folder | Source | Seasons | Accuracy vs baseline |
| --- | --- | --- | --- | --- |
| FA Cup | `data/fa-cup/` | `openfootball/england` `facup.txt` | 2021-22 to 2024-25 | 47.1% vs 43.8% |
| EFL Cup | `data/efl-cup/` | `openfootball/england` `eflcup.txt` | 2021-22 to 2024-25 | 47.3% vs **48.4%** |
| Coppa Italia | `data/coppa-italia/` | `openfootball/italy` `cup.txt` | 2021-22 to 2024-25 | 51.4% vs **51.4%** |
| Copa del Rey | `data/copa-del-rey/` | `openfootball/espana` `cup.txt` | 2021-22 to 2024-25 | 63.9% vs **64.8%** |
| Coupe de France | `data/coupe-de-france/` | `openfootball/france` `france/*_frcup.txt` | 2024-25 only | 32.6% vs **46.5%** |

**Three of these five lose to the "always predict home win" baseline, and
none of them clearly beats it.** Two things specific to cup competitions
explain why, and neither is a bug to fix:

1. **Most opponents have thin or zero Elo history.** A cup draw pits
   top-flight sides against lower-league or non-league teams this project
   has little or no history for (this repo has no Serie C, Championship-below
   data, etc.) — the model falls back to a neutral prior for one side, which
   isn't a real prediction.
2. **Small samples.** A league season is 380 matches; a cup competition is
   ~45-200 across a whole season, so there's far less to learn from — Coupe
   de France in particular is only one season (200 matches) since that's all
   `openfootball/france` has tracked so far.

`import_openfootball.py` does parse these correctly, though — extra time and
penalty-shootout scorelines (`2-1 a.e.t. (1-1, 0-1)`, `9-8 pen. (0-0)`) are
handled by `extract_score()`, always resolving to the actual 90+30 minute
result (a shootout only happens after a draw, so that's correctly a "D" for
training, not a coin-flip "win"). `predict.py` works the same as any other
competition — no current-season file exists for any of these five yet
(so no `fixtures.csv`/`predict_fixtures.py` next-round data), but naming two
teams still gets you a live prediction against these teams' cup Elo:

```bash
python src/predict.py "Palermo" "Modena" --data data/coppa-italia/matches.csv --model models/coppa-italia/model.joblib
```

Given the numbers above, treat a cup prediction as considerably less
trustworthy than a league one — this project's most useful call on Palermo
vs Modena is still the [Serie B one](#other-leagues), since both are
genuinely Serie B sides and that model actually beats its baseline.

### Thai League — no accessible data source

I looked for a Thai League 1 results dataset via GitHub search and general
web search and didn't find one — openfootball doesn't cover Thailand, and no
public GitHub CSV mirror turned up. `football-data.co.uk` might have it, but
this environment's network policy blocks that domain entirely (see
[Keeping the data current](#keeping-the-data-current)).

To add it: get match results into a CSV with `Date` (`DD/MM/YYYY`),
`HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR` columns — from Thai League's own
site, a stats provider, or by hand for a season or two — and save it as
`data/thai-league/matches.csv`. No import script needed for a file already in
that shape; `train.py --data data/thai-league/matches.csv --model-out
models/thai-league/model.joblib` picks it up as-is.

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

### "Today's matchday" + prediction history

The page also auto-predicts the whole next round (`data/fixtures.csv`, same
window `predict_fixtures.py` uses — "Today's matchday" is really "the next
scheduled round," which may be a few days out) and a **Predict & Save**
button that logs those predictions to the page's own shared history —
visible to anyone who opens the page, growing over time as it's clicked on
new matchdays.

This part **only works on the hosted (claude.ai) copy** — saving needs
Claude's `db` capability (a small per-artifact JSON store), which a page
opened from a local file has no access to at all. `web/index.html` still
shows the same predictions, just with the save button disabled and a note
explaining why — open the hosted link above for history.

It's a log of predictions made, not (yet) a scored track record — there's no
automatic check against actual results once a match is played. A page
declaring `db` can also read game results back in, so that's a real
follow-up if you want it, not a rebuild.

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

For the other leagues, swap in the matching repo/file and `--out`/`--fixtures-out`
(then `--model-out models/<league>/model.joblib` on the `train.py` step):

| League | Repo | File |
| --- | --- | --- |
| La Liga | `openfootball/espana` | `1-liga.txt` |
| Serie A | `openfootball/italy` | `1-seriea.txt` |
| Ligue 1 | `openfootball/france` | `france/*_fr1.txt` |
| Championship | `openfootball/england` | `2-championship.txt` |
| Segunda | `openfootball/espana` | `2-liga2.txt` |
| Serie B | `openfootball/italy` | `2-serieb.txt` |
| Ligue 2 | `openfootball/france` | `france/*_fr2.txt` |
| FA Cup | `openfootball/england` | `facup.txt` |
| EFL Cup | `openfootball/england` | `eflcup.txt` |
| Coppa Italia | `openfootball/italy` | `cup.txt` |
| Copa del Rey | `openfootball/espana` | `cup.txt` |
| Coupe de France | `openfootball/france` | `france/*_frcup.txt` |

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
data/la-liga/, data/serie-a/, data/ligue-1/               same pair of files, per top-flight league
data/championship/, data/segunda/, data/serie-b/, data/ligue-2/   same, per second division
data/fa-cup/, data/efl-cup/, data/coppa-italia/,
data/copa-del-rey/, data/coupe-de-france/                 same, per cup competition (no fixtures.csv yet)
src/import_openfootball.py  converts openfootball season files (any of the 13 competitions above)
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
models/<league>/              same, per league/division above (git-ignored)
```
