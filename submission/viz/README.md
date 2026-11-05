# submission/viz: figures and tables of the results section

Plot layer for the fresh pipeline (notebooks 01-03; 863,971 analysis games). Two steps:

1. `make_viz_data.py` reads the big tables of the pipeline and writes small CSV tables to `viz_data/` (about 3 MB) and a check log.
2. `viz.py` (pandas + plotly only) builds every figure and table from `viz_data/` plus the small model outputs in
   `../reports/model_outputs/`. Every `fig_*` function returns `(fig, caption)`; all numbers in captions and titles are computed
   from the tables.

## 1. Regenerate the tables

Needs pandas + pyarrow (the project venv). Run from `submission/viz/`:

```
# data root = the folder holding clean/ and interim/ written by notebooks 02 and 03 (default: ../data, i.e. submission/data)
python make_viz_data.py
CHESS_DATA_ROOT=C:\path\to\data python make_viz_data.py        # Windows: set CHESS_DATA_ROOT=... first
```

Inputs: `clean/analysis_games.parquet`, `clean/upsets_empirical.parquet`, `clean/upsets_empirical_interaction.parquet`, `clean/upset_audit_top.csv`,
`clean/model_coefficients_empirical(_interaction).csv`, `clean/critical_gap_empirical(_interaction).csv`; for cross-checks only
`clean/kappa_table.csv`, `clean/expected_upsets(_interaction).csv`, `clean/model_fit_by_band.csv`, `clean/ordering_check.csv`;
`interim/clean_games.parquet`, `interim/excluded_analysis_filters.parquet`. Optional `CHESS_VIZ_DATA` = output folder (default `viz_data/`).
Runtime about 1 minute, peak memory about 2.1 GB. Every sanity check is printed and stored in `viz_data/checks.csv` and
`viz_data/make_viz_data_log.txt` (expected, actual, and the value on the old 858,123-game table); the last line says how many mismatched
(it must be 0).

## 2. Build the figures

```python
import sys; sys.path.insert(0, "submission/viz")     # or run from submission/viz
import viz
T = viz.load2()                                      # viz_data/ + ../reports/model_outputs (override: CHESS_MODEL_OUTPUTS)
fig, cap = viz.fig_heatmap(T); fig.show()            # in a notebook: fig.update_layout(width=1300) for the 3-panel figures
df, md = viz.table_robustness(T)
```

`python viz.py` (or `viz.render_all2()`) writes `renders/<fig_name>.html` (plotly.js from the CDN), `renders/<fig_name>.md` (caption),
`renders/captions.md` (all captions) and `renders/tables.md` (all tables). PNGs (colour + greyscale) need a headless Chromium:
`pip install playwright pillow && playwright install chromium`, then `python render_png.py` (all figures) or
`python render_png.py fig_simpson` (some). Kaleido is not needed. The PNGs in `renders/` were made this way (1300 px wide, scale 1.5).

## 3. Functions

Observed data (`load()` tables are enough):

| function | question it answers |
|---|---|
| `fig_gap_distribution` | How far apart in rating are the two players in a typical game, per time control? |
| `fig_outcomes_by_gap` | As the gap grows, how often does the underdog win, draw and lose, per time control? |
| `fig_heatmap` | Does the underdog's chance depend on the rating level as well as on the gap? |
| `fig_simpson` | Does one pooled curve hide subgroups (underdog colour, rating band) that behave differently? (`tc=` rapid / blitz too) |
| `fig_upset_points` | Where in (gap, rating level) do the upsets at alpha 0.10 happen, and which 20 are the rarest? |
| `fig_set_compare` | Whom did the analysis filters remove, and does the analysis set still look like the larger cleaned set? |
| `fig_games_per_year` | How much data does each year and time control contribute? |

Model and upset rule (`load2()`):

| function | question it answers |
|---|---|
| `fig_model_vs_data` | Does the fitted model describe the games it is applied to, bin by bin (like-for-like prediction)? |
| `fig_verified_split` | Why did the old notebook plot show observed rates far above the curve? (its dots were the unverified-rating games only) |
| `fig_critical_gap_alpha` | Does the ordering classical < rapid < blitz depend on alpha, and where does the rule rest on extrapolation? |
| `fig_upset_rule` | What is the critical gap for any average rating, not only the three band reference ratings? |
| `fig_calibration_check` | On tournaments the model never saw, do predicted and observed underdog win rates agree? |
| `fig_model_misfit` | Is the model's tail right for strong players, does the rating x gap interaction model fix it, and how much do the critical gaps move at alpha 0.20 / 0.10 / 0.05? |
| `table_critical_gaps` | Critical gaps at alpha 0.10 with 95% intervals and whether each lies inside the observed data |
| `table_differences` | Time-control differences of the critical gap (median of the draws and 95% interval) |
| `table_robustness` | Critical gaps (2200-2499, alpha 0.10) in the main / verified / sensitivity / isotonic / interaction runs |
| `table_model_misfit` | Observed / expected underdog wins in the tail by band, main vs interaction model, plus fit by band (returns a third element: note) |
| `table_alpha005_caveat` | Main vs interaction critical gaps at alpha 0.20 / 0.10 / 0.05 and a summary of the change (third element) |
| `table_expected_upsets` | Expected (calibrated model) vs observed upsets, overall or `by_band=True` |
| `model_fit_stats`, `misfit_ratios`, `calibration_stats`, `ordering_check` | helper tables behind the captions |

Conventions: time-control order classical / rapid / blitz; HI (green) = the point of the plot, grey = context; bins / cells with fewer
than 100 games are dropped or hatched; underdog colour and model are encoded by marker shape and grey level so the figures survive
greyscale printing.

## 4. Short captions (rubric: two to three sentences under each plot)

`viz.short_captions(T)` (T = `load2()`; with `load()` only the data-layer figures) returns `{name: text}` in report order (`viz.REPORT_ORDER`, 13 figures + 6 tables): at most three sentences each (what the plot shows, the key finding with numbers, one caveat), every number computed from the tables like the long captions.
`viz.captions_all(T)` returns `{name: {"short": ..., "details": ...}}` in the same order; "details" is the unchanged long caption (figures) or the markdown table plus note (tables).
`viz.write_short_captions()` writes `renders/captions_short.md`; the long captions, titles and `render_all2()` outputs are unchanged.
