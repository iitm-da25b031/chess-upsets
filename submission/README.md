# Chess upsets over the board (DA2402, IIT Madras)

Team: consult kid + hack coord + high iq

| Member | Work |
|---|---|
| Pradish Gandhi S | Data collection, cleaning and processing |
| Mahir Ali T | Model and analysis |
| Siddharth Maira | Visualization and dashboard |

## Question

How large does the rating gap have to be before an underdog's win (or draw) counts as an upset in over-the-board chess, and does that gap depend on the time control (classical, rapid, blitz)?

## Data

About 1.26 million over-the-board games relayed on Lichess (monthly broadcast PGN files, 2020-01 to 2026-09), joined on the FIDE id to the FIDE monthly standard, rapid and blitz rating lists for the same months.

## Folder layout

| Path | Content |
|---|---|
| `notebooks/01_collection.ipynb` | downloads (resumable), robots.txt and terms evidence, manifest, PGN and XML parsing, combining, shape / dtype / meaning of every raw column |
| `notebooks/02_cleaning.ipynb` | time-control parser and clock estimate, cleaning rules with before/after counts, duplicates, ratings filled from FIDE, missing values (tests, MCAR/MAR/MNAR, imputation vs deletion), analysis filters, outliers (IQR, MAD, Isolation Forest), cleaning log, storage (format comparison, dtype round trip), data dictionary |
| `notebooks/03_model.ipynb` | why the Elo formula is not used, definition of an upset, logistic model per time control and underdog colour, critical gaps with tournament-clustered 95 % intervals, differences between time controls, held-out calibration, robustness runs (verified ratings, sensitivity, isotonic, rating x gap interaction), expected vs observed upsets, upset flags and audit |
| `notebooks/04_report.ipynb` | the report: introduction, data and methods, results (13 figures from `viz/`, critical-gap tables), conclusion, limitations, attributions, contributions. Reads only `viz/viz_data/` and `reports/` |
| `viz/` | the figure module `viz.py`, the script that builds its input tables (`make_viz_data.py`) and the static PNG copies of the figures |
| `dashboard/` | the interactive Dash app (see `dashboard/README.md`) |
| `data/raw/` | downloaded files, never edited; `MANIFEST.csv`, URL lists, `compliance/` (saved robots.txt and terms pages) |
| `data/interim/` | parsed and intermediate tables |
| `data/clean/` | the final table `analysis_games.parquet` |
| `reports/` | `cleaning_log.csv`, `data_dictionary.csv`, `dtype_roundtrip.txt`, `format_comparison.csv`, `missingness_tests.csv`, `outlier_summary.csv`, `raw_profile.csv`, `games_per_month.csv`, `model_outputs/` |
| `requirements.txt` | library versions |

## How to run

```
python -m venv .venv
.venv\Scripts\activate            (Windows)     source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
cd notebooks
jupyter lab
```

Open `01_collection.ipynb` and use Kernel > Restart & Run All, then do the same for `02_cleaning.ipynb`, `03_model.ipynb` and `04_report.ipynb`.

Headless alternative (from `notebooks/`): `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_collection.ipynb`, then the same for the other notebooks.

Each notebook starts with one configuration cell: `DATA_ROOT = os.environ.get("CHESS_DATA_ROOT", "../data")` (folders `raw/`, `interim/` and `clean/` below it) and `REPORTS` (`../reports`). Environment switches: `CHESS_DATA_ROOT` (another data folder), `CHESS_REPORTS`, `CHESS_MONTHS` (for example `2020-01,2023-01` for a quick test; default all 81 months) and `CHESS_DOWNLOAD=0` (never use the network).

Every step can be resumed: downloaded files, parsed months and clock estimates are skipped when they are up to date, so a stopped run can be restarted.

Run times, measured on a 2-CPU / 4 GB Linux VM (a normal laptop is similar or faster):

| Step | Time | Disk |
|---|---|---|
| 01: download 324 files | depends on the connection (2.6 GB; FIDE requests are 1 s apart) | `raw/` 2.6 GB |
| 01: parse 81 PGN months | about 5 to 6 min | |
| 01: parse 243 FIDE lists | about 20 to 25 min | |
| 01: manifest (SHA-256), combine, profiles | about 1 to 2 min | `interim/` 3.7 GB |
| 02: clock estimates (first run) | about 2.5 min | |
| 02: everything else | about 1.5 min | `clean/` 0.1 GB |
| 03: model, intervals, robustness runs | about 1 to 2 min (peak memory about 2.5 GB) | `clean/` +0.15 GB |

Peak memory stays below about 3.5 GB.

## Data download

The raw data (2.6 GB) is not in the repository. Two ways to get it:

1. Run notebook 01. It downloads every file from the original sources (URLs in `data/raw/urls_lichess.txt` and `urls_fide.txt`).
2. Use the team download link: https://drive.google.com/file/d/1Bll9kTp9R4b33JDkucQnjwjZtFPdsRb9/view?usp=sharing (about 790 MB: the 81 Lichess files and `clean/analysis_games.parquet`). Unzip it into `data/` so that `data/raw/lichess/` and `data/clean/` exist, then run the notebooks (finished steps are skipped). Because of the FIDE terms below, the link holds only the Lichess files and the tables derived from them. The FIDE lists are downloaded by notebook 01.

## Licences and attribution

* Lichess broadcast database (https://database.lichess.org/#broadcasts). The broadcast panel of the page states that broadcast games are released under the Creative Commons Attribution-ShareAlike 4.0 licence. The page header says database exports in general are CC0; the stricter, more specific statement is followed. Attribution: "Game data © Lichess.org contributors, CC BY-SA 4.0". Data derived from the games are shared under CC BY-SA 4.0. `lichess.org/robots.txt` disallows `/api/`, `/game/`, `/games/export/` and a few other paths, none of which is used; `database.lichess.org` publishes no robots.txt.
* FIDE rating lists (https://ratings.fide.com/download_lists.phtml). No robots.txt and no terms of use are published for the lists, and the site footer reserves all rights. The lists were downloaded once by script with a 1 s pause and are used for non-commercial coursework only. They are not redistributed.
* Saved evidence (2026-10-31): `data/raw/compliance/`, quoted in notebook 01, section 2.

## Notes on the numbers

* Notebooks 01 to 03 compute every number they show from the files on disk. The counts in the text are computed, not typed in.

## AI use

Claude (Anthropic) was used for code review and dashboard aesthetics. The team checked and ran the notebooks.
