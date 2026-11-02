# Chess upsets over the board (DA2402)

**Question.** How large a rating gap does it take for an upset, and does it change with the time control?

## Data
| Source | What | Licence / terms | Collected |
|---|---|---|---|
| Lichess broadcast database, https://database.lichess.org/#broadcasts | Monthly PGN files of over-the-board broadcast games, 2020-01 to 2026-09 (81 files) | CC0 (stated on the database page; saved copy in `data/raw/compliance/`) | 2026-10-24 |
| FIDE rating lists, https://ratings.fide.com/download_lists.phtml | Monthly standard, rapid and blitz lists (XML), same months (243 files) | FIDE publishes no terms of use for the lists; its footer reserves all rights. Used for non-commercial coursework only and not redistributed (`data/` is not in git). | 2026-10-24 |

Both sources were downloaded by script (`download_lichess.py`, `download_fide.py`), not by hand. Every raw file, its URL, period and download date is listed in `data/raw/MANIFEST.csv`. `robots.txt` and the terms pages were saved on 2026-10-31 in `data/raw/compliance/`: lichess.org disallows `/api/` and the game-export paths, which were not used (files come from `database.lichess.org`, which publishes no robots.txt); ratings.fide.com publishes none.

## Folders
| Folder | Contents |
|---|---|
| `data/raw/` | Downloaded files, never edited (`lichess/`, `fide/`, `compliance/`, `MANIFEST.csv`) |
| `data/interim/` | Parsed and combined tables (`all_games`, `clean_games`, FIDE tables) |
| `data/processed/` | The clean layer: final analysis table `analysis_games.parquet` (also `chess_upsets.sqlite`) and model outputs |
| `data/dashboard/` | Small files the dashboard reads |
| `reports/` | Text reports of every step, `cleaning_log.csv`, `data_dictionary.csv`, `dtype_roundtrip.txt`, `raw_profile.txt` |
| `dashboard/` | The Dash app |

Format: Parquet for the large tables (keeps dtypes; checked after reading back, see `reports/dtype_roundtrip.txt`), CSV for small result tables, SQLite for queries from the dashboard.

## Run order
1. `python download_lichess.py urls_lichess.txt`
2. `python download_fide.py`
3. `python run_all.py`
4. `python combine.py`
5. `python clock_features.py`
6. `python clean_games.py`
7. `python diag_elo.py`
8. `python missingness.py`
9. `python features.py`
10. `python empirical_upsets.py`, then `python empirical_upsets.py verified`, then `python empirical_upsets.py sens`
11. `python fit_model.py` and `python audit_and_store.py`
12. `python build_dashboard_data.py`
13. `python prep_page3.py`, `python prep_page4.py`, `python prep_page5.py`, `python prep_replay.py`
14. `python make_manifest.py` and `python make_data_docs.py`
15. `cd dashboard` then `python app.py`, and open http://127.0.0.1:8050

The dashboard reads from `dashboard/data/dashboard/` (a copy of `data/dashboard/`). Raw data is not in git.

## Libraries
requests, zstandard, pandas, numpy, pyarrow, scikit-learn, scipy, plotly, dash, dash-bootstrap-components, python-chess (data preparation only). Chess piece images: cburnett set (licence in `dashboard/chess_pieces.py`).

## AI use
Claude (Anthropic) was used for code, debugging and drafting text. Every script was run and checked by the team.

## Contributions
| Member | Contribution |
|---|---|
| | |
| | |
| | |
