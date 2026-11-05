# Upset Explorer (dashboard)

Run from this folder:

    pip install -r requirements.txt
    python app.py            # then open http://127.0.0.1:8050

Pages:

* Is it an upset?: enter two ratings, a time control and a threshold, and see the underdog's win chance and the critical gap.
* The data: games over time, before and after cleaning, colour and rating comparisons, observed outcomes by gap, and a win-rate heatmap.
* Can it be trusted?: prediction check on held-out tournaments, accuracy by rating level, and the effect of checking ratings against FIDE. This page uses a fixed 10 % upset threshold and starts with a Black underdog.
* The upsets: opening explorer (first four moves), game length, and a board replay of the rarest upsets.
* Federations: upset rate or count by the underdog's FIDE federation.

Folder layout:

* `app.py`: app shell and sidebar
* `common.py`: colours and helpers shared by the pages
* `viewpage.py` and `viz.py`: figures and captions shared with the report notebook (`viz.py` is identical to `../viz/viz.py`)
* `pages/`: one file per page
* `archive/pages/`: earlier pages (full data page, finding, model checks, who and where), kept for reference and not loaded by the app
* `assets/style.css`: styling
* `chess_pieces.py`: board artwork
* `data/`: the tables the app reads
* `screenshots/`: screenshots of the pages, shown in the report

Data refresh: `python refresh_data.py` copies the model tables from `../reports/model_outputs/` and `../viz/viz_data/`, then runs `prepare_fresh.py`. That script rebuilds `funnel.csv`, `before_after.csv`, `missing_by_year.csv`, `missing_tests.csv`, `length_bins.csv`, `tournaments.csv`, `players_by_class.csv` and `federations.csv` in `data/dashboard/` from the outputs of notebooks 01 to 03 (`CHESS_DATA_ROOT`, default `../data`). Run notebooks 01 to 03 and `python ../viz/make_viz_data.py` first.

The opening tables (`opening_lines.csv`, `opening_boards.csv`) and the replay database (`upsets.sqlite`) are included ready-made. They were built once from the clean tables with python-chess and are not rebuilt by `refresh_data.py`.

Data: Lichess broadcast games, CC BY-SA 4.0 (database.lichess.org). Ratings: FIDE rating lists, used for coursework only.
