# data/clean: the clean layer

`analysis_games.parquet`: the final analysis table written by notebook 02 (one row per game, key `game_url`; 90 columns; described in
`reports/data_dictionary.csv`). Notebooks 03 and 04 read it. Notebook 03 also writes its model outputs (small CSV files) here. Only `analysis_games.parquet` and the CSV files are in the repository; the other per-game parquet files are rebuilt by notebook 03.
