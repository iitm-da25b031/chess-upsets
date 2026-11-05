# data/clean: the clean layer

`analysis_games.parquet`: the final analysis table written by notebook 02 (one row per game, key `game_url`; 90 columns; described in
`reports/data_dictionary.csv`). Notebooks 03 and 04 read it. Notebook 03 also writes its model outputs (small CSV files) here. The CSV files are in the repository. The parquet files are not (`analysis_games.parquet` is 83 MB): get it from the download link in the main README, or rebuild it with notebooks 01 and 02.
