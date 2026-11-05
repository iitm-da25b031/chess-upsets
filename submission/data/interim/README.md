# data/interim: tables one step from the raw files (rebuilt by the notebooks, about 3.7 GB, not in the repository)

From notebook 01: `games_YYYY-MM.parquet`, `moves_YYYY-MM.parquet` (per month, raw values), `fide_<list>_YYYY-MM.parquet` (per list and month),
`all_games.parquet`, `fide_ratings.parquet`, `fide_players.parquet`.

From notebook 02: `clock_features/` and `clock_features.parquet` (time control estimated from the clocks), `clean_games.parquet` (games passing cleaning
rules 1-10), `excluded_games.parquet` and `excluded_analysis_filters.parquet` (every removed game with its reason), `analysis_derived.parquet`,
`outlier_flags.parquet`, `_log_*.csv` (parts of the cleaning log).
