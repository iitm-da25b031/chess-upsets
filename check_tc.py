"""Shows how the time control of one tournament was decided.   Usage:  python check_tc.py "Mexican World Cup"
Needs data/processed/analysis_games.parquet. Prints, for every tournament whose name contains the text:
games per tc_class, the raw time_control string(s) from the PGN, the base/increment used, how it was decided (tc_source)."""
import sys, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
key = sys.argv[1]
g = pd.read_parquet("data/processed/analysis_games.parquet")
d = g[g["broadcast_name"].astype(str).str.contains(key, case=False, regex=False)]
print(f"{len(d)} games in {d['broadcast_name'].nunique()} tournament(s) matching '{key}'\n")
cols = [c for c in ["broadcast_name", "tc_class", "time_control", "tc_base_s", "tc_inc_s", "tc_source"] if c in d.columns]
print(d.groupby(cols, dropna=False).size().rename("games").reset_index().to_string(index=False))
if "n_plies" in d.columns:
    print("\nmedian plies by tc_class:\n", d.groupby("tc_class")["n_plies"].median().to_string())
