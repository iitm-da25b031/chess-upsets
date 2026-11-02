"""
Data preparation for dashboard page 5 ("Who and where"): funnel plots for events and players.
Run after build_dashboard_data.py.   Reads data/processed/analysis_games.parquet and data/dashboard/upsets.parquet.
Writes data/dashboard/players_by_class.csv   key, name, tc_class, n (games as underdog), upsets, mean_elo   (players with >= 10 underdog games in a class)
(events come from data/dashboard/tournaments.csv, written by build_dashboard_data.py)

Players are identified by FIDE id when present, otherwise by name. Only public tournament records are used:
name, rating and results. No other personal attribute is stored.
"""
import os
import numpy as np, pandas as pd

D, P = os.path.join("data", "dashboard"), os.path.join("data", "processed")
g = pd.read_parquet(os.path.join(P, "analysis_games.parquet"))
g = g[g["has_underdog"]].copy()
u = pd.read_parquet(os.path.join(D, "upsets.parquet"))
u["is_upset"] = np.where(u["verified"], u["upset_verified"], u["upset_all"])
g["is_upset"] = g["game_url"].isin(set(u.loc[u["is_upset"], "game_url"]))
w = g["underdog_color"].astype(str).str.lower() == "white"
g["name"] = np.where(w, g["white"], g["black"])
g["fide"] = np.where(w, g["white_fide_id"], g["black_fide_id"])
g["key"] = pd.Series(g["fide"], index=g.index).astype("string").fillna(g["name"].astype("string"))
g["own_elo"] = np.where(w, g["white_elo_final"], g["black_elo_final"])
names = g.groupby("key")["name"].agg(lambda s: s.mode().iloc[0])
t = (g.groupby(["key", "tc_class"]).agg(n=("is_upset", "size"), upsets=("is_upset", "sum"), mean_elo=("own_elo", "mean")).reset_index())
t = t[t["n"] >= 10]
t["name"] = t["key"].map(names)
t["mean_elo"] = t["mean_elo"].round(0)
t[["key", "name", "tc_class", "n", "upsets", "mean_elo"]].to_csv(os.path.join(D, "players_by_class.csv"), index=False)
print(f"{len(t)} player x class rows ({t['key'].nunique()} players) -> players_by_class.csv")