"""Are there games whose time control was estimated from the clocks with an implausible increment?
Needs data/processed/analysis_games.parquet (+ upsets_empirical.parquet for the upset count).  Usage: python check_tc_global.py
Real increments are 0-30 s (rarely up to 60 s). An estimated increment far above that means the estimate is noise."""
import os, numpy as np, pandas as pd
pd.set_option("display.width", 200)
g = pd.read_parquet("data/processed/analysis_games.parquet")
R = []
def P(*a): R.append(" ".join(str(x) for x in a))
P(f"games {len(g)}\n\n== how the time control was decided\n" + g["tc_source"].value_counts().to_string())
clk = g["tc_source"].isin(["clock_raw"])
P("\n== clock_raw games (no usable time-control text, no snap to a known pair): estimated increment in seconds")
P(g.loc[clk, "tc_inc_s"].describe(percentiles=[.5, .9, .99]).round(1).to_string())
g["implausible"] = clk & (g["tc_inc_s"] > 60)
P(f"\nclock_raw games with estimated increment > 60 s: {int(g.implausible.sum())} ({g.implausible.mean():.2%} of all games)")
P("their assigned class:\n" + g.loc[g.implausible, "tc_class"].value_counts().to_string())
P("\ntournaments with the most such games:\n" + g.loc[g.implausible].groupby("broadcast_name").size().sort_values(ascending=False).head(15).to_string())
p = "data/processed/upsets_empirical.parquet"
if os.path.exists(p):
    u = pd.read_parquet(p)[["game_url", "upset_emp_0.10"]].merge(g[["game_url", "implausible"]], on="game_url")
    n = int(u["upset_emp_0.10"].sum())
    P(f"\nalpha=0.10 upsets with an implausible estimate: {int((u['upset_emp_0.10'] & u['implausible']).sum())} of {n}")
open("check_tc_global_report.txt", "w", encoding="utf-8").write("\n".join(R)); print("\n".join(R))
