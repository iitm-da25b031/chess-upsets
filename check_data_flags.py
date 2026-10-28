"""Three data checks suggested by the audit of the rarest upsets.   python check_data_flags.py   -> check_flags_report.txt
 1. Does the tournament NAME contradict the time-control class we assigned? (e.g. "... Open Blitz" labelled classical)
 2. Are there placeholder ratings (spikes at round numbers such as 1000) and are low ratings verifiable against FIDE?
 3. Are the rarest upsets mostly games whose ratings could not be checked against FIDE?"""
import os
import numpy as np, pandas as pd
OUT = os.path.join("data", "processed")
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
g = pd.read_parquet(os.path.join(OUT, "analysis_games.parquet"))
up = pd.read_parquet(os.path.join(OUT, "upsets_empirical.parquet"), columns=["game_url", "q_emp", "upset_emp_0.10"])
g = g.merge(up, on="game_url", how="left")

# ---- 1. name vs class ------------------------------------------------------------
nm = g["broadcast_name"].astype(str).str.lower()
kw = pd.DataFrame({"blitz": nm.str.contains("blitz|bullet"), "rapid": nm.str.contains("rapid|quick"),
                   "classical": nm.str.contains("classical|standard|classic ")})
one = kw.sum(axis=1) == 1                                   # names naming exactly one time control
g["name_class"] = np.where(one, kw.idxmax(axis=1), None)
d = g[g.name_class.notna()]
P(f"games whose tournament name names exactly one time control: {len(d)} of {len(g)}")
P("\nname says (rows) vs class we assigned (columns):"); P(pd.crosstab(d.name_class, d.tc_class).to_string())
bad = d[d.name_class != d.tc_class]
P(f"\nconflicts: {len(bad)} games ({len(bad)/max(len(d),1):.1%} of named games); by time-control source:")
P(bad.tc_source.value_counts().to_string())
P(f"conflicting games that are alpha=0.10 upsets: {int(bad['upset_emp_0.10'].fillna(False).astype(bool).sum())} of {int(g['upset_emp_0.10'].fillna(False).astype(bool).sum())}")
P("\n25 tournaments with most conflicts (name, we said, n):")
P(bad.groupby(["broadcast_name", "name_class", "tc_class"]).size().sort_values(ascending=False).head(25).to_string())

# ---- 2. placeholder ratings ---------------------------------------------------------
r = pd.concat([g["white_elo_final"], g["black_elo_final"]])
vc = r[(r >= 1000) & (r <= 1600)].round().value_counts().sort_index()
roll = vc.rolling(5, center=True, min_periods=3).median()
spk = (vc / roll).replace([np.inf], np.nan)
P("\n== rating values that occur far more often than their neighbours (possible placeholders), ratio > 3 and n >= 200")
P(vc[(spk > 3) & (vc >= 200)].to_string() or "none")
P(f"\nsides with rating exactly 1000: {int((r == 1000).sum())}")
lowside = np.where(g.white_elo_final < g.black_elo_final, g.white_elo_final, g.black_elo_final)
fid_known = np.where(g.white_elo_final < g.black_elo_final, g.white_elo_fide.notna(), g.black_elo_fide.notna())
bins = pd.cut(pd.Series(lowside), [0, 1400, 1600, 1800, 2000, 2200, 3000], right=False)
P("\nlower-rated player: share whose rating can be checked against FIDE, by rating:")
P(pd.Series(fid_known).groupby(bins.values, observed=True).agg(["size", "mean"]).rename(columns={"size": "games", "mean": "verifiable"}).round(3).to_string())

# ---- 3. rarest upsets vs verifiability ---------------------------------------------------
both_ok = g.white_elo_fide.notna() & g.black_elo_fide.notna()
u = g["upset_emp_0.10"].fillna(False).astype(bool)
top = g.nsmallest(500, "q_emp")
P(f"\nboth ratings verifiable against FIDE: all games {both_ok.mean():.1%}, alpha=0.10 upsets {both_ok[u].mean():.1%}, 500 rarest upsets {both_ok[top.index].mean():.1%}")
sh = (g.n_plies < 20)
P(f"games under 20 plies: all {sh.mean():.2%}, 500 rarest upsets {sh[top.index].mean():.1%}")
txt = "\n".join(out); open("check_flags_report.txt", "w", encoding="utf-8").write(txt); print(txt)
