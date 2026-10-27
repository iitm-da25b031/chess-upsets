"""Missingness analysis of Elo + check which FIDE list the PGN Elo follows.
Run after clean_games.py:  python missingness.py     Writes missingness_report.txt
Question: is "a player has no Elo" random (MCAR), explained by things we observe (MAR), or tied to
the missing value itself (MNAR, e.g. unrated = weaker/younger)? We cannot prove MNAR from the data,
but we can show missingness is NOT random, and describe who is missing."""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = os.path.join("data", "interim")
rng = np.random.default_rng(42)
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

g = pd.read_parquet(os.path.join(D, "clean_games.parquet"))
g["year"] = pd.to_numeric(g["period"].astype(str).str[:4], errors="coerce")   # game month, falls back to the file month
g = g[g["year"] >= 2019].reset_index(drop=True)
g["missing"] = ~g["analysis_ok"]
g["is_draw"] = (g["score_white"] == 0.5).astype(float)
P(f"games: {len(g)}   missing at least one Elo: {int(g.missing.sum())} ({g.missing.mean():.1%})")

# ---- 1. who is missing: rates by group --------------------------------------------
P("\n== missing rate by year");  P(g.groupby("year")["missing"].agg(["size", "mean"]).round(3).to_string())
P("\n== missing rate by time-control class"); P(g.groupby("tc_class")["missing"].agg(["size", "mean"]).round(3).to_string())
b = g.groupby("broadcast_name")["missing"].agg(["size", "mean"])
b = b[b["size"] >= 100]
P(f"\n== broadcasts with >=100 games: {len(b)}; missing rate quartiles (0/25/50/75/100%): {b['mean'].quantile([0,.25,.5,.75,1]).round(3).tolist()}")
P("   share of ALL missing games sitting in the 50 worst broadcasts:",
  f"{g.loc[g.missing & g.broadcast_name.isin(b.sort_values('mean', ascending=False).head(50).index)].shape[0] / max(g.missing.sum(), 1):.1%}")
# within a broadcast, is missingness all-or-nothing? (unrated players vs missing field in the whole event)
within = g.groupby("broadcast_name")["missing"].mean()
P("   broadcasts where >90% of games are missing:", int((within > .9).sum()), "| where 10-90% (mixed events):", int(((within > .1) & (within <= .9)).sum()))

# ---- 2. permutation tests: do games with a missing Elo differ in things we observe? ----
def perm_test(x, m, reps=1000):
    x = x.to_numpy(float); m = m.to_numpy(bool)
    obs = x[m].mean() - x[~m].mean()
    cnt = 0
    for _ in range(reps):
        pm = rng.permutation(m)
        cnt += abs(x[pm].mean() - x[~pm].mean()) >= abs(obs)
    return obs, (cnt + 1) / (reps + 1)
P("\n== permutation tests (1000 shuffles of the missing flag; difference = missing minus observed)")
for col, label in (("n_plies", "game length (plies)"), ("is_draw", "draw rate"), ("score_white", "White score")):
    d, p = perm_test(g[col], g["missing"])
    P(f"   {label:22s} diff = {d:+.4f}   p = {p:.4f}")
# one non-random-looking signal from the PGN itself: does the opponent have an Elo?
one = g[g.white_elo_source.eq("missing") ^ g.black_elo_source.eq("missing")]
both = g[g.white_elo_source.eq("missing") & g.black_elo_source.eq("missing")]
P(f"   exactly one player missing: {len(one)}   both missing: {len(both)}   (if independent, 'both' would be ~{g.missing.mean()**2*len(g):.0f})")
fide_known = {s: g[f"{s}_fide_id"].notna() for s in ("white", "black")}
P("   missing-Elo sides that still have a FIDE id (rated at some time, but not in that list/month):",
  int(((g.white_elo_source == 'missing') & fide_known['white']).sum() + ((g.black_elo_source == 'missing') & fide_known['black']).sum()),
  "of", int((g.white_elo_source == 'missing').sum() + (g.black_elo_source == 'missing').sum()))

# ---- 3. logistic model: which observed variables predict missingness? -------------------
s = g.sample(min(300_000, len(g)), random_state=1)
X = pd.get_dummies(s[["year", "tc_class"]].astype(str), drop_first=True).astype(float)
X["plies_z"] = (s.n_plies - s.n_plies.mean()) / s.n_plies.std()
X["is_draw"] = s["is_draw"]; X["has_clock"] = s["tc_source"].isin(["clock_raw", "clock_snapped", "clock_override"]).astype(float)
y = s["missing"].astype(int)
lr = LogisticRegression(max_iter=1000).fit(X, y)
auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
P(f"\n== logistic model of 'a player has no Elo' on {len(s)} games: AUC = {auc:.3f}  (0.5 = no pattern, 1 = perfectly predictable)")
P("   odds ratios (>1 = more likely missing):")
P(pd.Series(np.exp(lr.coef_[0]), index=X.columns).round(2).sort_values(ascending=False).to_string())
P("   reading: an AUC well above 0.5 rules out MCAR. Missingness depends on observed variables (year, event), i.e. at least MAR.")
P("   Whether it also depends on the missing Elo itself (MNAR) cannot be tested; external evidence says unrated players are mostly juniors/amateurs, so they are weaker on average.")

# ---- 4. which FIDE list does the PGN Elo follow when it disagrees with our list? ----------
fr = pd.read_parquet(os.path.join(D, "fide_ratings.parquet"), columns=["fideid", "period", "standard_rating", "rapid_rating", "blitz_rating"])
fr["fideid"], fr["period"] = fr["fideid"].astype(str), fr["period"].astype(str)
fr = fr.drop_duplicates(["fideid", "period"]).set_index(["fideid", "period"])
P("\n== PGN Elo vs FIDE lists: which list does a mismatching PGN Elo equal (within 25)?")
res = []
for side in ("white", "black"):
    mm = g[f"{side}_elo_mismatch"] & (g[f"{side}_elo_source"] == "pgn")
    d = g[mm]
    got = fr.reindex(pd.MultiIndex.from_arrays([d[f"{side}_fide_id"].astype(object).astype(str), d["period"].astype(str)]))
    pg = d[f"{side}_elo_final"].to_numpy(float)
    for lst in ("standard", "rapid", "blitz"):
        hit = (got[f"{lst}_rating"].to_numpy(float) - pg).__abs__() <= 25
        res.append(pd.DataFrame({"tc_class": d["tc_class"].to_numpy(), "list": lst, "hit": hit}))
r = pd.concat(res)
P(r.groupby(["tc_class", "list"])["hit"].mean().unstack().round(3).to_string())
P("   reading: if rapid/blitz games mostly hit 'standard', those organisers published the standard rating. Then the PGN Elo is a standard rating on a rapid game.")
txt = "\n".join(out); open("missingness_report.txt", "w", encoding="utf-8").write(txt); print(txt)
