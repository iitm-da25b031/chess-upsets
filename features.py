"""Stage 3: features, Elo-Davidson model, upset flags, first upset curve.
Run after clean_games.py:   python features.py
Reads  data/interim/clean_games.parquet
Writes data/processed/analysis_games.parquet, kappa_table.csv, upset_curve.csv, features_report.txt

PRE-REGISTERED CHOICES (fixed before looking at any upset result; change only with a note in the report):
  * near-equal games for kappa: rating gap <= 25
  * rating bands (mean of the two Elos): <2200, 2200-2499, >=2500
  * kappa cell needs >= 500 near-equal games, otherwise falls back to the time-control class value
  * upset threshold alpha: default 0.10, reported for 0.20 / 0.10 / 0.05 / 0.01
  * gap bins: 50 points wide, last bin 600+
"""
import os
import numpy as np, pandas as pd

D = os.path.join("data", "interim")
OUT = os.path.join("data", "processed"); os.makedirs(OUT, exist_ok=True)
NEAR, MIN_CELL, ALPHAS, BIN = 25, 500, (0.20, 0.10, 0.05, 0.01), 50
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

g = pd.read_parquet(os.path.join(D, "clean_games.parquet"))
log = [("start (clean games)", len(g))]
def keep(mask, name):
    global g
    g = g[mask].copy(); log.append((name, len(g)))
keep(g["game_date"].dt.year >= 2019, "drop games dated before 2019 (date artefacts)")
keep(g["analysis_ok"], "both Elos known")
keep(g["tc_class"] != "unknown", "time control known")
g = g.reset_index(drop=True)

# ---- basic features ---------------------------------------------------------------
we, be = g["white_elo_final"].astype(float), g["black_elo_final"].astype(float)
g["gap"] = (we - be).abs()
g["avg_elo"] = (we + be) / 2
g["has_underdog"] = g["gap"] > 0
g["underdog_is_white"] = we < be
g["fav_is_white"] = we > be
sw = g["score_white"].astype(float)
g["underdog_score"] = np.where(g["underdog_is_white"], sw, 1 - sw)       # 1 win, .5 draw, 0 loss
g["underdog_color"] = np.where(g["underdog_is_white"], "white", "black")
g["band"] = pd.cut(g["avg_elo"], [0, 2200, 2500, 5000], right=False, labels=["<2200", "2200-2499", "2500+"]).astype(str)
g["gap_bin"] = (np.minimum(g["gap"], 600) // BIN * BIN).astype(int)          # lower edge, 600 = "600+"
g["elo_mismatch_any"] = g.get("white_elo_mismatch", False) | g.get("black_elo_mismatch", False)

# ---- kappa (Davidson draw parameter) per time-control class and rating band ----------
near = g[g["gap"] <= NEAR]
def kappa_from(df):
    p = (df["score_white"] == 0.5).mean()
    return 2 * p / (1 - p), p, len(df)
k_class = {c: kappa_from(d) for c, d in near.groupby("tc_class")}
rows = []
for (c, b), d in near.groupby(["tc_class", "band"]):
    k = kappa_from(d) if len(d) >= MIN_CELL else k_class[c]
    rows.append(dict(tc_class=c, band=b, n_near_equal=len(d), draw_rate_equal=k[1], kappa=k[0],
                     fallback=len(d) < MIN_CELL))
ktab = pd.DataFrame(rows)
ktab.to_csv(os.path.join(OUT, "kappa_table.csv"), index=False)
g = g.merge(ktab[["tc_class", "band", "kappa"]], on=["tc_class", "band"], how="left")
g["kappa"] = g["kappa"].fillna(g["tc_class"].map({c: v[0] for c, v in k_class.items()}))

# ---- Elo-Davidson probabilities for the underdog ------------------------------------
# v = rating difference from the underdog's side = -gap
a, b = 10 ** (-g["gap"] / 800), 10 ** (g["gap"] / 800)
Dn = a + b + g["kappa"]
g["p_win_u"], g["p_draw"], g["p_loss_u"] = a / Dn, g["kappa"] / Dn, b / Dn
g["exp_score_u"] = g["p_win_u"] + 0.5 * g["p_draw"]
# q = probability of a result at least as good as the one that happened (for the underdog)
g["q"] = np.select([g["underdog_score"] == 1, g["underdog_score"] == 0.5], [g["p_win_u"], g["p_win_u"] + g["p_draw"]], 1.0)
for al in ALPHAS:
    g[f"upset_{al:.2f}"] = g["has_underdog"] & (g["underdog_score"] > 0) & (g["q"] < al)
# second definition: score-surprise against the plain Elo expectation
g["exp_score_elo"] = 1 / (1 + 10 ** (g["gap"] / 400))
g["surprise"] = g["underdog_score"] - g["exp_score_elo"]
g.to_parquet(os.path.join(OUT, "analysis_games.parquet"), index=False)

# ---- first look: observed vs model by time control and gap bin ----------------------
u = g[g["has_underdog"]]
def summ(df):
    n = len(df)
    return pd.Series(dict(n=n, u_win=(df.underdog_score == 1).mean(), u_draw=(df.underdog_score == .5).mean(),
        u_loss=(df.underdog_score == 0).mean(), u_score=df.underdog_score.mean(), model_score=df.exp_score_u.mean(),
        model_win=df.p_win_u.mean(), model_draw=df.p_draw.mean(), elo_score=df.exp_score_elo.mean(),
        upset_10=df["upset_0.10"].mean()))
curve = u.groupby(["tc_class", "gap_bin"]).apply(summ, include_groups=False).reset_index()
curve["score_residual"] = curve["u_score"] - curve["model_score"]
curve.to_csv(os.path.join(OUT, "upset_curve.csv"), index=False)

P("== exclusions"); [P(f"{n:55s} {m:>9d}") for n, m in log]
P(f"\nanalysis games: {len(g)}   with an underdog (gap>0): {len(u)}   exact ties: {int((~g.has_underdog).sum())}")
P("\n== kappa table (draw parameter)"); P(ktab.round(3).to_string(index=False))
P("\n== games by class"); P(g["tc_class"].value_counts().to_string())
P("\n== upset counts (underdog wins or draws with q < alpha)")
for al in ALPHAS:
    f = g[f"upset_{al:.2f}"]
    P(f"alpha={al:.2f}: " + ", ".join(f"{c} {int(f[g.tc_class == c].sum())} ({f[g.tc_class == c].mean():.2%})" for c in ("classical", "rapid", "blitz")))
P("\n== overall calibration (underdog, all gaps): observed vs model score by class")
cal = u.groupby("tc_class").agg(n=("q", "size"), observed=("underdog_score", "mean"), model=("exp_score_u", "mean"), plain_elo=("exp_score_elo", "mean"))
P(cal.round(4).to_string())
P("\n== by gap bin (all classes pooled)")
P(u.groupby("gap_bin").apply(summ, include_groups=False)[["n", "u_win", "u_draw", "u_score", "model_score", "elo_score", "upset_10"]].round(4).to_string())
P("\n== favourite's colour: underdog score when underdog has White vs Black, gap 50-199")
m = u[(u.gap >= 50) & (u.gap < 200)]
P(m.groupby(["tc_class", "underdog_color"])["underdog_score"].agg(["size", "mean"]).round(4).to_string())
P("\nRows with gap_bin n < 100 are unreliable; every figure must show n.")
txt = "\n".join(out); open("features_report.txt", "w", encoding="utf-8").write(txt); print(txt)
