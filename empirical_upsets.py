"""Stage 5: model-free upset curves, critical gaps with confidence intervals, and the final upset flags.
Run after features.py:   python empirical_upsets.py            (main analysis)
                         python empirical_upsets.py verified    (only games whose two ratings are both confirmed by the FIDE list)
                         python empirical_upsets.py sens        (sensitivity: drops games whose PGN Elo disagrees with FIDE
                                                                 by >100 and games whose time control is near a class boundary)
WHY: the fitted Davidson model gets the underdog's SCORE right but not its win/draw split at big gaps (it gives too few
wins and too many draws in classical). So the final answer does not lean on a draw model. For each time-control class
and each underdog colour we model directly, with logistic regression on the games themselves:
      P(underdog wins)           and          P(underdog does not lose)
as functions of  gap, gap^2, gap^3  and the players' average Elo.  Fitted on one half of the TOURNAMENTS, checked on the
other half (so the check is on events the model has never seen), then refitted on everything.
Confidence intervals are cluster-robust (games from one tournament are not independent).
UPSET (final definition): underdog wins and P(win) < alpha, or underdog draws and P(not lose) < alpha, i.e. the result was
rarer than alpha for a game with that gap, class, colour and strength level.
CRITICAL GAP: the smallest rating gap where P(underdog wins) falls below alpha, with a 95% interval from the
cluster-robust covariance (1000 draws). Also given: the same for the underdog's share of DECISIVE games (draws removed),
and a no-model cross-check by isotonic regression. Differences between classes get an interval too.
Writes data/processed/critical_gap_empirical[_sens].csv, calibration_empirical[_sens].csv, upsets_empirical[_sens].parquet,
empirical_report[_sens].txt"""
import os, sys
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
SENS, VER = MODE == "sens", MODE == "verified"
SUF = "_sens" if SENS else ("_verified" if VER else "")
OUT = os.path.join("data", "processed")
ALPHAS, GRID, REF_ELO, CAP = (0.20, 0.10, 0.05, 0.01), np.arange(0, 1505, 5), {"<2200": 2100, "2200-2499": 2350, "2500+": 2600}, 900
rng = np.random.default_rng(7)
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

g = pd.read_parquet(os.path.join(OUT, "analysis_games.parquet"))
g = g[g["has_underdog"]].copy()
if SENS:
    n0 = len(g)
    g = g[~g["elo_mismatch_any"].fillna(False) & ~g["tc_borderline"].fillna(False)].copy()
    P(f"SENSITIVITY RUN: dropped {n0 - len(g)} games (Elo mismatch >100 or borderline time control); {len(g)} left")
if VER:
    n0 = len(g)
    ok = lambda s: g[f"{s}_elo_fide"].notna() & ((g[f"{s}_elo_final"] - g[f"{s}_elo_fide"]).abs() <= 100)
    g = g[ok("white") & ok("black")].copy()
    P(f"VERIFIED RUN: keeping only games where BOTH ratings are confirmed by the FIDE list (within 100 points): {len(g)} of {n0} games")
# cluster = the tournament (broadcast_name). NOT broadcast_url: that is one URL per ROUND (about 12 games), so rounds of the
# same tournament would land in both halves and look "unseen". Series that repeat yearly are told apart by the year.
g["cluster"] = g["broadcast_name"].astype(str) + "|" + g["period"].astype(str).str[:4]
g["cl_code"] = pd.factorize(g["cluster"])[0]
g["train"] = pd.util.hash_array(g["cluster"].to_numpy(dtype=object)) % 2 == 0      # half of the tournaments
P(f"tournaments (clusters): {g['cl_code'].nunique()}; games in training half: {int(g.train.sum())}, in held-out half: {int((~g.train).sum())}")
g["win"] = (g["underdog_score"] == 1).astype(int)
g["notlose"] = (g["underdog_score"] > 0).astype(int)

def design(gap, avg):
    x = np.minimum(np.asarray(gap, float), CAP) / 300
    return np.column_stack([x, x ** 2, x ** 3, (np.asarray(avg, float) - 2300) / 100])

def fit_logit(X, y, cl=None):
    m = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    th = np.r_[m.intercept_, m.coef_[0]]
    if cl is None:
        return th, None
    p = m.predict_proba(X)[:, 1]
    Xd = np.column_stack([np.ones(len(X)), X])
    Ainv = np.linalg.inv((Xd * (p * (1 - p))[:, None]).T @ Xd)
    S = pd.DataFrame(Xd * (y - p)[:, None]).groupby(cl).sum().to_numpy()      # one score vector per tournament
    G = len(S)
    return th, Ainv @ (S.T @ S) @ Ainv * G / max(G - 1, 1)                     # cluster-robust (sandwich) covariance

def predict(th, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ th)))

classes, colours = ("classical", "rapid", "blitz"), ("white", "black")
models = {}                       # (class, colour, outcome, which) -> (theta, cov)
for c in classes:
    for col in colours:
        sub = g[(g.tc_class == c) & (g.underdog_color == col)]
        for which, d in (("train", sub[sub.train]), ("all", sub)):
            X = design(d.gap, d.avg_elo)
            for oc in ("win", "notlose"):
                models[(c, col, oc, which)] = fit_logit(X, d[oc].to_numpy(), d["cl_code"].to_numpy() if which == "all" else None)

# ---- out-of-sample calibration -----------------------------------------------------------
te = g[~g.train].copy()
te["p_win"] = np.nan; te["p_nl"] = np.nan
for c in classes:
    for col in colours:
        m = ((te.tc_class == c) & (te.underdog_color == col)).to_numpy()
        X = design(te.gap[m], te.avg_elo[m])
        te.loc[m, "p_win"] = predict(models[(c, col, "win", "train")][0], X)
        te.loc[m, "p_nl"] = predict(models[(c, col, "notlose", "train")][0], X)
te["gap100"] = (np.minimum(te.gap, 700) // 100 * 100).astype(int)
cal = (te.groupby(["tc_class", "underdog_color", "gap100"])
         .agg(n=("win", "size"), obs_win=("win", "mean"), pred_win=("p_win", "mean"),
              obs_notlose=("notlose", "mean"), pred_notlose=("p_nl", "mean")).reset_index())
cal.to_csv(os.path.join(OUT, f"calibration_empirical{SUF}.csv"), index=False)
P("== held-out calibration (tournaments not used for fitting): observed vs predicted, underdog win and not-lose")
P(cal.round(4).to_string(index=False))
w = lambda a, b: float(np.average(np.abs(cal[a] - cal[b]), weights=cal["n"]))
P(f"\nweighted mean |observed - predicted|: win {w('obs_win','pred_win'):.4f}   not-lose {w('obs_notlose','pred_notlose'):.4f}")

# ---- critical gaps with intervals --------------------------------------------------------
B = 1000
draws = {}                        # (class, colour, band, alpha) -> array of critical gaps
rows = []
for c in classes:
    for col in colours:
        th, cov = models[(c, col, "win", "all")]
        samp = rng.multivariate_normal(th, cov, B)
        for band, e in REF_ELO.items():
            Xg = design(GRID, np.full(len(GRID), e))
            Z = np.column_stack([np.ones(len(GRID)), Xg])
            pg = 1 / (1 + np.exp(-(samp @ Z.T)))                      # B x grid
            p0 = predict(th, Xg)
            for al in ALPHAS:
                def first_cross(p):
                    below = p < al
                    return np.where(below.any(axis=-1), GRID[below.argmax(axis=-1)], np.nan)
                cg = first_cross(pg); draws[(c, col, band, al)] = cg
                rows.append(dict(tc_class=c, underdog_colour=col, band=band, ref_avg_elo=e, alpha=al,
                                 critical_gap=float(first_cross(p0)),
                                 lo95=float(np.nanpercentile(cg, 2.5)) if np.isfinite(cg).any() else np.nan,
                                 hi95=float(np.nanpercentile(cg, 97.5)) if np.isfinite(cg).any() else np.nan))
crit = pd.DataFrame(rows)
crit.to_csv(os.path.join(OUT, f"critical_gap_empirical{SUF}.csv"), index=False)
P("\n== critical gap: smallest rating gap where P(underdog wins) < alpha  (95% interval); band = strength level of the players")
for al in ALPHAS[1:]:
    P(f"\n  alpha = {al}")
    t = crit[crit.alpha == al].copy()
    t["gap (95% CI)"] = t.apply(lambda r: f"{r.critical_gap:.0f} ({r.lo95:.0f}-{r.hi95:.0f})", axis=1)
    P(t.pivot_table(index=["tc_class", "band"], columns="underdog_colour", values="gap (95% CI)", aggfunc="first").to_string())
P("\n== does it change with time control? difference in critical gap (band 2200-2499, alpha=0.10); interval from the same draws")
for col in colours:
    for a_, b_ in (("classical", "rapid"), ("classical", "blitz"), ("rapid", "blitz")):
        d = draws[(a_, col, "2200-2499", 0.10)] - draws[(b_, col, "2200-2499", 0.10)]
        d = d[np.isfinite(d)]
        P(f"   {col:5s} underdog: {a_} minus {b_}: {np.median(d):+.0f}  (95% {np.percentile(d,2.5):+.0f} to {np.percentile(d,97.5):+.0f})")
P("   (nan in the tables above = the curve does not fall below alpha within a 900-point gap; there is too little data beyond that)")
P("   an interval that excludes 0 means the critical gap really differs between the two time controls.")


# ---- critical gap on DECISIVE games only (draws removed) ------------------------------------
P("\n== critical gap on decisive games: smallest gap where the underdog wins less than alpha of the games that are NOT drawn")
P("   (removes the effect of different draw rates; band 2200-2499, avg Elo 2350; interval from independent draws of the two fits)")
dres = []
for c in classes:
    for col in colours:
        thw, cw = models[(c, col, "win", "all")]; thn, cn = models[(c, col, "notlose", "all")]
        sw_, sn_ = rng.multivariate_normal(thw, cw, B), rng.multivariate_normal(thn, cn, B)
        Z = np.column_stack([np.ones(len(GRID)), design(GRID, np.full(len(GRID), 2350))])
        pw_, pn_ = 1 / (1 + np.exp(-(sw_ @ Z.T))), 1 / (1 + np.exp(-(sn_ @ Z.T)))
        pn_ = np.maximum(pn_, pw_)
        share = pw_ / np.clip(pw_ + 1 - pn_, 1e-9, None)
        p0w, p0n = predict(thw, design(GRID, np.full(len(GRID), 2350))), predict(thn, design(GRID, np.full(len(GRID), 2350)))
        sh0 = p0w / (p0w + 1 - np.maximum(p0n, p0w))
        for al in ALPHAS[:3]:
            f = lambda a_: np.where((a_ < al).any(axis=-1), GRID[(a_ < al).argmax(axis=-1)], np.nan)
            cg = f(share); pt = float(f(sh0))
            dres.append(dict(tc_class=c, underdog_colour=col, alpha=al, critical_gap_decisive=pt,
                             lo95=float(np.nanpercentile(cg, 2.5)) if np.isfinite(cg).any() else np.nan,
                             hi95=float(np.nanpercentile(cg, 97.5)) if np.isfinite(cg).any() else np.nan))
dres = pd.DataFrame(dres); dres.to_csv(os.path.join(OUT, f"critical_gap_decisive{SUF}.csv"), index=False)
dres["gap (95% CI)"] = dres.apply(lambda r: f"{r.critical_gap_decisive:.0f} ({r.lo95:.0f}-{r.hi95:.0f})", axis=1)
P(dres.pivot_table(index=["alpha", "tc_class"], columns="underdog_colour", values="gap (95% CI)", aggfunc="first").to_string())

# ---- no-model cross-check: isotonic regression ------------------------------------------------
from sklearn.isotonic import IsotonicRegression
iso = []
for c in classes:
    for col in colours:
        d = g[(g.tc_class == c) & (g.underdog_color == col)]
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(np.minimum(d.gap, 1500), d["win"])
        fitv = ir.predict(GRID)
        for al in ALPHAS[:3]:
            iso.append(dict(tc_class=c, underdog_colour=col, alpha=al, critical_gap_isotonic=float(GRID[(fitv < al).argmax()]) if (fitv < al).any() else np.nan))
iso = pd.DataFrame(iso); iso.to_csv(os.path.join(OUT, f"critical_gap_isotonic{SUF}.csv"), index=False)
P("\n== no-model cross-check: isotonic (monotone) curve of the observed win rate, all strength levels pooled (nan = never falls below alpha in the data)")
P(iso.pivot_table(index=["alpha", "tc_class"], columns="underdog_colour", values="critical_gap_isotonic").to_string())

# ---- the tail: what do underdogs still win at huge gaps? ---------------------------------------------
tail = g[g.gap >= 700].groupby(["tc_class", "underdog_color"]).agg(n=("win", "size"), win=("win", "mean"), not_lose=("notlose", "mean")).round(4)
P("\n== games with gap >= 700: how often do underdogs still win / not lose?  (tiny n means treat with care)")
P(tail.to_string())

# ---- final upset flags on every game ---------------------------------------------------------
g["p_win_emp"] = np.nan; g["p_notlose_emp"] = np.nan
for c in classes:
    for col in colours:
        m = ((g.tc_class == c) & (g.underdog_color == col)).to_numpy()
        X = design(g.gap[m], g.avg_elo[m])
        g.loc[m, "p_win_emp"] = predict(models[(c, col, "win", "all")][0], X)
        g.loc[m, "p_notlose_emp"] = predict(models[(c, col, "notlose", "all")][0], X)
g["p_notlose_emp"] = np.maximum(g["p_notlose_emp"], g["p_win_emp"])
us = g["underdog_score"].to_numpy()
g["q_emp"] = np.where(us == 1, g["p_win_emp"], np.where(us == 0.5, g["p_notlose_emp"], 1.0))
for al in ALPHAS:
    g[f"upset_emp_{al:.2f}"] = (us > 0) & (g["q_emp"] < al)
g[["game_url", "p_win_emp", "p_notlose_emp", "q_emp"] + [f"upset_emp_{a:.2f}" for a in ALPHAS]].to_parquet(
    os.path.join(OUT, f"upsets_empirical{SUF}.parquet"), index=False)
P("\n== upsets under the final (empirical) definition")
for al in ALPHAS:
    f = g[f"upset_emp_{al:.2f}"]
    P(f"alpha={al:.2f}: " + ", ".join(f"{c} {int(f[g.tc_class == c].sum())} ({f[g.tc_class == c].mean():.2%})" for c in classes)
      + "   | by underdog result: wins %d, draws %d" % (int((f & (us == 1)).sum()), int((f & (us == .5)).sum())))
P("\nmonotonicity check (P(win) should fall as the gap grows): share of fitted curves with a rise before gap 900 =",
  f"{np.mean([np.any(np.diff(predict(models[(c,col,'win','all')][0], design(GRID[GRID<=900], np.full((GRID<=900).sum(), 2350)))) > 1e-9) for c in classes for col in colours]):.0%}")
txt = "\n".join(out); open(f"empirical_report{SUF}.txt", "w", encoding="utf-8").write(txt); print(txt)
