"""Stage 4: fit the rating model to the data and re-define upsets with it.
Run after features.py:   python fit_model.py
Why: the fixed Elo-Davidson model (scale 400, no colour effect, kappa from near-equal games) over-predicts
the underdog in classical and under-predicts it in blitz, and ignores the White advantage (an underdog with White
scores ~7 points more than with Black). So we FIT three things per time-control class by maximum likelihood:
    s   rating scale (400 = textbook Elo; larger = ratings predict less)
    w   White's advantage in rating points
    kappa  draw parameter, one per rating band
Fitting uses odd calendar months, evaluation uses even months, so every comparison is out-of-sample.
Model:  v = Elo_white - Elo_black + w ;  a = 10^(v/(2s)) ;  D = a + 1/a + kappa
        P(White win) = a/D ,  P(draw) = kappa/D ,  P(White loss) = (1/a)/D
Writes data/processed/model_params.csv, critical_gap.csv, calibration_test.csv, upsets_fitted.parquet, fit_report.txt"""
import os
import numpy as np, pandas as pd
from scipy.optimize import minimize

OUT = os.path.join("data", "processed")
ALPHAS = (0.10, 0.05, 0.01)
BANDS = ["<2200", "2200-2499", "2500+"]
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

g = pd.read_parquet(os.path.join(OUT, "analysis_games.parquet"))
g["month"] = pd.to_numeric(g["period"].astype(str).str[-2:], errors="coerce")
g["train"] = g["month"] % 2 == 1
dif = (g["white_elo_final"] - g["black_elo_final"]).to_numpy(float)
res = g["score_white"].to_numpy(float)
bidx = g["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()

def probs(theta, dif, bidx, free):
    """theta = [log s, w, log k0, log k1, log k2]; entries not in `free` stay at their start values."""
    s, w = np.exp(theta[0]), theta[1]
    k = np.exp(np.asarray(theta[2:]))[bidx]
    a = np.exp((dif + w) * np.log(10) / (2 * s))
    Dn = a + 1 / a + k
    return a / Dn, k / Dn, (1 / a) / Dn          # P(White win), P(draw), P(White loss)

def nll(x, theta0, free, dif, bidx, res):
    th = theta0.copy(); th[free] = x
    pw, pd_, pl = probs(th, dif, bidx, free)
    p = np.where(res == 1, pw, np.where(res == 0.5, pd_, pl))
    return -np.log(np.clip(p, 1e-12, None)).mean()

def fit(theta0, free, dif, bidx, res):
    r = minimize(nll, theta0[free], args=(theta0, free, dif, bidx, res), method="L-BFGS-B")
    th = theta0.copy(); th[free] = r.x
    return th

def loglik(theta, dif, bidx, res):
    pw, pd_, pl = probs(theta, dif, bidx, None)
    p = np.where(res == 1, pw, np.where(res == 0.5, pd_, pl))
    return np.log(np.clip(p, 1e-12, None)).mean()

kt = pd.read_csv(os.path.join(OUT, "kappa_table.csv"))
rows, param_rows, fitted = [], [], {}
for cls in ("classical", "rapid", "blitz"):
    m = (g["tc_class"] == cls).to_numpy()
    tr, te = m & g["train"].to_numpy(), m & ~g["train"].to_numpy()
    k0 = np.array([kt[(kt.tc_class == cls) & (kt.band == b)]["kappa"].iloc[0] for b in BANDS])
    base = np.array([np.log(400.0), 0.0, *np.log(k0)])
    args = lambda sel: (dif[sel], bidx[sel], res[sel])
    models = {
        "A fixed: s=400, w=0, kappa from near-equal games": (base, None),
        "B kappa fitted": (fit(base, [2, 3, 4], *args(tr)), [2, 3, 4]),
    }
    models["C + White advantage w"] = (fit(models["B kappa fitted"][0], [1, 2, 3, 4], *args(tr)), None)
    models["D + scale s (full model)"] = (fit(models["C + White advantage w"][0], [0, 1, 2, 3, 4], *args(tr)), None)
    for name, (th, _) in models.items():
        rows.append(dict(tc_class=cls, model=name, n_test=int(te.sum()), test_loglik=loglik(th, *args(te)),
                         train_loglik=loglik(th, *args(tr))))
    th = models["D + scale s (full model)"][0]
    fitted[cls] = th
    param_rows.append(dict(tc_class=cls, scale_s=np.exp(th[0]), white_adv_points=th[1],
                           **{f"kappa_{b}": np.exp(th[2 + i]) for i, b in enumerate(BANDS)}))
cmp_ = pd.DataFrame(rows)
cmp_["gain_vs_A_per_game"] = cmp_["test_loglik"] - cmp_.groupby("tc_class")["test_loglik"].transform("first")
params = pd.DataFrame(param_rows); params.to_csv(os.path.join(OUT, "model_params.csv"), index=False)
P("== model comparison, mean log-likelihood per game on HELD-OUT (even-month) games; higher is better")
P(cmp_.round(4).to_string(index=False))
P("\n== fitted parameters (full model D)"); P(params.round(3).to_string(index=False))
P("   scale_s: 400 = textbook Elo. white_adv_points: rating points White is worth. kappa: draw parameter by rating band.")

# ---- apply the fitted model: P(underdog wins/draws), q, upset flags ---------------------
pw_u = np.zeros(len(g)); pd_u = np.zeros(len(g))
for cls, th in fitted.items():
    m = (g["tc_class"] == cls).to_numpy()
    pw, pdr, pl = probs(th, dif[m], bidx[m], None)
    uw = g.loc[m, "underdog_is_white"].to_numpy()
    pw_u[m] = np.where(uw, pw, pl); pd_u[m] = pdr
g["p_win_u_fit"], g["p_draw_fit"] = pw_u, pd_u
g["exp_score_u_fit"] = pw_u + 0.5 * pd_u
us = g["underdog_score"].to_numpy()
g["q_fit"] = np.where(us == 1, pw_u, np.where(us == 0.5, pw_u + pd_u, 1.0))
for al in ALPHAS:
    g[f"upset_fit_{al:.2f}"] = g["has_underdog"] & (us > 0) & (g["q_fit"] < al)
g[["game_url", "q_fit", "p_win_u_fit", "p_draw_fit", "exp_score_u_fit"] + [f"upset_fit_{a:.2f}" for a in ALPHAS]].to_parquet(
    os.path.join(OUT, "upsets_fitted.parquet"), index=False)

# ---- out-of-sample calibration (even months) ----------------------------------------------
u = g[g["has_underdog"] & ~g["train"]].copy()
u["gap100"] = (np.minimum(u["gap"], 600) // 100 * 100).astype(int)
def cal(df):
    return pd.Series(dict(n=len(df), observed=df.underdog_score.mean(), model_fixed=df.exp_score_u.mean(),
                          model_fitted=df.exp_score_u_fit.mean(), obs_win=(df.underdog_score == 1).mean(),
                          fit_win=df.p_win_u_fit.mean()))
c1 = u.groupby(["tc_class", "gap100"]).apply(cal, include_groups=False).reset_index()
c2 = u.groupby(["tc_class", "underdog_color", "gap100"]).apply(cal, include_groups=False).reset_index()
c1.to_csv(os.path.join(OUT, "calibration_test.csv"), index=False)
P("\n== held-out calibration: underdog score, observed vs fixed vs fitted model (gap bins of 100)")
P(c1.round(4).to_string(index=False))
err = lambda c, col: float(np.average(np.abs(c["observed"] - c[col]), weights=c["n"]))
P(f"\nweighted mean |observed - model| over those bins: fixed {err(c1,'model_fixed'):.4f}   fitted {err(c1,'model_fitted'):.4f}")
P("\n== same, split by the underdog's colour (gap 100-299 only)")
P(c2[c2.gap100.isin([100, 200])].round(4).to_string(index=False))

# ---- critical gap: the rating gap where the underdog's win probability falls to alpha ---------
def pwin_u(th, gap, band, underdog_white):
    dd = (-gap) if underdog_white else gap          # White minus Black
    pw, _, pl = probs(th, np.array([dd], float), np.array([band]), None)
    return float(pw[0] if underdog_white else pl[0])
crit = []
for cls, th in fitted.items():
    for bi, b in enumerate(BANDS):
        for uw in (True, False):
            for al in ALPHAS:
                lo, hi = 0.0, 2000.0
                if pwin_u(th, hi, bi, uw) > al: crit.append((cls, b, "white" if uw else "black", al, np.nan)); continue
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if pwin_u(th, mid, bi, uw) > al else (lo, mid)
                crit.append((cls, b, "white" if uw else "black", al, (lo + hi) / 2))
crit = pd.DataFrame(crit, columns=["tc_class", "band", "underdog_colour", "alpha", "critical_gap"])
crit.to_csv(os.path.join(OUT, "critical_gap.csv"), index=False)
P("\n== critical gap (rating points) at which the underdog's chance of WINNING falls to alpha, fitted model")
P(crit.pivot_table(index=["tc_class", "band", "underdog_colour"], columns="alpha", values="critical_gap").round(0).to_string())
P("\n== upsets under the fitted model (underdog wins/draws with q_fit < alpha)")
for al in ALPHAS:
    f = g[f"upset_fit_{al:.2f}"]
    P(f"alpha={al:.2f}: " + ", ".join(f"{c} {int(f[g.tc_class == c].sum())} ({f[g.tc_class == c].mean():.2%})" for c in ("classical", "rapid", "blitz")))
txt = "\n".join(out); open("fit_report.txt", "w", encoding="utf-8").write(txt); print(txt)
