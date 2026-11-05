"""Data layer for submission/viz/viz.py: small aggregate tables built from the FRESH pipeline outputs.

Reads only the needed columns of the big parquet files under the data root and writes small csv files to
viz_data/ next to this script. Nothing else is written.

    CHESS_DATA_ROOT=/path/to/data  python make_viz_data.py      # default data root: ../data relative to this folder
                                                                 # (i.e. submission/data when run from submission/viz)
Inputs (fresh layout written by notebooks 02 and 03):
    clean/analysis_games.parquet, clean/upsets_empirical.parquet, clean/upsets_empirical_interaction.parquet,
    clean/upset_audit_top.csv, clean/model_coefficients_empirical(_interaction).csv, clean/critical_gap_empirical(_interaction).csv,
    clean/kappa_table.csv, clean/expected_upsets(_interaction).csv, clean/model_fit_by_band.csv, clean/ordering_check.csv (cross-checks)
    interim/clean_games.parquet, interim/excluded_analysis_filters.parquet
Optional: CHESS_VIZ_DATA = output folder (default viz_data/ next to this script).

Tables (one row per group; counts are games):
  gap_hist.csv      tc_class, gap_lo (25-Elo lower edge; 900 = "900+"), n
  gap_summary.csv   tc_class ('all' = pooled), n, median, mean, p90, p99, max, share_ge_400, share_ge_600
  wdl_by_gap.csv    tc_class, underdog_color, band, gap_bin (50-Elo lower edge, 600 = "600+"), n, wins, draws, losses
  heat.csv          tc_class, underdog_color ('white'/'black'/'all'), elo_lo (100-Elo lower edge of the average rating), gap_bin, n, wins, draws
  upset_points.csv  every game with upset_emp_0.10 True: tc_class, underdog_color, band, gap, avg_elo, q_emp, underdog_score, year,
                    broadcast_name, game_url, white, black, white_elo, black_elo, rarest200 (bool), rank_q (1 = rarest overall)
  rarest_upsets.csv the 200 rows of clean/upset_audit_top.csv + rank_q + underdog/favourite names and ratings
  set_compare.csv   set ('clean' = interim/clean_games, 'analysis', 'dropped' = clean minus analysis), dimension, category, n, share
  drop_steps.csv    the sequential analysis filters of notebook 02 (from interim/excluded_analysis_filters.parquet): step, removed, remaining
  missing_elo_by_year.csv  year, clean_n, both_elo_n, analysis_n, share_missing_elo
  yearly_counts.csv year, tc_class, n, first_period, last_period, partial (True for 2026)
  monthly_counts.csv period, tc_class, n
  near_equal.csv    tc_class, band, n, draws, draw_rate  for games with gap <= 25 (ties included; as kappa in notebook 03)
  -- model / rule layer --
  model_bins.csv    tc_class, underdog_color, band, gap_bin (50-pt, 900 = "900+"), n, wins, draws, obs_win, obs_lo95, obs_hi95 (Wilson),
                    pred_win (mean main-model P(win) of the bin's games), pred_win_ref (main model at the band reference rating),
                    pred_win_int (mean interaction-model P(win)), mean_gap, mean_avg_elo
  model_bins_by_verified.csv  as model_bins split by the verified flag (n, wins, pred_win, mean_avg_elo, obs_win)
  data_support.csv  tc_class, underdog_color, band, n, last_bin_n100, support_limit, last_contig_bin_n100
  critical_gap_by_alpha.csv   tc_class, underdog_color, band, ref_avg_elo, alpha, critical_gap (main model, NaN = no crossing up to 900)
  critical_gap_by_elo.csv     tc_class, underdog_color, avg_elo (1600-2800 step 10), alpha (0.10), critical_gap (main model)
  avg_elo_hist.csv  tc_class, underdog_color, elo_lo (25-pt), n
  expected_upsets.csv / expected_upsets_interaction.csv  tc_class, band, alpha, games_pwin_below, exp_wins, obs_wins, sd_wins,
                    games_pnl_below, exp_draws, obs_draws, exp_total, obs_total, obs_flags  (main / interaction model)
  checks.csv        every sanity check: section, name, expected, actual, old (value on the old 858,123-game table), ok, kind
  make_viz_data_log.txt  the printed log
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.environ.get("CHESS_DATA_ROOT", os.path.join(HERE, "..", "data")))
P = lambda *a: os.path.join(DATA_ROOT, *a)
OUT = os.path.abspath(os.environ.get("CHESS_VIZ_DATA", os.path.join(HERE, "viz_data")))
os.makedirs(OUT, exist_ok=True)
TC = ["classical", "rapid", "blitz"]
BANDS = ["<2200", "2200-2499", "2500+"]
FIDE_TITLES = {"GM", "IM", "FM", "CM", "WGM", "WIM", "WFM", "WCM"}
log, checks = [], []
SECTION = ["data"]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.append(s)


def check(name, actual, expected, old=None, tol=0, kind="exact"):
    """kind: 'exact' (fresh reference value from notebooks 02/03 or BUILD_NOTES), 'cross-check' (equal to an independent
    file of the pipeline), 'stability' (no independent fresh reference: compared with the old value within tol)."""
    if isinstance(expected, (bool, str, list)):
        ok = actual == expected
    else:
        ok = abs(float(actual) - float(expected)) <= tol
    checks.append(dict(section=SECTION[0], name=name, expected=expected, actual=actual, old=old, ok=bool(ok), kind=kind))
    o = f" (old table: {old})" if old is not None else ""
    say(f"  CHECK [{kind}] {name}: actual={actual} expected={expected}{o} -> {'OK' if ok else 'MISMATCH'}")
    return ok


def write(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    say(f"  wrote {name}: {len(df)} rows, {os.path.getsize(path) / 1e6:.2f} MB")


say(f"data root: {DATA_ROOT}")
say(f"output:    {OUT}")

# ---------------------------------------------------------------- analysis set
cols = ["period", "game_url", "tc_class", "gap", "avg_elo", "has_underdog", "underdog_score", "underdog_color",
        "band", "gap_bin", "year", "broadcast_name", "white", "black", "white_elo_final", "black_elo_final",
        "white_title", "black_title", "n_plies"]
a = pd.read_parquet(P("clean", "analysis_games.parquet"), columns=cols)
say(f"analysis games: {len(a)}")
check("analysis rows", len(a), 863971, old=858123)
check("analysis game_url unique", bool(a.game_url.is_unique), True, kind="cross-check")
u = a[a["has_underdog"]].copy()
check("games with an underdog", len(u), 862599, old=856773)
vc0 = a.tc_class.value_counts()
for tc, exp, old in zip(TC, [613735, 142763, 107473], [609011, 141712, 107400]):
    check(f"analysis games {tc}", int(vc0.get(tc, 0)), exp, old=old)
u["win"] = (u["underdog_score"] == 1).astype(int)
u["draw"] = (u["underdog_score"] == 0.5).astype(int)
u["loss"] = (u["underdog_score"] == 0).astype(int)

# 1. gap histogram (25-point bins, last bin 900+) and summary
u["gap_lo"] = (np.minimum(u["gap"], 900) // 25 * 25).astype(int)
gh = u.groupby(["tc_class", "gap_lo"]).size().rename("n").reset_index()
write(gh, "gap_hist.csv")
rows = []
for tc, d in list(u.groupby("tc_class")) + [("all", u)]:
    g = d["gap"]
    rows.append(dict(tc_class=tc, n=len(g), median=g.median(), mean=g.mean(), p90=g.quantile(.9),
                     p99=g.quantile(.99), max=g.max(), share_ge_400=(g >= 400).mean(), share_ge_600=(g >= 600).mean()))
gs = pd.DataFrame(rows)
write(gs.round(4), "gap_summary.csv")
for tc, old in zip(TC, [152.0, 187.0, 174.0]):
    check(f"median gap {tc} (within 3 of old)", float(gs.set_index("tc_class").loc[tc, "median"]), old, old=old, tol=3, kind="stability")
check("share gap>=400 pooled (within 0.003 of old)", round(float(gs.set_index("tc_class").loc["all", "share_ge_400"]), 4), 0.0831,
      old=0.0831, tol=0.003, kind="stability")

# 2. W/D/L by gap
wdl = (u.groupby(["tc_class", "underdog_color", "band", "gap_bin"])
         .agg(n=("win", "size"), wins=("win", "sum"), draws=("draw", "sum"), losses=("loss", "sum")).reset_index())
write(wdl, "wdl_by_gap.csv")
check("wdl total", int(wdl.n.sum()), 862599, old=856773)

# 3. heat table
u["elo_lo"] = (u["avg_elo"] // 100 * 100).astype(int)
h1 = u.groupby(["tc_class", "underdog_color", "elo_lo", "gap_bin"]).agg(n=("win", "size"), wins=("win", "sum"), draws=("draw", "sum")).reset_index()
h2 = u.groupby(["tc_class", "elo_lo", "gap_bin"]).agg(n=("win", "size"), wins=("win", "sum"), draws=("draw", "sum")).reset_index()
h2["underdog_color"] = "all"
heat = pd.concat([h1, h2[h1.columns]], ignore_index=True)
write(heat, "heat.csv")

# 4. upsets at alpha = 0.10
e = pd.read_parquet(P("clean", "upsets_empirical.parquet"), columns=["game_url", "q_emp", "upset_emp_0.10"])
m = u.merge(e, on="game_url", how="inner", validate="one_to_one")
check("empirical join rows", len(m), 862599, old=856773)
up = m[m["upset_emp_0.10"]].copy()
check("upsets at 0.10", len(up), 11114, old=11049)
vc = up["tc_class"].value_counts()
for tc, exp, old in zip(TC, [9007, 1483, 624], [8952, 1479, 618]):
    check(f"upsets at 0.10 {tc}", int(vc.get(tc, 0)), exp, old=old)
check("draw upsets at 0.10 (BUILD_NOTES_03: wins / draws 10,510 / 604)", int((up.underdog_score == 0.5).sum()), 604, old=603)
up["rank_q"] = up["q_emp"].rank(method="first").astype(int)
top = pd.read_csv(P("clean", "upset_audit_top.csv"))
up["rarest200"] = up["game_url"].isin(top["game_url"])
check("rarest200 flagged", int(up["rarest200"].sum()), 200, old=200)
check("upset_audit_top = the 200 smallest q_emp among the 0.10 upsets", bool((up.loc[up.rarest200, "rank_q"] <= 200).all()), True, kind="cross-check")
say(f"  rarest200 by tc_class: {up.loc[up.rarest200, 'tc_class'].value_counts().to_dict()}  (old: all 200 classical)")
up = up.rename(columns={"white_elo_final": "white_elo", "black_elo_final": "black_elo"})
pc = ["tc_class", "underdog_color", "band", "gap", "avg_elo", "q_emp", "underdog_score", "year", "broadcast_name",
      "game_url", "white", "black", "white_elo", "black_elo", "rarest200", "rank_q"]
up = up[pc].sort_values("rank_q")
up["q_emp"] = up["q_emp"].round(6)
write(up, "upset_points.csv")
rr = top.merge(up[["game_url", "rank_q", "underdog_color", "underdog_score", "avg_elo", "year"]], on="game_url", how="left")
rr["underdog"] = np.where(rr.underdog_color == "white", rr.white, rr.black)
rr["favourite"] = np.where(rr.underdog_color == "white", rr.black, rr.white)
rr["underdog_elo"] = np.where(rr.underdog_color == "white", rr.white_elo_final, rr.black_elo_final)
rr["favourite_elo"] = np.where(rr.underdog_color == "white", rr.black_elo_final, rr.white_elo_final)
write(rr.sort_values("rank_q"), "rarest_upsets.csv")

# 6. yearly / monthly counts
yc = a.groupby(["year", "tc_class"]).agg(n=("game_url", "size"), first_period=("period", "min"), last_period=("period", "max")).reset_index()
yc["partial"] = yc["year"] == 2026
mc = a.groupby(["period", "tc_class"]).size().rename("n").reset_index()
write(mc, "monthly_counts.csv")
say(f"  2026 games by month: {a.loc[a.year == 2026, 'period'].value_counts().sort_index().to_dict()}")

# near-equal games (gap <= 25, ties included, as for kappa in notebook 03): draw rate by class x band
ne = a[a["gap"] <= 25].assign(draw=lambda d: (d["underdog_score"] == 0.5).astype(int))
ne = ne.groupby(["tc_class", "band"]).agg(n=("draw", "size"), draws=("draw", "sum")).reset_index()
ne["draw_rate"] = (ne.draws / ne.n).round(4)
write(ne, "near_equal.csv")
kt = pd.read_csv(P("clean", "kappa_table.csv")).set_index(["tc_class", "band"])
nei = ne.set_index(["tc_class", "band"])
check("near-equal n and draw rate equal kappa_table.csv (9 rows x 2 values)",
      int(sum(int(nei.loc[i, "n"] == kt.loc[i, "n_near_equal"]) + int(abs(nei.loc[i, "draw_rate"] - kt.loc[i, "draw_rate_equal"]) < 6e-5)
              for i in kt.index)), 18, kind="cross-check")
say("  near-equal classical draw rates by band: " + str(ne[ne.tc_class == "classical"].set_index("band").draw_rate.to_dict())
    + "  (old: 0.327 / 0.491 / 0.630)")
write(yc, "yearly_counts.csv")
check("yearly total", int(yc.n.sum()), 863971, old=858123)
a_tc = a.set_index("game_url")["tc_class"]
analysis_urls = set(a["game_url"])
del a, u, m, e, h1, h2

# 5. before/after: interim/clean_games vs analysis set
SECTION[0] = "set_compare"
cc = ["game_url", "period", "tc_class", "n_plies", "white_title", "black_title", "white_elo_final", "black_elo_final", "analysis_ok"]
c = pd.read_parquet(P("interim", "clean_games.parquet"), columns=cc + ["tc_source", "tc_base_s", "tc_inc_s", "broadcast_url", "broadcast_name"])
say(f"clean games: {len(c)}")
check("clean rows", len(c), 1129532, old=1117314)
check("clean game_url unique", bool(c.game_url.is_unique), True, kind="cross-check")
# Fresh notebook 02 stores the REPAIRED time class only for the analysis games (clean/analysis_games, interim/analysis_derived);
# interim/clean_games.tc_class is still the class before the glitchy-clock repair, and the dropped games' repaired class is not
# stored. So: analysis games take the class from the analysis table; for the dropped games the same repair rule as 02 section 7
# (identical code) is replayed, and the replay is checked against everything 02 does store.
bad = (c["tc_source"] == "clock_raw") & (c["tc_inc_s"] > 60)
good = c[c["tc_source"].isin(["string_verified_by_clock", "clock_snapped"]) & c["tc_base_s"].notna()]
def dominant(key):
    t = good.groupby([key, "tc_base_s", "tc_inc_s"]).size().rename("n").reset_index()
    t["share"] = t["n"] / t.groupby(key)["n"].transform("sum")
    return t.sort_values("n", ascending=False).drop_duplicates(key).query("share >= 0.8").set_index(key)[["tc_base_s", "tc_inc_s"]]
fb = pd.Series(np.nan, index=c.index); fi = pd.Series(np.nan, index=c.index)
for key in ("broadcast_url", "broadcast_name"):
    dom = dominant(key)
    mm = bad & fb.isna() & c[key].isin(dom.index)
    fb[mm] = c.loc[mm, key].map(dom["tc_base_s"]).astype(float); fi[mm] = c.loc[mm, key].map(dom["tc_inc_s"]).astype(float)
done = fb.notna()
tot_s = fb[done] + 60 * fi[done]
rep = c["tc_class"].astype(object).copy()
rep[done] = np.select([tot_s <= 600, tot_s < 3600], ["blitz", "rapid"], "classical")
rep[bad & ~done] = "unknown"
check("glitchy clock estimates (02 cleaning log: 12,828 + 962)", int(bad.sum()), 13790, old=13764)
check("repaired from round/tournament (02 cleaning log)", int(done.sum()), 12828, old=12802)
check("unresolved -> unknown (02 cleaning log)", int((bad & ~done).sum()), 962, old=962)
c = c.drop(columns=["tc_source", "tc_base_s", "tc_inc_s", "broadcast_url", "broadcast_name"])
del good
c["in_analysis"] = c["game_url"].isin(analysis_urls)
check("clean games that are in the analysis set", int(c.in_analysis.sum()), 863971, old=858123)
ia = c.in_analysis.to_numpy()
same = c.loc[ia, "game_url"].map(a_tc).to_numpy() == rep[ia].to_numpy()
check("replayed repair gives the stored analysis tc_class for every analysis game", int(same.sum()), 863971, kind="cross-check")
c["tc_class"] = rep.to_numpy()
c.loc[ia, "tc_class"] = c.loc[ia, "game_url"].map(a_tc).to_numpy()      # stored value (identical, see check)
del a_tc, rep
c["year"] = pd.to_numeric(c["period"].astype(str).str[:4], errors="coerce")

# drop steps: the reasons stored by notebook 02 (interim/excluded_analysis_filters.parquet), cross-checked by replaying the filters
ex = pd.read_parquet(P("interim", "excluded_analysis_filters.parquet"))
check("excluded games = clean minus analysis", len(ex), int((~c.in_analysis).sum()), kind="cross-check")
check("excluded game_urls are exactly the clean games not in the analysis set",
      bool(set(ex.game_url) == set(c.loc[~c.in_analysis, "game_url"])), True, kind="cross-check")
labels = {"11": "dated before 2019 (date artefacts)", "12": "missing Elo for at least one player",
          "13": "time control unknown", "14": "fewer than 10 plies (forfeits/walkovers)"}
ex["code"] = ex.reason.astype(str).str[:2]
steps, remaining = [], len(c)
for code_, name in labels.items():
    k = int((ex.code == code_).sum())
    remaining -= k
    steps.append(dict(step=name, removed=k, remaining=remaining))
ds = pd.DataFrame(steps)
write(ds, "drop_steps.csv")
cur, rep_steps = c, []
for keepf in [lambda d: d["year"] >= 2019, lambda d: d["analysis_ok"], lambda d: d["tc_class"] != "unknown", lambda d: d["n_plies"] >= 10]:
    k = keepf(cur).fillna(False).astype(bool)
    cur = cur[k]
    rep_steps.append(len(cur))
for i, (exp, old) in enumerate(zip([1129505, 891337, 867399, 863971], [1117287, 885313, 861534, 858123])):
    check(f"remaining after step {i + 1} (02 cleaning log)", int(ds.remaining.iloc[i]), exp, old=old)
check("replayed filter chain gives the same remaining counts", rep_steps, ds.remaining.tolist(), kind="cross-check")
del cur
c["drop_reason"] = c.game_url.map(ex.set_index("game_url").code).fillna("")
check("dropped for 'time control unknown' all have tc_class unknown (replayed repair)",
      bool((c.loc[c.drop_reason == "13", "tc_class"] == "unknown").all()), True, kind="cross-check")

def titled(t):
    return t.isin(FIDE_TITLES)
c["any_titled"] = np.where(titled(c.white_title) | titled(c.black_title), "at least one FIDE title", "no FIDE title")
c["any_gm"] = np.where((c.white_title == "GM") | (c.black_title == "GM"), "at least one GM", "no GM")
pb = [0, 10, 20, 40, 60, 80, 100, 120, 150, 200, 10 ** 6]
pl = ["0-9", "10-19", "20-39", "40-59", "60-79", "80-99", "100-119", "120-149", "150-199", "200+"]
c["plies_bin"] = pd.cut(c.n_plies, pb, right=False, labels=pl).astype(str)
both = c.white_elo_final.notna() & c.black_elo_final.notna()
avg = (c.white_elo_final.astype(float) + c.black_elo_final.astype(float)) / 2
c["avg_elo_bin"] = np.where(both, (np.clip(avg, 1000, 2799) // 200 * 200).fillna(0).astype(int).astype(str), "missing")
c["year_s"] = c["year"].astype("Int64").astype(str)
sets = {"clean": c, "analysis": c[c.in_analysis], "dropped": c[~c.in_analysis]}
rows = []
for sname, d in sets.items():
    for dim, col in [("year", "year_s"), ("tc_class", "tc_class"), ("plies", "plies_bin"), ("titled", "any_titled"),
                     ("gm", "any_gm"), ("avg_elo", "avg_elo_bin")]:
        vc2 = d[col].value_counts()
        for cat, n in vc2.items():
            rows.append(dict(set=sname, dimension=dim, category=cat, n=int(n), share=n / len(d)))
sc = pd.DataFrame(rows)
sc["share"] = sc["share"].round(6)
write(sc, "set_compare.csv")
my = c.groupby("year_s").agg(clean_n=("analysis_ok", "size"), both_elo_n=("analysis_ok", "sum"), analysis_n=("in_analysis", "sum")).reset_index().rename(columns={"year_s": "year"})
my["share_missing_elo"] = (1 - my.both_elo_n / my.clean_n).round(4)
write(my, "missing_elo_by_year.csv")
check("missing-Elo share of the cleaned set (02: 21.1%)", round(1 - float(c.analysis_ok.astype(bool).mean()), 3), 0.211, old=0.208, tol=0.0005)
del c, sets


# ==================================================================================================== model + rule layer
SECTION[0] = "model"
CAP2, GRID2 = 900, np.arange(0, 1505, 5)                       # as in notebook 03
REF2 = {"<2200": 2100, "2200-2499": 2350, "2500+": 2600}


def load_theta(suffix):
    coef = pd.read_csv(P("clean", f"model_coefficients_empirical{suffix}.csv"))
    names = ["intercept", "b_gap", "b_gap2", "b_gap3", "b_avgelo"] + (["b_gap_avgelo", "b_gap2_avgelo"] if "b_gap_avgelo" in coef else [])
    return {(r["tc_class"], r["underdog_colour"], r["outcome"]): r[names].to_numpy(float) for _, r in coef.iterrows()}


TH = {"main": load_theta(""), "interaction": load_theta("_interaction")}


def pwin(tc, col, gap, avg, oc="win", model="main"):
    """P(underdog result) from the stored coefficients; same design as notebook 03 (x = min(gap, 900)/300, e = (avg - 2300)/100)."""
    x = np.minimum(np.asarray(gap, float), CAP2) / 300
    ev = (np.asarray(avg, float) - 2300) / 100 + 0 * x
    th = TH[model][(tc, col, oc)]
    cols_ = [np.ones_like(x), x, x ** 2, x ** 3, ev] + ([x * ev, x ** 2 * ev] if len(th) == 7 else [])
    return 1 / (1 + np.exp(-(th @ np.vstack(cols_))))


def first_cross(p, al):
    below = p < al
    return float(GRID2[below.argmax()]) if below.any() else np.nan


g2 = pd.read_parquet(P("clean", "analysis_games.parquet"),
                     columns=["game_url", "tc_class", "underdog_color", "band", "gap", "avg_elo", "underdog_score", "has_underdog",
                              "white_elo_final", "white_elo_fide", "black_elo_final", "black_elo_fide", "verified"])
okv = lambda s_: g2[f"{s_}_elo_fide"].notna() & ((g2[f"{s_}_elo_final"] - g2[f"{s_}_elo_fide"]).abs() <= 100)
ver_re = (okv("white") & okv("black")).to_numpy()
check("verified recomputed from its definition equals the stored column", int((ver_re == g2.verified.to_numpy()).sum()), len(g2), kind="cross-check")
check("verified games, all analysis games (BUILD_NOTES_03)", int(g2.verified.sum()), 657365, old=657325)
g2 = g2[g2.has_underdog].drop(columns=["has_underdog", "white_elo_final", "white_elo_fide", "black_elo_final", "black_elo_fide"])
g2["verified"] = g2.verified.astype(bool)
e2 = pd.read_parquet(P("clean", "upsets_empirical.parquet"),
                     columns=["game_url", "p_win_emp", "p_notlose_emp", "upset_emp_0.20", "upset_emp_0.10", "upset_emp_0.05", "upset_emp_0.01"])
g2 = g2.merge(e2, on="game_url", how="inner", validate="one_to_one"); del e2
ei = pd.read_parquet(P("clean", "upsets_empirical_interaction.parquet"),
                     columns=["game_url", "p_win_emp", "p_notlose_emp", "upset_emp_0.20", "upset_emp_0.10", "upset_emp_0.05", "upset_emp_0.01"])
ei = ei.rename(columns={c_: c_ + "_int" for c_ in ei.columns if c_ != "game_url"})
g2 = g2.merge(ei, on="game_url", how="inner", validate="one_to_one"); del ei
check("model join rows (main + interaction)", len(g2), 862599, old=856773)
g2["win"] = (g2.underdog_score == 1).astype(int)
g2["draw"] = (g2.underdog_score == 0.5).astype(int)
g2["gap_bin"] = (np.minimum(g2.gap, 900) // 50 * 50).astype(int)
g2["p_ref"] = np.nan
md_main = md_int = 0.0
for (tc, col), idx in g2.groupby(["tc_class", "underdog_color"]).groups.items():
    d = g2.loc[idx]
    md_main = max(md_main, float(np.abs(pwin(tc, col, d.gap, d.avg_elo) - d.p_win_emp).max()))
    md_int = max(md_int, float(np.abs(pwin(tc, col, d.gap, d.avg_elo, model="interaction") - d.p_win_emp_int).max()))
    g2.loc[idx, "p_ref"] = pwin(tc, col, d.gap, d.band.map(REF2))
check("max |P(win) from main coefficients - p_win_emp| < 1e-9", md_main < 1e-9, True, kind="cross-check")
check("max |P(win) from interaction coefficients - p_win_emp_int| < 1e-9", md_int < 1e-9, True, kind="cross-check")

# ---- model_bins (main + interaction prediction)
mb = (g2.groupby(["tc_class", "underdog_color", "band", "gap_bin"])
        .agg(n=("win", "size"), wins=("win", "sum"), draws=("draw", "sum"), pred_win=("p_win_emp", "mean"),
             pred_win_ref=("p_ref", "mean"), pred_win_int=("p_win_emp_int", "mean"), mean_gap=("gap", "mean"),
             mean_avg_elo=("avg_elo", "mean")).reset_index())
z = 1.959964
ph = mb.wins / mb.n
den = 1 + z ** 2 / mb.n
cen = (ph + z ** 2 / (2 * mb.n)) / den
hw = z * np.sqrt(ph * (1 - ph) / mb.n + z ** 2 / (4 * mb.n ** 2)) / den
mb["obs_win"], mb["obs_lo95"], mb["obs_hi95"] = ph, cen - hw, cen + hw
check("model_bins total games", int(mb.n.sum()), 862599, old=856773)
mb = mb[["tc_class", "underdog_color", "band", "gap_bin", "n", "wins", "draws", "obs_win", "obs_lo95", "obs_hi95",
         "pred_win", "pred_win_ref", "pred_win_int", "mean_gap", "mean_avg_elo"]]
write(mb.round(6), "model_bins.csv")
# cross-check against notebook 03's model_fit_by_band.csv (bins with n >= 100, both colours counted separately)
fbb = pd.read_csv(P("clean", "model_fit_by_band.csv"))
ok_ = mb[mb.n >= 100]
agree = 0
for r in fbb.itertuples():
    d = ok_[(ok_.tc_class == r.tc_class) & (ok_.band == r.band)]
    pr = d.pred_win if r.model == "main" else d.pred_win_int
    inside = int(((pr >= d.obs_lo95) & (pr <= d.obs_hi95)).sum())
    agree += int(len(d) == r.bins and inside == r.inside and abs(np.average((d.obs_win - pr) * 100, weights=d.n) - r.mean_err_wtd_pp) < 1e-3)
check("model_bins reproduce model_fit_by_band.csv (bins, inside, n-weighted error; 18 rows)", agree, 18, kind="cross-check")
for model, col_ in (("main", "pred_win"), ("interaction", "pred_win_int")):
    ins = int(((ok_[col_] >= ok_.obs_lo95) & (ok_[col_] <= ok_.obs_hi95)).sum())
    check(f"bins with n>=100 whose {model} prediction is inside the Wilson interval (BUILD_NOTES_03)", ins,
          193 if model == "main" else 203, old=195 if model == "main" else 204)
c25 = ok_[(ok_.tc_class == "classical") & (ok_.band == "2500+")]
check("classical 2500+ n-weighted error, main (pp)", round(float(np.average((c25.obs_win - c25.pred_win) * 100, weights=c25.n)), 2), -2.03,
      old=-2.03, tol=0.01)

mv = (g2.groupby(["tc_class", "underdog_color", "band", "verified", "gap_bin"])
        .agg(n=("win", "size"), wins=("win", "sum"), pred_win=("p_win_emp", "mean"), mean_avg_elo=("avg_elo", "mean")).reset_index())
mv["obs_win"] = mv.wins / mv.n
write(mv.round(6), "model_bins_by_verified.csv")
say(f"  verified games with an underdog: {int(g2.verified.sum())} of {len(g2)} ({g2.verified.mean():.1%})  (old: 654,685 of 856,773, 76.4%)")
rv = mv.set_index(["tc_class", "underdog_color", "band", "verified", "gap_bin"])
r = mb.set_index(["tc_class", "underdog_color", "band", "gap_bin"])
check("UNVERIFIED-only observed win rapid/white/2200-2499 bin 250 (within 0.01 of old)",
      round(float(rv.loc[("rapid", "white", "2200-2499", False, 250), "obs_win"]), 3), 0.217, old=0.217, tol=0.01, kind="stability")
check("UNVERIFIED-only observed win blitz/white/2200-2499 bin 300 (within 0.01 of old)",
      round(float(rv.loc[("blitz", "white", "2200-2499", False, 300), "obs_win"]), 3), 0.201, old=0.201, tol=0.01, kind="stability")
say("  ALL games, observed win rapid/white/2200-2499 bin 250: "
    f"{r.loc[('rapid', 'white', '2200-2499', 250), 'obs_win']:.3f}; blitz bin 300: {r.loc[('blitz', 'white', '2200-2499', 300), 'obs_win']:.3f}  (old: 0.178 / 0.168)")

# ---- data support per slice
rows = []
for (tc, col, band), d in mb.groupby(["tc_class", "underdog_color", "band"]):
    d = d.sort_values("gap_bin")
    ok = d[(d.n >= 100) & (d.gap_bin < 900)]
    last = int(ok.gap_bin.max()) if len(ok) else np.nan
    contig = 0
    for b in range(0, 900, 50):
        nn = d.loc[d.gap_bin == b, "n"]
        if len(nn) and int(nn.iloc[0]) >= 100:
            contig = b
        else:
            break
    rows.append(dict(tc_class=tc, underdog_color=col, band=band, n=int(d.n.sum()), last_bin_n100=last,
                     support_limit=last + 50, last_contig_bin_n100=contig))
ds2 = pd.DataFrame(rows)
write(ds2, "data_support.csv")
say(f"  slices whose n>=100 bins are not one unbroken run from gap 0: {int((ds2.last_bin_n100 != ds2.last_contig_bin_n100).sum())} (old: 0)")

# ---- critical gap vs alpha (main model), reproduction of the stored tables (main and interaction)
alphas = np.round(np.r_[0.01, np.arange(0.02, 0.3001, 0.005)], 4)
rows = []
for tc in TC:
    for col in ("white", "black"):
        for band, ref in REF2.items():
            p = pwin(tc, col, GRID2, ref)
            for al in alphas:
                rows.append(dict(tc_class=tc, underdog_color=col, band=band, ref_avg_elo=ref, alpha=al, critical_gap=first_cross(p, al)))
cga = pd.DataFrame(rows)
write(cga, "critical_gap_by_alpha.csv")
for model, suf in (("main", ""), ("interaction", "_interaction")):
    stored = pd.read_csv(P("clean", f"critical_gap_empirical{suf}.csv"))
    re_ = np.array([first_cross(pwin(rw.tc_class, rw.underdog_colour, GRID2, rw.ref_avg_elo, model=model), rw.alpha) for rw in stored.itertuples()])
    same = (stored.critical_gap.to_numpy() == re_) | (stored.critical_gap.isna().to_numpy() & np.isnan(re_))
    check(f"stored {model} critical gaps reproduced from the coefficients (72 rows)", int(same.sum()), 72, kind="cross-check")
stored = pd.read_csv(P("clean", "critical_gap_empirical.csv"))
nonmono = 0
for _, d in cga.groupby(["tc_class", "underdog_color", "band"]):
    v = d.sort_values("alpha").critical_gap.to_numpy()
    v = v[np.isfinite(v)]
    nonmono += int((np.diff(v) > 0).sum())
check("critical gap never increases as alpha increases", nonmono, 0, old=0)
hl = stored[(stored.band == "2200-2499") & np.isclose(stored.alpha, 0.10)].set_index(["underdog_colour", "tc_class"])
for (col, tc), exp, old in [(("white", "classical"), 280, 275), (("white", "rapid"), 405, 405), (("white", "blitz"), 465, 465),
                            (("black", "classical"), 185, 180), (("black", "rapid"), 310, 315), (("black", "blitz"), 405, 405)]:
    check(f"headline critical gap {col} {tc} 2200-2499 alpha 0.10", float(hl.loc[(col, tc), "critical_gap"]), exp, old=old)

# ---- critical gap vs average rating at alpha 0.10 (main model), average-rating distribution
elos = np.arange(1600, 2801, 10)
rows = []
for tc in TC:
    for col in ("white", "black"):
        for el in elos:
            rows.append(dict(tc_class=tc, underdog_color=col, avg_elo=int(el), alpha=0.10, critical_gap=first_cross(pwin(tc, col, GRID2, el), 0.10)))
write(pd.DataFrame(rows), "critical_gap_by_elo.csv")
g2["elo_lo"] = (g2.avg_elo // 25 * 25).astype(int)
write(g2.groupby(["tc_class", "underdog_color", "elo_lo"]).size().rename("n").reset_index(), "avg_elo_hist.csv")


# ---- expected vs observed upsets, main and interaction
def expected(sfx):
    pw, pn = "p_win_emp" + sfx, "p_notlose_emp" + sfx
    rows = []
    for tc, band in [(t, b) for t in TC + ["all"] for b in ["all"] + BANDS]:
        d = g2 if tc == "all" else g2[g2.tc_class == tc]
        d = d if band == "all" else d[d.band == band]
        for al in (0.20, 0.10, 0.05, 0.01):
            bw, bn = d[pw] < al, d[pn] < al
            pdraw = (d[pn] - d[pw]).clip(lower=0)
            ew, ed = float(d[pw][bw].sum()), float(pdraw[bn].sum())
            rows.append(dict(tc_class=tc, band=band, alpha=al, games_pwin_below=int(bw.sum()), exp_wins=ew, obs_wins=int(d.win[bw].sum()),
                             sd_wins=float(np.sqrt((d[pw][bw] * (1 - d[pw][bw])).sum())),
                             games_pnl_below=int(bn.sum()), exp_draws=ed, obs_draws=int(d.draw[bn].sum()),
                             exp_total=ew + ed, obs_total=int(d.win[bw].sum() + d.draw[bn].sum()),
                             obs_flags=int(d[f"upset_emp_{al:.2f}{sfx}"].sum())))
    return pd.DataFrame(rows)


for model, sfx, fname in (("main", "", "expected_upsets.csv"), ("interaction", "_int", "expected_upsets_interaction.csv")):
    eu = expected(sfx)
    write(eu.round(2), fname)
    ref = pd.read_csv(P("clean", fname))
    mrg = eu.merge(ref, on=["tc_class", "band", "alpha"], suffixes=("", "_03"))
    diff = max(float((mrg[c_] - mrg[c_ + "_03"]).abs().max()) for c_ in ["exp_wins", "obs_wins", "exp_total", "obs_total", "obs_flags"])
    check(f"expected_upsets ({model}) equals notebook 03's {fname} (64 rows, max abs diff < 0.01)",
          bool(len(mrg) == 64 and diff < 0.01), True, kind="cross-check")
    t10 = eu[(eu.tc_class == "all") & (eu.band == "all") & (eu.alpha == 0.10)].iloc[0]
    check(f"{model}: observed upsets at 0.10 = wins below + draws below = flag count", int(t10.obs_total), int(t10.obs_flags), kind="cross-check")
    rb = eu[(eu.tc_class == "all") & (eu.alpha == 0.10) & (eu.band != "all")].set_index("band")
    ratio = (rb.obs_wins / rb.exp_wins).round(2)
    for b, exp, old in zip(BANDS, [1.05, 0.91, 0.77] if model == "main" else [1.04, 0.97, 0.78],
                           [1.05, 0.91, 0.77] if model == "main" else [1.04, 0.98, 0.78]):
        check(f"{model}: obs/exp underdog wins with P(win) < 0.10, band {b}", float(ratio[b]), exp, old=old, tol=0.005)
    say(f"  {model}: expected upsets at 0.10: {t10.exp_total:.0f} vs observed {int(t10.obs_total)} "
        f"(wins {t10.exp_wins:.0f} vs {int(t10.obs_wins)}, draws {t10.exp_draws:.0f} vs {int(t10.obs_draws)})"
        + ("  (old main: 11,203 vs 11,049)" if model == "main" else ""))
eu = pd.read_csv(os.path.join(OUT, "expected_upsets.csv"))
t10 = eu[(eu.tc_class == "all") & (eu.band == "all") & (eu.alpha == 0.10)].iloc[0]
check("main: expected upsets at 0.10 (BUILD_NOTES_03: 11,273)", round(float(t10.exp_total)), 11273, old=11203, tol=1)

# ---- ordering classical < rapid < blitz at 0.20/0.10/0.05 (main stored table) and notebook 03's ordering_check.csv
bad_ = []
for (col, band, al), d in stored[stored.alpha.isin([0.2, 0.1, 0.05])].groupby(["underdog_colour", "band", "alpha"]):
    v = d.set_index("tc_class").critical_gap
    if not (v["classical"] < v["rapid"] < v["blitz"]):
        bad_.append(f"{col} {band} alpha={al}: " + ", ".join(f"{t} {v[t]:.0f}" for t in TC))
check("ordering classical < rapid < blitz holds in all 18 cells (main)", len(bad_), 0, old=0)
oc = pd.read_csv(P("clean", "ordering_check.csv"))
check("ordering_check.csv: cells where the ordering holds, all 4 runs (72)", int(oc.ordering_holds.sum()), 72, old=72)

pd.DataFrame(checks).to_csv(os.path.join(OUT, "checks.csv"), index=False)
n_bad = sum(not c_["ok"] for c_ in checks)
say(f"checks: {len(checks)} run, {n_bad} mismatches")
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".csv"))
say(f"total size of viz_data csv: {tot / 1e6:.2f} MB")
with open(os.path.join(OUT, "make_viz_data_log.txt"), "w") as f:
    f.write("\n".join(log) + "\n")
