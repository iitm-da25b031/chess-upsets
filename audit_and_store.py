"""Stage 6: audit the upsets, flag statistical outliers, and build the SQLite database.
Run after empirical_upsets.py:   python audit_and_store.py
Reads  data/processed/analysis_games.parquet, upsets_empirical.parquet, critical_gap_*.csv, kappa_table.csv, model_params.csv
Writes data/processed/chess_upsets.sqlite, upset_audit_top.csv, giant_killers.csv, audit_report.txt

Three checks (all from the course's cleaning toolbox):
 1. Robust univariate outliers: game length (plies) per time control by IQR fence and by MAD z-score.
 2. Multivariate outliers: Isolation Forest on (gap, average Elo, plies, result), per time control.
 3. Upset audit: are the upsets mostly real, or do they pile up in suspicious games (very short, huge gap, one player)?
Nothing is deleted. Every game gets flags, so the analysis can be rerun with or without them."""
import os, sqlite3
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest

OUT = os.path.join("data", "processed")
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

g = pd.read_parquet(os.path.join(OUT, "analysis_games.parquet"))
up = pd.read_parquet(os.path.join(OUT, "upsets_empirical.parquet"))
g = g.merge(up, on="game_url", how="left")                    # games without an underdog (exact ties) get NaN here
for c in ("white", "black", "round", "termination", "white_fide_id", "black_fide_id", "broadcast_name"):
    if c not in g.columns: g[c] = None
P(f"games: {len(g)}   upset flags available for {int(g['q_emp'].notna().sum())}")

# ---- 1. univariate: game length per time control -------------------------------------------------
g["short_game"] = g["n_plies"] < 20                          # fewer than 10 moves each: possible walkover / agreed result
rows = []
g["len_outlier_iqr"] = False; g["len_outlier_mad"] = False
for c, d in g.groupby("tc_class"):
    x = d["n_plies"].astype(float)
    q1, q3 = x.quantile([.25, .75]); lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    med = x.median(); mad = (x - med).abs().median() * 1.4826
    z = (x - med).abs() / max(mad, 1e-9)
    g.loc[d.index, "len_outlier_iqr"] = (x < lo) | (x > hi)
    g.loc[d.index, "len_outlier_mad"] = z > 3.5
    rows.append(dict(tc_class=c, n=len(d), median=med, mad_sd=round(mad, 1), iqr_fence_lo=round(lo, 1), iqr_fence_hi=round(hi, 1),
                     iqr_outliers=int(((x < lo) | (x > hi)).sum()), mad_outliers=int((z > 3.5).sum()), short_games=int((x < 20).sum())))
P("\n== game length (plies) outliers per time control"); P(pd.DataFrame(rows).to_string(index=False))

# ---- 2. multivariate: Isolation Forest ---------------------------------------------------------------
g["iforest_outlier"] = False
for c, d in g.groupby("tc_class"):
    X = np.column_stack([d["gap"], d["avg_elo"], d["n_plies"], d["score_white"]]).astype(float)
    X = (X - X.mean(0)) / X.std(0)
    f = IsolationForest(n_estimators=200, contamination=0.01, random_state=0, n_jobs=-1).fit(X)
    g.loc[d.index, "iforest_outlier"] = f.predict(X) == -1
P("\n== Isolation Forest (1% most isolated games per time control): how many are upsets?")
for al in (0.20, 0.10, 0.05):
    col = f"upset_emp_{al:.2f}"
    if col in g.columns:
        a, b = g.loc[g.iforest_outlier, col].mean(), g.loc[~g.iforest_outlier, col].mean()
        P(f"   alpha={al:.2f}: upset rate among outliers {a:.2%} vs others {b:.2%}")

# ---- 3. upset audit (alpha = 0.10) --------------------------------------------------------------------
u = g[g.get("upset_emp_0.10", False).fillna(False).astype(bool)].copy()
P(f"\n== upset audit at alpha=0.10: {len(u)} upsets")
P(f"   short games (<20 plies): {int(u.short_game.sum())} ({u.short_game.mean():.1%}) vs {g.short_game.mean():.1%} of all games")
P(f"   length outliers (IQR / MAD): {u.len_outlier_iqr.mean():.1%} / {u.len_outlier_mad.mean():.1%} vs {g.len_outlier_iqr.mean():.1%} / {g.len_outlier_mad.mean():.1%} of all games")
P(f"   Isolation-Forest outliers: {u.iforest_outlier.mean():.1%} vs {g.iforest_outlier.mean():.1%}")
P(f"   PGN/FIDE Elo mismatch flag on either side: {u.elo_mismatch_any.fillna(False).mean():.1%} vs {g.elo_mismatch_any.fillna(False).mean():.1%}")
P(f"   time control borderline: {u.tc_borderline.fillna(False).mean():.1%} vs {g.tc_borderline.fillna(False).mean():.1%}")
P("   upsets by gap bin:"); P(u.groupby("gap_bin").size().to_string())
top = u.sort_values("q_emp").head(200)
cols = [c for c in ("game_url", "broadcast_name", "round", "white", "black", "white_elo_final", "black_elo_final", "gap", "score_white",
                    "tc_class", "n_plies", "termination", "q_emp", "short_game", "elo_mismatch_any") if c in top.columns]
top[cols].to_csv(os.path.join(OUT, "upset_audit_top.csv"), index=False)
P("   200 rarest upsets saved to upset_audit_top.csv: open a few game_url links in Lichess and check they are real games.")

# repeat winners: one player behind many upsets is either a rating error (junior improving fast) or a genuine giant-killer
u["underdog_name"] = np.where(u.underdog_color == "white", u.white, u.black)
u["underdog_fide"] = np.where(u.underdog_color == "white", u.white_fide_id, u.black_fide_id)
key = u["underdog_fide"].fillna(u["underdog_name"])
all_u = g[g.has_underdog].copy()
all_u["key"] = np.where(all_u.underdog_color == "white", all_u.white_fide_id, all_u.black_fide_id)
all_u["key"] = pd.Series(all_u["key"]).fillna(pd.Series(np.where(all_u.underdog_color == "white", all_u.white, all_u.black), index=all_u.index))
n_as_u = all_u.groupby("key").size()
n_up = key.value_counts()
gk = pd.DataFrame({"games_as_underdog": n_as_u, "upsets": n_up}).dropna()
gk = gk[gk.games_as_underdog >= 30]
gk["upset_rate"] = gk.upsets / gk.games_as_underdog
med = gk.upset_rate.median(); mad = (gk.upset_rate - med).abs().median() * 1.4826
gk["robust_z"] = (gk.upset_rate - med) / max(mad, 1e-9)
gk = gk.sort_values("upsets", ascending=False)
gk.head(300).to_csv(os.path.join(OUT, "giant_killers.csv"))
P(f"\n== players with >=30 games as underdog: {len(gk)}; median upset rate {med:.2%}; players with robust z > 3.5: {int((gk.robust_z > 3.5).sum())}")
P(f"   share of all alpha=0.10 upsets produced by the top 20 players: {gk.upsets.head(20).sum() / len(u):.1%}")

# ---- SQLite database ------------------------------------------------------------------------------------------
keep = [c for c in ("game_url", "broadcast_name", "round", "white", "black", "white_fide_id", "black_fide_id", "white_elo_final", "black_elo_final",
        "white_elo_source", "black_elo_source", "result", "score_white", "termination", "tc_class", "tc_source", "tc_borderline", "period",
        "n_plies", "gap", "gap_bin", "avg_elo", "band", "underdog_color", "underdog_score", "p_win_emp", "p_notlose_emp", "q_emp",
        "upset_emp_0.20", "upset_emp_0.10", "upset_emp_0.05", "elo_mismatch_any", "short_game", "len_outlier_iqr", "len_outlier_mad",
        "iforest_outlier") if c in g.columns]
db = os.path.join(OUT, "chess_upsets.sqlite")
if os.path.exists(db): os.remove(db)
con = sqlite3.connect(db)
gg = g[keep].copy()
gg.columns = [c.replace(".", "_") for c in gg.columns]
for c in gg.columns:
    if gg[c].dtype == bool: gg[c] = gg[c].astype(int)
gg.to_sql("games", con, index=False, chunksize=50000)
for name in ("kappa_table", "model_params", "critical_gap_empirical", "critical_gap_decisive", "critical_gap_isotonic"):
    p = os.path.join(OUT, name + ".csv")
    if os.path.exists(p): pd.read_csv(p).to_sql(name, con, index=False)
gk.reset_index().rename(columns={"index": "player_key"}).to_sql("giant_killers", con, index=False)
for ix in ("tc_class", "gap_bin", "underdog_color", "period", "upset_emp_0_10", "upset_emp_0_20", "game_url"):
    if ix in gg.columns: con.execute(f"CREATE INDEX idx_games_{ix} ON games({ix})")
con.commit()
n = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
chk = con.execute("SELECT tc_class, COUNT(*), SUM(upset_emp_0_10) FROM games GROUP BY tc_class").fetchall()
con.close()
P(f"\n== SQLite: {db}  ({os.path.getsize(db)/1e6:.0f} MB), table games = {n} rows; upsets at alpha=0.10 by class: {chk}")
txt = "\n".join(out); open("audit_report.txt", "w", encoding="utf-8").write(txt); print(txt)
