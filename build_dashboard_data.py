"""
Builds the small files the dashboard reads (data/dashboard/). Nothing is deleted: every game keeps a `verified` flag.

Needs (run these first): features.py, empirical_upsets.py, empirical_upsets.py verified
Writes data/dashboard/:
  upsets.parquet      one row per game that is an upset (alpha 0.10) in the all-games run or the verified run
  curve_bins.csv      class x underdog colour x band x verified x gap bin: games, underdog wins/draws/losses
  yearly.csv          class x year x verified: games, upsets (alpha 0.10), rate
  tournaments.csv     tournament x year: games, upsets, rate (tournaments with >= 50 games)
  critical gaps       copies of the critical_gap_*.csv files
"""
import os, shutil
import numpy as np, pandas as pd

P, OUT = os.path.join("data", "processed"), os.path.join("data", "dashboard")
os.makedirs(OUT, exist_ok=True)

g = pd.read_parquet(os.path.join(P, "analysis_games.parquet"))
g = g[g["has_underdog"]].copy()
assert g["game_url"].is_unique, "game_url is not unique"

# verified = both PGN ratings confirmed by the FIDE list within 100 points (same rule as empirical_upsets.py verified)
ok = lambda s: g[f"{s}_elo_fide"].notna() & ((g[f"{s}_elo_final"] - g[f"{s}_elo_fide"]).abs() <= 100)
g["verified"] = (ok("white") & ok("black")).to_numpy()

ua = pd.read_parquet(os.path.join(P, "upsets_empirical.parquet"))
g = g.merge(ua.rename(columns={"q_emp": "q_all", "upset_emp_0.10": "upset_all"})[["game_url", "q_all", "upset_all"]],
            on="game_url", how="left")
vp = os.path.join(P, "upsets_empirical_verified.parquet")
if os.path.exists(vp):
    uv = pd.read_parquet(vp)
    g = g.merge(uv.rename(columns={"q_emp": "q_verified", "upset_emp_0.10": "upset_verified"})[["game_url", "q_verified", "upset_verified"]],
                on="game_url", how="left")
else:
    print("WARNING: upsets_empirical_verified.parquet missing - run: python empirical_upsets.py verified")
    g["q_verified"], g["upset_verified"] = np.nan, False
g["upset_all"] = g["upset_all"].fillna(False).astype(bool)
g["upset_verified"] = g["upset_verified"].fillna(False).astype(bool)
g["year"] = g["year"].astype(int)
g["tournament"] = g["broadcast_name"].astype(str)

# ---- upsets table ------------------------------------------------------------------------------
want = ["game_url", "tournament", "year", "game_date", "tc_class", "band", "white", "black", "white_elo_final", "black_elo_final",
        "gap", "avg_elo", "underdog_color", "underdog_score", "n_plies", "result", "q_all", "q_verified", "verified",
        "upset_all", "upset_verified", "elo_mismatch_any", "tc_borderline"]
have = [c for c in want if c in g.columns]
print("columns not found (skipped):", [c for c in want if c not in g.columns])
u = g[g["upset_all"] | g["upset_verified"]][have].copy()
u["q_best"] = u["q_verified"].where(u["verified"], u["q_all"])
u = u.sort_values("q_best").reset_index(drop=True)
u["rarity_rank"] = np.arange(1, len(u) + 1)
u.to_parquet(os.path.join(OUT, "upsets.parquet"), index=False)

# ---- aggregates --------------------------------------------------------------------------------
g["u_win"] = (g["underdog_score"] == 1).astype(int)
g["u_draw"] = (g["underdog_score"] == 0.5).astype(int)
g["u_loss"] = (g["underdog_score"] == 0).astype(int)
cb = (g.groupby(["tc_class", "underdog_color", "band", "verified", "gap_bin"])
        .agg(games=("u_win", "size"), wins=("u_win", "sum"), draws=("u_draw", "sum"), losses=("u_loss", "sum")).reset_index())
cb.to_csv(os.path.join(OUT, "curve_bins.csv"), index=False)

g["is_upset"] = np.where(g["verified"], g["upset_verified"], g["upset_all"])      # headline flag: verified run for verified games
yr = (g.groupby(["tc_class", "year", "verified"]).agg(games=("is_upset", "size"), upsets=("is_upset", "sum")).reset_index())
yr["rate"] = yr["upsets"] / yr["games"]
yr.to_csv(os.path.join(OUT, "yearly.csv"), index=False)

tn = (g.groupby(["tournament", "year", "tc_class"]).agg(games=("is_upset", "size"), upsets=("is_upset", "sum"),
      mean_gap=("gap", "mean"), mean_elo=("avg_elo", "mean")).reset_index())
tn = tn[tn["games"] >= 50].copy(); tn["rate"] = tn["upsets"] / tn["games"]
tn.sort_values("rate", ascending=False).to_csv(os.path.join(OUT, "tournaments.csv"), index=False)

for f in ["critical_gap_empirical.csv", "critical_gap_empirical_verified.csv", "critical_gap_decisive.csv",
          "critical_gap_decisive_verified.csv", "critical_gap_isotonic_verified.csv", "critical_gap_diff_empirical.csv", "critical_gap_diff_empirical_verified.csv", "calibration_empirical.csv", "calibration_empirical_verified.csv", "kappa_table.csv",
          "fitted_curves.csv", "fitted_curves_verified.csv", "model_coefficients_empirical.csv", "model_coefficients_empirical_verified.csv"]:
    for src in (".", P):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), os.path.join(OUT, f)); break

print(f"games {len(g)}, verified {int(g.verified.sum())} ({g.verified.mean():.1%})")
print(f"upsets table: {len(u)} rows (all-run {int(u.upset_all.sum())}, verified-run {int(u.upset_verified.sum())})")
print("files in", OUT, ":", sorted(os.listdir(OUT)))
for f in os.listdir(OUT):
    print(f"  {f}: {os.path.getsize(os.path.join(OUT, f)) / 1e6:.2f} MB")