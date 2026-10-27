"""Why did FIDE recover so few missing Elos, and why do PGN and FIDE Elo sometimes differ?
Usage: python diag_elo.py       (run after clean_games.py)   Writes diag_elo_report.txt"""
import os
import numpy as np, pandas as pd

D = os.path.join("data", "interim")
k = pd.read_parquet(os.path.join(D, "clean_games.parquet"))
fr = pd.read_parquet(os.path.join(D, "fide_ratings.parquet"),
                     columns=["fideid", "period", "standard_rating", "rapid_rating", "blitz_rating"])
fr["fideid"], fr["period"] = fr["fideid"].astype(str), fr["period"].astype(str)
fr = fr.drop_duplicates(["fideid", "period"]).set_index(["fideid", "period"])
pl = set(pd.read_parquet(os.path.join(D, "fide_players.parquet"), columns=["fideid"])["fideid"].astype(str))
out = []
def P(*a): out.append(" ".join(str(x) for x in a))

for side in ("white", "black"):
    ids = k[f"{side}_fide_id"].astype(object).astype(str)
    has_id = k[f"{side}_fide_id"].notna()
    got = fr.reindex(pd.MultiIndex.from_arrays([ids, k["period"].astype(str)]))
    any_rating = got.notna().any(axis=1).to_numpy()
    in_row = got.index.map(lambda t: True) if False else ~got.isna().all(axis=1).to_numpy()
    miss = (k[f"{side}_elo_source"] == "missing") & has_id
    P(f"== {side}: Elo missing but FIDE id present: {int(miss.sum())} games")
    P("   id never appears in any FIDE list (unrated / below floor / invalid id):",
      int((miss & ~ids.isin(pl)).sum()))
    P("   id known, but game date outside the downloaded FIDE months:", int((miss & ids.isin(pl) & k["date_outside_fide_range"]).sum()))
    P("   id known, date in range, but no rating in that month's lists:",
      int((miss & ids.isin(pl) & ~k["date_outside_fide_range"] & ~any_rating).sum()))
    P("   a rating exists that month but not in the list matching the time control:",
      int((miss & any_rating).sum()), "| of these time control unknown:", int((miss & any_rating & (k.tc_class == 'unknown')).sum()))
    # PGN vs FIDE disagreement
    both = (k[f"{side}_elo_source"] == "pgn") & k[f"{side}_elo_fide"].notna()
    d = (k[f"{side}_elo_final"] - k[f"{side}_elo_fide"]).abs()
    big = both & (d > 100)
    P(f"   PGN Elo and FIDE Elo both present: {int(both.sum())}; differ by >100: {int(big.sum())} ({big.sum()/max(both.sum(),1):.1%})")
    P("   difference percentiles (p50, p90, p99):", [float(d[both].quantile(q)) for q in (.5, .9, .99)])
    if big.any():
        pg = k.loc[big, f"{side}_elo_final"].to_numpy(float)
        cols = got.loc[big.to_numpy()][["standard_rating", "rapid_rating", "blitz_rating"]].to_numpy(dtype="float64", na_value=np.nan)
        diffs = np.abs(cols - pg[:, None])
        diffs = np.where(np.isnan(diffs), 1e9, diffs)
        near = diffs.min(axis=1) <= 25
        P("   of the big differences, PGN Elo matches ANOTHER list within 25 points (list confusion):", int(near.sum()),
          "| matches none (genuine difference or stale rating):", int((~near).sum()))
        P("   big differences by time-control class:", k.loc[big, "tc_class"].value_counts().to_dict())

none = (k["white_elo_source"] == "missing") & (k["black_elo_source"] == "missing")
P(f"\n== both players without any Elo: {int(none.sum())} games ({none.mean():.1%} of clean games)")
yr = k["game_date"].dt.year
P("share of games missing at least one Elo, by year:")
P((~k["analysis_ok"]).groupby(yr).mean().round(3).to_string())
P("\nshare of games missing at least one Elo, by time-control class:")
P((~k["analysis_ok"]).groupby(k["tc_class"]).mean().round(3).to_string())
P("\ntop 12 broadcasts by games missing an Elo:")
P(k.loc[~k["analysis_ok"], "broadcast_name"].value_counts().head(12).to_string())
txt = "\n".join(out)
open("diag_elo_report.txt", "w", encoding="utf-8").write(txt)
print(txt)
