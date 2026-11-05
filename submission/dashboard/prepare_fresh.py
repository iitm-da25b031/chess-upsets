"""Builds the tables of the dashboard pages that are not copied from the model outputs.

Usage (from this folder):  python prepare_fresh.py
Reads the pipeline outputs of notebooks 01-03 below CHESS_DATA_ROOT (default ../data):
    interim/all_games.parquet, interim/clean_games.parquet, clean/analysis_games.parquet, clean/upsets_empirical.parquet
Writes data/dashboard/: funnel.csv, before_after.csv, missing_by_year.csv, missing_tests.csv, length_bins.csv,
    tournaments.csv, players_by_class.csv
An upset is a game with upset_emp_0.10 (alpha = 0.10 rule). Players are identified by FIDE id, otherwise by name;
only public tournament records (name, rating, results) are used.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.environ.get("CHESS_DATA_ROOT", os.path.join(os.path.dirname(HERE), "data")))
OUT = os.path.join(HERE, "data", "dashboard")
os.makedirs(OUT, exist_ok=True)

raw = pd.read_parquet(os.path.join(ROOT, "interim", "all_games.parquet"),
                      columns=["white_elo", "black_elo", "n_plies", "game_date", "source_month", "time_control"])
cln = pd.read_parquet(os.path.join(ROOT, "interim", "clean_games.parquet"), columns=["period", "analysis_ok", "tc_class", "broadcast_name"])
ana = pd.read_parquet(os.path.join(ROOT, "clean", "analysis_games.parquet"))
ups = pd.read_parquet(os.path.join(ROOT, "clean", "upsets_empirical.parquet"), columns=["game_url", "upset_emp_0.10"])
print(f"raw {len(raw):,}   clean {len(cln):,}   analysis {len(ana):,}")

# funnel -------------------------------------------------------------------------------------------
pd.DataFrame({"step": ["All games parsed", "After cleaning", "Both ratings known", "Analysis set", "Ratings checked against FIDE"],
              "n": [len(raw), len(cln), int(cln["analysis_ok"].sum()), len(ana), int(ana["verified"].sum())]}).to_csv(os.path.join(OUT, "funnel.csv"), index=False)


def hist(values, width, lo, hi, variable, stage):
    v = pd.Series(values).dropna().to_numpy(float)
    b = np.clip(np.floor(v / width) * width, lo, hi).astype(int)
    s = pd.Series(b).value_counts().sort_index()
    return pd.DataFrame({"variable": variable, "stage": stage, "bin_lo": s.index, "n": s.values})


# before / after: game length and average rating, raw files against the analysis set -----------------
we, be = pd.to_numeric(raw["white_elo"], errors="coerce"), pd.to_numeric(raw["black_elo"], errors="coerce")
raw_avg = ((we + be) / 2).where(we.gt(0) & be.gt(0))
pd.concat([hist(raw["n_plies"], 10, 0, 200, "length", "raw"), hist(ana["n_plies"], 10, 0, 200, "length", "clean"),
           hist(raw_avg, 100, 800, 3900, "rating", "raw"), hist(ana["avg_elo"], 100, 800, 3900, "rating", "clean")]
          ).to_csv(os.path.join(OUT, "before_after.csv"), index=False)

# missing share by year, in the source and after repair -----------------------------------------------
yr_raw = raw["game_date"].dt.year.fillna(pd.to_numeric(raw["source_month"].astype(str).str[:4], errors="coerce"))
yr_cln = pd.to_numeric(cln["period"].astype(str).str[:4], errors="coerce")
rows = []
for measure, year, flag in (("elo_source", yr_raw, ~(we.gt(0) & be.gt(0))), ("tc_source", yr_raw, (raw["time_control"].isna() | raw["time_control"].astype(str).str.strip().isin(["", "-", "?"]))),
                            ("elo_repaired", yr_cln, ~cln["analysis_ok"]), ("tc_repaired", yr_cln, cln["tc_class"].eq("unknown"))):
    d = pd.DataFrame({"year": year, "m": flag}).dropna(subset=["year"])
    d = d[d.year >= 2019].groupby("year")["m"].agg(n="size", missing="sum").reset_index()
    d.insert(1, "measure", measure)
    rows.append(d)
mb = pd.concat(rows)
mb["year"] = mb["year"].astype(int)
mb.to_csv(os.path.join(OUT, "missing_by_year.csv"), index=False)

# is the missingness random? chi-square test of independence with Cramer's V ----------------------------
c = cln.copy()
c["year"] = yr_cln
c = c[c["year"] >= 2019]
c["elo_missing"], c["tc_missing"] = ~c["analysis_ok"], c["tc_class"].eq("unknown")
big = c["broadcast_name"].map(c["broadcast_name"].value_counts()) >= 100
tests = []
for resp in ("elo_missing", "tc_missing"):
    for factor, d in (("year", c), ("event", c[big])):
        tab = pd.crosstab(d["year"] if factor == "year" else d["broadcast_name"], d[resp])
        if tab.shape[1] < 2:
            continue
        chi2, p, dof, _ = chi2_contingency(tab)
        v = np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1)))
        tests.append(dict(response=resp, factor=factor, n=int(tab.to_numpy().sum()), categories=tab.shape[0], chi2=chi2, dof=dof, p=p, cramers_v=v))
pd.DataFrame(tests).to_csv(os.path.join(OUT, "missing_tests.csv"), index=False)

# games with an underdog and their upset flag ----------------------------------------------------------
g = ana[ana["has_underdog"]].copy()
g["is_upset"] = g["game_url"].isin(set(ups.loc[ups["upset_emp_0.10"].astype(bool), "game_url"]))
g["underdog_color"] = g["underdog_color"].astype(str).str.lower()

lb = g.assign(bin=(np.minimum(g["n_plies"], 150) // 10 * 10).astype(int)).groupby(["tc_class", "underdog_color", "is_upset", "bin"]).size().rename("games").reset_index()
lb.to_csv(os.path.join(OUT, "length_bins.csv"), index=False)

tn = g.groupby(["broadcast_name", "year", "tc_class"]).agg(games=("is_upset", "size"), upsets=("is_upset", "sum"),
                                                           mean_gap=("gap", "mean"), mean_elo=("avg_elo", "mean")).reset_index()
tn = tn[tn["games"] >= 50].rename(columns={"broadcast_name": "tournament"})
tn["rate"] = tn["upsets"] / tn["games"]
tn.sort_values("rate", ascending=False).to_csv(os.path.join(OUT, "tournaments.csv"), index=False)

w = g["underdog_color"] == "white"
g["name"] = np.where(w, g["white"], g["black"])
g["fide"] = np.where(w, g["white_fide_id"], g["black_fide_id"])
g["key"] = pd.Series(g["fide"], index=g.index).astype("string").fillna(g["name"].astype("string"))
g["own_elo"] = np.where(w, g["white_elo_final"], g["black_elo_final"])
names = g.groupby("key")["name"].agg(lambda s: s.mode().iloc[0])
t = g.groupby(["key", "tc_class"]).agg(n=("is_upset", "size"), upsets=("is_upset", "sum"), mean_elo=("own_elo", "mean")).reset_index()
t = t[t["n"] >= 10]
t["name"] = t["key"].map(names)
t["mean_elo"] = t["mean_elo"].round(0)
t[["key", "name", "tc_class", "n", "upsets", "mean_elo"]].to_csv(os.path.join(OUT, "players_by_class.csv"), index=False)

# upsets by the underdog's federation: FIDE id of the underdog joined to the FIDE player list ---------------
fp = pd.read_parquet(os.path.join(ROOT, "interim", "fide_players.parquet"), columns=["fideid", "country"])
fed = fp.dropna(subset=["fideid"]).assign(fideid=lambda d: d["fideid"].astype(str)).drop_duplicates("fideid").set_index("fideid")["country"]
g["fed"] = g["fide"].astype("string").map(fed)
fd = g.dropna(subset=["fed"]).groupby("fed").agg(games=("is_upset", "size"), upsets=("is_upset", "sum"), players=("key", "nunique")).reset_index()
fd["rate"] = fd["upsets"] / fd["games"]
fd.sort_values("games", ascending=False).to_csv(os.path.join(OUT, "federations.csv"), index=False)
print(f"{len(tn)} tournaments, {len(t)} player x class rows, {len(lb)} length bins; tables written to {OUT}")
