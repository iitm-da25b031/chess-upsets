"""Data for page 3 (before/after cleaning, missing data).   Run after features.py:   python prep_page3.py
Reads  data/interim/all_games.parquet (every parsed game), clean_games.parquet, data/processed/analysis_games.parquet
Writes data/dashboard/before_after.csv    variable, stage (raw | clean), bin_lo, n      (game length in half-moves, average rating)
       data/dashboard/missing_by_year.csv year, measure, n, missing                      (share of games with a missing value, source vs repaired)
       data/dashboard/missing_tests.csv   response, factor, n, categories, chi2, dof, p, cramers_v   (is missingness random?)
Nothing is changed in the input files."""
import os
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency

D, PR, OUT = os.path.join("data", "interim"), os.path.join("data", "processed"), os.path.join("data", "dashboard")
os.makedirs(OUT, exist_ok=True)

raw = pd.read_parquet(os.path.join(D, "all_games.parquet"), columns=["white_elo", "black_elo", "n_plies", "game_date", "source_month", "tc_base_s"])
cln = pd.read_parquet(os.path.join(D, "clean_games.parquet"), columns=["period", "analysis_ok", "tc_class", "broadcast_name"])
ana = pd.read_parquet(os.path.join(PR, "analysis_games.parquet"), columns=["n_plies", "avg_elo"])
print(f"raw {len(raw):,}   clean {len(cln):,}   analysis {len(ana):,}   (analysis set should be 858,123)")

# ---- 1. before / after: two distributions -------------------------------------------------------
def hist(values, width, lo, hi, variable, stage):
    v = pd.Series(values).dropna().to_numpy(float)
    b = np.clip(np.floor(v / width) * width, lo, hi).astype(int)
    s = pd.Series(b).value_counts().sort_index()
    return pd.DataFrame({"variable": variable, "stage": stage, "bin_lo": s.index, "n": s.values})
we = pd.to_numeric(raw["white_elo"], errors="coerce"); be = pd.to_numeric(raw["black_elo"], errors="coerce")
raw_avg = ((we + be) / 2).where(we.gt(0) & be.gt(0))
ba = pd.concat([hist(raw["n_plies"], 10, 0, 200, "length", "raw"), hist(ana["n_plies"], 10, 0, 200, "length", "clean"),
                hist(raw_avg, 100, 800, 3900, "rating", "raw"), hist(ana["avg_elo"], 100, 800, 3900, "rating", "clean")])
ba.to_csv(os.path.join(OUT, "before_after.csv"), index=False)

# ---- 2. missing share by year, in the source and after repair ------------------------------------
yr_raw = raw["game_date"].dt.year.fillna(pd.to_numeric(raw["source_month"].astype(str).str[:4], errors="coerce"))
yr_cln = pd.to_numeric(cln["period"].astype(str).str[:4], errors="coerce")
rows = []
for measure, year, flag in (("elo_source", yr_raw, we.isna() | be.isna()), ("tc_source", yr_raw, raw["tc_base_s"].isna()),
                            ("elo_repaired", yr_cln, ~cln["analysis_ok"]), ("tc_repaired", yr_cln, cln["tc_class"].eq("unknown"))):
    d = pd.DataFrame({"year": year, "m": flag}).dropna(subset=["year"])
    d = d[d.year >= 2019].groupby("year")["m"].agg(n="size", missing="sum").reset_index()
    d.insert(1, "measure", measure); rows.append(d)
mb = pd.concat(rows); mb["year"] = mb["year"].astype(int)
mb.to_csv(os.path.join(OUT, "missing_by_year.csv"), index=False)

# ---- 3. is the missingness random?  chi-square test of independence, with Cramer's V as the effect size ------------
c = cln.copy(); c["year"] = yr_cln
c = c[c["year"] >= 2019]
c["elo_missing"], c["tc_missing"] = ~c["analysis_ok"], c["tc_class"].eq("unknown")
big = c["broadcast_name"].map(c["broadcast_name"].value_counts()) >= 100
tests = []
for resp in ("elo_missing", "tc_missing"):
    for factor, d in (("year", c), ("event", c[big])):
        key = d["year"] if factor == "year" else d["broadcast_name"]
        tab = pd.crosstab(key, d[resp])
        if tab.shape[1] < 2: continue
        chi2, p, dof, _ = chi2_contingency(tab)
        v = np.sqrt(chi2 / (tab.to_numpy().sum() * (min(tab.shape) - 1)))
        tests.append(dict(response=resp, factor=factor, n=int(tab.to_numpy().sum()), categories=tab.shape[0], chi2=chi2, dof=dof, p=p, cramers_v=v))
pd.DataFrame(tests).to_csv(os.path.join(OUT, "missing_tests.csv"), index=False)
print(pd.DataFrame(tests).round(4).to_string()); print("written to", OUT)
