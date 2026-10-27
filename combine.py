"""Combine all months into three tables (run AFTER run_all.py has converted everything).
Usage: python combine.py
Writes (in data/interim/):
  all_games.parquet     every game from every month (no move text)
  fide_ratings.parquet  one row per (fideid, period): standard/rapid/blitz rating, games, k, flag
                        -- only for players who appear in at least one game
  fide_players.parquet  one row per fideid: name, country, sex, birth year, title (latest seen)
Also prints a games-per-month table."""
import glob, os, re
import pandas as pd

D = os.path.join("data", "interim")

# 1. games -------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(D, "games_*.parquet")))
games = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
games.to_parquet(os.path.join(D, "all_games.parquet"), index=False)
print(f"{len(games)} games from {len(files)} months -> all_games.parquet")
print(games.groupby("source_month").size().to_string())

ids = set(games.get("white_fide_id", pd.Series(dtype="string")).dropna()) | \
      set(games.get("black_fide_id", pd.Series(dtype="string")).dropna())
print(f"\n{len(ids)} distinct FIDE ids appear in games")

# 2. FIDE: keep only players who appear in games -------------------------------
periods = sorted({re.search(r"(\d{4}-\d{2})", f).group(1) for f in glob.glob(os.path.join(D, "fide_*_*.parquet"))
                  if re.search(r"fide_(standard|rapid|blitz)_", os.path.basename(f))})
static, monthly = [], []
for p in periods:
    merged = None
    for lst in ("standard", "rapid", "blitz"):
        f = os.path.join(D, f"fide_{lst}_{p}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        d = d[d.fideid.isin(ids)]
        static.append(d[["fideid", "name", "country", "sex", "birthday", "title", "period"]])
        keep = d[["fideid"] + [c for c in (f"{lst}_rating", f"{lst}_games", f"{lst}_k", "flag") if c in d]]
        keep = keep.rename(columns={"flag": f"{lst}_flag"})
        merged = keep if merged is None else merged.merge(keep, on="fideid", how="outer")
    if merged is not None:
        merged["period"] = p
        monthly.append(merged)
ratings = pd.concat(monthly, ignore_index=True)
ratings.to_parquet(os.path.join(D, "fide_ratings.parquet"), index=False)
players = (pd.concat(static, ignore_index=True).sort_values("period")
             .drop_duplicates("fideid", keep="last").drop(columns="period"))
players.to_parquet(os.path.join(D, "fide_players.parquet"), index=False)
print(f"fide_ratings.parquet: {len(ratings)} rows over {len(periods)} periods")
print(f"fide_players.parquet: {len(players)} players")
print(f"ids in games but never found in any FIDE list: {len(ids - set(players.fideid))}")
