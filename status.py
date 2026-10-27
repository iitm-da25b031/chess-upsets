"""Show what has been downloaded and converted, and what is still missing.
Usage: python status.py        (writes status_report.txt too)"""
import glob, os, re

def n(pattern): return sorted(glob.glob(pattern))
def gb(files): return sum(os.path.getsize(f) for f in files) / 1e9
out = []
def P(s=""): out.append(s)

expected = []
if os.path.exists("urls_lichess.txt"):
    expected = [re.search(r"(\d{4}-\d{2})", u).group(1) for u in open("urls_lichess.txt") if u.strip()]
raw = n("data/raw/lichess/*.pgn.zst")
raw_m = {re.search(r"(\d{4}-\d{2})", f).group(1): f for f in raw}
part = n("data/raw/lichess/*.part")
games = {re.search(r"(\d{4}-\d{2})", f).group(1) for f in n("data/interim/games_*.parquet")}
moves = {re.search(r"(\d{4}-\d{2})", f).group(1) for f in n("data/interim/moves_*.parquet")}
fz = n("data/raw/fide/*.zip")
fp = n("data/interim/fide_*.parquet")

P(f"Lichess raw files:      {len(raw)} of {len(expected) or '?'}  ({gb(raw):.2f} GB)")
P(f"Lichess games parquet:  {len(games)}   moves parquet: {len(moves)}")
P(f"FIDE zips:              {len(fz)} of 243   ({gb(fz):.2f} GB)")
P(f"FIDE parquet files:     {len(fp)}")
miss_raw = [m for m in expected if m not in raw_m]
miss_conv = [m for m in raw_m if m not in games or m not in moves]
P(f"\nLichess months NOT downloaded ({len(miss_raw)}): {', '.join(miss_raw) if miss_raw else 'none'}")
P(f"Lichess months downloaded but NOT converted ({len(miss_conv)}): {', '.join(sorted(miss_conv)) if miss_conv else 'none'}")
if part: P("unfinished Lichess download(s): " + ", ".join(os.path.basename(p) for p in part))
small = [os.path.basename(f) + f" ({os.path.getsize(f)/1e6:.1f} MB)" for f in raw if os.path.getsize(f) < 1e6]
if small: P("suspiciously small Lichess files: " + ", ".join(small))
txt = "\n".join(out)
open("status_report.txt", "w", encoding="utf-8").write(txt)
print(txt)
