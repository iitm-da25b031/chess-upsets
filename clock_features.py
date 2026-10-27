"""Estimate each game's time control from its clock readings.

For every game: base_hat (starting time, seconds) and inc_hat (increment, seconds).
 * inc_hat  = largest clock INCREASE between two consecutive moves of the same player,
              looking at the first 30 moves of each side (before any move-40 time bonus);
              jumps above 600 s are ignored as clock glitches.
 * base_hat = white's first clock reading - inc_hat + 1
              (after the first move the clock shows base + increment - time spent).
Usage: python clock_features.py       (reads data/interim/moves_*.parquet)
Writes data/interim/clock_features.parquet"""
import glob, os
import pandas as pd
from parse_pgn import MOVE, clk_seconds

def estimate(text):
    if not isinstance(text, str) or "%clk" not in text:
        return None, None, 0
    w, b = [], []
    for num, dots, san, com in MOVE.findall(text):
        if not com:
            continue
        c = clk_seconds(com)
        if c is not None:
            (w if dots == "." else b).append(c)
    deltas = []
    for s in (w[:30], b[:30]):
        deltas += [s[i] - s[i - 1] for i in range(1, len(s))]
    pos = [d for d in deltas if 0 < d <= 600]
    inc = max(pos) if pos else 0.0
    first = w[0] if w else (b[0] if b else None)
    base = first - inc + 1 if first is not None else None
    if base is not None and not (30 <= base <= 14400):
        base = None
    return base, inc, len(w) + len(b)

if __name__ == "__main__":
    rows = []
    files = sorted(glob.glob(os.path.join("data", "interim", "moves_*.parquet")))
    for i, f in enumerate(files, 1):
        m = pd.read_parquet(f)
        est = [estimate(t) for t in m["movetext"]]
        rows.append(pd.DataFrame({"game_url": m["game_url"],
                                  "base_hat": [e[0] for e in est],
                                  "inc_hat": [e[1] for e in est],
                                  "n_clk": [e[2] for e in est]}))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}  {len(m)} games", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(os.path.join("data", "interim", "clock_features.parquet"), index=False)
    print(len(out), "games ->", "data/interim/clock_features.parquet")
    print("with a clock estimate:", out.base_hat.notna().sum())
