"""Parse one Lichess broadcast file into two Parquet tables.

Usage:  python parse_pgn.py data/raw/lichess/lichess_db_broadcast_2026-09.pgn.zst
Writes: data/interim/games_2026-09.parquet   (one row per game, no move text)
        data/interim/moves_2026-09.parquet   (game_url + raw movetext, for the board later)
Streams the file, so memory stays small. Needs: pandas, pyarrow, zstandard (for .zst).
"""
import sys, os, re, io, bz2
import pandas as pd
from tc_parse import parse_tc, tc_class

TAG = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
MOVE = re.compile(r'(\d+)(\.{1,3})\s*([^\s{]+)((?:\s*\{[^}]*\})*)')
CLK = re.compile(r'\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]')
EVAL = re.compile(r'\[%eval ([^\]]+)\]')

def open_text(p):
    if p.endswith(".zst"):
        try:
            import zstandard
            r = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(open(p, "rb"))
            return io.TextIOWrapper(r, encoding="utf-8", errors="replace")
        except ImportError:
            from compression import zstd              # Python 3.14+
            return io.TextIOWrapper(zstd.open(p, "rb"), encoding="utf-8", errors="replace")
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")

def snake(t):                      # WhiteFideId -> white_fide_id, GameURL -> game_url
    t = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", t)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", t).lower()

def clk_seconds(comments):
    m = CLK.search(comments)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None

def analyse_moves(text):
    """Counts and clock facts from the movetext. No chess rules needed."""
    out = dict(n_plies=0, has_clk=False, has_eval=False,
               first_clk_w=None, first_clk_b=None, last_clk_w=None, last_clk_b=None,
               n_inacc_w=0, n_mistake_w=0, n_blunder_w=0,
               n_inacc_b=0, n_mistake_b=0, n_blunder_b=0)
    for num, dots, san, comments in MOVE.findall(text):
        side = "w" if dots == "." else "b"
        out["n_plies"] += 1
        if san.endswith("??"):   out[f"n_blunder_{side}"] += 1
        elif san.endswith("?!"): out[f"n_inacc_{side}"] += 1
        elif san.endswith("?"):  out[f"n_mistake_{side}"] += 1
        if comments:
            if EVAL.search(comments): out["has_eval"] = True
            c = clk_seconds(comments)
            if c is not None:
                out["has_clk"] = True
                if out[f"first_clk_{side}"] is None: out[f"first_clk_{side}"] = c
                out[f"last_clk_{side}"] = c
    return out

def games(f):
    tags, moves = {}, []
    for line in f:
        if line.startswith("[Event ") and (tags or moves):
            yield tags, " ".join(moves).strip()
            tags, moves = {}, []
        m = TAG.match(line)
        if m:
            tags[m.group(1)] = m.group(2)
        elif line.strip():
            moves.append(line.strip())
    if tags:
        yield tags, " ".join(moves).strip()

def main(path):
    tag = re.search(r"(\d{4}-\d{2})", os.path.basename(path)).group(1)
    rows, mv = [], []
    with open_text(path) as f:
        for i, (tags, text) in enumerate(games(f), 1):
            row = {snake(k): v for k, v in tags.items()}
            row.update(analyse_moves(text))
            rows.append(row)
            mv.append((tags.get("GameURL"), text))
            if i % 5000 == 0: print(f"  {i} games...", flush=True)
    df = pd.DataFrame(rows)
    for c in ("white_elo", "black_elo"):
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ("white_fide_id", "black_fide_id"):
        if c in df: df[c] = df[c].astype("string")
    d = pd.to_datetime(df.get("utc_date", pd.Series(index=df.index, dtype=object)), format="%Y.%m.%d", errors="coerce")
    d = d.fillna(pd.to_datetime(df.get("date", pd.Series(index=df.index, dtype=object)), format="%Y.%m.%d", errors="coerce"))
    df["game_date"] = d
    p = [parse_tc(s) for s in df.get("time_control", pd.Series([None] * len(df)))]
    df["tc_base_s"] = pd.array([x[0] for x in p], dtype="Int64")
    df["tc_inc_s"]  = pd.array([x[1] for x in p], dtype="Int64")
    df["tc_method"] = [x[2] for x in p]
    df["tc_class"]  = [tc_class(x[0], x[1]) for x in p]
    df["source_month"] = tag
    os.makedirs(os.path.join("data", "interim"), exist_ok=True)
    out_g = os.path.join("data", "interim", f"games_{tag}.parquet")
    out_m = os.path.join("data", "interim", f"moves_{tag}.parquet")
    df.to_parquet(out_g, index=False)
    pd.DataFrame(mv, columns=["game_url", "movetext"]).to_parquet(out_m, index=False)
    print(f"{len(df)} games -> {out_g}\n{len(mv)} movetexts -> {out_m}")

if __name__ == "__main__":
    main(sys.argv[1])
