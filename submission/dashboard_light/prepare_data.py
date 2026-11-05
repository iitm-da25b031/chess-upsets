"""Builds the tables in data/ that the dashboard reads.

Usage (from this folder):
    python prepare_data.py              all parts
    python prepare_data.py tables       copy the figure tables (../viz/viz_data, ../reports)
    python prepare_data.py openings     opening lines and board positions for the opening pages
    python prepare_data.py replay       the rarest upsets with every board position, for the replay page

The openings and replay parts read the pipeline outputs of notebooks 01-03 below CHESS_DATA_ROOT (default ../data):
clean/analysis_games.parquet, clean/upsets_empirical.parquet and interim/moves_*.parquet. They need python-chess,
which the dashboard itself does not use. An upset is a game with upset_emp_0.10 (the default alpha = 0.10 rule).
"""
import glob
import io
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")
SUB = os.path.dirname(HERE)
DATA_ROOT = os.path.abspath(os.environ.get("CHESS_DATA_ROOT", os.path.join(SUB, "data")))

MODEL_TABLES = ["critical_gap_empirical", "critical_gap_empirical_verified", "critical_gap_empirical_sens",
                "critical_gap_isotonic", "critical_gap_isotonic_verified", "critical_gap_isotonic_sens",
                "critical_gap_diff_empirical", "calibration_empirical", "fitted_curves",
                "critical_gap_empirical_interaction", "critical_gap_main_vs_interaction", "model_fit_by_band", "ordering_check"]
REPORT_TABLES = ["cleaning_log", "missingness_tests"]
MAX_PLIES = 8           # opening depth stored (four moves each side)
MIN_LINE = 30           # a line is stored when at least this many games follow it
PER_GROUP = 60          # rarest upsets kept per time control and underdog colour for the replay

COMMENT, VARIATION, NAG, NUM = re.compile(r"\{[^}]*\}"), re.compile(r"\([^)]*\)"), re.compile(r"\$\d+"), re.compile(r"^\d+\.+")
RESULTS = ("1-0", "0-1", "1/2-1/2", "*")
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)


def tables():
    """Copy the small figure tables next to the app so that the dashboard folder is self-contained."""
    os.makedirs(os.path.join(OUT, "viz_data"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "model_outputs"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "reports"), exist_ok=True)
    for f in glob.glob(os.path.join(SUB, "viz", "viz_data", "*.csv")):
        shutil.copy2(f, os.path.join(OUT, "viz_data"))
    for t in MODEL_TABLES:
        shutil.copy2(os.path.join(SUB, "reports", "model_outputs", t + ".csv"), os.path.join(OUT, "model_outputs"))
    for t in REPORT_TABLES:
        shutil.copy2(os.path.join(SUB, "reports", t + ".csv"), os.path.join(OUT, "reports"))
    print("figure tables copied")


def games_with_upset_flag():
    cols = ["game_url", "tc_class", "underdog_color", "has_underdog", "white", "black", "white_elo_final", "black_elo_final",
            "gap", "score_white", "broadcast_name", "year", "set_up"]
    g = pd.read_parquet(os.path.join(DATA_ROOT, "clean", "analysis_games.parquet"), columns=cols)
    g = g[g["has_underdog"]].drop(columns="has_underdog")
    u = pd.read_parquet(os.path.join(DATA_ROOT, "clean", "upsets_empirical.parquet"), columns=["game_url", "q_emp", "upset_emp_0.10"])
    g = g.merge(u, on="game_url", how="inner")
    g["is_upset"] = g["upset_emp_0.10"].astype(bool)
    g["underdog_color"] = g["underdog_color"].astype(str).str.lower()
    g["tc_class"] = g["tc_class"].astype(str)
    return g


def first_plies(text, k=MAX_PLIES):
    """First k moves (SAN) of a PGN movetext; comments, variations, NAGs and move numbers are removed."""
    for head in (text[:2500], text):
        t = NAG.sub(" ", VARIATION.sub(" ", COMMENT.sub(" ", head)))
        out = []
        for tok in t.split():
            tok = NUM.sub("", tok).rstrip("!?")
            if not tok or tok in RESULTS:
                continue
            out.append(tok)
            if len(out) == k:
                return out
        if len(head) == len(text):
            break
    return out


def movetexts(urls):
    """Movetext of the given games, read month by month."""
    urls = set(urls)
    parts = []
    for f in sorted(glob.glob(os.path.join(DATA_ROOT, "interim", "moves_*.parquet"))):
        m = pd.read_parquet(f)
        parts.append(m[m["game_url"].isin(urls)])
    return pd.concat(parts).drop_duplicates("game_url")


def openings():
    import chess
    g = games_with_upset_flag()
    g = g[g["set_up"].astype(str).isin(["", "0", "None", "nan", "False"]) | g["set_up"].isna()]
    keep = set(g["game_url"])
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA_ROOT, "interim", "moves_*.parquet"))):
        m = pd.read_parquet(f)
        m = m[m["game_url"].isin(keep)]
        for url, text in zip(m["game_url"], m["movetext"]):
            pl = first_plies(text)
            if len(pl) == MAX_PLIES:
                rows.append((url, " ".join(pl)))
    ln = pd.DataFrame(rows, columns=["game_url", "line8"]).drop_duplicates("game_url")
    x = g[["game_url", "tc_class", "underdog_color", "is_upset"]].merge(ln, on="game_url")
    print(f"opening line found for {len(x):,} of {len(g):,} games with an underdog")
    out = []
    split = x["line8"].str.split()
    for depth in range(1, MAX_PLIES + 1):
        x["line"] = split.str[:depth].str.join(" ")
        t = x.groupby(["tc_class", "underdog_color", "line"]).agg(games=("is_upset", "size"), upsets=("is_upset", "sum")).reset_index()
        t.insert(0, "depth", depth)
        out.append(t[t["games"] >= MIN_LINE])
    ol = pd.concat(out, ignore_index=True)
    boards, cache = [], {"": chess.Board()}
    for line in sorted(ol["line"].unique(), key=lambda s: (s.count(" "), s)):
        prefix, _, last = line.rpartition(" ")
        if prefix not in cache:
            continue
        b = cache[prefix].copy(stack=False)
        try:
            b.push_san(last)
        except ValueError:
            continue
        cache[line] = b
        boards.append((line, b.board_fen()))
    bd = pd.DataFrame(boards, columns=["line", "placement"])
    ol = ol[ol["line"].isin(set(bd["line"]))]
    totals = x.groupby(["tc_class", "underdog_color"]).agg(games=("is_upset", "size"), upsets=("is_upset", "sum")).reset_index()
    totals.insert(0, "depth", 0)
    totals.insert(3, "line", "")
    ol = pd.concat([totals, ol], ignore_index=True)
    os.makedirs(os.path.join(OUT, "openings"), exist_ok=True)
    ol.to_csv(os.path.join(OUT, "openings", "opening_lines.csv"), index=False)
    bd.to_csv(os.path.join(OUT, "openings", "opening_boards.csv"), index=False)
    print(f"{len(ol):,} opening-line rows, {len(bd):,} board positions")


def replay():
    import chess
    import chess.pgn
    g = games_with_upset_flag()
    up = g[g["is_upset"]].sort_values("q_emp")
    cand = up.groupby(["tc_class", "underdog_color"], group_keys=False).head(PER_GROUP * 2)
    mv = movetexts(cand["game_url"])
    cand = cand.merge(mv, on="game_url")
    rows = []
    for r in cand.itertuples(index=False):
        try:
            game = chess.pgn.read_game(io.StringIO(r.movetext))
            board = game.board()
            if board.board_fen() != chess.STARTING_BOARD_FEN or game.errors:
                continue
            san, pos = [], [board.board_fen()]
            for m in game.mainline_moves():
                san.append(board.san(m))
                board.push(m)
                pos.append(board.board_fen())
        except Exception:
            continue
        if len(san) < 10:
            continue
        res = {1.0: "1-0", 0.0: "0-1", 0.5: "1/2-1/2"}.get(float(r.score_white), "")
        rows.append(dict(game_url=r.game_url, tournament=r.broadcast_name, year=int(r.year), tc_class=r.tc_class,
                         underdog_color=r.underdog_color, white=r.white, black=r.black, white_elo=int(r.white_elo_final),
                         black_elo=int(r.black_elo_final), gap=int(r.gap), result=res, q_emp=float(r.q_emp),
                         n_plies=len(san), moves_san=" ".join(san), positions="|".join(pos)))
    d = pd.DataFrame(rows).sort_values("q_emp")
    d = d.groupby(["tc_class", "underdog_color"], group_keys=False).head(PER_GROUP).reset_index(drop=True)
    d.insert(0, "rank", d.groupby(["tc_class", "underdog_color"]).cumcount() + 1)
    os.makedirs(os.path.join(OUT, "replay"), exist_ok=True)
    path = os.path.join(OUT, "replay", "upsets.sqlite")
    tmp = os.path.join(tempfile.mkdtemp(), "upsets.sqlite")
    con = sqlite3.connect(tmp)
    d.to_sql("upsets", con, index=False)
    con.execute("CREATE INDEX ix_upsets_tc ON upsets (tc_class, underdog_color, rank)")
    con.commit()
    con.close()
    shutil.copyfile(tmp, path)
    shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
    print(f"{len(d)} games -> data/replay/upsets.sqlite ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(d.groupby(["tc_class", "underdog_color"]).size().to_string())


if __name__ == "__main__":
    parts = sys.argv[1:] or ["tables", "openings", "replay"]
    for p in parts:
        {"tables": tables, "openings": openings, "replay": replay}[p]()
