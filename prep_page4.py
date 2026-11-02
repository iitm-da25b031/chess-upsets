"""
Data preparation for dashboard page 4 ("The upsets"). Run after build_dashboard_data.py and prep_replay.py.
python-chess is used here only (data preparation); the dashboard itself never imports it.

Reads : data/processed/analysis_games.parquet, data/dashboard/upsets.parquet, data/dashboard/upset_moves.parquet,
        data/interim/moves_*.parquet (movetext of every game)
Writes: data/dashboard/opening_lines.csv   depth(plies 1-8), tc_class, underdog_color, line, games, upsets   (lines with >= 30 games)
        data/dashboard/opening_boards.csv  line, placement   (board position after each line, for the opening explorer)
        data/dashboard/length_bins.csv     tc_class, underdog_color, is_upset, bin (plies, lower edge, 150 = "150+"), games
        data/dashboard/upsets.sqlite       the 500 rarest upsets with every board position (read-only in the app)
"""
import glob, os, re, sqlite3
import numpy as np, pandas as pd
import chess

D, P = os.path.join("data", "dashboard"), os.path.join("data", "processed")
TOP = 500          # games kept for the replay
MIN_LINE = 30      # storage floor only: the app shows a line when its expected number of upsets is at least 5
MAXD = 8           # plies stored (4 moves each)

g = pd.read_parquet(os.path.join(P, "analysis_games.parquet"))
g = g[g["has_underdog"]].copy()
u = pd.read_parquet(os.path.join(D, "upsets.parquet"))
u["is_upset"] = np.where(u["verified"], u["upset_verified"], u["upset_all"])
g["is_upset"] = g["game_url"].isin(set(u.loc[u["is_upset"], "game_url"]))
g["underdog_color"] = g["underdog_color"].astype(str).str.lower()
print(f"{len(g)} games with an underdog, {int(g.is_upset.sum())} upsets")

# ---- game length ---------------------------------------------------------------------------------
g["bin"] = (np.minimum(g["n_plies"], 150) // 10 * 10).astype(int)
lb = g.groupby(["tc_class", "underdog_color", "is_upset", "bin"]).size().rename("games").reset_index()
lb.to_csv(os.path.join(D, "length_bins.csv"), index=False)

# ---- opening lines: first 8 plies from the movetext (text only, no chess rules needed) -------------
COMMENT, VARIATION, NAG, NUM = re.compile(r"\{[^}]*\}"), re.compile(r"\([^)]*\)"), re.compile(r"\$\d+"), re.compile(r"^\d+\.+")

def first_plies(text, k=8):
    text = NAG.sub(" ", VARIATION.sub(" ", COMMENT.sub(" ", text)))
    out = []
    for tok in text.split():
        tok = NUM.sub("", tok).rstrip("!?")
        if not tok or tok in ("1-0", "0-1", "1/2-1/2", "*"):
            continue
        out.append(tok)
        if len(out) == k:
            break
    return out

assert first_plies("1. e4 { [%clk 0:03:00] } c5 { [%clk 0:03:00] } 2. Nf3 d6 3. d4?! cxd4 1-0", 6) == ["e4", "c5", "Nf3", "d6", "d4", "cxd4"]
assert first_plies("1.e4 e5 2.Nf3 Nc6") == ["e4", "e5", "Nf3", "Nc6"]

keep = set(g["game_url"])
rows = []
for f in sorted(glob.glob(os.path.join("data", "interim", "moves_*.parquet"))):
    m = pd.read_parquet(f)
    m = m[m["game_url"].isin(keep)]
    for url, text in zip(m["game_url"], m["movetext"]):
        pl = first_plies(text)
        if len(pl) == MAXD:
            rows.append((url, " ".join(pl)))
    print(f"  {os.path.basename(f)}: {len(m)} games", flush=True)
ln = pd.DataFrame(rows, columns=["game_url", "line8"]).drop_duplicates("game_url")
x = g[["game_url", "tc_class", "underdog_color", "is_upset"]].merge(ln, on="game_url")
print(f"opening line found for {len(x)} of {len(g)} games")
out = []
for depth in range(1, MAXD + 1):
    x["line"] = x["line8"].str.split().str[:depth].str.join(" ")
    t = x.groupby(["tc_class", "underdog_color", "line"]).agg(games=("is_upset", "size"), upsets=("is_upset", "sum")).reset_index()
    t.insert(0, "depth", depth)
    out.append(t[t["games"] >= MIN_LINE])
    cov = t.loc[t["games"] >= MIN_LINE, "games"].sum() / len(x)
    print(f"depth {depth} plies: {len(t)} distinct lines, {cov:.1%} of games are in a stored line (>= {MIN_LINE} games)")
ol = pd.concat(out)
ol.to_csv(os.path.join(D, "opening_lines.csv"), index=False)
boards = []
for line in sorted(ol["line"].unique()):
    b = chess.Board()
    try:
        for s_ in line.split():
            b.push_san(s_)
        boards.append((line, b.board_fen()))
    except ValueError:
        print("illegal line skipped:", line)
pd.DataFrame(boards, columns=["line", "placement"]).to_csv(os.path.join(D, "opening_boards.csv"), index=False)
print(f"{len(ol)} opening-line rows, {len(boards)} board positions")

# ---- replay database: the rarest upsets with a board position for every ply -------------------------
mv = pd.read_parquet(os.path.join(D, "upset_moves.parquet"))
top = u.sort_values("q_best").head(TOP * 2).merge(mv[mv["replay_ok"]], on="game_url").head(TOP).reset_index(drop=True)
pos = []
for san in top["moves_san"]:
    b = chess.Board()
    fens = [b.board_fen()]
    for s in san.split():
        b.push_san(s)
        fens.append(b.board_fen())
    pos.append("|".join(fens))
top["positions"] = pos
top["rank"] = np.arange(1, len(top) + 1)
top["underdog_color"] = top["underdog_color"].astype(str).str.lower()
ren = {"white_elo_final": "white_elo", "black_elo_final": "black_elo"}
top = top.rename(columns=ren)
cols = ["rank", "game_url", "tournament", "year", "tc_class", "underdog_color", "white", "black", "white_elo", "black_elo", "result", "q_best", "n_plies", "moves_san", "positions"]
top = top[[c for c in cols if c in top.columns]]
path = os.path.join(D, "upsets.sqlite")
if os.path.exists(path):
    os.remove(path)
con = sqlite3.connect(path)
top.to_sql("upsets", con, index=False)
con.execute("CREATE INDEX ix_upsets_tc ON upsets (tc_class, underdog_color, rank)")
con.commit(); con.close()
print(f"{len(top)} games -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")