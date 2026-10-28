"""
Prepares the board replay for the upsets (python-chess is used here only, for data preparation).
Needs: data/dashboard/upsets.parquet (build_dashboard_data.py) and data/interim/moves_*.parquet (parse_pgn.py)
Writes data/dashboard/upset_moves.parquet: game_url, moves_san (space separated), n_moves_ok, replay_ok, final_fen
The dashboard then steps through moves_san one by one; clock comments and annotations are already removed.
"""
import glob, io, os, sys
import pandas as pd, chess, chess.pgn

D = os.path.join("data", "dashboard")
urls = set(pd.read_parquet(os.path.join(D, "upsets.parquet"), columns=["game_url"])["game_url"])
print(f"{len(urls)} upset games to replay")

parts = []
for f in sorted(glob.glob(os.path.join("data", "interim", "moves_*.parquet"))):
    m = pd.read_parquet(f, filters=[("game_url", "in", list(urls))])
    if len(m): parts.append(m)
    print(f"  {os.path.basename(f)}: {len(m)} found", flush=True)
mv = pd.concat(parts).drop_duplicates("game_url")
print(f"movetext found for {len(mv)} of {len(urls)} games")

rows = []
for url, text in zip(mv["game_url"], mv["movetext"]):
    san, ok, fen = [], True, ""
    try:
        game = chess.pgn.read_game(io.StringIO(text))
        board = game.board()
        for mvv in game.mainline_moves():
            san.append(board.san(mvv)); board.push(mvv)
        fen = board.fen()
        ok = len(game.errors) == 0 and len(san) > 0
    except Exception:
        ok = False
    rows.append((url, " ".join(san), len(san), ok, fen))
out = pd.DataFrame(rows, columns=["game_url", "moves_san", "n_moves_ok", "replay_ok", "final_fen"])
out.to_parquet(os.path.join(D, "upset_moves.parquet"), index=False)
print(f"replayable: {int(out.replay_ok.sum())} of {len(out)}  ->  {os.path.join(D, 'upset_moves.parquet')}")
print(f"size {os.path.getsize(os.path.join(D, 'upset_moves.parquet')) / 1e6:.1f} MB")
