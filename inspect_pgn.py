"""Peek at one Lichess PGN file (.pgn, .pgn.zst or .pgn.bz2) without loading it all.
Usage:  python inspect_pgn.py path/to/file.pgn.zst
Writes sample_head.txt (first 200 lines) and inspect_report.txt in the current folder."""
import sys, re, io, bz2, collections

path = sys.argv[1]

def open_text(p):
    if p.endswith(".zst"):
        try:
            import zstandard
            return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(p, "rb")),
                                    encoding="utf-8", errors="replace")
        except ImportError:
            from compression import zstd          # Python 3.14+
            return io.TextIOWrapper(zstd.open(p, "rb"), encoding="utf-8", errors="replace")
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")

TAG = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
head, n_games, zero_move = [], 0, 0
tag_present = collections.Counter()
tc_vals, event_samples = collections.Counter(), []
cur_tags, cur_moves = {}, []

def finish():
    global n_games, zero_move
    if not cur_tags: return
    n_games += 1
    tag_present.update(cur_tags.keys())
    tc_vals[cur_tags.get("TimeControl", "<missing>")] += 1
    if len(event_samples) < 15 and cur_tags.get("Event") not in event_samples:
        event_samples.append(cur_tags.get("Event"))
    text = " ".join(cur_moves).strip()
    if text in ("", "1-0", "0-1", "1/2-1/2", "*"): zero_move += 1

with open_text(path) as f:
    for i, line in enumerate(f):
        if i < 200: head.append(line)
        m = TAG.match(line)
        if m:
            if cur_moves and not cur_tags.get("_open"):   # tags after movetext => new game
                finish(); cur_tags, cur_moves = {}, []
            cur_tags[m.group(1)] = m.group(2)
        elif line.strip():
            cur_moves.append(line.strip())
            cur_tags["_open"] = True
        elif cur_moves:
            finish(); cur_tags, cur_moves = {}, []
finish()

open("sample_head.txt", "w", encoding="utf-8").writelines(head)
with open("inspect_report.txt", "w", encoding="utf-8") as out:
    out.write(f"file: {path}\ngames: {n_games}\nzero-move games: {zero_move}\n\n")
    out.write("tag presence (count / share of games):\n")
    for t, c in tag_present.most_common():
        if t != "_open": out.write(f"  {t:<22}{c:>9}  {c/max(n_games,1):6.1%}\n")
    out.write("\nTimeControl values (top 40):\n")
    for v, c in tc_vals.most_common(40): out.write(f"  {v!r:<28}{c}\n")
    out.write("\nsample Event names:\n")
    for e in event_samples: out.write(f"  {e}\n")
print(open("inspect_report.txt", encoding="utf-8").read())
print("wrote sample_head.txt and inspect_report.txt")
