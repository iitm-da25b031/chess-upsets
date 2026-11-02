"""Lists every raw file with its source, URL, period and download date.   Usage (project root):   python make_manifest.py
Writes data/raw/MANIFEST.csv (one row per file) and prints a summary for the report.
Lichess files:  data/raw/lichess/lichess_db_broadcast_YYYY-MM.pgn.zst   from https://database.lichess.org/broadcast/
FIDE files:     data/raw/fide/<standard|rapid|blitz>_<mon><yy>frl_xml.zip from https://ratings.fide.com/download/
The download date is the file's last-modified date (the files were written once by download_lichess.py / download_fide.py and never edited)."""
import os, re, datetime
import pandas as pd

MON = "jan feb mar apr may jun jul aug sep oct nov dec".split()
rows = []
for f in sorted(os.listdir(os.path.join("data", "raw", "lichess"))):
    m = re.fullmatch(r"lichess_db_broadcast_(\d{4})-(\d{2})\.pgn\.zst", f)
    if m: rows.append(("Lichess broadcasts", "PGN games", f"{m[1]}-{m[2]}", f"https://database.lichess.org/broadcast/{f}", os.path.join("lichess", f)))
for f in sorted(os.listdir(os.path.join("data", "raw", "fide"))):
    m = re.fullmatch(r"(standard|rapid|blitz)_([a-z]{3})(\d{2})frl_xml\.zip", f)
    if m: rows.append(("FIDE rating lists", f"{m[1]} list (XML)", f"20{m[3]}-{MON.index(m[2]) + 1:02d}", f"https://ratings.fide.com/download/{f}", os.path.join("fide", f)))
d = pd.DataFrame(rows, columns=["source", "content", "period", "url", "path"])
p = d["path"].map(lambda x: os.path.join("data", "raw", x))
d["size_mb"] = p.map(lambda x: round(os.path.getsize(x) / 1e6, 2))
d["downloaded"] = p.map(lambda x: datetime.date.fromtimestamp(os.path.getmtime(x)).isoformat())
d.to_csv(os.path.join("data", "raw", "MANIFEST.csv"), index=False)
print(d.groupby(["source", "content"]).agg(files=("path", "size"), first=("period", "min"), last=("period", "max"), mb=("size_mb", "sum"),
                                           downloaded_from=("downloaded", "min"), downloaded_to=("downloaded", "max")).round(0).to_string())
print("\nMANIFEST.csv written;", len(d), "files")
if os.path.exists(os.path.join("data", "raw", "fide", "_missing.txt")):
    print("\nmonths FIDE did not offer (data/raw/fide/_missing.txt):"); print(open(os.path.join("data", "raw", "fide", "_missing.txt")).read()[:800])
