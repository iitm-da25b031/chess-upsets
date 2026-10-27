"""Convert every downloaded file into Parquet (skips months already done).
Usage: python run_all.py        (run AFTER the downloads; safe to stop and re-run)"""
import glob, os, re, traceback
import parse_pgn, parse_fide

errors = []
for f in sorted(glob.glob(os.path.join("data", "raw", "lichess", "*.pgn.zst"))):
    tag = re.search(r"(\d{4}-\d{2})", os.path.basename(f)).group(1)
    if os.path.exists(os.path.join("data", "interim", f"games_{tag}.parquet")):
        continue
    print("lichess", tag, flush=True)
    try:
        parse_pgn.main(f)
    except Exception:
        errors.append(f"{f}\n{traceback.format_exc()}")

for z in sorted(glob.glob(os.path.join("data", "raw", "fide", "*.zip"))):
    base = os.path.basename(z)
    period, lst = parse_fide.guess_period(base), parse_fide.guess_list(base)
    if period and os.path.exists(os.path.join("data", "interim", f"fide_{lst}_{period}.parquet")):
        continue
    print("fide", lst, period, flush=True)
    try:
        parse_fide.convert(z)
    except Exception:
        errors.append(f"{z}\n{traceback.format_exc()}")

open("run_all_errors.txt", "w").write("\n\n".join(errors))
print(f"finished with {len(errors)} errors (see run_all_errors.txt)")
