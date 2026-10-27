"""Download the FIDE monthly rating lists (standard, rapid, blitz; XML) for a range of months.
Usage:  python download_fide.py                 # 2020-01 .. 2026-09 (the months of the Lichess files)
        python download_fide.py 2020-01 2020-01 # just one month, to test
Files go to data/raw/fide/. Re-running skips finished files. Months FIDE does not offer
are listed in data/raw/fide/_missing.txt (that is information for the report, not an error)."""
import os, sys, time, requests

MON = "jan feb mar apr may jun jul aug sep oct nov dec".split()
OUT = os.path.join("data", "raw", "fide")
HEAD = {"User-Agent": "Mozilla/5.0 (university course project, DA2402)"}
os.makedirs(OUT, exist_ok=True)

def months(a, b):
    y, m = a
    while (y, m) <= b:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

def url(lst, y, m):
    return f"https://ratings.fide.com/download/{lst}_{MON[m-1]}{y % 100:02d}frl_xml.zip"

def get(u, path):
    r = requests.get(u, headers=HEAD, stream=True, timeout=120)
    if r.status_code == 404:
        return "404"
    r.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with open(tmp, "rb") as f:
        if f.read(2) != b"PK":                 # not a zip (probably an error page)
            os.remove(tmp)
            return "not-a-zip"
    os.replace(tmp, path)
    return "ok"

def parse_ym(s):
    y, m = s.split("-")
    return int(y), int(m)

if __name__ == "__main__":
    a = parse_ym(sys.argv[1]) if len(sys.argv) > 1 else (2020, 1)
    b = parse_ym(sys.argv[2]) if len(sys.argv) > 2 else (2026, 9)
    missing = []
    for y, m in months(a, b):
        for lst in ("standard", "rapid", "blitz"):
            u = url(lst, y, m)
            path = os.path.join(OUT, os.path.basename(u))
            if os.path.exists(path):
                print(f"[skip] {os.path.basename(u)}"); continue
            try:
                res = get(u, path)
            except Exception as e:
                res = f"error: {e}"
            size = f"{os.path.getsize(path)/1e6:.1f} MB" if res == "ok" else ""
            print(f"[{res}] {os.path.basename(u)} {size}", flush=True)
            if res != "ok":
                missing.append(f"{lst} {y}-{m:02d} {res}")
            time.sleep(1)                       # be polite to the server
    with open(os.path.join(OUT, "_missing.txt"), "w") as f:
        f.write("\n".join(missing))
    print(f"done. {len(missing)} files not obtained (see {OUT}/_missing.txt)")
