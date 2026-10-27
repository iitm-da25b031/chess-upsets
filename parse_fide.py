"""Parse one FIDE rating-list zip (XML) into a Parquet table.

Usage:  python parse_fide.py path/to/standard_rating_list_xml.zip --period 2026-10
        (the period is read from the file name when it contains e.g. 'oct26' or '2026-10')
List type (standard / rapid / blitz / combined) is read from the file name, or pass --list.
Output: data/interim/fide_<list>_<period>.parquet
Streams the XML, so memory stays small even for the ~1.5 million players."""
import sys, os, re, zipfile, argparse
import xml.etree.ElementTree as ET
import pandas as pd

MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()

def guess_period(name):
    m = re.search(r"(\d{4})[-_](\d{2})", name)
    if m: return f"{m.group(1)}-{m.group(2)}"
    m = re.search(r"(" + "|".join(MONTHS) + r")(\d{2})", name.lower())
    if m: return f"20{m.group(2)}-{MONTHS.index(m.group(1)) + 1:02d}"
    return None

def guess_list(name):
    n = name.lower()
    for k in ("standard", "rapid", "blitz"):
        if k in n: return k
    return "combined"

def convert(zip_path, period=None, lst=None):
    """Convert one FIDE zip to Parquet; returns the output path."""
    base = os.path.basename(zip_path)
    period = period or guess_period(base)
    if not period: raise ValueError("Could not read the period from the file name; pass --period YYYY-MM")
    lst = lst or guess_list(base)
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        member = [i.filename for i in z.infolist() if i.filename.lower().endswith(".xml")][0]
        with z.open(member) as f:
            for ev, el in ET.iterparse(f, events=("end",)):
                if el.tag == "player":
                    rows.append({c.tag: (c.text or "").strip() for c in el})
                    el.clear()
    df = pd.DataFrame(rows)
    df["fideid"] = df["fideid"].astype("string")
    prefix = "standard" if lst == "combined" else lst
    df = df.rename(columns={c: f"{prefix}_{c}" for c in ("rating", "games", "k") if c in df})
    for c in df.columns:
        if c == "birthday" or re.search(r"(^|_)(rating|games|k)$", c):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in [c for c in df.columns if c.endswith("_rating")]:   # 0 = unrated in the combined file
        df.loc[df[c] == 0, c] = pd.NA
    df["period"] = period
    os.makedirs(os.path.join("data", "interim"), exist_ok=True)
    out = os.path.join("data", "interim", f"fide_{lst}_{period}.parquet")
    df.to_parquet(out, index=False)
    return out, df, lst, period

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--period"); ap.add_argument("--list")
    a = ap.parse_args()
    try:
        out, df, lst, period = convert(a.zip, a.period, a.list)
    except ValueError as e:
        sys.exit(str(e))
    print(f"{len(df)} players, list={lst}, period={period} -> {out}")
    print("columns:", list(df.columns))
    rc = [c for c in df.columns if c.endswith("rating")]
    if rc: print(df[rc].describe().round(0).to_string())
    print("duplicate fideid:", int(df.fideid.duplicated().sum()), "| missing rating:", int(df[rc[0]].isna().sum()) if rc else "n/a")

if __name__ == "__main__":
    main()
