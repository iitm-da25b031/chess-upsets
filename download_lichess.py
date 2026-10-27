"""Resumable downloader. Usage:
    python download_lichess.py urls.txt            # urls.txt = one URL per line
Files go to data/raw/lichess/. Re-running skips finished files and resumes partial ones."""
import sys, os, requests

OUT = os.path.join("data", "raw", "lichess")
os.makedirs(OUT, exist_ok=True)

def fetch(url):
    name = url.rstrip("/").split("/")[-1]
    path = os.path.join(OUT, name)
    total = None
    head = requests.head(url, allow_redirects=True, timeout=60)
    if head.ok:
        total = int(head.headers.get("Content-Length", 0)) or None
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if total and have == total:
        print(f"[skip] {name} ({have/1e6:.1f} MB)")
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:          # already complete
            print(f"[skip] {name}"); return
        r.raise_for_status()
        mode = "ab" if (have and r.status_code == 206) else "wb"
        with open(path, mode) as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    size = os.path.getsize(path)
    ok = (total is None) or (size == total)
    print(f"[{'ok' if ok else 'SIZE MISMATCH'}] {name} {size/1e6:.1f} MB")

if __name__ == "__main__":
    urls = [u.strip() for u in open(sys.argv[1]) if u.strip() and not u.startswith("#")]
    print(f"{len(urls)} files")
    for u in urls:
        try:
            fetch(u)
        except Exception as e:
            print(f"[FAILED] {u}: {e}")
