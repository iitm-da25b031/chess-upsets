"""Peek inside a FIDE rating-list zip. Usage:  python peek_fide.py path/to/list.zip
Writes fide_head.txt (first 60 lines of the first file inside) in the current folder."""
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
print("contents:", [(i.filename, i.file_size) for i in z.infolist()])
name = z.infolist()[0].filename
with z.open(name) as f:
    lines = [next(f, b"").decode("utf-8", "replace") for _ in range(60)]
open("fide_head.txt", "w", encoding="utf-8").writelines(lines)
print("".join(lines[:15]))
print("wrote fide_head.txt")
