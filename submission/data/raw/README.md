# data/raw: files exactly as downloaded (never edited)

Written by notebook 01 (section 3). Only `MANIFEST.csv`, the URL lists and `compliance/` are in the repository. The data files (2.6 GB) are downloaded by notebook 01, or from the link in the main README; the FIDE lists may not be redistributed.

| Path | Content |
|---|---|
| `lichess/lichess_db_broadcast_YYYY-MM.pgn.zst` | 81 monthly Lichess broadcast PGN files, 2020-01 to 2026-09 (744 MB) |
| `fide/<standard\|rapid\|blitz>_<mon><yy>frl_xml.zip` | 243 FIDE rating lists, same months (1.9 GB) |
| `fide/_missing.txt` | FIDE files not offered (empty for this project) |
| `compliance/` | robots.txt and terms pages of both sources, saved 2026-10-31 (quoted in notebook 01, section 2) |
| `MANIFEST.csv` | one row per raw file: source, period, URL, size, download date, SHA-256 (rewritten by notebook 01) |
| `urls_lichess.txt`, `urls_fide.txt` | the 81 + 243 download URLs |

The copy of `MANIFEST.csv` here lists the files used for the project, downloaded on 2026-10-24.
