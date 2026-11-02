"""Documentation helper for the rubric (components 1-2).   Usage (from the project root):   python make_data_docs.py
Writes reports/data_dictionary.csv    one row per column of the final table: dtype, unit, source, meaning, allowed range, missing count
       reports/raw_profile.txt        shape, dtypes and missing counts of every table read straight from the raw files (before any cleaning)
       reports/dtype_roundtrip.txt    dtypes before writing vs after reading back (Parquet, and the SQLite copy of the final table)
       data/raw/compliance/           robots.txt and terms pages of each source, saved as served, with the date in the file name
Nothing in data/ is changed except the compliance folder."""
import os, sqlite3, datetime
import numpy as np, pandas as pd

TODAY = datetime.date.today().isoformat()
FINAL = os.path.join("data", "processed", "analysis_games.parquet")
SQLITE = os.path.join("data", "processed", "chess_upsets.sqlite")
os.makedirs("reports", exist_ok=True)

# column -> (unit, source, meaning).  Columns not listed are written with "TODO" so that nothing is silently skipped.
LICHESS, FIDE, DERIVED = "Lichess broadcast PGN", "FIDE rating list", "derived (this project)"
M = {
 "game_url": ("-", LICHESS, "Lichess URL of the game; unique key"),
 "event": ("-", LICHESS, "PGN Event tag"), "site": ("-", LICHESS, "PGN Site tag"), "round": ("-", LICHESS, "PGN Round tag"),
 "white": ("-", LICHESS, "White player's name as written in the PGN"), "black": ("-", LICHESS, "Black player's name as written in the PGN"),
 "result": ("-", LICHESS, "Result string from the PGN (1-0, 0-1, 1/2-1/2)"),
 "white_elo": ("Elo", LICHESS, "White's rating tag as written in the PGN"), "black_elo": ("Elo", LICHESS, "Black's rating tag as written in the PGN"),
 "white_fide_id": ("-", LICHESS, "White's FIDE id from the PGN; join key to the FIDE lists"), "black_fide_id": ("-", LICHESS, "Black's FIDE id from the PGN"),
 "time_control": ("-", LICHESS, "PGN TimeControl string as written (e.g. 5400+30)"),
 "broadcast_name": ("-", LICHESS, "Name of the Lichess broadcast (tournament)"), "broadcast_url": ("-", LICHESS, "URL of the broadcast"),
 "game_date": ("date", LICHESS, "Game date from UTCDate, else Date tag"), "source_month": ("YYYY-MM", LICHESS, "Month of the PGN archive file the game came from"),
 "n_plies": ("half-moves", DERIVED, "Number of half-moves in the game"),
 "period": ("YYYY-MM", DERIVED, "Month used to look up FIDE ratings: game month, else archive month"),
 "year": ("year", DERIVED, "Year of the game"),
 "tc_base_s": ("seconds", DERIVED, "Base time of the time control, from the PGN string or estimated from the clocks"),
 "tc_inc_s": ("seconds", DERIVED, "Increment per move"),
 "tc_class": ("-", DERIVED, "classical / rapid / blitz, from base + 60 x increment (<=600 s blitz, <3600 s rapid)"),
 "tc_source": ("-", DERIVED, "How the time control was obtained (string verified by clock, clock estimate, event name, ...)"),
 "tc_borderline": ("bool", DERIVED, "Class lies within 3 minutes of a class threshold"),
 "score_white": ("points", DERIVED, "White's score: 1, 0.5 or 0"),
 "white_elo_final": ("Elo", DERIVED, "White rating used: valid PGN rating, else FIDE list for that month and time control"),
 "black_elo_final": ("Elo", DERIVED, "Black rating used (same rule)"),
 "white_elo_source": ("-", DERIVED, "pgn / fide / missing"), "black_elo_source": ("-", DERIVED, "pgn / fide / missing"),
 "white_elo_fide": ("Elo", FIDE, "White's rating in the FIDE list of that month for this time control"), "black_elo_fide": ("Elo", FIDE, "Black's FIDE rating (same rule)"),
 "white_elo_mismatch": ("bool", DERIVED, "PGN and FIDE ratings differ by more than 100"), "black_elo_mismatch": ("bool", DERIVED, "same for Black"),
 "white_elo_invalid": ("bool", DERIVED, "PGN rating present but outside 1000-2900"), "black_elo_invalid": ("bool", DERIVED, "same for Black"),
 "analysis_ok": ("bool", DERIVED, "Both ratings known"),
 "avg_elo": ("Elo", DERIVED, "Mean of the two ratings"), "gap": ("Elo", DERIVED, "Absolute rating difference"),
 "band": ("-", DERIVED, "Rating band of avg_elo: <2200, 2200-2499, 2500+"),
 "has_underdog": ("bool", DERIVED, "gap > 0"), "underdog_color": ("-", DERIVED, "Colour of the lower-rated player"),
 "underdog_score": ("points", DERIVED, "Underdog's score: 1, 0.5 or 0"),
 "kappa": ("-", DERIVED, "Draw parameter of the Elo-Davidson model for the class and band"),
 "p_win_u": ("probability", DERIVED, "Model probability that the underdog wins"), "p_draw": ("probability", DERIVED, "Model probability of a draw"),
 "p_loss_u": ("probability", DERIVED, "Model probability that the underdog loses"), "exp_score_u": ("points", DERIVED, "Model expected score of the underdog"),
 "q": ("probability", DERIVED, "Probability of a result at least as good as the one that happened, for the underdog"),
 "exp_score_elo": ("points", DERIVED, "Plain Elo expected score of the underdog (400-point logistic)"),
 "surprise": ("points", DERIVED, "underdog_score minus exp_score_elo"),
 "verified": ("bool", DERIVED, "Both PGN ratings within 100 points of the FIDE list"),
}
M.update({
 "date": ("date", LICHESS, "PGN Date tag, as written (may be partial, e.g. 2024.??.??)"),
 "utc_date": ("date", LICHESS, "PGN UTCDate tag: date the game was played, UTC"), "utc_time": ("time", LICHESS, "PGN UTCTime tag: start time, UTC"),
 "white_title": ("-", LICHESS, "FIDE title of White from the PGN (GM, IM, ...); empty if untitled"), "black_title": ("-", LICHESS, "FIDE title of Black from the PGN"),
 "variant": ("-", LICHESS, "PGN Variant tag; empty or Standard for ordinary chess (other variants are removed in cleaning)"),
 "eco": ("-", LICHESS, "ECO opening code from the PGN, where given"), "opening": ("-", LICHESS, "Opening name from the PGN, where given"),
 "study_name": ("-", LICHESS, "Name of the Lichess study that holds the broadcast round"), "chapter_name": ("-", LICHESS, "Name of the study chapter (one game or board)"),
 "chapter_url": ("-", LICHESS, "URL of the study chapter"), "annotator": ("-", LICHESS, "PGN Annotator tag"),
 "termination": ("-", LICHESS, "PGN Termination tag (e.g. Normal, Unterminated, Time forfeit)"),
 "fen": ("-", LICHESS, "Starting position if not the standard one (such games are removed in cleaning)"), "set_up": ("-", LICHESS, "PGN SetUp tag; 1 means a non-standard start"),
 "white_team": ("-", LICHESS, "Team of White in team events"), "black_team": ("-", LICHESS, "Team of Black in team events"),
 "board": ("-", LICHESS, "Board number in team events"), "game_id": ("-", LICHESS, "Lichess game id, where the PGN gives one"),
 "white_clock": ("-", LICHESS, "PGN tag for White's clock, as written"), "black_clock": ("-", LICHESS, "PGN tag for Black's clock, as written"),
 "white_country": ("-", LICHESS, "Federation of White from the PGN, where given"), "black_country": ("-", LICHESS, "Federation of Black from the PGN, where given"),
 "has_clk": ("bool", DERIVED, "Move text contains clock readings"), "has_eval": ("bool", DERIVED, "Move text contains engine evaluations"),
 "first_clk_w": ("seconds", DERIVED, "White's clock after its first move"), "first_clk_b": ("seconds", DERIVED, "Black's clock after its first move"),
 "last_clk_w": ("seconds", DERIVED, "White's last recorded clock reading"), "last_clk_b": ("seconds", DERIVED, "Black's last recorded clock reading"),
 "n_inacc_w": ("moves", DERIVED, "Number of White moves annotated ?! (inaccuracy)"), "n_mistake_w": ("moves", DERIVED, "Number of White moves annotated ? (mistake)"),
 "n_blunder_w": ("moves", DERIVED, "Number of White moves annotated ?? (blunder)"), "n_inacc_b": ("moves", DERIVED, "Number of Black moves annotated ?!"),
 "n_mistake_b": ("moves", DERIVED, "Number of Black moves annotated ?"), "n_blunder_b": ("moves", DERIVED, "Number of Black moves annotated ??"),
 "tc_method": ("-", DERIVED, "How the PGN TimeControl string was parsed (pattern name, missing, unparsed)"),
 "base_hat": ("seconds", DERIVED, "Base time estimated from the clocks: White's first reading minus the increment plus 1"),
 "inc_hat": ("seconds", DERIVED, "Increment estimated from the clocks: largest clock increase between a player's consecutive moves"),
 "date_outside_fide_range": ("bool", DERIVED, "Game month lies outside the FIDE lists downloaded (before 2020-01 or after 2026-09)"),
 "underdog_is_white": ("bool", DERIVED, "White has the lower rating"), "fav_is_white": ("bool", DERIVED, "White has the higher rating"),
 "gap_bin": ("Elo", DERIVED, "Lower edge of the 50-point rating-gap bin; 600 means a gap of 600 or more"),
 "elo_mismatch_any": ("bool", DERIVED, "PGN and FIDE ratings differ by more than 100 for either player (missing where no FIDE rating exists)"),
})
for a in ("0.20", "0.10", "0.05", "0.01"):
    M[f"upset_{a}"] = ("bool", DERIVED, f"Underdog scored and q < {a}")

def kind(s):
    return "string" if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s) else str(s.dtype)

# ---------------------------------------------------------------- 1. data dictionary of the final table
g = pd.read_parquet(FINAL)
rows = []
for c in g.columns:
    s = g[c]
    unit, src, meaning = M.get(c, ("TODO", "TODO", "TODO: describe"))
    if pd.api.types.is_bool_dtype(s): rng = "True / False"
    elif pd.api.types.is_numeric_dtype(s): rng = f"{s.min():.6g} to {s.max():.6g}" if s.notna().any() else "all missing"
    elif pd.api.types.is_datetime64_any_dtype(s): rng = f"{s.min()} to {s.max()}"
    else:
        u = s.nunique(dropna=True)
        rng = ", ".join(map(str, s.dropna().unique()[:8])) if u <= 8 else f"{u:,} distinct values"
    rows.append(dict(column=c, dtype=kind(s), unit=unit, source=src, meaning=meaning, allowed_range_observed=rng, n_missing=int(s.isna().sum())))
dd = pd.DataFrame(rows)
dd.to_csv(os.path.join("reports", "data_dictionary.csv"), index=False)
todo = dd[dd.meaning.str.startswith("TODO")].column.tolist()
print(f"data_dictionary.csv: {len(dd)} columns, {len(dd) - len(todo)} described.   To describe by hand ({len(todo)}):", todo)

# ---------------------------------------------------------------- 2. profile of the tables built straight from the raw files
out = []
for name in ("all_games.parquet", "fide_ratings.parquet", "fide_players.parquet"):
    p = os.path.join("data", "interim", name)
    if not os.path.exists(p): out.append(f"{name}: not found\n"); continue
    d = pd.read_parquet(p)
    out.append(f"== {name}   shape {d.shape}   ({os.path.getsize(p) / 1e6:.0f} MB)   read on {TODAY}")
    out.append(pd.DataFrame({"dtype": [kind(d[c]) for c in d.columns], "n_missing": d.isna().sum().values,
                             "share_missing": (d.isna().mean().round(4)).values}, index=d.columns).to_string() + "\n")
    del d
open(os.path.join("reports", "raw_profile.txt"), "w", encoding="utf-8").write("\n".join(out))
print("raw_profile.txt written")

# ---------------------------------------------------------------- 3. dtypes after reading back
rt = ["== Parquet: write the final table to a temporary file, read it back, compare dtypes and values"]
tmp = os.path.join("data", "processed", "_roundtrip_tmp.parquet")
g.to_parquet(tmp, index=False); b = pd.read_parquet(tmp); os.remove(tmp)
bad = [c for c in g.columns if kind(g[c]) != kind(b[c])]
rt.append(f"rows {len(g)} -> {len(b)}; columns whose dtype changed: {bad or 'none'}; frames equal: {g.reset_index(drop=True).equals(b)}")
if os.path.exists(SQLITE):
    rt.append("\n== SQLite: table 'games' read back (SQLite has no bool/datetime type, so changes are expected here)")
    con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    n = con.execute("select count(*) from games").fetchone()[0]
    s = pd.read_sql("select * from games limit 20000", con); con.close()
    rt.append(f"rows in table {n:,} (final table {len(g):,})")
    ch = [(c, kind(g[c]), kind(s[c])) for c in g.columns if c in s.columns and kind(g[c]) != kind(s[c])]
    rt.append("dtype differences (column, parquet, sqlite): " + (str(ch) if ch else "none"))
    rt.append("reading: Parquet keeps every dtype; SQLite stores bool as 0/1, turns missing values in nullable columns into NaN and has no date type, so the dashboard reads only the columns it needs from the Parquet and CSV files.")
open(os.path.join("reports", "dtype_roundtrip.txt"), "w", encoding="utf-8").write("\n".join(rt))
print("\n".join(rt))

# ---------------------------------------------------------------- 4. robots.txt and terms, saved as served
import requests
PAGES = {"lichess_robots": "https://lichess.org/robots.txt", "lichess_database_robots": "https://database.lichess.org/robots.txt",
         "lichess_terms": "https://lichess.org/terms-of-service", "lichess_database_page": "https://database.lichess.org/",
         "fide_robots": "https://ratings.fide.com/robots.txt", "fide_ratings_page": "https://ratings.fide.com/download_lists.phtml"}
cdir = os.path.join("data", "raw", "compliance"); os.makedirs(cdir, exist_ok=True)
for k, url in PAGES.items():
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "DA2402-student-project"})
        ext = "txt" if "robots" in k else "html"
        open(os.path.join(cdir, f"{k}_{TODAY}.{ext}"), "wb").write(r.content)
        print(f"{k}: HTTP {r.status_code}, {len(r.content):,} bytes -> {k}_{TODAY}.{ext}")
    except Exception as e:
        print(f"{k}: FAILED ({e}); save the page by hand from {url}")
