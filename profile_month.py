"""Quality profile of one parsed month.
Usage: python profile_month.py data/interim/games_2026-09.parquet
Writes profile_report.txt next to where you run it. Send that file back."""
import sys
import pandas as pd

df = pd.read_parquet(sys.argv[1])
n = len(df)
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
def share(x): return f"{int(x):>8}  {x/n:6.1%}"
def col(c): return df[c] if c in df else pd.Series([None] * n)

P(f"file: {sys.argv[1]}   games: {n}")
P("\n== Result"); P(col("result").value_counts(dropna=False).to_string())
P("\n== Variant"); P(col("variant").value_counts(dropna=False).to_string())
P("\n== Termination"); P(col("termination").value_counts(dropna=False).head(15).to_string())
zero = (df.n_plies == 0)
P("\n== Zero-move games", share(zero.sum()))
P("   of which result '*':", int((zero & (col("result") == "*")).sum()),
  "  decisive/draw result with no moves (forfeits?):", int((zero & col("result").isin(["1-0", "0-1", "1/2-1/2"])).sum()))
P("games with a starting FEN (non-standard start):", share(col("fen").notna().sum()))
P("ECO '?':", share((col("eco") == "?").sum()))
P("event names that look like a pairing ('Round N: A - B'):", share(col("event").astype(str).str.match(r"^Round \d+:").sum()))
P("\n== Duplicates")
P("duplicated game_url:", int(df.duplicated("game_url").sum()))
key = ["white_fide_id", "black_fide_id", "round", "broadcast_url"]
if all(k in df for k in key):
    sub = df.dropna(subset=["white_fide_id", "black_fide_id"])
    P("duplicated (white id, black id, round, broadcast):", int(sub.duplicated(key).sum()))

P("\n== Ratings and IDs")
we, be = col("white_elo"), col("black_elo")
P("Elo missing for white:", share(we.isna().sum()), "| black:", share(be.isna().sum()),
  "| either:", share((we.isna() | be.isna()).sum()), "| both:", share((we.isna() & be.isna()).sum()))
wi, bi = col("white_fide_id"), col("black_fide_id")
P("FIDE id missing for white:", share(wi.isna().sum()), "| black:", share(bi.isna().sum()))
P("Elo missing BUT FIDE id present (recoverable from FIDE list), white:", share((we.isna() & wi.notna()).sum()))
P("Elo missing AND FIDE id missing (needs name matching), white:", share((we.isna() & wi.isna()).sum()))
for c in ("white_elo", "black_elo"):
    s = col(c).dropna().astype(float)
    P(f"{c}: min {s.min():.0f}  p1 {s.quantile(.01):.0f}  median {s.median():.0f}  p99 {s.quantile(.99):.0f}  max {s.max():.0f}  zeros {(s==0).sum()}  below 1000: {(s<1000).sum()}")
both = df.dropna(subset=["white_elo", "black_elo"])
g = (both.white_elo.astype(float) - both.black_elo.astype(float)).abs()
P("rating gap |W-B|: median", g.median(), " p90", g.quantile(.9), " p99", g.quantile(.99), " max", g.max())

P("\n== Time control")
P("missing:", share(df.tc_method.eq("missing").sum()))
P(df.tc_method.value_counts().to_string())
P("class counts:"); P(df.tc_class.value_counts(dropna=False).to_string())
P("\nstrings that could NOT be parsed (top 25):")
P(col("time_control")[df.tc_method == "unparsed"].value_counts().head(25).to_string())
P("\nambiguous numeric strings (121..299 base):")
P(col("time_control")[df.tc_method == "numeric_ambiguous"].value_counts().head(15).to_string())

P("\n== Clocks and engine evals")
P("games with %clk:", share(df.has_clk.sum()), "| with %eval:", share(df.has_eval.sum()))
ok = df[df.tc_base_s.notna() & df.first_clk_w.notna()].copy()
if len(ok):
    exp = ok.tc_base_s.astype(float) + ok.tc_inc_s.astype(float)
    ok["clk_ok"] = (ok.first_clk_w - exp).abs() <= 120
    P("parsed TC agrees with white's first clock (+-120 s):", f"{ok.clk_ok.mean():.1%} of {len(ok)} games")
    P("strings that DISAGREE most often:"); P(ok.loc[~ok.clk_ok, "time_control"].value_counts().head(15).to_string())
miss = df[df.tc_method.eq("missing") & df.first_clk_w.notna()]
P(f"\nmissing-TC games that still have clocks: {len(miss)} of {df.tc_method.eq('missing').sum()}")
P("white's first clock rounded to 10 s, top 15 (hints at the hidden time control):")
P((miss.first_clk_w / 10).round().mul(10).astype(int).value_counts().head(15).to_string())

P("\n== Missing time control: where is it concentrated? (top broadcasts)")
m = df[df.tc_method.eq("missing")]
P(m.broadcast_name.value_counts().head(15).to_string())
P("\n== Engine-flagged errors per game (mean)")
P(df[["n_inacc_w", "n_mistake_w", "n_blunder_w", "n_inacc_b", "n_mistake_b", "n_blunder_b"]].mean().round(2).to_string())

txt = "\n".join(out)
open("profile_report.txt", "w", encoding="utf-8").write(txt)
print(txt)
