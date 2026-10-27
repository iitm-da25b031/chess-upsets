"""Cleaning pipeline, stage 1.  Usage: python clean_games.py
Needs (in data/interim/): all_games.parquet, fide_ratings.parquet, clock_features.parquet
Writes: clean_games.parquet     games that survive every exclusion (Elo may still be missing -> analysis_ok False)
        excluded_games.parquet  every removed game with its reason (for audit)
        cleaning_log.csv, clean_report.txt
Each game is removed for the FIRST reason that applies, in the order below."""
import os, re
import numpy as np, pandas as pd
from tc_parse import parse_tc

D = os.path.join("data", "interim")
g = pd.read_parquet(os.path.join(D, "all_games.parquet"))
cf = pd.read_parquet(os.path.join(D, "clock_features.parquet")).drop_duplicates("game_url")
g = g.merge(cf[["game_url", "base_hat", "inc_hat"]], on="game_url", how="left")
n0 = len(g)
rep = []
def P(*a): rep.append(" ".join(str(x) for x in a))
def col(name):
    return g[name] if name in g.columns else pd.Series(pd.NA, index=g.index, dtype="object")

g["reason"] = pd.Series(None, index=g.index, dtype="object")
log = []
def drop(mask, name):
    mask = mask.fillna(False).astype(bool)
    new = mask & g["reason"].isna()
    g.loc[new, "reason"] = name
    log.append((name, int(new.sum())))

# ---- 1. results --------------------------------------------------------------
PLAYED = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "0.5-0.5": 0.5}
FORFEIT = {"+--", "--+", "+/-", "-/+", "-+", "+-"}
def norm_result(s):
    if s == "<NA>": return "invalid", np.nan
    t = (str(s).strip().replace("½", "1/2").replace("–", "-").replace("—", "-")
         .replace(" ", "").replace(":", "-").replace("_", "-").lower())
    if t in PLAYED: return "played", PLAYED[t]
    if t == "*": return "unfinished", np.nan
    if t in FORFEIT or re.fullmatch(r"[01]f-[01]f", t): return "forfeit", np.nan
    return "invalid", np.nan
r = g["result"].astype("object").where(g["result"].notna(), "<NA>")
rmap = {u: norm_result(u) for u in r.unique()}
kind = r.map(lambda u: rmap[u][0])
g["score_white"] = r.map(lambda u: rmap[u][1]).astype(float)
drop(kind == "unfinished", "1 unfinished result (*)")
drop(kind == "forfeit", "2 forfeit result code")
drop(kind == "invalid", "3 invalid result string")

# ---- 2. variant and starting position -------------------------------------------
var = col("variant").astype("string").str.lower()
nonstd = (var.notna() & (var != "standard")) | col("fen").notna() | (col("set_up").astype("string") == "1")
drop(nonstd, "4 non-standard variant / Chess960 / start position")

# ---- 3. unplayed / no moves -------------------------------------------------------
term = col("termination").astype("string").str.lower()
drop((g["n_plies"] == 0) | term.isin(["unplayed", "unterminated"]), "5 no moves / unplayed (incl. forfeits)")

# ---- 4. engines and adjudicated games --------------------------------------------
we_raw = pd.to_numeric(col("white_elo"), errors="coerce")
be_raw = pd.to_numeric(col("black_elo"), errors="coerce")
eng_elo = ((we_raw > 2900) & (we_raw <= 4000)) | ((be_raw > 2900) & (be_raw <= 4000))
adj = term.str.startswith("adjudication").fillna(False)
bshare = (eng_elo | adj).groupby(g["broadcast_name"]).transform("mean").fillna(0)
drop(eng_elo, "6 engine-level Elo (2900-4000)")
drop(adj, "7 adjudicated game")
drop(bshare >= 0.5, "8 engine/adjudicated event (>=50% of its games flagged)")

# ---- 5. duplicates ---------------------------------------------------------------
rem = g[g["reason"].isna()]
dup = rem.index[rem.duplicated("game_url", keep="first")]
drop(pd.Series(g.index.isin(dup), index=g.index), "9 duplicate game_url")
key = ["white_fide_id", "black_fide_id", "round", "broadcast_url", "score_white", "n_plies"]
rem = g[g["reason"].isna() & col("white_fide_id").notna() & col("black_fide_id").notna()]
dup = rem.index[rem.duplicated(key, keep="first")]
drop(pd.Series(g.index.isin(dup), index=g.index), "10 duplicate (same players, round, broadcast, result, length)")

# ---- 6. time control: string, checked against the first clock -----------------------
tcs = g["time_control"].astype("object").where(g["time_control"].notna(), None)
tmap = {u: parse_tc(u) for u in pd.unique(tcs)}
sb = pd.to_numeric(tcs.map(lambda u: tmap[u][0]), errors="coerce")
si = pd.to_numeric(tcs.map(lambda u: tmap[u][1]), errors="coerce")
g["tc_method"] = tcs.map(lambda u: tmap[u][2])
string_ok = sb.notna() & ~g["tc_method"].isin(["missing", "unparsed"])
clk0 = pd.to_numeric(g["first_clk_w"], errors="coerce")
clk0 = clk0.fillna(pd.to_numeric(g["first_clk_b"], errors="coerce"))
agrees = string_ok & clk0.notna() & ((clk0 - (sb + si)).abs() <= 120)
use_string = string_ok & (agrees | clk0.isna())

# The clock estimate is noisy by a few seconds of increment, and class depends on base + 60*increment.
# So snap it to the nearest time control that was verified on many games (e.g. 5400+30).
def build_snapper(pairs, min_count=200):
    known = [(b, i, c) for (b, i), c in pairs.items() if c >= min_count]
    def snap(b, i):
        best = None
        for cb, ci, c in known:
            if abs(i - ci) <= 3 and abs(b - cb) <= max(90, 0.04 * cb):
                if best is None or c > best[2]:
                    best = (cb, ci, c)
        return (best[0], best[1]) if best else None
    return snap
def apply_snap(snap, base_hat, inc_hat):
    keys = pd.DataFrame({"b": base_hat.round(), "i": inc_hat.round()})
    uk = keys.dropna().drop_duplicates().reset_index(drop=True)
    res = [snap(b, i) for b, i in zip(uk.b, uk.i)]
    uk["sb"] = [r[0] if r else np.nan for r in res]
    uk["si"] = [r[1] if r else np.nan for r in res]
    out = keys.merge(uk, on=["b", "i"], how="left")
    out.index = base_hat.index
    return out["sb"], out["si"]
snap_all = build_snapper(pd.DataFrame({"b": sb[agrees], "i": si[agrees]}).value_counts())
snap_b, snap_i = apply_snap(snap_all, g["base_hat"], g["inc_hat"])
is_snapped = snap_b.notna()
base_c, inc_c = snap_b.fillna(g["base_hat"]), snap_i.fillna(g["inc_hat"])
use_clock = ~use_string & base_c.notna()
def klass(base, inc):
    total = base + 60 * inc
    return pd.Series(np.select([total <= 600, total < 3600], ["blitz", "rapid"], "classical"),
                     index=base.index).where(base.notna())
g["tc_base_s"] = np.where(use_string, sb, np.where(use_clock, base_c, np.nan))
g["tc_inc_s"] = np.where(use_string, si, np.where(use_clock, inc_c, np.nan))
g["tc_class"] = klass(g["tc_base_s"], g["tc_inc_s"])
g["tc_source"] = np.select(
    [use_string & agrees, use_string & ~agrees, use_clock & string_ok, use_clock & is_snapped, use_clock],
    ["string_verified_by_clock", "string_unverified(no clock)", "clock_override", "clock_snapped", "clock_raw"], "none")
_tot = g["tc_base_s"] + 60 * g["tc_inc_s"]
g["tc_borderline"] = (np.minimum((_tot - 600).abs(), (_tot - 3600).abs()) <= 180)
name = col("broadcast_name").astype("string").str.lower().fillna("")
hr, hb = name.str.contains("rapid"), name.str.contains("blitz")
hc = name.str.contains("classical|klassisch|classic|standard")
only = (hr.astype(int) + hb.astype(int) + hc.astype(int)) == 1
need = g["tc_class"].isna()
g.loc[need & only & hr, "tc_class"] = "rapid"
g.loc[need & only & hb, "tc_class"] = "blitz"
g.loc[need & only & hc, "tc_class"] = "classical"
g.loc[need & only & g["tc_class"].notna(), "tc_source"] = "event_name"
g["tc_class"] = g["tc_class"].fillna("unknown")
cs = klass(sb, si).where(string_ok)
cc = klass(g["base_hat"], g["inc_hat"])                 # raw clock estimate
both = cs.notna() & cc.notna()
# out-of-sample check of snapping: learn the verified pairs from ODD months, test on EVEN months
mon = pd.to_numeric(g["source_month"].astype(str).str[-2:], errors="coerce")
learn, test = agrees & (mon % 2 == 1), agrees & (mon % 2 == 0)
snap_odd = build_snapper(pd.DataFrame({"b": sb[learn], "i": si[learn]}).value_counts())
tb, ti = apply_snap(snap_odd, g["base_hat"][test], g["inc_hat"][test])
cc_snap_test = klass(tb.fillna(g["base_hat"][test]), ti.fillna(g["inc_hat"][test]))

# ---- 7. Elo: validate, then fill from the FIDE list of the game's month -------------
fr = pd.read_parquet(os.path.join(D, "fide_ratings.parquet"),
                     columns=["fideid", "period", "standard_rating", "rapid_rating", "blitz_rating"])
fr["fideid"], fr["period"] = fr["fideid"].astype(str), fr["period"].astype(str)
fr = fr.drop_duplicates(["fideid", "period"]).set_index(["fideid", "period"])
period = g["game_date"].dt.strftime("%Y-%m").fillna(g["source_month"]).astype(str)
g["period"] = period
g["date_outside_fide_range"] = (period < "2020-01") | (period > "2026-09")
valid = lambda x: x.where((x >= 1000) & (x <= 2900))
def fide_lookup(side):
    idx = pd.MultiIndex.from_arrays([g[f"{side}_fide_id"].astype(object).astype(str), period])
    got = fr.reindex(idx)
    a, b, c = (got[k].to_numpy(dtype="float64", na_value=np.nan)
               for k in ("standard_rating", "rapid_rating", "blitz_rating"))
    cls = g["tc_class"]
    return pd.Series(np.select([cls == "classical", cls == "rapid", cls == "blitz"], [a, b, c], np.nan), index=g.index)
for side, raw in (("white", we_raw), ("black", be_raw)):
    pgn = valid(raw)
    fide = fide_lookup(side)
    final = pgn.fillna(fide)
    g[f"{side}_elo_invalid"] = raw.notna() & pgn.isna()
    g[f"{side}_elo_fide"] = fide
    g[f"{side}_elo_final"] = final
    g[f"{side}_elo_source"] = np.select([pgn.notna(), fide.notna()], ["pgn", "fide"], "missing")
    g[f"{side}_elo_mismatch"] = (pgn - fide).abs() > 100
g["analysis_ok"] = g["white_elo_final"].notna() & g["black_elo_final"].notna()

# ---- outputs ---------------------------------------------------------------------
kept = g[g["reason"].isna()].drop(columns="reason")
kept.to_parquet(os.path.join(D, "clean_games.parquet"), index=False)
g.loc[g["reason"].notna(), ["game_url", "reason", "broadcast_name", "source_month"]].to_parquet(
    os.path.join(D, "excluded_games.parquet"), index=False)
lg = pd.DataFrame(log, columns=["step", "removed"])
lg["left"] = n0 - lg["removed"].cumsum()
lg.to_csv("cleaning_log.csv", index=False)

P(f"start: {n0} games")
P(lg.to_string(index=False))
P(f"\nclean games kept: {len(kept)}   with both Elos (analysis_ok): {int(kept.analysis_ok.sum())}"
  f" ({kept.analysis_ok.mean():.1%})")
P("\n== time control source (kept games)"); P(kept.tc_source.value_counts().to_string())
P("\n== time control class (kept games)"); P(kept.tc_class.value_counts().to_string())
kb = both[kept.index]
P(f"\nclass agreement string vs clock, games with both: {int((cs[kb.index][kb] == cc[kb.index][kb]).mean()*100)}% of {int(kb.sum())}" if kb.any() else "")
if kb.any():
    P(pd.crosstab(cs[kb.index][kb], cc[kb.index][kb], rownames=["string"], colnames=["clock"]).to_string())
ktest = cs[test & g["reason"].isna()].index
ktest = ktest[cc_snap_test.reindex(ktest).notna().to_numpy() & cc.reindex(ktest).notna().to_numpy()]
P(f"\nout-of-sample class agreement on even-month verified games (n={len(ktest)}): "
  f"raw clock {(cs[ktest] == cc[ktest]).mean():.1%} -> snapped {(cs[ktest] == cc_snap_test[ktest]).mean():.1%}")
P("borderline share (class within 3 min of a threshold), by source:")
P(kept.groupby("tc_source")["tc_borderline"].mean().round(3).to_string())
P("\n== Elo (kept games)")
for s in ("white", "black"):
    P(f"{s}: source", kept[f'{s}_elo_source'].value_counts().to_dict(),
      "| invalid raw values:", int(kept[f'{s}_elo_invalid'].sum()),
      "| PGN vs FIDE differ by >100:", int(kept[f'{s}_elo_mismatch'].sum()))
x = kept.dropna(subset=["white_elo_fide", "white_elo_final"])
x = x[x.white_elo_source == "pgn"]
if len(x): P("median |PGN - FIDE| (white, where both):", float((x.white_elo_final - x.white_elo_fide).abs().median()))
kept["year"] = kept["game_date"].dt.year
P("\n== usable games per year (both Elos)")
P(kept.groupby("year").agg(games=("analysis_ok", "size"), usable=("analysis_ok", "sum")).to_string())
txt = "\n".join(rep)
open("clean_report.txt", "w", encoding="utf-8").write(txt)
print(txt)
