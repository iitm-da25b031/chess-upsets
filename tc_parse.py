"""Parse the messy TimeControl strings of Lichess broadcasts.

parse_tc(s) -> (base_seconds, increment_seconds, method)
tc_class(base, inc) -> 'blitz' | 'rapid' | 'classical' | None   (FIDE definition)

Number rule used everywhere: a number <= 120 is MINUTES, >= 300 is SECONDS,
121..299 is ambiguous (treated as seconds, method 'numeric_ambiguous').
The first clock of the game is used later to check / override these parses.
"""
import re

NUM    = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*$")                    # 90+30, 900+10
SINGLE = re.compile(r"^\s*(\d+)\s*$")                                  # 600, 1500, 90
STAGED = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*\+\s*(\d+)\s*:")         # 5400/40+30:1800, 40/5400+30:1800
TRIPLE = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*$")        # 5400+30+1800
COLON  = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")                      # 5400:30
MIN    = re.compile(r"(\d+)\s*(?:min\w*|m\b|mn\b|')", re.I)
SEC    = re.compile(r"(\d+)\s*(?:sec\w*|seg\w*|sg\b|s\b|\")", re.I)
GAME   = re.compile(r"\bg\s*:?\s*(\d+)", re.I)                         # "G90' + 30", "G: 20 minutes"
LOOSE  = re.compile(r"(\d+)\s*[+´`']\s*(\d+)")                         # '90+30´ 30', '900 + 5 sec'

def _num_rule(a, b, tag="numeric"):
    if a <= 120:
        return a * 60, b, f"{tag}_min"
    if a >= 300:
        return a, b, f"{tag}_sec"
    return a, b, f"{tag}_ambiguous"

def parse_tc(s):
    if s is None or s != s or str(s).strip() in ("", "-", "?"):
        return None, None, "missing"
    s = str(s).strip()
    m = NUM.match(s)
    if m:
        return _num_rule(int(m.group(1)), int(m.group(2)))
    m = SINGLE.match(s)
    if m:
        return _num_rule(int(m.group(1)), 0)
    m = STAGED.match(s)
    if m:
        x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if x <= 60 and y >= 300: return y, z, "staged"
        if y <= 60 and x >= 300: return x, z, "staged"
        return None, None, "unparsed"
    m = TRIPLE.match(s)
    if m:
        return _num_rule(int(m.group(1)), int(m.group(2)))
    m = COLON.match(s)
    if m and int(m.group(1)) >= 300:
        return int(m.group(1)), int(m.group(2)), "numeric_sec"
    base = inc = None
    mm, ms = MIN.search(s), SEC.search(s)
    if mm:
        base = int(mm.group(1)) * 60
    else:
        g = GAME.search(s)
        if g:
            base = int(g.group(1)) * 60
    if ms:
        inc = int(ms.group(1))
    if base is not None and inc is None:
        after = re.search(r"\+\s*(\d+)", s)
        if after:
            inc = int(after.group(1))
    if base is None:
        lm = LOOSE.search(s)
        if lm:
            b, i, meth = _num_rule(int(lm.group(1)), int(lm.group(2)), "text_loose")
            return b, i, "text_loose"
        return None, None, "unparsed"
    if inc is None:
        inc = 0
    return base, inc, "text"

def tc_class(base, inc):
    """FIDE: blitz <= 10 min, rapid 10..60 min, classical >= 60 min, where
    total = base + 60 * increment."""
    if base is None or base != base:
        return None
    total = base + 60 * (inc or 0)
    if total <= 600:
        return "blitz"
    if total < 3600:
        return "rapid"
    return "classical"
