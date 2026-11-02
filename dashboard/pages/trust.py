"""Page 3: "Can it be trusted?"   What data stands behind the results, whether they move when only rating-checked games are used, and whether the model predicts held-out tournaments.

Reads (data/dashboard/): critical_gap_empirical.csv, critical_gap_empirical_verified.csv, calibration_empirical.csv
Five views behind one toggle so the page fits one screen:
    Data kept     games remaining after each cleaning step
    Before/after  game length and average rating before and after cleaning (before_after.csv)
    Missing data  share of games with a missing Elo or time control, in the source and after repair, with chi-square tests (missing_by_year.csv, missing_tests.csv)
    Rating check  critical gap on all games against games whose ratings were confirmed against FIDE lists
    Model check   predicted against observed underdog win rate on tournaments excluded from fitting
Rule used on every page: a control that does not change what is on screen is disabled.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from common import DATA, TCS, TC_COL, GREEN, GREY, PAPER, GRID, GCFG, seg, view_toggle, kpi_static, notes_card, style_figure

dash.register_page(__name__, path="/trust", name="Can it be trusted?", title="Upset Explorer: can it be trusted?", order=2)
P = "p3_"

# Game counts from the cleaning logs (features.py / clean_games.py). Edit here if a rerun changes them.
FUNNEL = [("All games parsed", 1_249_480), ("After cleaning", 1_117_314), ("Both ratings known", 885_340),
          ("Analysis set", 858_123), ("Ratings checked against FIDE", 654_685)]
REMOVED = {"After cleaning": "unfinished games, chess variants, games with no moves, engine-level ratings and duplicates",
           "Both ratings known": "a rating was missing for one or both players",
           "Analysis set": "unknown time control, games shorter than 10 half-moves, games before 2019"}

CG = pd.read_csv(os.path.join(DATA, "critical_gap_empirical.csv"))
CGV = pd.read_csv(os.path.join(DATA, "critical_gap_empirical_verified.csv")) if os.path.exists(os.path.join(DATA, "critical_gap_empirical_verified.csv")) else None
CAL = pd.read_csv(os.path.join(DATA, "calibration_empirical.csv"))
BA = pd.read_csv(os.path.join(DATA, "before_after.csv"))
MB = pd.read_csv(os.path.join(DATA, "missing_by_year.csv"))
MT = pd.read_csv(os.path.join(DATA, "missing_tests.csv"))
CAL_ERR = float(np.average(np.abs(CAL.obs_win - CAL.pred_win), weights=CAL.n)) * 100          # weighted mean error, percentage points


COLOUR_OPTS = [{"label": "White", "value": "white"}, {"label": "Black", "value": "black"}]
BAND_OPTS = [{"label": "Under 2200", "value": "<2200"}, {"label": "2200–2499", "value": "2200-2499"}, {"label": "2500+", "value": "2500+"}]
ALPHA_OPTS = [{"label": "20%", "value": 0.2}, {"label": "10%", "value": 0.1}, {"label": "5%", "value": 0.05}]



controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label("Underdog plays"), seg(P + "colour", COLOUR_OPTS, "white")], width=3),
    dbc.Col([html.Label("Players' average rating"), seg(P + "band", BAND_OPTS, "2200-2499")], width=5),
    dbc.Col([html.Label("Upset threshold"), seg(P + "alpha", ALPHA_OPTS, 0.1)], width=4),
], className="g-3 align-items-start")), className="mb-3")

verified_share = FUNNEL[-1][1] / FUNNEL[3][1]
layout = html.Div([
    dbc.Row(dbc.Col(html.H2("Can it be trusted?"), width=12), className="mb-2"),
    controls,
    dbc.Row([dbc.Col(kpi_static("Games analysed", f"{FUNNEL[3][1]:,}"), width=4),
             dbc.Col(kpi_static("Games with a rating checked against FIDE", f"{verified_share:.0%}"), width=4),
             dbc.Col(kpi_static("Mean absolute calibration error (held-out games)", f"{CAL_ERR:.2f} pts"), width=4)], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(view_toggle(P + "view", [{"label": "Data kept", "value": "kept"}, {"label": "Before/after", "value": "ba"},
                                                                  {"label": "Missing data", "value": "miss"}, {"label": "Rating check", "value": "move"},
                                                                  {"label": "Model check", "value": "cal"}], "kept")),
                          dbc.CardBody([html.Div(id=P + "q", className="q-title"), dcc.Graph(id=P + "fig", config=GCFG)])], className="h-100"), width=8),
        dbc.Col(notes_card(P + "notes"), width=4)], className="g-3"),
], className="p-4")

QUESTION = {"kept": "How many games remain after each cleaning step?",
            "ba": "What did cleaning change in the data?",
            "miss": "Where are values missing, and is it random?",
            "move": "Does restricting to rating-checked games change the critical gap?",
            "cal": "Do predicted win rates match unseen tournaments?"}

def _share(stage, var):
    d = BA[(BA.variable == var) & (BA.stage == stage)].sort_values("bin_lo")
    return d.bin_lo.to_numpy(), d.n.to_numpy()


def _median(stage, var):
    x, n = _share(stage, var)
    return float(x[np.searchsorted(np.cumsum(n), n.sum() / 2)])


def _miss(measure):
    d = MB[MB.measure == measure]
    return d.missing.sum() / d.n.sum()


def _test(resp, factor):
    r = MT[(MT.response == resp) & (MT.factor == factor)].iloc[0]
    p = "p < 0.001" if r.p < 0.001 else f"p = {r.p:.3f}"
    return f"χ² = {r.chi2:,.0f}, {p}, Cramér's V = {r.cramers_v:.2f}"


def ba_numbers():
    lx, ln = _share("raw", "length"); short = ln[lx < 10].sum() / ln.sum()
    rx, rn = _share("raw", "rating"); high = rn[rx >= 2900].sum() / rn.sum(); low = rn[rx < 1000].sum() / rn.sum()
    return dict(short=short, high=high, low=low, med_raw=_median("raw", "length"), med_clean=_median("clean", "length"),
                r_raw=_median("raw", "rating"), r_clean=_median("clean", "rating"))


NOTE_SHOWS = {"kept": "Number of games remaining after each cleaning step. The green bar is the analysis set on which every result in the dashboard is based.",
              "ba": "Distribution of game length (half-moves) and of the players' average rating, as a share of games. The white line is every parsed game before cleaning, the green bars the analysis set. Differences between them are what the cleaning rules removed.",
              "miss": "Share of games with a missing Elo (left) or time control (right), by year. Dashed line: missing in the source files. Green line: still missing after repair (FIDE rating-list lookup for Elo; clock data and event name for time control). Games removed during cleaning are excluded from the green line.",
              "move": "Critical gap (the smallest rating gap at which an underdog win is classed as an upset) estimated on all analysed games (open circle) and on the subset whose ratings were confirmed against FIDE lists (filled dot). A small difference indicates the result is insensitive to the rating check.",
              "cal": "Calibration on held-out data. Games are binned by rating gap (100-point bins); each point compares the model's mean predicted underdog win probability with the observed win frequency in that bin. The held-out games come from tournaments excluded when fitting. Points on the dashed line indicate perfect calibration."}
NOTE_LIMIT = {"kept": "Rating availability is not random: missing ratings are concentrated among unrated and casual players, so the results describe rated players only.",
              "ba": "The rating distribution before cleaning covers only games in which both ratings are present in the source, so it is not the full raw set. The length rule (at least 10 half-moves) also removes some genuine short draws.",
              "miss": "A chi-square test shows missingness depends on observed variables, which rules out MCR but not dependence on the missing value itself (MNAR). That cannot be tested from the data; unrated players are probably weaker, so the results describe rated players.",
              "move": "The confirmed-rating subset under-represents rapidly improving players, whose ratings lag their strength and who win more often as underdogs. Gaps estimated on this subset are probably 20 to 45 points too small for rapid and blitz.",
              "cal": "Bins at large gaps contain few games, so their observed frequencies have wide sampling error. The headline error above is the game-weighted mean absolute difference between predicted and observed rates."}
NOTE_LIMIT["miss"] = NOTE_LIMIT["miss"].replace("MCR", "MCAR")


def numbers(view, colour, band, alpha):
    """One sentence with the figures behind the chart, computed from the files on screen."""
    if view == "kept":
        steps = [(FUNNEL[i + 1][0], FUNNEL[i][1] - FUNNEL[i + 1][1]) for i in range(3)]
        big = max(steps, key=lambda t: t[1])
        return (f"Of {FUNNEL[0][1]:,} parsed games, {FUNNEL[3][1]:,} ({FUNNEL[3][1] / FUNNEL[0][1]:.0%}) form the analysis set; "
                f"the largest loss is '{big[0]}' ({big[1]:,} games).")
    if view == "ba":
        n = ba_numbers()
        return (f"Before cleaning, {n['short']:.1%} of games have fewer than 10 half-moves and {n['high']:.1%} an average rating of 2900 or more (engine level). "
                f"After cleaning, median length is about {n['med_clean']:.0f} half-moves (before: {n['med_raw']:.0f}) and median average rating about {n['r_clean']:.0f} (before: {n['r_raw']:.0f}).")
    if view == "miss":
        return (f"Elo is missing in {_miss('elo_source'):.1%} of source games and in {_miss('elo_repaired'):.1%} after repair; time control in {_miss('tc_source'):.1%} and {_miss('tc_repaired'):.1%}. "
                f"Missing Elo varies by event ({_test('elo_missing', 'event')}), so it is not completely random (MCAR).")
    if view == "move":
        sel = lambda d: d[(d.underdog_colour == colour) & (d.band == band) & np.isclose(d.alpha, alpha)].set_index("tc_class")
        a = sel(CG)
        if CGV is None: return ""
        v = sel(CGV)
        diff = {t: v.loc[t, "critical_gap"] - a.loc[t, "critical_gap"] for t in TCS if t in a.index and t in v.index and np.isfinite(a.loc[t, "critical_gap"]) and np.isfinite(v.loc[t, "critical_gap"])}
        if not diff: return ""
        t = max(diff, key=lambda k: abs(diff[k]))
        return f"For this selection the checked-ratings gap differs from the all-games gap by {abs(diff[t]):.0f} Elo at most ({t}); classical: {diff.get('classical', float('nan')):+.0f}."
    err = (CAL.obs_win - CAL.pred_win).abs()
    return f"Mean absolute error is {CAL_ERR:.2f} percentage points over {int(CAL.n.sum()):,} held-out games; the largest single bin is off by {err.max() * 100:.1f} points ({int(CAL.n[err.idxmax()]):,} games)."


def notes_for(view, colour="white", band="2200-2499", alpha=0.1):
    out = [html.H6("What it shows"), html.P(NOTE_SHOWS[view]), html.P(numbers(view, colour, band, alpha))]
    if view == "kept":
        out += [html.H6("Removal reasons")]
        for i, (k, v) in enumerate(REMOVED.items()):
            out.append(html.P(f"{k} (−{FUNNEL[i][1] - FUNNEL[i + 1][1]:,}): {v}.", style={"marginBottom": "6px"}))
        out.append(html.Div(style={"height": "6px"}))
    out += [html.H6("Limitation"), html.P(NOTE_LIMIT[view])]
    return out


NONE = dict(colour=False, band=False, alpha=False)
USES = {"kept": NONE, "ba": NONE, "miss": NONE, "move": dict(colour=True, band=True, alpha=True), "cal": dict(colour=True, band=False, alpha=False)}


def fig_kept():
    names = [n for n, _ in FUNNEL]
    vals = [v for _, v in FUNNEL]
    labels = [f"{v:,}" for v in vals]
    colors = [GREY] * len(vals)
    colors[3] = GREEN                                                       # the analysis set every result uses
    fig = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker_color=colors, text=labels, textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:,} games<extra></extra>"))
    style_figure(fig, 410)
    fig.update_layout(bargap=0.35, margin=dict(l=190, r=190, t=10, b=50))
    fig.update_xaxes(title="Games", range=[0, FUNNEL[0][1] * 1.02])
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_ba():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=["Game length", "Average rating of the two players"])
    for j, (var, width) in enumerate((("length", 10), ("rating", 100)), start=1):
        for stage in ("clean", "raw"):
            x, n = _share(stage, var)
            share = n / n.sum()
            if stage == "clean":
                fig.add_trace(go.Bar(x=x + width / 2, y=share, width=width * 0.9, marker_color=GREEN, name="After cleaning", legendgroup="clean", showlegend=(j == 1),
                                     customdata=n, hovertemplate="%{x:.0f}+: %{y:.1%} (%{customdata:,} games)<extra>after cleaning</extra>"), row=1, col=j)
            else:
                fig.add_trace(go.Scatter(x=x + width / 2, y=share, mode="lines", line=dict(color="#e8e7e5", width=2, shape="hvh"), name="Before cleaning", legendgroup="raw",
                                         showlegend=(j == 1), customdata=n, hovertemplate="%{x:.0f}+: %{y:.1%} (%{customdata:,} games)<extra>before cleaning</extra>"), row=1, col=j)
    style_figure(fig, 410)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=1.16, font=dict(color="#c3c2c1")), margin=dict(l=60, r=20, t=70, b=55), bargap=0)
    fig.update_xaxes(title="Half-moves (last bin: 200+)", range=[0, 215], row=1, col=1)
    fig.update_xaxes(title="Average rating (Elo)", range=[800, 3950], row=1, col=2)
    fig.update_yaxes(title="Share of games", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    for a in fig.layout.annotations:
        a.font.color = "#c3c2c1"
    return fig


def fig_miss():
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05, subplot_titles=["Elo", "Time control"])
    for j, (src, rep) in enumerate((("elo_source", "elo_repaired"), ("tc_source", "tc_repaired")), start=1):
        for m, color, dash, nm in ((src, "#e8e7e5", "dash", "In the source files"), (rep, GREEN, "solid", "After repair")):
            d = MB[MB.measure == m].sort_values("year")
            fig.add_trace(go.Scatter(x=d.year, y=d.missing / d.n, mode="lines+markers", line=dict(color=color, width=3, dash=dash), marker=dict(size=8), name=nm,
                                     legendgroup=m[-6:], showlegend=(j == 1), customdata=np.c_[d.missing, d.n],
                                     hovertemplate="%{x}: %{y:.1%} (%{customdata[0]:,} of %{customdata[1]:,} games)<extra>" + nm + "</extra>"), row=1, col=j)
    style_figure(fig, 410)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=1.16, font=dict(color="#c3c2c1")), margin=dict(l=60, r=20, t=70, b=55))
    fig.update_yaxes(title="Share of games with a missing value", tickformat=".0%", rangemode="tozero", row=1, col=1)
    fig.update_xaxes(title="Year", dtick=1, tickformat="d")
    for a in fig.layout.annotations:
        a.font.color = "#c3c2c1"
    return fig


def fig_move(colour, band, alpha):
    fig = go.Figure()
    sel = lambda d: d[(d.underdog_colour == colour) & (d.band == band) & np.isclose(d.alpha, alpha)].set_index("tc_class")
    a, v = sel(CG), (sel(CGV) if CGV is not None else None)
    for i, t in enumerate(TCS):
        if t not in a.index or not np.isfinite(a.loc[t, "critical_gap"]):
            continue
        ga = a.loc[t, "critical_gap"]
        if v is not None and t in v.index and np.isfinite(v.loc[t, "critical_gap"]):
            gv = v.loc[t, "critical_gap"]
            fig.add_trace(go.Scatter(x=[ga, gv], y=[t.title()] * 2, mode="lines", line=dict(color=GREY, width=3), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=[gv], y=[t.title()], mode="markers+text", text=[f"{gv:.0f}"], textposition="bottom center", showlegend=False, legendgroup="checked",
                                     marker=dict(size=14, color=TC_COL[t]), hovertemplate=f"{t.title()}, ratings checked: {gv:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[ga], y=[t.title()], mode="markers+text", text=[f"{ga:.0f}"], textposition="top center", showlegend=False, legendgroup="all",
                                 marker=dict(size=14, color=PAPER, line=dict(color=TC_COL[t], width=3)),
                                 hovertemplate=f"{t.title()}, all games: {ga:.0f}<extra></extra>"))
    for nm, grp, sym in (("All games", "all", dict(color=PAPER, line=dict(color="#c3c2c1", width=3))), ("Rating-checked games", "checked", dict(color="#c3c2c1"))):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm, legendgroup=grp, marker=dict(size=12, **sym), showlegend=True))
    style_figure(fig, 410)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.08, font=dict(color="#c3c2c1")))
    xs = [g for g in list(a.critical_gap) + (list(v.critical_gap) if v is not None else []) if np.isfinite(g)]
    lo = 50 * int((min(xs) - 60) // 50) if xs else 0
    hi = 50 * int((max(xs) + 60) // 50 + 1) if xs else 1000
    fig.update_xaxes(title="Critical gap (Elo points)", range=[max(lo, 0), hi])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=[t.title() for t in TCS])
    fig.update_layout(margin=dict(l=90, r=40, t=10, b=50))
    return fig


def fig_cal(colour):
    fig = go.Figure()
    top = float(max(CAL.pred_win.max(), CAL.obs_win.max()) * 1.08)
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="#989795", dash="dash"), hoverinfo="skip", showlegend=False))
    c = CAL[CAL.underdog_color == colour]
    for t in TCS:
        o = c[c.tc_class == t]
        fig.add_trace(go.Scatter(x=o.pred_win, y=o.obs_win, mode="markers", marker=dict(size=10, color=TC_COL[t], line=dict(color=PAPER, width=1)),
                                 name=t.title(), showlegend=True, customdata=np.c_[o.n, o.gap100], hovertemplate=t.title() + ": predicted %{x:.1%}, observed %{y:.1%}<br>gap %{customdata[1]}+, %{customdata[0]:,} games<extra></extra>"))
    style_figure(fig, 410)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=1.08, font=dict(color="#c3c2c1")))
    fig.update_xaxes(title="Predicted chance the underdog wins", tickformat=".0%", range=[0, top])
    fig.update_yaxes(title="Observed", tickformat=".0%", range=[0, top])
    fig.update_layout(margin=dict(l=70, r=40, t=10, b=50))
    return fig


@callback(Output(P + "fig", "figure"), Output(P + "q", "children"), Output(P + "notes", "children"),
          Output(P + "colour", "options"), Output(P + "band", "options"), Output(P + "alpha", "options"),
          Input(P + "view", "value"), Input(P + "colour", "value"), Input(P + "band", "value"), Input(P + "alpha", "value"))
def update(view, colour, band, alpha):
    fig = {"kept": fig_kept, "ba": fig_ba, "miss": fig_miss}.get(view, lambda: fig_move(colour, band, alpha) if view == "move" else fig_cal(colour))()
    u = USES[view]                                  # a control that does not change this view is disabled
    dis = lambda opts, key: [{**o, "disabled": not u[key]} for o in opts]
    return fig, QUESTION[view], notes_for(view, colour, band, alpha), dis(COLOUR_OPTS, "colour"), dis(BAND_OPTS, "band"), dis(ALPHA_OPTS, "alpha")


