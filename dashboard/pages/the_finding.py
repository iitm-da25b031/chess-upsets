"""Page 2: "The finding".   The underdog's chance falls with the rating gap; the gap at which a win becomes surprising is larger in faster games.

Reads (data/dashboard/, written by empirical_upsets.py and build_dashboard_data.py):
    critical_gap_empirical.csv   critical gap + 95% interval, per class x underdog colour x band x alpha
    fitted_curves.csv            fitted P(underdog wins) with 95% band, per class x colour x band
    curve_bins.csv               games / wins / draws / losses per class x colour x band x gap bin
Two views behind one toggle: the chance of winning (curves) and what replaces the wins (wins, draws, losses).
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from common import DATA, TCS, TC_COL, GREEN, GREY, PAPER, GCFG, seg, view_toggle, kpi, notes_children, notes_card, style_figure

dash.register_page(__name__, path="/finding", name="The finding", title="Upset Explorer: the finding", order=1)
P = "p2_"
MIN_GAMES = 100                                                                   # a gap bin with fewer games is not drawn

CG = pd.read_csv(os.path.join(DATA, "critical_gap_empirical.csv"))
FC = pd.read_csv(os.path.join(DATA, "fitted_curves.csv"))
CB = pd.read_csv(os.path.join(DATA, "curve_bins.csv"))
BIN = int(np.diff(np.sort(CB.gap_bin.unique())).min())                            # width of a gap bin, read from the data
MAXBIN = int(CB.gap_bin.max())

COLOUR_OPTS = [{"label": "White", "value": "white"}, {"label": "Black", "value": "black"}]
BAND_OPTS = [{"label": "Under 2200", "value": "<2200"}, {"label": "2200–2499", "value": "2200-2499"}, {"label": "2500+", "value": "2500+"}]
ALPHA_OPTS = [{"label": "20%", "value": 0.2}, {"label": "10%", "value": 0.1}, {"label": "5%", "value": 0.05}]

controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label("Underdog plays"), seg(P + "colour", COLOUR_OPTS, "white")], width=3),
    dbc.Col([html.Label("Players' average rating"), seg(P + "band", BAND_OPTS, "2200-2499")], width=5),
    dbc.Col([html.Label("Upset threshold"), seg(P + "alpha", ALPHA_OPTS, 0.1)], width=4),
], className="g-3 align-items-start")), className="mb-3")

layout = html.Div([
    dbc.Row(dbc.Col(html.H2("The finding"), width=12), className="mb-2"),
    controls,
    dbc.Row([dbc.Col(kpi(P + "l1", P + "k1", TC_COL["classical"]), width=4), dbc.Col(kpi(P + "l2", P + "k2", TC_COL["rapid"]), width=4),
             dbc.Col(kpi(P + "l3", P + "k3", TC_COL["blitz"]), width=4)], id=P + "kpi_row", className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(view_toggle(P + "view", [{"label": "Chance of winning", "value": "curves"}, {"label": "Wins, draws, losses", "value": "wdl"}], "curves")),
                          dbc.CardBody([html.Div(id=P + "q", className="q-title"), dcc.Graph(id=P + "fig", config=GCFG)])], className="h-100"), width=8),
        dbc.Col(notes_card(P + "notes"), width=4)], className="g-3"),
], className="p-4")

QUESTION = {"curves": "How fast does the underdog's chance fall as the gap grows?", "wdl": "What replaces the underdog's wins as the gap grows?"}
NOTES = {"curves": ("Fitted probability that the underdog wins against rating gap, with a 95% band, for each time control at the middle rating of the selected band. Dots are observed win "
                    "rates in gap bins with at least 100 games. Rings mark where each fit crosses the threshold (dashed line), with the 95% interval of that crossing.",
                    "Beyond about 600 points few games remain and the bands widen. Each curve is evaluated at one average rating, so games at the edges of the band differ slightly."),
         "wdl": ("Share of the underdog's results that were wins, draws and losses by rating gap, one panel per time control. Bars are scaled to 100% with losses at the baseline.",
                 "Bins with fewer than 100 games are omitted. The last bin pools every game with a gap of 600 or more.")}


@callback(Output(P + "l1", "children"), Output(P + "k1", "children"), Output(P + "l2", "children"), Output(P + "k2", "children"),
          Output(P + "l3", "children"), Output(P + "k3", "children"),
          Output(P + "fig", "figure"), Output(P + "q", "children"), Output(P + "notes", "children"),
          Output(P + "alpha", "options"), Output(P + "kpi_row", "style"),
          Input(P + "colour", "value"), Input(P + "band", "value"), Input(P + "alpha", "value"), Input(P + "view", "value"))
def update(colour, band, alpha, view):
    cg = CG[(CG.underdog_colour == colour) & (CG.band == band) & np.isclose(CG.alpha, alpha)].set_index("tc_class")

    def card(tc):                                   # label (with the 95% interval) and value of one critical-gap card
        if tc not in cg.index or not np.isfinite(cg.loc[tc, "critical_gap"]):
            return f"{tc.title()} critical gap", "n/a"
        r = cg.loc[tc]
        return f"{tc.title()} critical gap (95%: {r.lo95:.0f} to {r.hi95:.0f})", f"{r.critical_gap:.0f} Elo"
    k = {t: card(t) for t in TCS}
    fig = fig_curves(colour, band, alpha, cg) if view == "curves" else fig_wdl(colour, band)
    uses_alpha = view == "curves"                   # a control that does not change what is on screen is disabled, and returns with its view
    alpha_opts = [{**o, "disabled": not uses_alpha} for o in ALPHA_OPTS]
    s, l = NOTES[view]
    if view == "curves":
        g = {t: cg.loc[t, "critical_gap"] for t in TCS if t in cg.index and np.isfinite(cg.loc[t, "critical_gap"])}
        fact = (f"At {alpha:.0%} and an average rating of {band}, the critical gap is {g['classical']:.0f} Elo in classical, {g['rapid']:.0f} in rapid and {g['blitz']:.0f} in blitz, "
                f"so blitz needs {g['blitz'] - g['classical']:.0f} Elo more than classical.") if len(g) == 3 else ""
    else:
        o = observed(colour, band)
        o = o[o.tc_class == "classical"].sort_values("gap_bin")
        fact = (f"In classical games the underdog wins {o.wins.iloc[0] / o.games.iloc[0]:.0%} of games at a gap of {int(o.gap_bin.iloc[0])} to {int(o.gap_bin.iloc[0]) + 50} Elo "
                f"and {o.wins.iloc[-1] / o.games.iloc[-1]:.0%} at {int(o.gap_bin.iloc[-1])} or more.") if len(o) else ""
    return (*k["classical"], *k["rapid"], *k["blitz"], fig, QUESTION[view], notes_children(s, l, fact), alpha_opts, {} if uses_alpha else {"opacity": 0.35})


def observed(colour, band):
    ob = (CB[(CB.underdog_color == colour) & (CB.band == band)].groupby(["tc_class", "gap_bin"], as_index=False)[["games", "wins", "draws", "losses"]].sum())
    return ob[ob.games >= MIN_GAMES]


def fig_curves(colour, band, alpha, cg):
    f1 = go.Figure()
    fc = FC[(FC.underdog_colour == colour) & (FC.band == band) & (FC.gap <= 800)]
    ob = observed(colour, band).assign(rate=lambda x: x.wins / x.games)
    ends = {}
    for t in TCS:
        c = fc[fc.tc_class == t]
        f1.add_trace(go.Scatter(x=np.r_[c.gap, c.gap[::-1]], y=np.r_[c.p_win_hi95, c.p_win_lo95[::-1]], fill="toself", line=dict(width=0),
                                fillcolor=TC_COL[t] + "33", hoverinfo="skip"))
        f1.add_trace(go.Scatter(x=c.gap, y=c.p_win, mode="lines", line=dict(color=TC_COL[t], width=3), hoverinfo="skip"))
        o = ob[ob.tc_class == t]
        f1.add_trace(go.Scatter(x=o.gap_bin + BIN / 2, y=o.rate, mode="markers", marker=dict(color=TC_COL[t], size=7, line=dict(color=PAPER, width=1)),
                                customdata=o.games, hovertemplate=t.title() + ": %{y:.1%} of %{customdata:,} games<extra></extra>"))
        if t in cg.index and np.isfinite(cg.loc[t, "critical_gap"]):        # ring on the threshold line where the fit crosses it, with its 95% interval
            r = cg.loc[t]
            f1.add_trace(go.Scatter(x=[r.critical_gap], y=[alpha], mode="markers", marker=dict(size=13, color=PAPER, line=dict(color=TC_COL[t], width=3)),
                                    error_x=dict(type="data", symmetric=False, array=[r.hi95 - r.critical_gap], arrayminus=[r.critical_gap - r.lo95],
                                                 color=TC_COL[t], thickness=4, width=0),
                                    hovertemplate=f"{t.title()} critical gap {r.critical_gap:.0f} (95%: {r.lo95:.0f}-{r.hi95:.0f})<extra></extra>"))
        if len(c):
            ends[t] = float(c.p_win.iloc[-1])
    lab_y, last = {}, None                                   # direct labels at the line ends, nudged apart so they do not overlap
    for t in sorted(ends, key=ends.get):
        y = ends[t] if last is None else max(ends[t], last + 0.016)
        lab_y[t], last = y, y
    for t, y in lab_y.items():
        f1.add_annotation(x=800, y=y, text=t.title(), xanchor="left", showarrow=False, font=dict(color=TC_COL[t], size=12))
    f1.add_hline(y=alpha, line=dict(color="#989795", dash="dash"))
    style_figure(f1, 410)
    f1.update_layout(margin=dict(l=55, r=80, t=10, b=50))
    f1.update_xaxes(title="Rating gap (Elo points)", range=[0, 800])
    f1.update_yaxes(title="P(underdog wins)", tickformat=".0%", rangemode="tozero")
    return f1


def fig_wdl(colour, band):
    """Small multiples, one panel per class, bars scaled to 100% with LOSS at the baseline (a growing loss share is read against the axis)."""
    f3 = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[t.title() for t in TCS], horizontal_spacing=0.02)
    ob = observed(colour, band)
    for j, t in enumerate(TCS, start=1):
        o = ob[ob.tc_class == t]
        x = [f"{g}+" if g == MAXBIN else str(g) for g in o.gap_bin]
        for name, col_, share in (("Loss", "#4a4845", o.losses / o.games), ("Draw", "#989795", o.draws / o.games), ("Win", GREEN, o.wins / o.games)):
            f3.add_trace(go.Bar(x=x, y=share, name=name, marker_color=col_, customdata=o.games, showlegend=(j == 1),
                                hovertemplate=name + " %{y:.1%} of %{customdata:,} games<extra></extra>"), row=1, col=j)
    style_figure(f3, 410)
    f3.update_layout(barmode="stack", bargap=0.1, margin=dict(l=55, r=20, t=90, b=50), showlegend=True,
                     legend=dict(orientation="h", y=1.2, x=0, traceorder="reversed"))
    f3.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=1)
    f3.update_xaxes(title="Rating gap (lower edge of bin)", tickangle=0, tickmode="array", tickvals=["0","200","400","600+"], ticktext=["0","200","400","600+"])
    for a in f3.layout.annotations:
        a.font.color = "#c3c2c1"
    return f3


