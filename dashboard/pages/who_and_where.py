"""Page 5: "Who and where".   Funnel plots for events and players (Spiegelhalter 2005).

Reads (data/dashboard/): tournaments.csv (build_dashboard_data.py), players_by_class.csv (prep_page5.py)
Each unit (an event or a player) is a point: x = number of underdog games, y = share that were upsets. The funnel is the range of rates that chance alone
would produce if every game had the overall upset probability (square-root limits for a count). Points outside it are flagged; they are candidates for
explanation, not explanations.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from common import DATA, GREEN, GREY, PAPER, GRID, GCFG, seg, kpi, notes_children

dash.register_page(__name__, path="/who", name="Who and where", title="Upset Explorer: who and where", order=4)
P = "p5_"

EV = pd.read_csv(os.path.join(DATA, "tournaments.csv"))
EV = EV.assign(name=EV.tournament.astype(str) + " " + EV.year.astype(str), n=EV.games)[["name", "tc_class", "n", "upsets"]]
PL = pd.read_csv(os.path.join(DATA, "players_by_class.csv"))[["name", "key", "tc_class", "n", "upsets"]]
LIMITS = {0.05: ("95%", 1.96), 0.002: ("99.8%", 3.09)}      # label, z (two-sided)


def limit_counts(n, p0, z):
    """Count range expected from chance alone: sqrt(count) is roughly normal with variance 1/4 (Poisson, variance-stabilised).
    Smooth and sensible at small counts, where the normal approximation of the rate itself is poor."""
    m = np.sqrt(np.asarray(n) * p0)
    return np.maximum(m - z / 2, 0) ** 2, (m + z / 2) ** 2


def units(view, tc):
    if view == "events":
        d = EV if tc == "all" else EV[EV.tc_class == tc]
        return d.copy()
    d = PL if tc == "all" else PL[PL.tc_class == tc]
    if tc == "all":                                                   # one point per player across classes
        d = d.groupby("key").agg(name=("name", "first"), n=("n", "sum"), upsets=("upsets", "sum")).reset_index()
    return d[["name", "n", "upsets"]].copy()


def classify(d, alpha):
    p0 = d.upsets.sum() / d.n.sum()
    lo, hi = limit_counts(d.n, p0, LIMITS[alpha][1])
    d = d.assign(rate=d.upsets / d.n, z=(d.upsets - d.n * p0) / np.sqrt(d.n * p0 * (1 - p0)), high=d.upsets > hi, low=d.upsets < lo)
    return d, p0



TC_OPTS = [{"label": "All", "value": "all"}, {"label": "Classical", "value": "classical"}, {"label": "Rapid", "value": "rapid"}, {"label": "Blitz", "value": "blitz"}]
LIM_OPTS = [{"label": "95%", "value": 0.05}, {"label": "99.8%", "value": 0.002}]
VIEW_OPTS = [{"label": "Events", "value": "events"}, {"label": "Players", "value": "players"}]


controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label("Time control"), seg(P + "tc", TC_OPTS, "all")], width=5),
    dbc.Col([html.Label("Funnel limits"), seg(P + "alpha", LIM_OPTS, 0.002)], width=3),
    dbc.Col([html.Label("View"), seg(P + "view", VIEW_OPTS, "events")], width=4),
], className="g-3 align-items-start")), className="mb-3")


layout = html.Div([
    dbc.Row(dbc.Col(html.H2("Who and where"), width=12), className="mb-2"),
    controls,
    dbc.Row([dbc.Col(kpi(P + "l1", P + "k1"), width=4), dbc.Col(kpi(P + "l2", P + "k2"), width=4), dbc.Col(kpi(P + "l3", P + "k3"), width=4)], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(id=P + "chart_head"), dbc.CardBody(dcc.Graph(id=P + "fig_main", config=GCFG))], className="h-100"), width=7),
        dbc.Col(dbc.Card([dbc.CardHeader("Largest deviations"), dbc.CardBody([html.Div(id=P + "rank"), html.Div(id=P + "notes", className="notes mt-3")])], className="h-100"), width=5)],
        className="g-3"),
], className="p-4")

NOTES = {"events": ("Each point is one event (tournament, year, time control): underdog games played against the share that were upsets. The funnel is the range of rates chance alone would produce at the overall upset probability. Green points lie above it, open circles below it.",
                    "Events differ in rating spread, so part of the scatter reflects who played. A point outside the funnel is a flag to investigate, not an explanation."),
         "players": ("Each point is one player (at least 10 games as the lower-rated side): games as underdog against the share that were upsets. The funnel is the range chance alone would produce at the overall rate. Green points lie above it, open circles below it.",
                     "A high rate can reflect rating lag (an improving player is rated below true strength) as well as strength. Names come from public tournament records; a flag is statistical, not a statement about conduct.")}
HEAD = {"events": "Do some events produce more upsets than chance allows?", "players": "Do some players beat higher-rated opponents more often than chance allows?"}


@callback(Output(P + "fig_main", "figure"), Output(P + "chart_head", "children"), Output(P + "rank", "children"), Output(P + "notes", "children"),
          Output(P + "l1", "children"), Output(P + "k1", "children"), Output(P + "l2", "children"), Output(P + "k2", "children"), Output(P + "l3", "children"), Output(P + "k3", "children"),
          Input(P + "view", "value"), Input(P + "tc", "value"), Input(P + "alpha", "value"))
def update(view, tc, alpha):
    d, p0 = classify(units(view, tc), alpha)
    lab = LIMITS[alpha][0]
    out = d.high | d.low
    ns = np.unique(np.round(np.geomspace(max(d.n.min(), 1), d.n.max(), 160)).astype(int))
    fig = go.Figure()
    lo_c, up_c = limit_counts(ns, p0, LIMITS[alpha][1])
    up, lo = up_c / ns, lo_c / ns
    fig.add_trace(go.Scatter(x=ns, y=up, mode="lines", line=dict(color="#c3c2c1", width=1.5, dash="dot"), name=f"{lab} limits", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ns, y=lo, mode="lines", line=dict(color="#c3c2c1", width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[ns.min(), ns.max()], y=[p0, p0], mode="lines", line=dict(color="#989795", width=1.5), name="Overall rate", hoverinfo="skip"))
    for mask, nm, mk in (((~d.high) & (~d.low), "Within", dict(color=GREY, size=6, opacity=0.55, line=dict(width=0))),
                         (d.high, "Above", dict(color=GREEN, size=8, opacity=0.95, line=dict(width=0))),
                         (d.low, "Below", dict(color="rgba(0,0,0,0)", size=8, line=dict(color="#c3c2c1", width=1.5)))):
        s_ = d[mask]
        fig.add_trace(go.Scatter(x=s_.n, y=s_.rate, mode="markers", name=nm, marker=mk,
                                 customdata=np.c_[s_.name, s_.upsets], hovertemplate="%{customdata[0]}<br>%{customdata[1]:.0f} upsets in %{x:,} games (%{y:.1%})<extra></extra>"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor=PAPER, height=440, margin=dict(l=60, r=20, t=10, b=50),
                      legend=dict(orientation="h", x=0, y=1.1, font=dict(color="#c3c2c1", size=11)))
    fig.update_xaxes(type="log", title="Underdog games (log scale)", gridcolor=GRID, tickvals=[10, 30, 100, 300, 1000, 3000], ticktext=["10", "30", "100", "300", "1,000", "3,000"], minor=dict(showgrid=False, ticks=""))
    fig.update_yaxes(title="Upset rate", tickformat=".0%", rangemode="tozero", gridcolor=GRID)
    top = d[d.high].sort_values("z", ascending=False).head(5)
    rows = [html.Div([html.Div("Name", className="rk-name"), html.Div("Upsets / games", className="rk-n"), html.Div("Rate", className="rk-rate"), html.Div("z", className="rk-z")], className="rk-head")]
    rows += [html.Div([html.Div(r["name"], className="rk-name", title=r["name"]), html.Div(f"{int(r.upsets)} / {int(r.n)}", className="rk-n"),
                       html.Div(f"{r.rate:.1%}", className="rk-rate"), html.Div(f"{r.z:.1f}", className="rk-z")], className="rk-row") for _, r in top.iterrows()]
    s, l = NOTES[view]
    unit = "events" if view == "events" else "players"
    exp = alpha * 100
    notes = notes_children(s, l, f"{int(out.sum())} of {len(d):,} {unit} lie outside the {lab} funnel, against about {exp:.1f}% expected by chance; the most extreme has z = {d.z.max():.1f}.")
    kp = (f"{unit.title()} shown", f"{len(d):,}", f"Outside the {lab} funnel (chance alone: about {exp:.1f}%)", f"{out.sum():,} ({out.mean():.1%})",
          "Share of upsets in units above the funnel", f"{d.upsets[d.high].sum() / d.upsets.sum():.1%}")
    return fig, HEAD[view], html.Div(rows), notes, *kp


