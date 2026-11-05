"""Presentation evidence: held-out predictions and robustness."""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from common import DATA, TCS, TC_COL, GREEN, GREY, PAPER, GCFG, seg, view_toggle, kpi_static, notes_card, style_figure

dash.register_page(__name__, path="/trust", name="Can it be trusted?", title="Upset Explorer: can it be trusted?", order=2)
P = "p3_"
FUNNEL = [(a, int(b)) for a, b in pd.read_csv(os.path.join(DATA, "funnel.csv")).itertuples(index=False, name=None)]
CG = pd.read_csv(os.path.join(DATA, "critical_gap_empirical.csv"))
CGV = pd.read_csv(os.path.join(DATA, "critical_gap_empirical_verified.csv"))
CAL = pd.read_csv(os.path.join(DATA, "calibration_empirical.csv"))
CAL_ERR = float(np.average(np.abs(CAL.obs_win - CAL.pred_win), weights=CAL.n)) * 100
ANALYSED = dict(FUNNEL)["Analysis set"]
CHECKED = dict(FUNNEL)["Ratings checked against FIDE"]
COLOUR_OPTS = [{"label": "White", "value": "white"}, {"label": "Black", "value": "black"}]
BAND_OPTS = [{"label": "Under 2200", "value": "<2200"}, {"label": "2200-2499", "value": "2200-2499"}, {"label": "2500+", "value": "2500+"}]

# Threshold sensitivity and strong-player misfit survive as concise findings.
ROOT = os.path.dirname(DATA)
GRID = pd.read_csv(os.path.join(ROOT, "viz_data", "critical_gap_by_alpha.csv"))
grid = GRID[GRID.alpha.between(0.05 - 1e-9, 0.30 + 1e-9)].pivot_table(
    index=["underdog_color", "band", "alpha"], columns="tc_class", values="critical_gap").dropna()
ORDERED = int(((grid.classical < grid.rapid) & (grid.rapid < grid.blitz)).sum())
TOTAL = len(grid)
EXPECTED = pd.read_csv(os.path.join(ROOT, "viz_data", "expected_upsets.csv"))
tail = EXPECTED[np.isclose(EXPECTED.alpha, 0.1) & (EXPECTED.band == "2500+") & (EXPECTED.tc_class != "all")]
TAIL_RATIO = tail.obs_wins.sum() / tail.exp_wins.sum()

layout = html.Div([
    html.H2("Can it be trusted?"),
    html.P("Check the predictions on new tournaments. See where the model needs care.", className="lead-text"),
    dbc.Card(dbc.CardBody(view_toggle(P + "view", [
        {"label": "Prediction check", "value": "cal"},
        {"label": "Accuracy by rating", "value": "rating"},
        {"label": "Check the ratings", "value": "move"}], "cal")), className="mb-3"),
    dbc.Row([
        dbc.Col(kpi_static("Games analysed", f"{ANALYSED:,}"), width=4),
        dbc.Col(kpi_static("Games with ratings checked against FIDE", f"{CHECKED / ANALYSED:.0%}"), width=4),
        dbc.Col(kpi_static("Test error in percentage points", f"{CAL_ERR:.2f}"), width=4),
    ], id=P + "kpi_row", className="mb-3 g-3"),
    html.Div(html.Details([
        html.Summary("Change colour or rating"),
        dbc.Row([
            dbc.Col([html.Label("Underdog plays"), seg(P + "colour", COLOUR_OPTS, "black")], md=4),
            dbc.Col([html.Label("Players' average rating"), seg(P + "band", BAND_OPTS, "2200-2499")], md=8),
        ], className="g-3 mt-1"),
    ], className="explore-settings"), id=P + "settings", style={"display": "none"}),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(id=P + "q", className="q-title"),
            html.P(id=P + "selection", className="selection-label"),
            dcc.Graph(id=P + "fig", config=GCFG),
        ]), className="h-100"), width=8),
        dbc.Col(notes_card(P + "notes"), width=4),
    ], className="g-3"),
], className="p-4")


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
    for nm, grp, sym in (("All games", "all", dict(color=PAPER, line=dict(color="#c3c2c1", width=3))), ("Ratings checked against FIDE", "checked", dict(color="#c3c2c1"))):
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

def fig_cal():
    fig = go.Figure()
    top = float(max(CAL.pred_win.max(), CAL.obs_win.max()) * 1.08)
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="#989795", dash="dash"), hoverinfo="skip", showlegend=False))
    c = CAL.assign(pred_total=CAL.pred_win * CAL.n, obs_total=CAL.obs_win * CAL.n)
    c = c.groupby(["tc_class", "gap100"], as_index=False)[["n", "pred_total", "obs_total"]].sum()
    c = c.assign(pred_win=c.pred_total / c.n, obs_win=c.obs_total / c.n)
    for t in TCS:
        o = c[c.tc_class == t]
        fig.add_trace(go.Scatter(x=o.pred_win, y=o.obs_win, mode="markers", marker=dict(size=10, color=TC_COL[t], line=dict(color=PAPER, width=1)),
                                 name=t.title(), showlegend=True, customdata=np.c_[o.n, o.gap100], hovertemplate=t.title() + ": predicted %{x:.1%}, observed %{y:.1%}<br>gap %{customdata[1]}+, %{customdata[0]:,} games<extra></extra>"))
    style_figure(fig, 410)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=1.08, font=dict(color="#c3c2c1")))
    fig.update_xaxes(title="Predicted chance the underdog wins", tickformat=".0%", range=[0, top], constrain="domain")
    fig.update_yaxes(title="Observed win rate", tickformat=".0%", range=[0, top], scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(margin=dict(l=70, r=40, t=10, b=50))
    return fig

def fig_rating_fit():
    """Observed / expected wins in the model's under-10% tail, pooled across time controls."""
    d = EXPECTED[np.isclose(EXPECTED.alpha, 0.1) & (EXPECTED.band != "all") & (EXPECTED.tc_class != "all")]
    d = d.groupby("band")[["obs_wins", "exp_wins", "games_pwin_below"]].sum().reindex(["<2200", "2200-2499", "2500+"])
    ratio = d.obs_wins / d.exp_wins
    fig = go.Figure(go.Bar(
        x=["Under 2200", "2200-2499", "2500+"], y=ratio,
        marker_color=[GREY, GREY, GREEN], text=[f"{r:.2f}×" for r in ratio], textposition="outside",
        customdata=np.c_[d.obs_wins, d.exp_wins, d.games_pwin_below],
        hovertemplate="%{x}: %{y:.2f}× expected wins<br>Observed: %{customdata[0]:,.0f}<br>Expected: %{customdata[1]:,.0f}<br>%{customdata[2]:,.0f} games<extra></extra>",
    ))
    fig.add_hline(y=1, line=dict(color="#c3c2c1", dash="dash"),
                  annotation_text="Observed = expected", annotation_position="top right")
    style_figure(fig, 410)
    fig.update_layout(margin=dict(l=70, r=25, t=35, b=60), bargap=0.5)
    fig.update_xaxes(title="Players' average rating")
    fig.update_yaxes(title="Observed wins / expected wins", range=[0, max(1.25, ratio.max() * 1.2)])
    return fig


def notes_for(view, colour, band):
    if view == "rating":
        return [
            html.P("At 1, actual wins match the model's expected count. Below 1, the model predicts too many wins."),
            html.P(f"In the 2500+ average-rating band, underdogs win {TAIL_RATIO:.2f} times as often as expected."),
            html.H6("Keep in mind"), html.P("The model overestimates rare wins at the highest rating level. A good overall score hides this weakness."),
            html.Details([html.Summary("How we checked"), html.P("We use games where the model gave the underdog less than a 10% win chance. We add those chances to get the expected number of wins, then count the actual wins. Each band uses the players' average Elo. We combine colours and time controls. This check uses the analysis games, not the separate test set.")]),
        ]
    if view == "cal":
        return [
            html.P("Each dot compares a predicted win rate with the actual rate. Dots near the dashed line agree."),
            html.P(f"The average error is {CAL_ERR:.2f} percentage points across {int(CAL.n.sum()):,} test games."),
            html.H6("Keep in mind"), html.P("This chart combines rating levels. It hides the model's weaker predictions for the 2500+ group."),
            html.Details([html.Summary("How we tested"), html.P("We kept half the tournaments out of training and tested on their games. Each dot groups games by rating gap and time control. It combines both colours. The error uses the original groups split by colour. Groups with more games count more. We used one train/test split, and small groups give less precise results.")]),
        ]
    a = CG[(CG.underdog_colour == colour) & (CG.band == band) & np.isclose(CG.alpha, 0.1)].set_index("tc_class")
    v = CGV[(CGV.underdog_colour == colour) & (CGV.band == band) & np.isclose(CGV.alpha, 0.1)].set_index("tc_class")
    delta = (v.critical_gap - a.critical_gap).abs().dropna()
    return [
        html.P(f"Using only games with checked ratings moves these critical gaps by at most {delta.max():.0f} Elo."),
        html.P(f"The gap is smallest in classical and largest in blitz in {ORDERED} of {TOTAL} tested combinations."),
        html.H6("Keep in mind"), html.P("The checked games may include fewer fast-improving players. At a 5% threshold, the choice of model changes the gap estimates more."),
        html.Details([html.Summary("How to read this"), html.P("Open circles show all analysis games. Filled dots show games with ratings checked against FIDE lists. The critical gap is where the predicted win chance falls below 10%. This tests how much rating checks change the result. We also compared colours, rating bands and thresholds from 5% to 30%. The ordering fails only when gaps tie at zero.")]),
    ]


@callback(Output(P + "fig", "figure"), Output(P + "q", "children"),
          Output(P + "notes", "children"), Output(P + "settings", "style"),
          Output(P + "selection", "children"), Output(P + "kpi_row", "style"),
          Input(P + "view", "value"), Input(P + "colour", "value"), Input(P + "band", "value"))
def update(view, colour, band):
    if view == "cal":
        fig, title, selection = fig_cal(), "Predictions broadly match new tournaments", "Both colours · all rating levels · one series per time control"
    elif view == "rating":
        fig, title = fig_rating_fit(), "The model predicts too many rare wins in the 2500+ group"
        selection = "Predicted win chance below 10% · all time controls and colours"
    else:
        fig, title = fig_move(colour, band, 0.1), "Do checked ratings change the result?"
        selection = f"{colour.title()} underdog · average rating {band} · upset threshold 10%"
    return fig, title, notes_for(view, colour, band), ({} if view == "move" else {"display": "none"}), selection, ({} if view == "cal" else {"display": "none"})
