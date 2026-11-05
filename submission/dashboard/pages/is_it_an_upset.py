"""Page 1: "Is it an upset?"   Enter two ratings and a time control; the page says how likely the lower-rated player's win is and whether it counts as an upset.

Reads data/dashboard/model_coefficients_empirical.csv  (written by empirical_upsets.py; the verified-ratings file is optional)

The model (explain this in the viva)
  For each time control and underdog colour a logistic regression was fitted:
      P(underdog wins) = sigmoid( b0 + b1*x + b2*x^2 + b3*x^3 + b4*(avg_elo - 2300)/100 ),   x = min(gap, 900)/300
  and the same for P(underdog does not lose). A game is an upset at level alpha when the underdog wins (or draws) AND the model gave
  that result a probability below alpha. The critical gap is the smallest rating gap at which P(win) falls below alpha.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from common import DATA, TCS, TC_COL, GREEN, GREY, GCFG, seg, view_toggle, kpi, notes_children, notes_card, style_figure

dash.register_page(__name__, path="/", name="Is it an upset?", title="Upset Explorer: is it an upset?", order=0)
P = "p1_"
GAPS = np.arange(0, 1505, 5)


def load(name):
    df = pd.read_csv(os.path.join(DATA, name))
    df["outcome"] = df["outcome"].astype(str).str.lower()
    df["underdog_colour"] = df["underdog_colour"].astype(str).str.lower()
    return df


COEF = {"all": load("model_coefficients_empirical.csv")}
if os.path.exists(os.path.join(DATA, "model_coefficients_empirical_verified.csv")):
    COEF["verified"] = load("model_coefficients_empirical_verified.csv")


def prob(coef, tc, colour, kind, gap, avg_elo):
    """P(underdog wins) (kind='win') or P(underdog does not lose) (kind='notlose') from the fitted logistic model."""
    row = coef[(coef.tc_class == tc) & (coef.underdog_colour == colour) & (coef.outcome == kind)]
    if row.empty:
        return np.full(np.shape(gap), np.nan)
    r = row.iloc[0]
    x = np.minimum(gap, 900) / 300.0
    z = r.intercept + r.b_gap * x + r.b_gap2 * x**2 + r.b_gap3 * x**3 + r.b_avgelo * (avg_elo - 2300) / 100.0
    return 1 / (1 + np.exp(-z))


def critical_gap(coef, tc, colour, avg_elo, alpha):
    p = prob(coef, tc, colour, "win", GAPS, avg_elo)
    idx = np.where(p < alpha)[0]
    return int(GAPS[idx[0]]) if len(idx) else None


# ---------------------------------------------------------------- layout
controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label("White Elo"), dbc.Input(id=P + "w", type="number", value=2450, min=1000, max=2900, step=1, debounce=True)], width=2),
    dbc.Col([html.Label("Black Elo"), dbc.Input(id=P + "b", type="number", value=2150, min=1000, max=2900, step=1, debounce=True)], width=2),
    dbc.Col([html.Label("Time control"), seg(P + "tc", [{"label": t.title(), "value": t} for t in TCS], "classical")], width=4),
    dbc.Col([html.Label(["Upset threshold: ", html.Span(id=P + "alpha_lbl", style={"color": GREEN, "fontWeight": 700})]),
             html.Div(dcc.Slider(id=P + "alpha", min=0.05, max=0.20, step=0.01, value=0.10, updatemode="drag", marks={0.05: "5%", 0.10: "10%", 0.20: "20%"}),
                      className="slider-wrap")], width=4),
], className="g-3 align-items-start")), className="mb-3")

layout = html.Div([
    dcc.Store(id=P + "sel"),                       # shared choices (time control, threshold) in one Store
    dbc.Row(dbc.Col(html.H2("Is it an upset?"), width=12), className="mb-2"),
    controls,
    dbc.Row([dbc.Col(kpi(P + "l1", P + "k1"), width=4), dbc.Col(kpi(P + "l2", P + "k2"), width=4), dbc.Col(kpi(P + "l3", P + "k3"), width=4)], className="mb-3 g-3"),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(view_toggle(P + "view", [{"label": "Chance of winning", "value": "curves"}, {"label": "Gap by time control", "value": "bars"}], "curves")),
                          dbc.CardBody([html.Div(id=P + "q", className="q-title"), dcc.Graph(id=P + "fig", config=GCFG)])], className="h-100"), width=8),
        dbc.Col(notes_card(P + "notes"), width=4)], className="g-3"),
], className="p-4")

QUESTION = {"curves": "When does the underdog's chance fall below the threshold?", "bars": "How much more rating gap do faster games need?"}
NOTES = {
    "curves": (
        "Each line shows the underdog's predicted win chance. Green marks the selected time control. The dot marks your game. The dashed line marks the upset threshold.",
        "The model uses the rating gap, average rating and underdog's colour. Games with gaps above 900 Elo are scarce.",
    ),
    "bars": (
        "The critical gap is where the predicted win chance first falls below your threshold. Longer bars mean a larger gap is needed. Labels show the difference from classical.",
        "These estimates use your game's average rating and underdog colour. They describe how surprising a win would be.",
    ),
}


# ---------------------------------------------------------------- figures
def fig_curves(coef, tc, colour, avg, alpha, gap, p_win):
    fig = go.Figure()
    for t in TCS:                                   # grey context first, the selected class last and in the highlight colour
        if t == tc:
            continue
        y = prob(coef, t, colour, "win", GAPS, avg)
        fig.add_trace(go.Scatter(x=GAPS, y=y, mode="lines", line=dict(color=GREY, width=2), hovertemplate=f"{t}: " + "%{y:.1%}<extra></extra>"))
        fig.add_annotation(x=1000, y=float(prob(coef, t, colour, "win", np.array([1000]), avg)[0]), text=t.title(), xanchor="left",
                           showarrow=False, font=dict(color="#989795", size=12))
    y = prob(coef, tc, colour, "win", GAPS, avg)
    fig.add_trace(go.Scatter(x=GAPS, y=y, mode="lines", line=dict(color=GREEN, width=4), hovertemplate=f"{tc}: " + "%{y:.1%}<extra></extra>"))
    fig.add_annotation(x=1000, y=float(prob(coef, tc, colour, "win", np.array([1000]), avg)[0]), text=tc.title(), xanchor="left", showarrow=False, font=dict(color=GREEN, size=13))
    fig.add_hline(y=alpha, line=dict(color="#989795", dash="dash"))
    if gap > 0:
        fig.add_trace(go.Scatter(x=[gap], y=[p_win], mode="markers", marker=dict(size=14, color="#fff", line=dict(color=GREEN, width=3)),
                                 hovertemplate="This game: gap %{x}, %{y:.1%}<extra></extra>"))
    style_figure(fig, 400)
    fig.update_layout(margin=dict(l=55, r=70, t=10, b=50))
    fig.update_xaxes(title="Rating gap (Elo points)", range=[0, 1000])
    fig.update_yaxes(title="Underdog's win chance", tickformat=".0%", rangemode="tozero")
    return fig


def fig_gap_bars(coef, tc, colour, avg, alpha):
    """Bars start at 0 and the axis is fixed at 0 to 1000, the same range as the curve chart, so changing the threshold or the ratings
    visibly lengthens or shortens the bars (an auto-scaled axis would make the longest bar look the same every time)."""
    vals = [critical_gap(coef, t, colour, avg, alpha) for t in TCS]
    base = vals[0]
    shown = [min(v, 1000) if v is not None else 1000 for v in vals]

    def label(v):
        if v is None or v > 1000:
            return "> 1000"
        return f"{v}" if v == base or base is None else f"{v}  (+{v - base})"

    fig = go.Figure(go.Bar(y=[t.title() for t in TCS], x=shown, orientation="h", marker_color=[GREEN if t == tc else GREY for t in TCS],
                           text=[label(v) for v in vals], textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x} Elo points<extra></extra>"))
    style_figure(fig, 400)
    fig.update_layout(bargap=0.45, margin=dict(l=75, r=90, t=10, b=50))
    fig.update_xaxes(title="Critical gap (Elo points)", range=[0, 1000])
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------- callbacks
@callback(Output(P + "alpha_lbl", "children"), Input(P + "alpha", "value"))
def show_alpha(alpha):
    """Shows the slider's value next to its name while it is being dragged."""
    return f"{alpha:.0%}"


@callback(Output(P + "sel", "data"), Input(P + "tc", "value"), Input(P + "alpha", "value"))
def update_store(tc, alpha):
    """Shared choices live in one Store so that every chart reads the same selection."""
    return {"tc": tc, "src": "all", "alpha": alpha}


@callback(Output(P + "l1", "children"), Output(P + "k1", "children"), Output(P + "l2", "children"), Output(P + "k2", "children"),
          Output(P + "l3", "children"), Output(P + "k3", "children"), Output(P + "k3", "className"),
          Output(P + "fig", "figure"), Output(P + "q", "children"), Output(P + "notes", "children"),
          Input(P + "sel", "data"), Input(P + "w", "value"), Input(P + "b", "value"), Input(P + "view", "value"))
def update_page(sel, w, b, view):
    """The rating boxes use debounce=True: the page updates when typing is finished (Enter or leaving the box), not on every key."""
    tc, src, alpha = sel["tc"], sel["src"], sel["alpha"]
    coef = COEF.get(src, COEF["all"])
    w = 2300 if w is None else float(w)
    b = 2300 if b is None else float(b)
    gap, avg = abs(w - b), (w + b) / 2
    colour = "white" if w < b else "black"                    # colour of the underdog (the lower-rated side)
    under = colour.title()
    p_win = float(prob(coef, tc, colour, "win", np.array([gap]), avg)[0])
    cg = critical_gap(coef, tc, colour, avg, alpha)
    if gap == 0:
        verdict, vclass = "No underdog", "kpi-value verdict-off"
    elif p_win < alpha:
        verdict, vclass = "Upset zone", "kpi-value verdict-on"
    else:
        verdict, vclass = "Above threshold", "kpi-value verdict-off"
    chance = f"{p_win:.1%}" + (f"  ·  1 in {round(1 / p_win)}" if p_win > 0 else "")
    fig = fig_curves(coef, tc, colour, avg, alpha, gap, p_win) if view == "curves" else fig_gap_bars(coef, tc, colour, avg, alpha)
    s, l = NOTES[view]
    if view == "curves":
        fact = (f"With a {gap:.0f} Elo gap, the {under.lower()} underdog has a {p_win:.1%} win chance. "
                f"At a {alpha:.0%} threshold, the {tc} critical gap is {cg if cg is not None else 'above 1500'} Elo.")
    else:
        cgs = {t: critical_gap(coef, t, colour, avg, alpha) for t in TCS}
        fact = "At the " + f"{alpha:.0%} threshold and an average rating of {avg:.0f}, the critical gap is " + ", ".join(f"{t} {cgs[t] if cgs[t] is not None else '>1500'}" for t in TCS) + " Elo."
    return (f"{under} underdog: chance of winning", chance, f"Critical gap, {tc}", (f"{cg} Elo" if cg is not None else "> 1500 Elo"),
            f"This game ({under} win)", verdict, vclass, fig, QUESTION[view], notes_children(s, l, fact))
