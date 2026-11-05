"""Shared constants and helpers used by every page of the Upset Explorer."""
import os
import dash_bootstrap_components as dbc
from dash import html

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dashboard")
TCS = ["classical", "rapid", "blitz"]
TC_COL = {"classical": "#2f6fd0", "rapid": "#e8743b", "blitz": "#5fe0b5"}      # same time control, same colour on every page; lightness steps (dark, mid, light) keep them apart in greyscale
GREEN, GREY, PAPER, GRID = "#81b64c", "#6f6e6c", "#2b2a27", "#3a3835"          # one highlight colour, grey for context
GCFG = {"displayModeBar": False, "showTips": False}                              # plotly config used by every graph

PAGES = [("/", "Is it an upset?"), ("/data", "The data"), ("/trust", "Can it be trusted?"), ("/upsets", "The upsets"), ("/map", "Federations")]


def seg(cid, options, value):
    """Button group (radio items). Used for every control so that all pages look and behave alike."""
    return dbc.RadioItems(id=cid, options=options, value=value, class_name="btn-group tc-group", input_class_name="btn-check",
                          label_class_name="btn btn-outline-secondary", label_checked_class_name="active")


def view_toggle(cid, options, value):
    """Smaller button group used to switch between views of one chart."""
    return dbc.RadioItems(id=cid, options=options, value=value, class_name="btn-group view-group", input_class_name="btn-check",
                          label_class_name="btn btn-outline-secondary", label_checked_class_name="active")


def kpi(label_id, value_id, color=None):
    """Number card: a label and one value, nothing else. Both are filled by the page's callback."""
    return dbc.Card(dbc.CardBody([html.P(id=label_id, className="kpi-label"), html.H3(id=value_id, className="kpi-value", style={"color": color} if color else {})]),
                    className="h-100")


def notes_children(shows, limitation, fact=None):
    """Text block of the notes panel: what the chart shows (with one sentence of the figures behind it) and what it cannot show."""
    return [html.H6("How to read it"), html.P(shows)] + ([html.P(fact)] if fact else []) + [html.H6("Keep in mind"), html.P(limitation)]


def notes_card(notes_id):
    return dbc.Card([dbc.CardHeader("About this chart"), dbc.CardBody(id=notes_id, className="notes")], className="h-100")


def style_figure(fig, height, **kw):
    """Dark theme, shared grid colour, no legend unless the page switches it on."""
    fig.update_layout(template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor=PAPER, height=height, showlegend=False,
                      margin=dict(l=55, r=40, t=10, b=50), **kw)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig


def kpi_static(label, value):
    """Number card whose text does not change (no callback)."""
    return dbc.Card(dbc.CardBody([html.P(label, className="kpi-label"), html.H3(value, className="kpi-value")]), className="h-100")
