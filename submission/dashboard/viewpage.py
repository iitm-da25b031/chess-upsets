"""Pages made of one chart per view: a view selector, a figure from the figure module (viz.py) and its short caption."""
import os
import re

import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html

import viz
from common import DATA, GCFG, view_toggle

ROOT = os.path.dirname(DATA)
PAPER, GRID, LIGHT, GREEN = "#2b2a27", "#3a3835", "#e5e4e3", "#81b64c"
TC_COL = {"classical": "#2f6fd0", "rapid": "#e8743b", "blitz": "#5fe0b5"}                    # the time-control colours of the other pages
viz.TC_COL = viz.TC_GREY = viz.TC_LINE = TC_COL
viz.BAND_COL = {"<2200": "#cfe0ff", "2200-2499": "#6f9be8", "2500+": "#2f6fd0"}
viz.PURPLES = [[0.0, "#33322f"], [0.25, "#3f5a2f"], [0.5, "#5f8a3a"], [0.75, "#81b64c"], [1.0, "#d6f5a8"]]   # win-rate scale: dark (low) to green (high)
for _key, _col in (("white", "#f0b429"), ("black", "#9b7bff")):                       # underdog colour: amber for White, violet for Black
    viz.UC_STYLE[_key]["color"] = _col
for _i, _key in enumerate(viz.MODEL_STYLE):                                         # main model and interaction model
    viz.MODEL_STYLE[_key]["color"] = ("#2f6fd0", "#e8743b")[_i % 2]
viz.HI = GREEN                                       # the highlight colour of the figure module is the dashboard green
T = viz.load2(os.path.join(ROOT, "viz_data"), os.path.join(ROOT, "model_outputs"))
CAPS = viz.captions_all(T)
TOP_K = 20
_NAMES = [(re.compile(r"\s*\((?:see )?(?:fig|table)_\w+\)"), ""), (re.compile(r"[;,] (?:see )?(?:fig|table)_\w+"), ""),
          (re.compile(r"[;,]?\s*full links are in \w+\.csv\.?"), "."), (re.compile(r"(?:fig|table)_\w+"), "the matching view")]


def clean(text):
    """Captions come from the figure module; references to its function and file names are removed for the dashboard."""
    for pattern, repl in _NAMES:
        text = pattern.sub(repl, text)
    return text


_HEX, _RGB = re.compile(r"^#[0-9a-f]{6}$"), re.compile(r"^rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)$")


def _dark_colour(c):
    """Ink and white fills of the light figures become light ink and dark paper; saturated colours stay."""
    s = c.strip().lower()
    if s == "white":
        return PAPER
    if s == "black":
        return LIGHT
    m = _RGB.match(s)
    if _HEX.match(s):
        r, g, b, a = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16), None
    elif m:
        r, g, b, a = int(m[1]), int(m[2]), int(m[3]), m[4]
    else:
        return c
    lum, sat = (0.299 * r + 0.587 * g + 0.114 * b) / 255, (max(r, g, b) - min(r, g, b)) / max(r, g, b, 1)
    if sat < 0.35 and lum < 0.4:
        new = (229, 228, 227)
    elif sat < 0.15 and lum > 0.85:
        new = (43, 42, 39)
    else:
        return c
    return f"rgba({new[0]},{new[1]},{new[2]},{a})" if a is not None else "#%02x%02x%02x" % new


def _recolour(o):
    if isinstance(o, dict):
        return {k: _recolour(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_recolour(v) for v in o]
    return _dark_colour(o) if isinstance(o, str) else o


TINT = {"fig_upset_points", "fig_calibration_check"}                  # figures with one panel per time control: marker colour = time-control colour


def _is_neutral(c):
    m = _RGB.match(c.strip().lower())
    if _HEX.match(c.strip().lower()):
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    elif m:
        r, g, b = int(m[1]), int(m[2]), int(m[3])
    else:
        return False
    return (max(r, g, b) - min(r, g, b)) / max(r, g, b, 1) < 0.2


def tint(fig):
    for t in fig.data:
        if t.type not in ("scatter", "scattergl") or "markers" not in (t.mode or ""):
            continue
        col = t.marker.color if isinstance(t.marker.color, str) else (t.marker.line.color if t.marker.line is not None else None)
        if not isinstance(col, str) or not _is_neutral(col):
            continue
        axis = t.xaxis or "x"
        c = list(TC_COL.values())[(int(axis[1:] or 1) - 1) % 3]
        t.update(marker=dict(color=c, line=dict(color=c)))
        if t.error_y is not None and t.error_y.visible:
            t.update(error_y=dict(color=c))
    return fig


def semantic(fig, name):
    """Series that the figure module draws in grey get a colour of their own: outcomes, kept and dropped games."""
    if name == "fig_outcomes_by_gap":
        pal = {"loses": "#e8743b", "draw": "#6f9be8", "wins": GREEN}
        for t in fig.data:
            key = next((k for k in pal if k in (t.name or "")), None)
            if key and t.type == "scatter":
                t.update(line=dict(color=pal[key]), marker=dict(color=pal[key]), textfont=dict(color=pal[key]))
    if name == "fig_set_compare":
        for t in fig.data:
            if t.type == "bar":
                c = "#f0b429" if "time control" in (t.name or "") else "#9b7bff"      # two reasons for dropping a game: amber and violet
                t.update(marker=dict(color=c, line=dict(color=c)))
    return fig


TC_LIST = list(TC_COL.values())
WIN_SCALE = [[0.0, "#26354a"], [0.3, "#2f6fd0"], [0.65, "#5fe0b5"], [1.0, "#d6f5a8"]]    # win rate: dark blue (rare) to mint (common)
TEAL_SCALE = [[0.0, "#1d2b3a"], [0.25, "#214e73"], [0.5, "#2f6fd0"], [0.75, "#5fe0b5"], [1.0, "#d9fff0"]]   # game counts: dark blue to mint


def _axis_index(t):
    axis = t.xaxis or "x"
    return int(axis[1:] or 1) - 1


def colourise(fig, name):
    """Last pass: panels, bars and lines the figure module draws in grey get chess.com colours."""
    if name == "fig_gap_distribution":
        for t in fig.data:
            if t.type == "bar":
                c = TC_LIST[_axis_index(t) % 3]
                t.update(marker=dict(color=c, line=dict(color=PAPER, width=1)))
    if name == "fig_simpson":
        col = {"white": "#f0b429", "black": "#9b7bff"}
        for t in fig.data:
            if t.type != "scatter" or not t.line or not t.line.color:
                continue
            if t.line.width in (6,):                              # wide casing under the pooled line, drawn in the paper colour
                t.update(line=dict(color=GREEN))
            elif t.line.width == 2.5 and t.line.dash is None:     # pooled curve (also shown in the other panels)
                t.update(line=dict(color=GREEN))
            elif t.line.color == "#e5e4e3" and t.line.dash == "solid":
                t.update(line=dict(color=col["white"]), marker=dict(color=col["white"]))
            elif t.line.color == "#7a7977":
                t.update(line=dict(color=col["black"]), marker=dict(color=col["black"]))
        for a in fig.layout.annotations:
            txt = str(a.text)
            if "has White" in txt:
                a.update(font=dict(color=col["white"]), arrowcolor=col["white"])
            elif "has Black" in txt:
                a.update(font=dict(color=col["black"]), arrowcolor=col["black"])
    if name == "fig_heatmap":
        for t in fig.data:
            if t.type == "heatmap" and t.colorbar is not None and "Games" in (t.colorbar.title.text or ""):
                t.update(colorscale=TEAL_SCALE)
    if name == "fig_set_compare":
        for t in fig.data:
            if t.type == "scatter" and "markers" in (t.mode or ""):
                c = "#e8743b" if t.marker.symbol == "circle" else "#2f6fd0"      # filled = analysis set, open = larger cleaned set
                t.update(marker=dict(color=c, line=dict(color=c)))
            elif t.type == "scatter":
                t.update(line=dict(color="#6f9be8"))
    if name == "fig_upset_rule":
        for t in fig.data:
            if t.type == "bar":
                t.update(marker=dict(color="#2f6fd0", line=dict(color=PAPER, width=1)))
            elif t.type == "scatter" and t.line and t.line.color == "#e5e4e3":
                t.update(line=dict(color="#5fe0b5"))
    if name == "fig_critical_gap_alpha":
        fig.data = [t for t in fig.data if not (t.type == "scatter" and t.line and t.line.width == 1 and t.line.color == "#d6d5d3")]
        fig.layout.annotations = [a for a in fig.layout.annotations if not str(a.text).startswith("data end")]
        for t in fig.data:
            if t.type == "scatter" and t.line and t.line.color == "#7a7977":
                t.update(line=dict(color="#e5e4e3"))
            if t.type == "scatter" and t.marker and t.marker.color == "#7a7977":
                t.update(marker=dict(color="#e5e4e3"))
    if name == "fig_model_misfit":
        for t in fig.data:
            if t.type == "scatter" and t.marker is not None and t.marker.color == "#e5e4e3":
                t.update(marker=dict(color="#5fe0b5"))

    if name == "fig_heatmap":
        fig.update_layout(coloraxis=dict(colorscale=WIN_SCALE), coloraxis2=dict(colorscale=TEAL_SCALE))
    if name == "fig_outcomes_by_gap":
        for t in fig.data:
            if t.type == "bar":
                t.update(marker=dict(color=TC_LIST[_axis_index(t) % 3], line=dict(color=PAPER, width=1)))
    if name in ("fig_simpson", "fig_outcomes_by_gap", "fig_critical_gap_alpha", "fig_model_vs_data"):
        for a in fig.layout.annotations:                                  # text labels sit on a paper-coloured plate so lines do not run through them
            if a.showarrow is not True and a.text and "<b>" not in str(a.text) and a.xref != "paper" and a.textangle is None:
                a.update(bgcolor="rgba(43,42,39,0.85)", borderpad=2)
    if name == "fig_critical_gap_alpha":
        fig.layout.annotations = [a for a in fig.layout.annotations if str(a.text) not in ("blitz", "rapid", "classical")]
        for tc, col in TC_COL.items():
            fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=col, width=3), name=tc, showlegend=True)
    if name == "fig_upset_rule":
        fig.update_layout(legend=dict(y=-0.22), margin=dict(b=150))
    if name == "fig_calibration_check":
        for a in fig.layout.annotations:
            if "perfect calibration" in str(a.text):
                a.update(yshift=-22, xshift=-6)
    return fig


def dark(fig, name=None):
    """Dashboard look for a figure of the figure module: dark paper, light text, shared grid colour."""
    fig = go.Figure(_recolour(fig.to_dict()))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PAPER, plot_bgcolor=PAPER, font=dict(color=LIGHT), autosize=True, width=None)
    fig.update_xaxes(gridcolor=GRID, linecolor="#4a4845", zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor="#4a4845", zerolinecolor=GRID)
    fig = colourise(semantic(fig, name), name)
    return tint(fig) if name in TINT else fig


def figure(name, height):
    fig, _ = getattr(viz, name)(T)
    fig.update_layout(height=height)
    return dark(fig, name)


def upset_scatter(height):
    """The upset scatter without the table panel of the figure; the table is listed under the caption instead."""
    fig, _ = viz.fig_upset_points(T)
    fig.data = [t for t in fig.data if t.type != "table"]
    fig.update_yaxes(domain=[0, 1])
    fig.update_layout(height=height, margin=dict(b=120), legend=dict(y=-0.2))
    return dark(fig, "fig_upset_points")


def rarest_table():
    r = T["rarest_upsets"].sort_values("rank_q").head(TOP_K)
    head = ["#", "Underdog (Elo)", "Favourite (Elo)", "Gap", "Result", "Time control", "Event", "Year", "Model chance"]
    rows = [html.Tr([html.Td(int(x.rank_q)), html.Td(f"{x.underdog} ({int(x.underdog_elo)})"), html.Td(f"{x.favourite} ({int(x.favourite_elo)})"),
                     html.Td(int(x.gap)), html.Td("win" if x.underdog_score == 1 else "draw"), html.Td(x.tc_class), html.Td(x.broadcast_name),
                     html.Td(int(x.year)), html.Td(f"{x.q_emp:.1%}")]) for x in r.itertuples()]
    return html.Details([html.Summary(f"The {TOP_K} rarest upsets (numbered in the plot)"),
                         html.Table([html.Thead(html.Tr([html.Th(h) for h in head])), html.Tbody(rows)], className="mini")])


def build(prefix, title, intro, views):
    """Layout of a page with one view per figure; views = [(figure function name, button label, height in px)]."""
    heights = {k: h for k, _, h in views}
    layout = html.Div([
        html.H2(title), html.P(intro, className="lead-text"),
        dbc.Card(dbc.CardBody([html.Label("View"), view_toggle(prefix + "view", [{"label": lab, "value": k} for k, lab, _ in views], views[0][0])]), className="mb-3"),
        dbc.Card(dbc.CardBody([html.Div(dcc.Graph(id=prefix + "fig", config=GCFG), className="fig-card"), html.Div(id=prefix + "cap", className="caps")]))],
        className="p-4")

    @callback(Output(prefix + "fig", "figure"), Output(prefix + "fig", "style"), Output(prefix + "cap", "children"), Input(prefix + "view", "value"))
    def show(view):
        if view not in heights:
            view = views[0][0]
        cap = CAPS[view]
        body = [html.P(clean(cap["short"])), html.Details([html.Summary("Full caption"), html.P(clean(cap["details"]))])]
        style = {"height": f"{heights[view]}px"}
        if view == "fig_upset_points":
            return upset_scatter(heights[view]), style, body + [rarest_table()]
        return figure(view, heights[view]), style, body

    return layout
