"""Page 4: "The upsets".   Opening explorer, game length and a replay of the rarest upsets.

Reads (data/dashboard/): opening_lines.csv, opening_boards.csv, length_bins.csv, upsets.sqlite   (all written by prep_page4.py)
Three views behind one toggle (the toggle sits in the control card so it is always visible):
    Openings     opening explorer: the board shows a line, the list shows the upset rate of every possible next move (first 4 moves only)
    Game length  length distribution of upsets against all other games
    Replay       the rarest upsets on a chessboard (SQLite query with parameters, read-only connection)
python-chess is NOT used here: board positions were precomputed in prep_page4.py. Piece artwork: see chess_pieces.py.
"""
import base64
import os
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, dash_table, Input, Output, State, ALL, callback, ctx
import dash_bootstrap_components as dbc

from common import DATA, TCS, GREEN, GREY, PAPER, GCFG, seg, kpi, notes_children, style_figure
from chess_pieces import PIECES                       # piece artwork (cburnett set)

dash.register_page(__name__, path="/upsets", name="The upsets", title="Upset Explorer: the upsets", order=3)
P = "p4_"
DB = os.path.join(DATA, "upsets.sqlite")
TOP_MOVES = 8            # continuations drawn per position
MAX_PLIES = 8            # opening explorer depth (4 moves each side)
MIN_EXPECTED = 5         # a line is listed when games x overall upset rate >= 5, so its rate rests on about 5 or more upsets
BOARD_PX = 380
LIGHT, DARK, HILITE = "#ebecd0", "#739552", "#f6f669"

LINES = pd.read_csv(os.path.join(DATA, "opening_lines.csv"))
BOARDS = dict(pd.read_csv(os.path.join(DATA, "opening_boards.csv")).values)
BOARDS[""] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
LEN = pd.read_csv(os.path.join(DATA, "length_bins.csv"))


def pretty(line):
    """'e4 c5 Nf3 d6' -> '1.e4 c5 2.Nf3 d6'"""
    t = line.split()
    return " ".join((f"{i // 2 + 1}." if i % 2 == 0 else "") + m for i, m in enumerate(t))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def overall(tc, colour):
    d = LEN[(LEN.tc_class == tc) & (LEN.underdog_color == colour)]
    return d[d.is_upset].games.sum(), d.games.sum()


def median_plies(d):
    d = d.sort_values("bin")
    cum = d.games.cumsum() / d.games.sum()
    return float(d.loc[cum >= 0.5, "bin"].iloc[0]) + 5          # bin midpoint



def _grid(p):
    cells = []
    for row in p.split("/"):
        for ch in row:
            cells += [""] * int(ch) if ch.isdigit() else [ch]
    return cells


def move_squares(before, after):
    """(from, to) square indices of the move that turns placement `before` into `after`, read from the two positions (no chess rules needed)."""
    a, b = _grid(before), _grid(after)
    ch = [i for i in range(64) if a[i] != b[i]]
    emptied = [i for i in ch if b[i] == ""]
    filled = [i for i in ch if b[i] != ""]
    if len(filled) == 2:                                           # castling: use the king
        to = next(i for i in filled if b[i].upper() == "K")
        return next(i for i in emptied if a[i].upper() == "K"), to
    to = filled[0]
    if len(emptied) == 2:                                          # en passant: the mover started on a different file
        return next(i for i in emptied if i % 8 != to % 8), to
    return emptied[0], to


def arrow_svg(frm, to, cls):
    """chess.com-style arrow between two squares; knight moves are drawn L-shaped."""
    c = lambda i: ((i % 8) * 60 + 30, (i // 8) * 60 + 30)
    (x0, y0), (x1, y1) = c(frm), c(to)
    dx, dy = abs(x1 - x0) / 60, abs(y1 - y0) / 60
    pts = [(x0, y0), (x1, y0), (x1, y1)] if (dx, dy) == (1, 2) else [(x0, y0), (x0, y1), (x1, y1)] if (dx, dy) == (2, 1) else [(x0, y0), (x1, y1)]
    if (dx, dy) == (1, 2):                                         # long leg first
        pts = [(x0, y0), (x0, y1), (x1, y1)]
    elif (dx, dy) == (2, 1):
        pts = [(x0, y0), (x1, y0), (x1, y1)]
    (px, py), (qx, qy) = pts[-2], pts[-1]
    L = max(((qx - px) ** 2 + (qy - py) ** 2) ** .5, 1)
    ux, uy = (qx - px) / L, (qy - py) / L
    head = 26
    bx, by = qx - ux * head, qy - uy * head
    shaft = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts[:-1] + [(bx, by)])
    hw = 17
    tri = f"{qx:.1f},{qy:.1f} {bx - uy * hw:.1f},{by + ux * hw:.1f} {bx + uy * hw:.1f},{by - ux * hw:.1f}"
    return (f'<g><polyline points="{shaft}" fill="none" stroke="#f08c28" stroke-width="13" stroke-linejoin="round"/>'
            f'<polygon points="{tri}" fill="#f08c28"/></g>')


def board_svg(placement, prev=None, arrows=()):
    """Board as one SVG string: squares, coordinates, last-move highlight, pieces, then optional arrows [(from, to, css class)]."""
    cur = _grid(placement)
    old = _grid(prev) if prev else cur
    sq = 60
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 480" width="{BOARD_PX}" height="{BOARD_PX}">']
    for i in range(64):
        r, c = divmod(i, 8)
        out.append(f'<rect x="{c * sq}" y="{r * sq}" width="{sq}" height="{sq}" fill="{LIGHT if (r + c) % 2 == 0 else DARK}"/>')
        if cur[i] != old[i]:                                                    # squares that changed = last move
            out.append(f'<rect x="{c * sq}" y="{r * sq}" width="{sq}" height="{sq}" fill="{HILITE}" fill-opacity="0.55"/>')
    shade = lambda r, c: DARK if (r + c) % 2 == 0 else LIGHT                    # coordinates in the opposite colour of their square
    for i in range(8):
        out.append(f'<text x="4" y="{i * sq + 15}" font-size="13" font-weight="700" font-family="Arial" fill="{shade(i, 0)}">{8 - i}</text>')
        out.append(f'<text x="{i * sq + sq - 5}" y="{8 * sq - 5}" font-size="13" font-weight="700" font-family="Arial" text-anchor="end" fill="{shade(7, i)}">{"abcdefgh"[i]}</text>')
    for i, ch in enumerate(cur):
        if ch:
            r, c = divmod(i, 8)
            out.append(f'<g transform="translate({c * sq},{r * sq}) scale({sq / 45})">{PIECES[("w" if ch.isupper() else "b") + ch.upper()]}</g>')
    out += [arrow_svg(f, t, cls) for f, t, cls in arrows]
    out.append("</svg>")
    return "".join(out)


def board_src(placement, prev=None):
    return "data:image/svg+xml;base64," + base64.b64encode(board_svg(placement, prev).encode()).decode()


def player_bar(name, elo, colour, underdog, score):
    return html.Div([html.Div(name[:1].upper(), className="avatar " + ("av-w" if colour == "white" else "av-b")),
                     html.Span(name, className="pname"), html.Span(f"({elo})", className="pelo"),
                     html.Span("Underdog", className="chip") if underdog else None,
                     html.Span(score, className="pscore")], className="pbar")


TC_OPTS = [{"label": t.title(), "value": t} for t in TCS]
COLOUR_OPTS = [{"label": "White", "value": "white"}, {"label": "Black", "value": "black"}]
VIEW_OPTS = [{"label": "Openings", "value": "open"}, {"label": "Game length", "value": "len"}, {"label": "Replay", "value": "replay"}]

controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([html.Label("Time control"), seg(P + "tc", TC_OPTS, "classical")], width=5),
    dbc.Col([html.Label("Underdog plays"), seg(P + "colour", COLOUR_OPTS, "white")], width=3),
    dbc.Col([html.Label("View"), seg(P + "view", VIEW_OPTS, "open")], width=4),
], className="g-3 align-items-start")), className="mb-3")

btn = lambda i, t, cls="": html.Button(t, id=i, n_clicks=0, className=f"nav-btn {cls}")

explorer_row = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([html.Div(id=P + "crumb", className="crumb"),
                                   html.Div(id=P + "board_open", className="board-wrap"),
                                   html.Div([btn(P + "b_back", "‹ Back", "nav-wide"), btn(P + "b_reset", "Reset", "nav-wide")], className="ctrl-row", style={"width": "380px", "margin": "10px auto 0"})]),
                     className="h-100"), width=5),
    dbc.Col(dbc.Card([dbc.CardHeader("Opening explorer (first 4 moves): hover a move to preview, click to play"),
                      dbc.CardBody([html.Div(id=P + "mv_list"), html.Div(id=P + "notes_o", className="notes mt-3")])], className="h-100"), width=7)],
    className="g-3", id=P + "explorer_row")

chart_row = dbc.Row([
    dbc.Col(dbc.Card([dbc.CardHeader("Game length"), dbc.CardBody([html.Div("Do upsets differ in length from other games?", className="q-title"), dcc.Graph(id=P + "fig_main", config=GCFG)])]), width=8),
    dbc.Col(dbc.Card([dbc.CardHeader("Notes"), dbc.CardBody(id=P + "notes", className="notes")], className="h-100"), width=4)], className="g-3", id=P + "chart_row")

replay_row = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([html.Div(id=P + "bar_top"), html.Div(html.Img(id=P + "board_replay"), className="board-wrap"), html.Div(id=P + "bar_bottom"),
                                   html.Div([btn(P + "b_start", "«"), btn(P + "b_prev", "‹"), btn(P + "b_play", "▶\ufe0e", "nav-play"), btn(P + "b_next", "›"), btn(P + "b_end", "»"),
                                             html.Div(id=P + "move_line", className="move-label", style={"flex": "1"})], className="ctrl-row", style={"width": "380px", "margin": "10px auto 0"}),
                                   html.Div(dcc.Slider(id=P + "ply", min=0, max=1, step=1, value=0, marks=None), style={"width": "380px", "margin": "6px auto 0"})]),
                     className="h-100"), width=5),
    dbc.Col(dbc.Card([dbc.CardHeader("Rarest upsets"),
                      dbc.CardBody([dash_table.DataTable(
                          id=P + "tbl", row_selectable="single", selected_rows=[0], page_action="none", fixed_rows={"headers": True}, style_table={"maxHeight": "250px", "overflowY": "auto"},
                          columns=[{"name": "#", "id": "rank"}, {"name": "Game", "id": "game"}, {"name": "Result", "id": "result"}],
                          style_as_list_view=True,
                          style_header={"backgroundColor": PAPER, "color": "#989795", "border": "none", "fontWeight": "600", "fontSize": "12px"},
                          style_cell_conditional=[{"if": {"column_id": "game"}, "maxWidth": "230px", "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}],
                          style_cell={"backgroundColor": PAPER, "color": "#e5e4e3", "border": "none", "fontSize": "13px", "textAlign": "left", "padding": "6px 8px"},
                          style_data_conditional=[{"if": {"state": "selected"}, "backgroundColor": "#3a3835", "border": "none"}]),
                          html.Div(id=P + "notes_r", className="notes mt-3")])], className="h-100"), width=7)], className="g-3", id=P + "replay_row")


layout = html.Div([
    dbc.Row(dbc.Col(html.H2("The upsets"), width=12), className="mb-2"),
    controls,
    dbc.Row([dbc.Col(kpi(P + "l1", P + "k1"), width=4), dbc.Col(kpi(P + "l2", P + "k2"), width=4), dbc.Col(kpi(P + "l3", P + "k3"), width=4)], className="mb-3 g-3"),
    explorer_row, chart_row, replay_row,
    dcc.Store(id=P + "line", data=""), dcc.Interval(id=P + "auto", interval=700, disabled=True),
], className="p-4")

NOTES = {"open": ("Board: position after the selected line; hovering over a move draws it as an arrow. Bars: upset rate of games continuing with that move, with a 95% Wilson interval. Green: the whole interval lies above the overall rate (dashed line).",
                  "Only the first four moves are covered, and a move is listed only with about five or more expected upsets. Some green bars arise by chance when many moves are compared."),
         "len": ("Share of games by length in 10-ply bins (a ply is one half-move), for upsets and for all other games. Each group sums to 100%.",
                 "Upsets combine wins and draws, which end at different lengths. Length is the number of plies recorded; games over 150 plies are pooled in the last bin."),
         "replay": ("The rarest upsets: the games whose result had the lowest probability under the fitted model. Select a game and step through it with the buttons or the slider; the play button runs the game.",
                    "Rarity depends on the fitted model and on the ratings recorded in the game file. The list is restricted to the selected time control and colour.")}


def move_list(tc, colour, tokens):
    """Returns (rows for the list, arrows for the board). Each row: move, games, mini bar of the upset rate with its 95% interval."""
    k = len(tokens)
    prefix = " ".join(tokens)
    d = LINES[(LINES.depth == k + 1) & (LINES.tc_class == tc) & (LINES.underdog_color == colour)]
    if k:
        d = d[d.line.str.startswith(prefix + " ")]
    k0_, n0_ = overall(tc, colour)
    d = d[d.line.isin(BOARDS) & (d.games * k0_ / n0_ >= MIN_EXPECTED)].sort_values("games", ascending=False).head(TOP_MOVES)
    if len(d) == 0:
        return html.Div("End of the opening explorer (first 4 moves), or too few games to show a rate. Complete games are in the Replay view.", style={"color": "#989795", "padding": "30px 10px"}), []
    k0, n0 = overall(tc, colour)
    base_rate = k0 / n0
    rate = (d.upsets / d.games).to_numpy()
    ci = np.array([wilson(a, b) for a, b in zip(d.upsets, d.games)])
    top = float(max(ci[:, 1].max(), base_rate) * 1.05)
    pct = lambda v: f"{100 * v / top:.2f}%"
    before = BOARDS.get(prefix, BOARDS[""])
    rows, arrows = [], []
    for i, (line, g, u) in enumerate(zip(d.line, d.games, d.upsets)):
        mv_ = line.split()[-1]
        green = ci[i, 0] > base_rate
        frm, to = move_squares(before, BOARDS[line])
        arrows.append((frm, to, f"a{i}"))
        label = f"{k // 2 + 1}{'.' if k % 2 == 0 else '...'}{mv_}"
        track = html.Div([html.Div(className="mv-int", style={"left": pct(ci[i, 0]), "width": f"{100 * (ci[i, 1] - ci[i, 0]) / top:.2f}%"}),
                          html.Div(className="mv-bar", style={"width": pct(rate[i]), "background": GREEN if green else GREY}),
                          html.Div(className="mv-base", style={"left": pct(base_rate)})], className="mv-track")
        rows.append(html.Div([html.Div(label, className="mv-move"), html.Div(f"{g:,}", className="mv-games"), track, html.Div(f"{rate[i]:.1%}", className="mv-rate")],
                             id={"type": "mv", "move": mv_}, n_clicks=0, className=f"mv-row mv-i{i}", title=f"{u:,} upsets in {g:,} games"))
    head = html.Div([html.Div("Move", className="mv-move"), html.Div("Games", className="mv-games"),
                     html.Div("Upset rate, 95% interval (dashed line: overall rate)", className="mv-track", style={"height": "auto"}), html.Div("", className="mv-rate")], className="mv-head")
    return html.Div([head, *rows]), arrows


def fig_len(tc, colour):
    d = LEN[(LEN.tc_class == tc) & (LEN.underdog_color == colour)]
    fig = go.Figure()
    for flag, name, col, w in ((False, "Other games", GREY, 2), (True, "Upsets", GREEN, 3)):
        x = d[d.is_upset == flag].sort_values("bin")
        if x.games.sum() == 0:
            continue
        fig.add_trace(go.Scatter(x=x.bin + 5, y=x.games / x.games.sum(), mode="lines+markers", name=name, line=dict(color=col, width=w), marker=dict(size=7),
                                 hovertemplate=name + ": %{y:.1%} of games<extra></extra>"))
    style_figure(fig, 400)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.08, font=dict(color="#c3c2c1")))
    fig.update_xaxes(title="Game length (plies)", range=[0, 160])
    fig.update_yaxes(title="Share of games", tickformat=".0%", rangemode="tozero")
    return fig


def query(sql, params=()):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)           # read-only connection, parameterised queries only
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def line_counts(tc, colour, tokens):
    if not tokens:
        k0, n0 = overall(tc, colour)
        return k0, n0
    r = LINES[(LINES.depth == len(tokens)) & (LINES.tc_class == tc) & (LINES.underdog_color == colour) & (LINES.line == " ".join(tokens))]
    return (int(r.upsets.iloc[0]), int(r.games.iloc[0])) if len(r) else (0, 0)


@callback(Output(P + "fig_main", "figure"), Output(P + "mv_list", "children"), Output(P + "board_open", "children"), Output(P + "crumb", "children"),
          Output(P + "notes", "children"), Output(P + "notes_o", "children"), Output(P + "notes_r", "children"),
          Output(P + "l1", "children"), Output(P + "k1", "children"), Output(P + "l2", "children"), Output(P + "k2", "children"), Output(P + "l3", "children"), Output(P + "k3", "children"),
          Output(P + "explorer_row", "style"), Output(P + "chart_row", "style"), Output(P + "replay_row", "style"),
          Input(P + "view", "value"), Input(P + "tc", "value"), Input(P + "colour", "value"), Input(P + "line", "data"))
def update(view, tc, colour, line):
    tokens = line.split() if line else []
    k0, n0 = overall(tc, colour)
    base_rate = k0 / n0 if n0 else float("nan")
    if view == "open":
        ku, nu = line_counts(tc, colour, tokens)
        kp = ("Games in this line", f"{nu:,}", "Upsets in this line", f"{ku:,}",
              f"Upset rate (overall {base_rate:.2%})", f"{ku / nu:.2%}" if nu else "n/a")
    else:
        d = LEN[(LEN.tc_class == tc) & (LEN.underdog_color == colour)]
        up, oth = d[d.is_upset], d[~d.is_upset]
        med = f"{median_plies(up) / 2:.0f} / {median_plies(oth) / 2:.0f} moves" if len(up) and len(oth) else "n/a"
        kp = ("Upsets", f"{k0:,}", "Share of games that are upsets", f"{base_rate:.2%}", "Median length: upsets / other games", med)
    ko, no = line_counts(tc, colour, tokens)
    fact_o = f"Overall, {base_rate:.2%} of the {n0:,} games in this class are upsets; the line above covers {no:,} games." if n0 else ""
    d_ = LEN[(LEN.tc_class == tc) & (LEN.underdog_color == colour)]
    u_, o_ = d_[d_.is_upset], d_[~d_.is_upset]
    fact_l = (f"The median upset lasts {median_plies(u_) / 2:.0f} moves, against {median_plies(o_) / 2:.0f} for other games ({len(u_):,} upsets compared)." if len(u_) and len(o_) else "")
    fact_r = f"This class holds {k0:,} upsets ({base_rate:.2%} of its games); the list shows those with the lowest model probability."
    prev = " ".join(tokens[:-1])
    rows, arrows = move_list(tc, colour, tokens)
    board_children = [html.Img(src=board_src(BOARDS.get(line, BOARDS[""]), BOARDS.get(prev, BOARDS[""]) if tokens else None))]
    for f_, t_, cls in arrows:                                   # one transparent layer per candidate move; CSS shows it while its list row is hovered
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 480" width="{BOARD_PX}" height="{BOARD_PX}">{arrow_svg(f_, t_, cls)}</svg>'
        board_children.append(html.Img(src="data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode(), className=f"arrow-img {cls}"))
    crumb = ("Opening line: " + pretty(line)) if tokens else "Opening line: start position"
    show, hide = {}, {"display": "none"}
    return (fig_len(tc, colour), rows, board_children, crumb,
            notes_children(*NOTES["len"], fact_l), notes_children(*NOTES["open"], fact_o), notes_children(*NOTES["replay"], fact_r), *kp,
            show if view == "open" else hide, show if view == "len" else hide, show if view == "replay" else hide)


@callback(Output(P + "line", "data"),
          Input({"type": "mv", "move": ALL}, "n_clicks"), Input(P + "b_back", "n_clicks"), Input(P + "b_reset", "n_clicks"), State(P + "line", "data"))
def explore(_clicks, _b, _r, line):
    tokens = line.split() if line else []
    t = ctx.triggered_id
    if isinstance(t, dict):
        if ctx.triggered[0]["value"] and len(tokens) < MAX_PLIES:           # a real click (rows are re-created with n_clicks = 0)
            tokens.append(t["move"])
    elif t == P + "b_back":
        tokens = tokens[:-1]
    elif t == P + "b_reset":
        tokens = []
    return " ".join(tokens)


@callback(Output(P + "tbl", "data"), Output(P + "tbl", "selected_rows"), Input(P + "tc", "value"), Input(P + "colour", "value"))
def games(tc, colour):
    rows = query("SELECT rank, game_url, white, black, white_elo, black_elo, result FROM upsets "
                 "WHERE tc_class = ? AND underdog_color = ? ORDER BY rank LIMIT 24", (tc, colour))
    data = [dict(rank=r[0], url=r[1], game=f"{r[2]} ({r[4]}) – {r[3]} ({r[5]})", result=r[6]) for r in rows]
    return data, ([0] if data else [])


@callback(Output(P + "ply", "max"), Output(P + "ply", "value"), Output(P + "auto", "disabled"), Output(P + "b_play", "children"),
          Input(P + "tbl", "selected_rows"), Input(P + "b_start", "n_clicks"), Input(P + "b_prev", "n_clicks"), Input(P + "b_next", "n_clicks"), Input(P + "b_end", "n_clicks"),
          Input(P + "b_play", "n_clicks"), Input(P + "auto", "n_intervals"),
          State(P + "tbl", "data"), State(P + "ply", "value"), State(P + "auto", "disabled"))
def navigate(sel, _a, _b, _c, _d, _e, _f, data, ply, stopped):
    PLAY, PAUSE = "▶︎", "❚❚"
    if not data or not sel:
        return 1, 0, True, PLAY
    n = query("SELECT n_plies FROM upsets WHERE game_url = ?", (data[sel[0]]["url"],))[0][0]
    t, ply, stop = ctx.triggered_id, int(ply or 0), True
    if t in (P + "tbl", None) or t == P + "b_end":
        ply = n
    elif t == P + "b_start":
        ply = 0
    elif t == P + "b_prev":
        ply = max(0, ply - 1)
    elif t == P + "b_next":
        ply = min(n, ply + 1)
    elif t == P + "b_play":
        if stopped:                                      # start playing; from the end, start again from the beginning
            ply = 0 if ply >= n else ply
            stop = False
    elif t == P + "auto":
        ply = min(n, ply + 1)
        stop = ply >= n
    return n, ply, stop, (PLAY if stop else PAUSE)


@callback(Output(P + "board_replay", "src"), Output(P + "move_line", "children"), Output(P + "bar_top", "children"), Output(P + "bar_bottom", "children"),
          Input(P + "ply", "value"), Input(P + "tbl", "selected_rows"), State(P + "tbl", "data"))
def board(ply, sel, data):
    if not data or not sel:
        return board_src(BOARDS[""]), "", None, None
    pos, san, w, b, we, be, ud, res = query(
        "SELECT positions, moves_san, white, black, white_elo, black_elo, underdog_color, result FROM upsets WHERE game_url = ?", (data[sel[0]]["url"],))[0]
    pos, san = pos.split("|"), san.split()
    ply = min(int(ply or 0), len(pos) - 1)
    label = "Start" if ply == 0 else f"{(ply + 1) // 2}{'.' if ply % 2 else '...'} {san[ply - 1]}"
    sw, sb = {"1-0": ("1", "0"), "0-1": ("0", "1")}.get(res, ("½", "½"))
    done = ply == len(pos) - 1
    top = player_bar(b, be, "black", ud == "black", sb if done else "")
    bottom = player_bar(w, we, "white", ud == "white", sw if done else "")
    return board_src(pos[ply], pos[ply - 1] if ply else None), label, top, bottom
