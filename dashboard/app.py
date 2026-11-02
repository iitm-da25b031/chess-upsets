"""Upset Explorer: five pages in one Dash app (multi-page, Dash "pages").

Run:   python app.py          then open http://127.0.0.1:8050
Needs: pip install dash dash-bootstrap-components plotly pandas numpy       (chess libraries are NOT needed here)
Files: app.py (shell, sidebar)  common.py (colours and helpers)  assets/style.css (all styling)  chess_pieces.py (board artwork)
       pages/*.py (one file per page)   data/dashboard/*  (written by the data-preparation scripts)

How the pieces connect
  - Every file in pages/ calls dash.register_page(...) with its URL, and defines `layout` (what the page shows).
  - Callbacks are written with @callback in the page files. Component ids carry the page prefix (p1_ ... p5_) so that two pages never clash.
  - The sidebar is built here from the page list in common.py; a callback on the URL (dcc.Location) marks the current page.
"""
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from common import PAGES

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, title="Upset Explorer")

app.layout = html.Div([
    dcc.Location(id="url"),
    dbc.Container(dbc.Row([
        dbc.Col(html.Div(id="sidebar"), width=2, className="p-0"),
        dbc.Col(dash.page_container, width=10)], className="g-0"), fluid=True, className="p-0")])


@app.callback(Output("sidebar", "children"), Input("url", "pathname"))
def sidebar(pathname):
    """Brand plus one link per page; the link of the page on screen is highlighted."""
    pathname = pathname or "/"
    return html.Div([html.Div(["Upset", html.Span("Explorer")], className="brand"),
                     *[dcc.Link(name, href=path, className="active" if path == pathname else "") for path, name in PAGES]], className="side")


if __name__ == "__main__":
    app.run(debug=False, port=8050)
