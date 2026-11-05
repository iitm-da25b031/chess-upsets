"""Page: "Federations".   Upset rate of the underdog's federation on a world map.

Reads (data/dashboard/): federations.csv (prepare_fresh.py): games, upsets and players per federation of the underdog.
The federation is taken from the FIDE player list through the underdog's FIDE id, which about 80% of the analysis games have.
Federations with fewer than MIN_GAMES underdog games are left blank. England, Scotland and Wales are drawn together as the United Kingdom.
"""
import os
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from common import DATA, GCFG, PAPER, GRID, view_toggle

dash.register_page(__name__, path="/map", name="Federations", title="Upset Explorer: federations", order=5)

MIN_GAMES = 200
LIGHT = "#e5e4e3"
# FIDE federation codes that differ from ISO 3166 alpha-3
ISO = dict(ALG="DZA", ANG="AGO", ANT="ATG", ARU="ABW", BAH="BHS", BAN="BGD", BAR="BRB", BER="BMU", BIZ="BLZ", BOT="BWA", BRN="BHR", BRU="BRN", BUL="BGR",
           BUR="BFA", CAM="KHM", CAY="CYM", CHA="TCD", CHI="CHL", CRC="CRI", CRO="HRV", CUR="CUW", DEN="DNK", ESA="SLV", GEQ="GNQ", GER="DEU", GRE="GRC",
           GRN="GRD", GUA="GTM", GUI="GIN", HAI="HTI", HON="HND", INA="IDN", IOM="IMN", IRI="IRN", ISV="VIR", IVB="VGB", KSA="SAU", KUW="KWT", LAT="LVA",
           LBA="LBY", LES="LSO", MAD="MDG", MAS="MYS", MAW="MWI", MGL="MNG", MNC="MCO", MRI="MUS", MTN="MRT", MYA="MMR", NCA="NIC", NED="NLD", NEP="NPL",
           NGR="NGA", NIG="NER", PAR="PRY", PHI="PHL", PLE="PSE", POR="PRT", PUR="PRI", RSA="ZAF", SEY="SYC", SKN="KNA", SLO="SVN", SOL="SLB", SRI="LKA",
           SUD="SDN", SUI="CHE", TAN="TZA", TGA="TON", TOG="TGO", TPE="TWN", UAE="ARE", URU="URY", VAN="VUT", VIN="VCT", VIE="VNM", ZAM="ZMB", ZIM="ZWE",
           KOS="XKX", FAI="FRO", ENG="GBR", SCO="GBR", WLS="GBR", GCI="GGY", JCI="JEY")
SCALE = [[0.0, "#1d2b3a"], [0.25, "#2f6fd0"], [0.55, "#5fe0b5"], [1.0, "#d6f5a8"]]

_raw = pd.read_csv(os.path.join(DATA, "federations.csv"))
_raw["iso"] = _raw["fed"].map(lambda f: ISO.get(f, f))
_raw = _raw[_raw["fed"] != "FID"]
FED = (_raw.groupby("iso").agg(games=("games", "sum"), upsets=("upsets", "sum"), fed=("fed", lambda s: " + ".join(s))).reset_index())
FED["rate"] = FED["upsets"] / FED["games"]
SHOWN = FED[FED["games"] >= MIN_GAMES]
OVERALL = FED["upsets"].sum() / FED["games"].sum()

layout = html.Div([
    html.H2("Federations"),
    html.P(f"See where underdogs are registered. Compare upset rates or counts. We leave countries blank when they have fewer than {MIN_GAMES} underdog games.", className="lead-text"),
    dbc.Card(dbc.CardBody([html.Label("Colour by"), view_toggle("pf_view", [{"label": "Upset rate", "value": "rate"}, {"label": "Number of upsets", "value": "upsets"}], "rate")]), className="mb-3"),
    dbc.Card(dbc.CardBody([html.Div(dcc.Graph(id="pf_map", config=GCFG, style={"height": "560px"}), className="fig-card"), html.Div(id="pf_cap", className="caps")])),
], className="p-4")


@callback(Output("pf_map", "figure"), Output("pf_cap", "children"), Input("pf_view", "value"))
def show(view):
    d = SHOWN
    if view == "rate":
        z, title, fmt = d["rate"] * 100, "Upsets per 100 underdog games", ".1f"
    else:
        z, title, fmt = d["upsets"], "Upsets", ",d"
    fig = go.Figure(go.Choropleth(
        locations=d["iso"], z=z, colorscale=SCALE, marker_line_color=PAPER, marker_line_width=0.5, colorbar=dict(title=dict(text=title), thickness=14, len=0.7),
        customdata=d[["fed", "games", "upsets", "rate"]].to_numpy(),
        hovertemplate="<b>%{customdata[0]}</b><br>underdog games: %{customdata[1]:,}<br>upsets: %{customdata[2]:,}<br>rate: %{customdata[3]:.2%}<extra></extra>"))
    fig.update_geos(showframe=False, showcoastlines=False, showcountries=True, countrycolor=GRID, bgcolor=PAPER, landcolor="#33322f",
                    projection_type="natural earth", lataxis_range=[-58, 85])
    fig.update_layout(paper_bgcolor=PAPER, plot_bgcolor=PAPER, font=dict(color=LIGHT), margin=dict(l=0, r=0, t=10, b=0), height=560)
    top = d[d["games"] >= 1000].sort_values("rate", ascending=False)
    hi, lo = top.iloc[0], top.iloc[-1]
    cap = [html.P(f"Overall, {OVERALL:.2%} of underdog games are upsets. {len(d)} countries have at least {MIN_GAMES} games."),
           html.Details([html.Summary("How to read this"),
                         html.P(f"Among federations with at least 1,000 games, upset rates range from {lo['rate']:.2%} ({lo['fed']}) to {hi['rate']:.2%} ({hi['fed']})."),
                         html.P("We use each underdog's FIDE federation, not the location of the game. About 80% of games have the FIDE ID we need. Small samples and uneven broadcast coverage affect the rates. The map shows patterns, not their causes.")])]
    return fig, cap
