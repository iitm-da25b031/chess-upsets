"""Descriptive analytics and the effect of cleaning, before model interpretation."""
import dash
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, callback, dcc, html
import dash_bootstrap_components as dbc
from common import GCFG, view_toggle, style_figure
from viewpage import figure, T, WIN_SCALE

dash.register_page(__name__, path="/data", name="The data", title="Upset Explorer: the data", order=1)

VIEWS = {
    "fig_games_per_year": ("Games over time", 430,
        "The number of games changes as broadcast coverage grows. It does not measure how much chess people played."),
    "fig_set_compare": ("Before / after cleaning", 530,
        "Cleaning removes many games with missing ratings. The remaining games include a larger share of titled players."),
    "colour_rating": ("Colour & rating", 450,
        "Underdogs win more often with White. Their win rate also changes with the players' rating level."),
    "fig_outcomes_by_gap": ("Observed outcomes", 510,
        "Underdogs win less often as the rating gap grows. These lines show actual results."),
    "heatmap": ("Win-rate heatmap", 470,
        "Read across to compare rating gaps. Read up to compare rating levels. Brighter cells show more underdog wins."),
}

layout = html.Div([
    html.H2("The data"),
    html.P("Explore the games, compare results and see what cleaning changed.", className="lead-text"),
    dbc.Card(dbc.CardBody(view_toggle("pd_view", [
        {"label": label, "value": key} for key, (label, _, _) in VIEWS.items()
    ], "fig_games_per_year")), className="mb-3"),
    dbc.Card(dbc.CardBody([
        html.Div(id="pd_takeaway", className="q-title"),
        dcc.Graph(id="pd_fig", config=GCFG),
        html.Div(id="pd_cap", className="caps"),
    ])),
], className="p-4")


def fig_colour_rating():
    """Two observed comparisons in classical games, without the duplicate pooled/draw panels."""
    games = T["wdl_by_gap"]
    games = games[games.tc_class == "classical"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=["Underdog's colour", "Players' average rating"], horizontal_spacing=0.12)
    groups = [
        (1, "underdog_color", [("white", "White", "#f0b429", "solid"), ("black", "Black", "#9b7bff", "dash")]),
        (2, "band", [("<2200", "Under 2200", "#cfe0ff", "dot"), ("2200-2499", "2200-2499", "#6f9be8", "dash"), ("2500+", "2500+", "#2f6fd0", "solid")]),
    ]
    for col, dimension, options in groups:
        grouped = games.groupby([dimension, "gap_bin"], as_index=False)[["n", "wins"]].sum()
        for key, label, colour, dash_style in options:
            d = grouped[(grouped[dimension] == key) & (grouped.n >= 100)].sort_values("gap_bin")
            fig.add_trace(go.Scatter(
                x=d.gap_bin + 25, y=d.wins / d.n, name=label, mode="lines+markers",
                line=dict(color=colour, dash=dash_style, width=3), marker=dict(size=5), customdata=d.n,
                hovertemplate=label + ": %{y:.1%}<br>Gap-bin midpoint %{x} Elo · %{customdata:,} games<extra></extra>",
            ), row=1, col=col)
    style_figure(fig, 450)
    for trace in fig.data:
        trace.legend = "legend" if trace.xaxis == "x" else "legend2"
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", x=0, xanchor="left", y=-0.25, yanchor="top"),
                      legend2=dict(orientation="h", x=0.56, xanchor="left", y=-0.25, yanchor="top"),
                      margin=dict(l=65, r=25, t=45, b=115))
    fig.update_xaxes(title="Rating gap (Elo)", range=[0, 660])
    fig.update_yaxes(title="Observed underdog win rate", tickformat=".0%", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=1, col=2)
    return fig


def fig_win_heatmap():
    """Observed win rates with a shared scale and the first observed 10% crossing."""
    h = T["heat"]
    h = h[(h.underdog_color == "all") & (h.elo_lo >= 1200) & (h.elo_lo < 2800)].copy()
    h["rate"] = h.wins / h.n
    gaps = sorted(h.gap_bin.unique())
    ratings = list(range(1200, 2800, 100))
    maximum = float(np.ceil(h.loc[h.n >= 100, "rate"].max() * 10) / 10)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["Classical", "Rapid", "Blitz"])
    for col, tc in enumerate(["classical", "rapid", "blitz"], 1):
        d = h[h.tc_class == tc]
        counts = d.pivot(index="elo_lo", columns="gap_bin", values="n").reindex(index=ratings, columns=gaps).fillna(0)
        rates = d.pivot(index="elo_lo", columns="gap_bin", values="rate").reindex(index=ratings, columns=gaps)
        shown = rates.where(counts >= 100)
        labels = [[[f"{gap}-{gap + 49}" if gap < 600 else "600+", f"{rating}-{rating + 99}", int(counts.loc[rating, gap])]
                   for gap in gaps] for rating in ratings]
        fig.add_trace(go.Heatmap(
            x=np.asarray(gaps) + 25, y=np.asarray(ratings) + 50, z=shown.to_numpy(),
            coloraxis="coloraxis", xgap=1, ygap=1, customdata=labels, hoverongaps=False,
            hovertemplate="Rating gap: %{customdata[0]} Elo<br>Average rating: %{customdata[1]}<br>Underdog wins: %{z:.1%}<br>%{customdata[2]:,} games<extra></extra>",
        ), row=1, col=col)
        # Mark the first reliable cell below 10%, at its lower gap edge.
        # Keep separate segments where a rating row has no observed crossing.
        boundary_x, boundary_y = [], []
        for rating in ratings:
            below = shown.loc[rating].dropna()
            below = below[below < 0.10]
            if len(below):
                gap = float(below.index[0])
                boundary_x.extend([gap, gap])
                boundary_y.extend([rating, rating + 100])
            else:
                boundary_x.append(None)
                boundary_y.append(None)
        for colour, width in [("#21201d", 6), ("#ffffff", 2.5)]:
            fig.add_trace(go.Scatter(
                x=boundary_x, y=boundary_y, mode="lines", line=dict(color=colour, width=width),
                hoverinfo="skip", showlegend=col == 1 and width == 2.5,
                name="First observed win rate below 10%", legendgroup="threshold",
            ), row=1, col=col)
    style_figure(fig, 470)
    fig.update_layout(
        margin=dict(l=75, r=100, t=45, b=100),
        showlegend=True, legend=dict(orientation="h", x=0, y=-0.23, yanchor="top"),
        coloraxis=dict(colorscale=WIN_SCALE, cmin=0, cmax=maximum,
                       colorbar=dict(title=dict(text="Win rate"), tickformat=".0%", thickness=14, len=0.85)),
    )
    fig.update_xaxes(title="Rating gap (Elo)", range=[0, 650],
                     tickvals=[25, 225, 425, 625], ticktext=["0", "200", "400", "600+"])
    fig.update_yaxes(range=[1200, 2800], dtick=200)
    fig.update_yaxes(title="Players' average rating (Elo)", row=1, col=1)
    return fig


def position_panel_legends(fig, view):
    """Put each panel's own key beneath it; remove floating series labels."""
    fig.update_layout(title=None)
    if view == "fig_set_compare":
        labels = ["Games dropped by year", "Time-control mix", "Titled players"]
        for annotation, label in zip(fig.layout.annotations[:3], labels):
            annotation.text = label
        seen = set()
        for trace in fig.data:
            panel = int((trace.xaxis or "x")[1:] or "1")
            trace.legend = "legend" if panel == 1 else f"legend{panel}"
            if trace.type == "bar":
                trace.name = "Missing rating" if "missing Elo" in trace.name else "Other exclusions"
                trace.showlegend = True
            elif trace.type == "scatter" and "markers" in (trace.mode or ""):
                symbol = trace.marker.symbol
                trace.name = "Larger cleaned set" if symbol == "circle-open" else "Analysis set"
                trace.legendgroup = f"panel{panel}-{symbol}"
                # One entry per dataset per panel, including the titled-player panel.
                trace.showlegend = trace.legendgroup not in seen
                seen.add(trace.legendgroup)
        for panel in range(1, 4):
            axis = fig.layout["xaxis" if panel == 1 else f"xaxis{panel}"]
            fig.update_layout(**{("legend" if panel == 1 else f"legend{panel}"): dict(
                x=axis.domain[0], xanchor="left", y=-0.27, yanchor="top", orientation="v",
                font=dict(size=11), traceorder="normal", groupclick="togglegroup")})
        fig.update_layout(showlegend=True, margin=dict(l=65, r=20, t=45, b=145))
    elif view == "fig_outcomes_by_gap":
        # Keep panel titles, replacing direct labels on the lines with local keys.
        fig.layout.annotations = [a for a in fig.layout.annotations if a.xref == "paper"]
        for trace in fig.data:
            if trace.type != "scatter":
                continue
            panel = int((trace.xaxis or "x")[1:] or "1")
            trace.legend = "legend" if panel == 1 else f"legend{panel}"
            trace.showlegend = True
            trace.name = {"underdog loses": "Loss", "draw": "Draw", "underdog wins": "Win"}.get(trace.name, trace.name)
        for panel in range(1, 4):
            axis = fig.layout["xaxis" if panel == 1 else f"xaxis{panel}"]
            fig.update_layout(**{("legend" if panel == 1 else f"legend{panel}"): dict(
                x=axis.domain[0], xanchor="left", y=-0.23, yanchor="top", orientation="h", font=dict(size=11))})
        fig.update_layout(showlegend=True, margin=dict(l=65, r=20, t=45, b=125))
    return fig


@callback(Output("pd_fig", "figure"), Output("pd_takeaway", "children"),
          Output("pd_cap", "children"), Input("pd_view", "value"))
def show(view):
    if view not in VIEWS:
        view = "fig_games_per_year"
    _, height, takeaway = VIEWS[view]
    if view == "heatmap":
        return fig_win_heatmap(), "How rating gap and rating level shape results", [
            html.P(takeaway),
            html.P("The white line marks the first cell below a 10% win rate in each rating row. Blank cells have fewer than 100 games."),
            html.Details([html.Summary("How to read this"), html.P(
                "Each cell groups a 50-point rating gap and a 100-point average rating range. "
                "We combine both underdog colours. Draws count as games the underdog did not win. "
                "The last column combines gaps of 600 Elo or more. We show average ratings from 1200 to 2799. "
                "All panels use the same colour scale. The white line follows observed results, not a fitted model. "
                "Rates can rise again after the first crossing. Cells still have sampling uncertainty, even with 100 games.")]),
        ]
    if view == "colour_rating":
        return fig_colour_rating(), "Observed win rates by colour and rating", [
            html.P(takeaway),
            html.P("Classical games. Each point groups a 50-point rating gap range."),
            html.Details([html.Summary("How to read this"), html.P(
                "The left panel combines rating levels. The right panel combines colours. Rating bands use the two players' average Elo. "
                "Each point needs at least 100 games. The last point combines gaps of 600 Elo or more. "
                "These comparisons show patterns, not causes. The lines have no uncertainty bands.")]),
        ]
    details = {
        "fig_games_per_year": "Bars count games by year and time control. Labels show each year's total. The hatched bar covers a partial year, so compare it with care.",
        "fig_outcomes_by_gap": "Each point groups games in a 50-point rating gap range. The final point combines gaps of 600 Elo or more. The lower bars count games. Their scale differs by time control. Each rate needs at least 100 games. Colours and rating levels are combined, and no uncertainty bands are shown.",
        "fig_set_compare": "The left panel shows which games the analysis filters removed. Other exclusions cover unknown time controls and games shorter than five full moves. The middle and right panels compare the larger cleaned set with the final analysis set. A titled-player game has at least one player with a recorded FIDE title. The comparison starts after the first cleaning stage, not with the raw files.",
    }
    notes = [html.P(takeaway), html.Details([
        html.Summary("How to read this"), html.P(details[view]),
    ])]
    fig = position_panel_legends(figure(view, height), view)
    if view == "fig_games_per_year":
        # Keep totals and the partial-year note, replacing only the category labels.
        fig.layout.annotations = [a for a in fig.layout.annotations
                                  if str(a.text).lower() not in {"classical", "rapid", "blitz"}]
        for trace in fig.data:
            trace.update(showlegend=True, name=trace.name.title())
        fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0, y=-0.22),
                          margin=dict(b=110))
    return fig, VIEWS[view][0], notes
