"""Figures and tables for the results section (submission version, fresh pipeline outputs).

Plain module (pandas + plotly only). Tables come from make_viz_data.py (viz_data/ next to this file) and from the
small model outputs of notebook 03 (../reports/model_outputs, or the folder in CHESS_MODEL_OUTPUTS):

    import viz                                     # with submission/viz on sys.path
    T = viz.load2()                                # all tables (load() = only the data-layer tables)
    fig, caption = viz.fig_gap_distribution(T)
    df, markdown = viz.table_robustness(T)

Part 1 (fig_gap_distribution ... fig_games_per_year) shows the observed data; part 2 (fig_model_vs_data ...
fig_model_misfit, table_*) shows the model and the upset rule against the data.
Every fig_* function returns (fig, caption); the caption is markdown and all of its numbers (and those in the
titles) are computed from the tables. Conventions follow chess_upsets.ipynb: time-control order
classical / rapid / blitz, TC_COL palette, HI = the point of the plot, GREY = context,
LIGHT = background marks, style() puts the title top-left and the legend in a row below.
Cells / bins with fewer than MIN_N games are dropped (lines) or hatched (heatmap cells).
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TC = ["classical", "rapid", "blitz"]
TC_COL = {"classical": "#1f4fa8", "rapid": "#e8743b", "blitz": "#2fb58a"}
TC_DASH = {"classical": "solid", "rapid": "dash", "blitz": "dot"}
HI, GREY, LIGHT = "#5b8c2a", "#7a7977", "#d6d5d3"
INK, INK2 = "#2b2b2b", "#555453"            # text / dark marks
MIN_N = 100                                  # minimum games per bin or cell before a rate is shown
BANDS = ["<2200", "2200-2499", "2500+"]
GAP_TICKS = dict(tickvals=[0, 100, 200, 300, 400, 500, 625], ticktext=["0", "100", "200", "300", "400", "500", "600+"])
GAP_TITLE = "Rating gap (Elo points)"
TABLES = ["gap_hist", "gap_summary", "wdl_by_gap", "heat", "upset_points", "rarest_upsets", "set_compare",
          "drop_steps", "missing_elo_by_year", "yearly_counts", "monthly_counts", "near_equal"]


HERE = os.path.dirname(os.path.abspath(__file__))


def load(data_dir=os.path.join(HERE, "viz_data")):
    """Read every table written by make_viz_data.py into a dict of DataFrames."""
    return {t: pd.read_csv(os.path.join(data_dir, t + ".csv")) for t in TABLES}


def style(fig, h=420, legend_y=-0.2, **kw):
    """Same layout rule as the notebook: title top-left, legend in a row under the plot, fixed margins.
    legend_y: legend top in paper coordinates of the plot area (negative = below it)."""
    title = kw.pop("title", None)
    leg = {k: v for k, v in (kw.pop("legend", None) or {}).items() if k not in ("x", "y", "xanchor", "yanchor", "orientation")}
    n_leg = sum(1 for t in fig.data if getattr(t, "showlegend", False) is not False and getattr(t, "name", None))
    margin = dict(l=75, r=30, t=95, b=100 if n_leg > 1 else 70)
    margin.update(kw.pop("margin", {}))
    fig.update_layout(template="simple_white", height=h + 40, font=dict(size=13), margin=margin, **kw)
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=15)))
    fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=legend_y, yanchor="top", **leg))
    return fig


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _gap_x(gap_bin):
    """x position of a 50-point gap bin: its centre; the pooled 600+ bin sits at 625."""
    return np.asarray(gap_bin, dtype=float) + 25


def _rates(df, by):
    g = df.groupby(by, as_index=False)[["n", "wins", "draws"] + (["losses"] if "losses" in df else [])].sum()
    g["win"] = g.wins / g.n
    g["draw"] = g.draws / g.n
    g["score"] = (g.wins + 0.5 * g.draws) / g.n
    if "losses" in g:
        g["loss"] = g.losses / g.n
    return g


# ---------------------------------------------------------------------------------------------------- 1
def fig_gap_distribution(T, bin_width=25, max_gap=900):
    """Question: how far apart in rating are the two players in a typical game, per time control?
    Shows the share of each class's underdog games in each gap bin (same bins, same y scale, so shapes compare
    directly; absolute counts differ ~6x between classical and blitz and are given in the panel titles).
    Hides: who the players are, and the far tail (>= 600 points holds 1-3% of games and is barely visible on a
    linear axis; it is quantified in the caption rather than shown with a log axis).
    Bin width: 25 Elo points = half the 50-point bins used for the outcome curves; even the sparsest class (blitz,
    ~107k games) has hundreds of games in every 25-point bin below 600, so the shape is not noise, and 25 resolves
    the steep rise near gap 0 that 50-point bins flatten. Pass bin_width (a multiple of 25) to rebin."""
    gh, gs = T["gap_hist"].copy(), T["gap_summary"].set_index("tc_class")
    assert bin_width % 25 == 0
    cap = max_gap
    gh["lo"] = np.minimum(gh.gap_lo, cap) // bin_width * bin_width
    gh.loc[gh.gap_lo >= cap, "lo"] = cap
    h = gh.groupby(["tc_class", "lo"], as_index=False).n.sum()
    h["share"] = h.n / h.groupby("tc_class").n.transform("sum")
    ymax = h.share.max() * 100 * 1.12
    fig = make_subplots(1, 3, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"<b>{tc}</b>  (n = {int(gs.loc[tc, 'n']):,} games)" for tc in TC])
    for i, tc in enumerate(TC, 1):
        d = h[h.tc_class == tc]
        is_tail = d.lo >= cap
        fig.add_bar(x=d.lo + bin_width / 2, y=d.share * 100, width=bin_width * 0.92, row=1, col=i, showlegend=False,
                    marker=dict(color=np.where(is_tail, "#e6e5e3", "#b9b8b5"), line_width=0),   # light bars so the HI median line stands out in greyscale
                    customdata=np.c_[d.lo, d.lo + bin_width, d.n],
                    hovertemplate="gap %{customdata[0]:.0f}-%{customdata[1]:.0f}: %{y:.2f}% (%{customdata[2]:,} games)<extra></extra>")
        med = gs.loc[tc, "median"]
        fig.add_scatter(x=[med, med], y=[0, ymax * 0.93], mode="lines", line=dict(color="white", width=6), showlegend=False,
                        hoverinfo="skip", row=1, col=i)
        fig.add_scatter(x=[med, med], y=[0, ymax * 0.93], mode="lines", line=dict(color=HI, width=3), showlegend=False,
                        hoverinfo="skip", row=1, col=i)
        fig.add_annotation(x=med, y=ymax * 0.97, text=f"median {med:.0f}", showarrow=False, xanchor="left", xshift=4,
                           font=dict(color=HI, size=13), row=1, col=i)
        fig.add_annotation(x=cap + bin_width / 2, y=d.loc[is_tail, "share"].sum() * 100, text=f"{cap}+", showarrow=False,
                           yanchor="bottom", font=dict(size=11, color=INK2), row=1, col=i)
    fig.update_xaxes(title_text=GAP_TITLE, range=[0, cap + bin_width + 10], tickvals=list(range(0, cap + 1, 200)))
    fig.update_yaxes(title_text=f"Games (% of class, per {bin_width}-pt bin)", range=[0, ymax], col=1)
    fig.update_yaxes(ticksuffix="%")
    lt200 = gh[gh.gap_lo < 200].n.sum() / gh.n.sum()
    wider = all(gs.loc[t, "median"] > gs.loc["classical", "median"] for t in ("rapid", "blitz"))
    style(fig, 380, title=f"{lt200:.0%} of games pair players less than 200 points apart"
                          + ("; rapid and blitz pairings are wider than classical" if wider else ""))
    a = gs.loc["all"]
    sparse = h[(h.lo < 600)].groupby("tc_class").n.min()
    cap_txt = (f"Rating gap |Elo(white) - Elo(black)| for the {int(a.n):,} analysis games that have an underdog, in {bin_width}-point bins "
               f"(bar height = % of that time control's games; the last bar pools gaps of {cap}+). "
               f"Median gap: classical {gs.loc['classical', 'median']:.0f}, rapid {gs.loc['rapid', 'median']:.0f}, blitz {gs.loc['blitz', 'median']:.0f} points; "
               f"{_pct(a.share_ge_400)} of all games have a gap of 400+ and {_pct(a.share_ge_600)} of 600+ "
               f"(classical {_pct(gs.loc['classical', 'share_ge_400'])}, rapid {_pct(gs.loc['rapid', 'share_ge_400'])}, blitz {_pct(gs.loc['blitz', 'share_ge_400'])} at 400+). "
               f"Limitation: the large-gap tail, where the most surprising upsets live, is too thin to see on this linear scale "
               f"(the sparsest {bin_width}-point bin below 600 still has {int(sparse.min()):,} games); the gaps are between published FIDE ratings, "
               f"which lag a player's current strength, especially for fast-improving juniors.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- 2
def fig_outcomes_by_gap(T, min_n=MIN_N):
    """Question: as the rating gap grows, how often does the underdog win, draw and lose, per time control?
    Shows observed rates (three lines on one 0-100% baseline) per 50-point gap bin, pooled over the underdog's colour
    and the rating band, with the number of games per bin as a strip under each panel.
    Hides: the colour and rating-level effects (see fig_simpson) and sampling noise (no intervals; bins under
    min_n games are dropped and drawn as light bars in the count strip)."""
    w = _rates(T["wdl_by_gap"], ["tc_class", "gap_bin"])
    fig = make_subplots(2, 3, shared_xaxes=True, shared_yaxes=False, row_heights=[0.78, 0.22], vertical_spacing=0.05,
                        horizontal_spacing=0.03, subplot_titles=[f"<b>{tc}</b>" for tc in TC] + [""] * 3)
    spec = [("loss", "underdog loses", INK2, "dot"), ("draw", "draw", GREY, "dash"), ("win", "underdog wins", HI, "solid")]
    for i, tc in enumerate(TC, 1):
        d = w[w.tc_class == tc].sort_values("gap_bin")
        ok = d[d.n >= min_n]
        for key, lab, col, dash in spec:
            fig.add_scatter(x=_gap_x(ok.gap_bin), y=ok[key] * 100, mode="lines+markers", name=lab, showlegend=False,
                            line=dict(color=col, dash=dash, width=2.5 if key == "win" else 2), marker=dict(size=5, color=col),
                            customdata=ok.n, hovertemplate=f"{lab}: %{{y:.1f}}% (n=%{{customdata:,}})<extra>{tc}</extra>", row=1, col=i)
        # direct labels at the 200-249 bin: the higher of win/draw labelled above its line, the lower one below
        at = ok[ok.gap_bin == 200].iloc[0]
        upper = "win" if at.win >= at.draw else "draw"
        for key, lab, col, dash in spec:
            if key == "loss":
                fig.add_annotation(x=_gap_x([at.gap_bin])[0], y=at[key] * 100, text=lab, showarrow=False, xanchor="left", yanchor="top",
                                   xshift=6, yshift=-6, font=dict(color=INK2, size=12), row=1, col=i)
            else:
                above = key == upper
                fig.add_annotation(x=_gap_x([at.gap_bin])[0], y=at[key] * 100, text=lab, showarrow=False,
                                   xanchor="left" if above else "right", yanchor="bottom" if above else "top",
                                   xshift=4 if above else -4, yshift=5 if above else -5,
                                   font=dict(color=col if key == "win" else INK2, size=12), row=1, col=i)
        fig.add_bar(x=_gap_x(d.gap_bin), y=d.n / 1000, width=44, showlegend=False, row=2, col=i,
                    marker=dict(color=np.where(d.n >= min_n, GREY, LIGHT), line_width=0),
                    customdata=d.n, hovertemplate="%{customdata:,} games<extra></extra>")
    fig.update_yaxes(range=[0, 100], ticksuffix="%", row=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(rangemode="tozero", nticks=3, row=2)      # own count scale per panel (classical has ~5x the games)
    fig.update_yaxes(title_text="Share of the underdog's games (%)", row=1, col=1)
    fig.update_yaxes(title_text="Games<br>(thousands)", row=2, col=1)
    fig.update_xaxes(range=[0, 660], **GAP_TICKS)
    fig.update_xaxes(title_text=GAP_TITLE, row=2)
    style(fig, 480, title="The underdog's win rate falls steadily with the gap; in classical the draw takes up much of the slack")
    sm = {tc: w[(w.tc_class == tc)].set_index("gap_bin") for tc in TC}
    dropped = int((w.n < min_n).sum())
    cap_txt = (f"Observed outcome rates of the lower-rated player per 50-point gap bin (the last bin pools 600+), all {int(w.n.sum()):,} games with an underdog, "
               f"pooled over the underdog's colour and the rating band; the grey strip counts the games per bin (its scale differs per panel: classical has "
               f"{w[w.tc_class == 'classical'].n.sum() / w[w.tc_class == 'blitz'].n.sum():.1f}x the games of blitz). "
               f"Underdog win rate in the 0-49 bin vs the 400-449 bin: classical {_pct(sm['classical'].loc[0, 'win'])} vs {_pct(sm['classical'].loc[400, 'win'])}, "
               f"rapid {_pct(sm['rapid'].loc[0, 'win'])} vs {_pct(sm['rapid'].loc[400, 'win'])}, blitz {_pct(sm['blitz'].loc[0, 'win'])} vs {_pct(sm['blitz'].loc[400, 'win'])}; "
               f"the classical draw rate is {_pct(sm['classical'].loc[0, 'draw'])} in the 0-49 bin. "
               f"Limitation: pooling mixes rating levels and colours (see the Simpson figure); "
               + (f"{dropped} bins with fewer than {min_n} games are not drawn, " if dropped else f"every bin has at least {min_n} games, ")
               + "and no confidence intervals are shown (n per bin is in the strip).")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- 3
PURPLES = [[0.0, "#efedf5"], [0.25, "#bcbddc"], [0.5, "#9e9ac8"], [0.75, "#6a51a3"], [1.0, "#3f007d"]]


def fig_heatmap(T, min_n=MIN_N, color="all", threshold=0.10, elo_range=(1200, 2800)):
    """Question: is the underdog's chance the same at every rating level, or does it depend on how strong both players are?
    Shows the observed underdog win rate in each (gap, average rating) cell, one panel per time control, with a
    companion count map below; cells with fewer than min_n games are hatched; the HI step line marks, in each rating
    row, the first gap bin whose observed win rate is below `threshold`.
    Hides: draws (a draw counts as not-a-win here), the colour effect (pooled unless color='white'/'black'), and noise
    (a cell of 100 games has a standard error of about 3 percentage points at a 10% rate). Rows outside elo_range
    are not shown (they hold few games)."""
    h = T["heat"]
    h = h[(h.underdog_color == color) & (h.elo_lo >= elo_range[0]) & (h.elo_lo < elo_range[1])].copy()
    h["rate"] = h.wins / h.n
    xs = sorted(h.gap_bin.unique())
    ys = list(range(elo_range[0], elo_range[1], 100))
    zmax = float(np.ceil(h.loc[h.n >= min_n, "rate"].max() * 20) / 20)
    fig = make_subplots(2, 3, shared_xaxes=True, shared_yaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.09,
                        horizontal_spacing=0.025,
                        subplot_titles=[f"<b>{tc}</b>: underdog win rate" for tc in TC] + [f"{tc}: games per cell" for tc in TC])
    cross = {}
    for i, tc in enumerate(TC, 1):
        d = h[h.tc_class == tc]
        n = d.pivot(index="elo_lo", columns="gap_bin", values="n").reindex(index=ys, columns=xs).fillna(0)
        r = d.pivot(index="elo_lo", columns="gap_bin", values="rate").reindex(index=ys, columns=xs)
        rz = r.where(n >= min_n)
        X, Y = _gap_x(xs), np.array(ys) + 50
        fig.add_heatmap(x=X, y=Y, z=rz.values * 100, coloraxis="coloraxis", xgap=1, ygap=1, row=1, col=i,
                        customdata=n.values, hovertemplate="gap bin %{x}, avg Elo %{y}: %{z:.1f}% win (n=%{customdata:,.0f})<extra></extra>")
        lo = (n > 0) & (n < min_n)
        yy, xx = np.where(lo.values)
        fig.add_scatter(x=X[xx], y=Y[yy], mode="markers", marker=dict(symbol="line-ne", size=11, line=dict(color="#a9a8a6", width=1.2)),
                        hoverinfo="skip", showlegend=(i == 1), name=f"fewer than {min_n} games (rate not shown)", row=1, col=i)
        # first reliable bin per row whose win rate is below the threshold -> step line at its lower edge
        px, py = [], []
        cross[tc] = {}
        for j, y0 in enumerate(ys):
            row_ok = rz.loc[y0].dropna()
            below = row_ok[row_ok < threshold]
            if len(below) == 0:
                if px:
                    px.append(None); py.append(None)
                continue
            xe = float(below.index[0])
            cross[tc][y0] = xe
            px += [xe, xe]; py += [y0, y0 + 100]
        fig.add_scatter(x=px, y=py, mode="lines", line=dict(color="white", width=6), hoverinfo="skip", showlegend=False, row=1, col=i)
        fig.add_scatter(x=px, y=py, mode="lines", line=dict(color=HI, width=3), hoverinfo="skip", row=1, col=i,
                        showlegend=(i == 1), name=f"first gap bin with an observed win rate below {threshold:.0%}")
        nz = np.log10(n.where(n > 0))
        fig.add_heatmap(x=X, y=Y, z=nz.values, coloraxis="coloraxis2", xgap=1, ygap=1, row=2, col=i, customdata=n.values,
                        hovertemplate="gap bin %{x}, avg Elo %{y}: %{customdata:,.0f} games<extra></extra>")
    fig.update_layout(
        coloraxis=dict(colorscale=PURPLES, cmin=0, cmax=zmax * 100,
                       colorbar=dict(title=dict(text="Underdog<br>win rate (%)", side="top"), ticksuffix="%", len=0.5, y=1.0, yanchor="top", x=1.01, thickness=14)),
        coloraxis2=dict(colorscale=[[0, "#f2f2f1"], [1, "#2b2b2b"]], cmin=0, cmax=float(np.log10(max(h.n.max(), 10))),
                        colorbar=dict(title=dict(text="Games per<br>cell (count)", side="top"), tickvals=[0, 1, 2, 3, 4], ticktext=["1", "10", "100", "1k", "10k"],
                                      len=0.32, y=0.0, yanchor="bottom", x=1.01, thickness=14)),
        plot_bgcolor="white")
    fig.update_xaxes(range=[0, 650], **GAP_TICKS)
    fig.update_xaxes(title_text=GAP_TITLE, row=2)
    fig.update_yaxes(title_text="Average rating of the two players (Elo)", row=1, col=1)
    fig.update_yaxes(title_text="Average rating (Elo)", row=2, col=1)
    fig.update_yaxes(range=[elo_range[0], elo_range[1]], dtick=200)
    style(fig, 730, legend_y=-0.115, title="Observed underdog win rate by gap and rating level: at higher ratings the same gap leaves the underdog less chance",
          margin=dict(r=110, b=60))
    # caption numbers
    def med_cross(tc, lo, hi):
        v = [x for y, x in cross[tc].items() if lo <= y < hi]
        return np.median(v) if v else np.nan
    shown = h[h.n >= min_n]
    cov = shown.n.sum() / h.n.sum()
    c = "classical"
    lowr, highr = med_cross(c, 1600, 2000), med_cross(c, 2200, 2600)
    cap_txt = (f"Each cell is the observed share of games won by the lower-rated player (draws count as not winning), by 50-point gap bin and 100-point bin of the "
               f"average rating ({'both colours pooled' if color == 'all' else 'underdog has ' + color}); the lower row counts the games per cell. "
               f"The green step marks, per rating row, the first gap bin whose win rate is below {threshold:.0%}: in classical it sits at a median gap of "
               f"{lowr:.0f} points for average ratings 1600-1999 but {highr:.0f} for 2200-2599. "
               f"Limitation: cells with fewer than {min_n} games are hatched (the coloured cells hold {_pct(cov)} of the games shown); "
               f"the step is the first crossing of a noisy observed rate, not a fitted boundary, and draws, which rise with rating, are ignored.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- 4
BAND_COL = {"<2200": "#9db6e0", "2200-2499": "#4f78c4", "2500+": "#142f66"}
BAND_DASH = {"<2200": "dot", "2200-2499": "dash", "2500+": "solid"}


def fig_simpson(T, tc="classical", min_n=MIN_N):
    """Question: does one pooled curve describe every game, or does it hide subgroups that behave differently?
    Shows, for one time control, the underdog's win rate (top row) and draw rate (bottom row) against the gap:
    pooled (left), split by the underdog's colour (middle) and by rating band of the average rating (right), with the
    pooled curve repeated in light grey behind each split.
    Hides: the other two time controls (call with tc='rapid' / 'blitz'), the joint split colour x band (the table has it),
    and noise (bins with fewer than min_n games are not drawn)."""
    w = T["wdl_by_gap"]
    w = w[w.tc_class == tc]
    pooled = _rates(w, ["gap_bin"])
    pooled = pooled[pooled.n >= min_n]
    bycol = _rates(w, ["underdog_color", "gap_bin"])
    byband = _rates(w, ["band", "gap_bin"])
    fig = make_subplots(2, 3, shared_xaxes=True, shared_yaxes="rows", vertical_spacing=0.08, horizontal_spacing=0.03,
                        subplot_titles=["<b>pooled</b> (all games)", "<b>split by the underdog's colour</b>", "<b>split by rating band</b> (average Elo)",
                                        "", "", ""])
    for r, key in enumerate(["win", "draw"], 1):
        for c in (1, 2, 3):
            fig.add_scatter(x=_gap_x(pooled.gap_bin), y=pooled[key] * 100, mode="lines", showlegend=False,
                            line=dict(color=LIGHT if c > 1 else INK, width=6 if c > 1 else 2.5),
                            hovertemplate=f"pooled {key}: %{{y:.1f}}%<extra></extra>", row=r, col=c)
        for colr, dash, colour in [("white", "solid", INK), ("black", "dash", GREY)]:
            d = bycol[(bycol.underdog_color == colr) & (bycol.n >= min_n)]
            fig.add_scatter(x=_gap_x(d.gap_bin), y=d[key] * 100, mode="lines+markers", showlegend=False, marker=dict(size=4),
                            line=dict(color=colour, dash=dash, width=2.2), customdata=d.n,
                            hovertemplate=f"underdog has {colr}: %{{y:.1f}}% (n=%{{customdata:,}})<extra></extra>", row=r, col=2)
            if r == 1:
                lab = d[d.gap_bin == (50 if colr == "white" else 300)]
                fig.add_annotation(x=_gap_x(lab.gap_bin)[0], y=lab[key].iloc[0] * 100, text=f"underdog has {colr.capitalize()}",
                                   showarrow=True, arrowhead=0, ax=60 if colr == "white" else -10, ay=-30 if colr == "white" else 32,
                                   font=dict(color=colour, size=12), arrowcolor=colour, xanchor="left" if colr == "white" else "right",
                                   row=r, col=2)
        for b in BANDS:
            d = byband[(byband.band == b) & (byband.n >= min_n)]
            fig.add_scatter(x=_gap_x(d.gap_bin), y=d[key] * 100, mode="lines+markers", showlegend=False, marker=dict(size=4),
                            line=dict(color=BAND_COL[b], dash=BAND_DASH[b], width=2.2), customdata=d.n,
                            hovertemplate=f"band {b}: %{{y:.1f}}% (n=%{{customdata:,}})<extra></extra>", row=r, col=3)
            if len(d):
                first = d.iloc[0]
                fig.add_annotation(x=_gap_x([first.gap_bin])[0], y=first[key] * 100, text=b, showarrow=False, xanchor="right", xshift=-6,
                                   font=dict(color=BAND_COL[b] if b != "<2200" else "#5e7cb0", size=12), row=r, col=3)
    fig.update_yaxes(ticksuffix="%")
    fig.update_yaxes(title_text="Underdog win rate (% of games)", row=1, col=1)
    fig.update_yaxes(title_text="Draw rate (% of games)", row=2, col=1)
    ymax = max(pooled.win.max(), bycol[bycol.n >= min_n].win.max(), byband[byband.n >= min_n].win.max()) * 100
    fig.update_yaxes(range=[0, ymax * 1.1], row=1)
    dmax = max(byband[byband.n >= min_n].draw.max(), bycol[bycol.n >= min_n].draw.max()) * 100
    fig.update_yaxes(range=[0, dmax * 1.1], row=2)
    fig.update_xaxes(range=[-85, 660], **GAP_TICKS)
    fig.update_xaxes(range=[-215, 660], col=3)           # room for the band labels left of the curves (they overlapped the y-axis)
    fig.update_xaxes(title_text=GAP_TITLE, row=2)
    style(fig, 600, title=f"One pooled curve hides two strong effects ({tc}): the underdog's colour and the rating level")
    # numbers
    sub = w[w.gap_bin.isin([50, 100, 150])]
    sc = _rates(sub, ["underdog_color"]).set_index("underdog_color")
    ne = T["near_equal"].set_index(["tc_class", "band"])
    dr = [ne.loc[(tc, b), "draw_rate"] for b in BANDS]
    ww = _rates(sub, ["band"]).set_index("band")
    cap_txt = (f"Observed {tc} results of the lower-rated player against the rating gap (50-point bins), pooled (left) and split (middle, right; the light grey line repeats the pooled curve). "
               f"At gaps of 50-199 the underdog scores {sc.loc['white', 'score']:.3f} points per game with White but {sc.loc['black', 'score']:.3f} with Black "
               f"(win rate {_pct(sc.loc['white', 'win'])} vs {_pct(sc.loc['black', 'win'])}); between near-equal players (gap of 25 or less) the draw rate rises with the rating band "
               f"from {_pct(dr[0], 0)} (<2200) to {_pct(dr[1], 0)} (2200-2499) and {_pct(dr[2], 0)} (2500+), so at the same gap a strong underdog wins less often "
               f"({_pct(ww.loc['2500+', 'win'])} vs {_pct(ww.loc['<2200', 'win'])} at 50-199 for 2500+ vs <2200) but draws far more. "
               f"Limitation: the band is defined by the average rating, so bands also differ in their gap mix and in who plays (juniors, open events); "
               f"colour and band are shown one at a time, not jointly, and bins with fewer than {min_n} games are not drawn.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- 5
def _short_url(u):
    parts = str(u).rstrip("/").split("/")
    return "lichess.org/…/" + "/".join(parts[-2:])


def _place_labels(x, y, xmax, xr=(0, None), yr=(1150, 2900), w_px=400, h_px=420, lab=(16, 14)):
    """Greedy label placement for numbered points: try offsets on growing rings (pixels) and keep the first that
    overlaps no earlier label and no highlighted point. Returns (ax, ay) pixel offsets for plotly annotations."""
    sx, sy = w_px / xmax, h_px / (yr[1] - yr[0])
    P = np.c_[np.asarray(x) * sx, -(np.asarray(y) - yr[0]) * sy]          # pixel coords, y down
    boxes, out = [], []
    cands = [(r * np.cos(a), r * np.sin(a)) for r in (18, 30, 44, 60, 78) for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    for k, (px, py) in enumerate(P):
        best = cands[-1]
        for dx, dy in cands:
            cx, cy = px + dx, py + dy
            b = (cx - lab[0] / 2, cy - lab[1] / 2, cx + lab[0] / 2, cy + lab[1] / 2)
            clash = any(not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3]) for o in boxes)
            clash = clash or any(abs(cx - qx) < lab[0] / 2 + 5 and abs(cy - qy) < lab[1] / 2 + 5 for qx, qy in P)
            if not clash:
                best = (dx, dy)
                break
        cx, cy = px + best[0], py + best[1]
        boxes.append((cx - lab[0] / 2, cy - lab[1] / 2, cx + lab[0] / 2, cy + lab[1] / 2))
        out.append((float(best[0]), float(best[1])))
    return out


def fig_upset_points(T, top_k=20):
    """Question: where in (gap, rating level) do the upsets happen, and which are the most extreme?
    Shows every upset at alpha = 0.10 as one light point (opacity shows density) per time control, with the top_k
    rarest (smallest q_emp) numbered in HI and listed in the table below.
    'Upset' is a model definition: the underdog won although the empirical model gave P(win) < 10%, or drew although
    it gave P(not losing) < 10% (q_emp < 0.10).
    Hides: the ~846k games that are not upsets (so the plot shows where upsets are, not how likely they are; see
    fig_heatmap for rates) and the underdog's colour (no marker shape; it is in the hover and the table)."""
    u, rr = T["upset_points"], T["rarest_upsets"].sort_values("rank_q").head(top_k)
    top = u.nsmallest(top_k, "q_emp")
    xmax = float(np.ceil(u.gap.max() / 100) * 100)
    fig = make_subplots(2, 3, shared_xaxes=False, shared_yaxes=True, row_heights=[0.475, 0.525], vertical_spacing=0.08, horizontal_spacing=0.03,
                        specs=[[{}, {}, {}], [{"type": "table", "colspan": 3}, None, None]],
                        subplot_titles=[f"<b>{tc}</b>  ({(u.tc_class == tc).sum():,} upsets)" for tc in TC] + [""])
    for i, tc in enumerate(TC, 1):
        d = u[u.tc_class == tc]
        fig.add_scattergl(x=d.gap, y=d.avg_elo, mode="markers", marker=dict(size=4, color=GREY, opacity=0.22), row=1, col=i,
                          name="upset (q < 0.10)", showlegend=(i == 1),
                          customdata=np.c_[d.white, d.black, d.white_elo, d.black_elo, d.q_emp, d.broadcast_name],
                          hovertemplate="%{customdata[0]} (%{customdata[2]}) - %{customdata[1]} (%{customdata[3]})<br>%{customdata[5]}<br>q = %{customdata[4]:.3f}<extra></extra>")
        t = top[top.tc_class == tc]
        if len(t):
            for (gx, gy, rk), (ax, ay) in zip(t[["gap", "avg_elo", "rank_q"]].values, _place_labels(t.gap.values, t.avg_elo.values, xmax)):
                fig.add_annotation(x=gx, y=gy, text=str(int(rk)), ax=ax, ay=ay, showarrow=True, arrowhead=0, arrowwidth=0.8,
                                   arrowcolor=HI, standoff=4, font=dict(color=HI, size=11), bgcolor="rgba(255,255,255,0.85)", borderpad=0,
                                   row=1, col=i)
            fig.add_scatter(x=t.gap, y=t.avg_elo, mode="markers", text=t.rank_q.astype(str),
                            marker=dict(size=8, color=HI, line=dict(color="white", width=1.5)),
                            name=f"{top_k} rarest upsets (numbered, see table)", showlegend=(i == 1), row=1, col=i,
                            customdata=np.c_[t.white, t.black, t.q_emp], hovertemplate="#%{text}: %{customdata[0]} - %{customdata[1]}, q = %{customdata[2]:.4f}<extra></extra>")
    fig.update_xaxes(range=[0, xmax], title_text=GAP_TITLE, row=1)
    fig.update_yaxes(title_text="Average rating (Elo)", row=1, col=1)
    res = np.where(rr.underdog_score == 1, "win", "draw")
    fig.add_table(row=2, col=1, columnwidth=[22, 150, 150, 40, 40, 230, 38, 52, 150],
                  header=dict(values=["#", "Underdog (Elo)", "Favourite (Elo)", "Gap", "Result", "Event", "Year", "model q", "Lichess game"],
                              fill_color="#efeeed", align="left", font=dict(size=11, color=INK)),
                  cells=dict(values=[rr.rank_q, [f"{a} ({int(b)})" for a, b in zip(rr.underdog, rr.underdog_elo)],
                                     [f"{a} ({int(b)})" for a, b in zip(rr.favourite, rr.favourite_elo)], rr.gap.astype(int), res,
                                     rr.broadcast_name.str.slice(0, 48), rr.year, rr.q_emp.round(4), [_short_url(x) for x in rr.game_url]],
                             align="left", font=dict(size=10, color=INK), height=19, fill_color="white", line_color="#e6e5e3"))
    q1, q3 = u.gap.quantile([0.25, 0.75])
    tcs = top.tc_class.unique()
    who = f"all {tcs[0]}" if len(tcs) == 1 else "mixed classes"
    style(fig, 1010, legend_y=-0.015, margin=dict(b=60), title=f"Where the {len(u):,} upsets happen: half have gaps of {q1:.0f}-{q3:.0f} points; "
                           f"the {top_k} rarest are {who} games with gaps of {int(top.gap.min())}+")
    n_draw = int((u.underdog_score == 0.5).sum())
    tt = top.tc_class.value_counts()
    cap_txt = (f"Every upset at the 10% level ({len(u):,} games: {(u.tc_class == 'classical').sum():,} classical, {(u.tc_class == 'rapid').sum():,} rapid, {(u.tc_class == 'blitz').sum():,} blitz) "
               f"plotted at its rating gap and average rating; an upset is defined by the empirical model: the underdog won although the model gave it P(win) < 10%, "
               f"or drew although P(not losing) < 10% ({n_draw:,} of the upsets are such draws). "
               f"The {top_k} rarest (smallest model probability, q from {top.q_emp.min():.4f} to {top.q_emp.max():.4f}) are highlighted and listed; "
               f"{'all' if tt.get('classical', 0) == top_k else tt.get('classical', 0)} of them are classical, with gaps of {int(top.gap.min())}-{int(top.gap.max())} points. "
               f"Limitation: this shows where upsets occur, not how likely they are (dense regions are dense mainly because many games are played there), "
               f"and the empty band along the left edge is built into the definition (where the model gives the underdog 10% or more, no result counts as an upset); a very large gap often means an outdated or wrong rating "
               f"rather than a sensational game; full links are in rarest_upsets.csv.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- 6
def fig_set_compare(T):
    """Question: who did the cleaning filters remove, and does the analysis set still look like the larger cleaned set?
    Shows (a) the share of each year's cleaned games that is dropped, split into 'missing Elo' and the other filters,
    (b) the time-control mix and (c) the share of games with a titled player, larger cleaned set vs analysis set.
    Hides: games removed even earlier (parsing / first cleaning: all_games -> clean_games), the game-length and
    rating mixes (in set_compare.csv; they barely change because the analysis set is defined by having ratings),
    and 2010 / 2019 (27 and 256 games)."""
    sc, my, ds = T["set_compare"], T["missing_elo_by_year"].copy(), T["drop_steps"]
    my = my[my.clean_n >= 1000].copy()
    my["year"] = my.year.astype(int)
    my["miss"] = 1 - my.both_elo_n / my.clean_n
    my["other"] = (my.both_elo_n - my.analysis_n) / my.clean_n
    fig = make_subplots(1, 3, column_widths=[0.42, 0.29, 0.29], horizontal_spacing=0.12,
                        subplot_titles=["<b>(a)</b> games dropped, by year", "<b>(b)</b> time-control mix", "<b>(c)</b> games with a titled player"])
    fig.add_bar(x=my.year, y=my.miss * 100, name="dropped: missing Elo for a player", marker=dict(color=HI, line=dict(color="white", width=1)), row=1, col=1,
                customdata=my.clean_n, hovertemplate="%{x}: %{y:.1f}% of %{customdata:,} games<extra></extra>")
    fig.add_bar(x=my.year, y=my.other * 100, name="dropped: unknown time control or < 10 plies",
                marker=dict(color=LIGHT, line=dict(color="white", width=1)), row=1, col=1, hovertemplate="%{x}: %{y:.1f}%<extra></extra>")
    tot = my.miss + my.other
    for x, y, nn in zip(my.year, tot, my.clean_n):
        fig.add_annotation(x=x, y=y * 100, text=f"{y:.0%}", showarrow=False, yanchor="bottom", font=dict(size=11, color=INK2), row=1, col=1)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Share of the year's cleaned games (%)", ticksuffix="%", range=[0, 50], row=1, col=1)
    fig.update_xaxes(title_text="Year (game date)", tickvals=list(my.year), ticks="", row=1, col=1)

    n_cl, n_an_ = int(ds.removed.sum() + ds.remaining.iloc[-1]), int(ds.remaining.iloc[-1])
    big = lambda n: f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"

    def dumbbell(dim, cats, labels, col, first):
        d = sc[sc.dimension == dim].pivot(index="category", columns="set", values="share").reindex(cats).fillna(0)
        y = labels
        for k in range(len(cats)):
            fig.add_scatter(x=[d.clean.iloc[k] * 100, d.analysis.iloc[k] * 100], y=[y[k]] * 2, mode="lines", line=dict(color=LIGHT, width=4),
                            showlegend=False, hoverinfo="skip", row=1, col=col)
        fig.add_scatter(x=d.clean * 100, y=y, mode="markers", marker=dict(symbol="circle-open", size=11, color=INK2, line=dict(width=2)),
                        name=f"larger cleaned set ({big(n_cl)} games)" if first else None, showlegend=first, row=1, col=col,
                        hovertemplate="cleaned set: %{x:.1f}%<extra></extra>")
        fig.add_scatter(x=d.analysis * 100, y=y, mode="markers", marker=dict(symbol="circle", size=11, color=INK),
                        name=f"analysis set ({big(n_an_)} games)" if first else None, showlegend=first, row=1, col=col,
                        hovertemplate="analysis set: %{x:.1f}%<extra></extra>")
        fig.update_yaxes(autorange="reversed", row=1, col=col)
        return d
    tcd = dumbbell("tc_class", TC + ["unknown"], ["classical", "rapid", "blitz", "unknown"], 2, True)
    tit = dumbbell("titled", ["at least one FIDE title"], ["any FIDE title"], 3, False)
    gm = sc[sc.dimension == "gm"].pivot(index="category", columns="set", values="share").loc["at least one GM"]
    fig.add_scatter(x=[gm.clean * 100, gm.analysis * 100], y=["a GM"] * 2, mode="lines", line=dict(color=LIGHT, width=4), showlegend=False, hoverinfo="skip", row=1, col=3)
    fig.add_scatter(x=[gm.clean * 100], y=["a GM"], mode="markers", marker=dict(symbol="circle-open", size=11, color=INK2, line=dict(width=2)), showlegend=False, row=1, col=3)
    fig.add_scatter(x=[gm.analysis * 100], y=["a GM"], mode="markers", marker=dict(symbol="circle", size=11, color=INK), showlegend=False, row=1, col=3)
    fig.update_yaxes(categoryorder="array", categoryarray=["any FIDE title", "a GM"], autorange=False, range=[2.2, -1.2], row=1, col=3)
    fig.update_xaxes(title_text="Share of the set's games (%)", ticksuffix="%", rangemode="tozero", row=1, col=2)
    fig.update_xaxes(title_text="Share of the set's games (%)", ticksuffix="%", range=[0, 65], row=1, col=3)
    hy = my[(my.miss + my.other) >= 0.3].year
    yrs = (f"{hy.min()}-{hy.max()}" if hy.max() - hy.min() == len(hy) - 1 else ", ".join(map(str, hy))) if len(hy) else "no year"
    style(fig, 480, title=f"The filters mostly remove games without ratings: {yrs} lose 30%+ of their games, untitled players lose most")
    drop = sc[(sc.set == "dropped") & (sc.dimension == "titled")].set_index("category").share
    worst = my.loc[my.miss.idxmax()]
    ds = ds.set_index("step")
    n_clean, n_an = int(ds.removed.sum() + ds.remaining.iloc[-1]), int(ds.remaining.iloc[-1])
    cap_txt = (f"Larger cleaned set ({n_clean:,} games, interim/clean_games) vs the analysis set ({n_an:,}). "
               f"Most of the {int(ds.removed.sum()):,} dropped games lack a FIDE rating for a player ({int(ds.loc['missing Elo for at least one player', 'removed']):,}); "
               f"that share peaks in {int(worst.year)} ({_pct(worst.miss, 0)} of the year's games). {_pct(tcd.loc['unknown', 'clean'])} of the cleaned set has an unknown time control; "
               f"{int(ds.loc['time control unknown', 'removed']):,} of these games still had both ratings and were removed by that filter. "
               f"Games with at least one FIDE-titled player rise from {_pct(tit.loc['at least one FIDE title', 'clean'], 0)} to {_pct(tit.loc['at least one FIDE title', 'analysis'], 0)}, "
               f"because only {_pct(drop.get('at least one FIDE title', 0), 0)} of the dropped games have one: the analysis set over-represents titled, rated players. "
               f"Limitation: 'titled' counts FIDE titles (GM, IM, FM, CM and the women's titles) recorded in the PGN, and the rating and game-length mixes are not shown.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- 7
TC_GREY = {"classical": "#3b3a39", "rapid": "#8a8987", "blitz": "#cfcecb"}


def fig_games_per_year(T):
    """Question: how much data does each year (and time control) contribute?
    Shows analysis games per year stacked by time control (grey levels, labelled directly); the last year is hatched
    because it is incomplete.
    Hides: within-year seasonality (monthly_counts.csv) and the events behind the growth (more broadcasts on Lichess,
    not more chess played)."""
    yc, mc = T["yearly_counts"], T["monthly_counts"]
    years = sorted(yc.year.unique())
    piv = yc.pivot(index="year", columns="tc_class", values="n").reindex(years).fillna(0)
    part = set(yc.loc[yc.partial, "year"])
    py = max(part) if part else None
    months = mc[mc.period.str.startswith(str(py))].groupby("period").n.sum() if py else pd.Series(dtype=float)
    full_months = months[months >= 100]
    span = ""
    if len(full_months):
        mname = lambda p: pd.Timestamp(p + "-01").strftime("%b")
        span = f"{mname(full_months.index.min())}-{mname(full_months.index.max())}"
    xs = [int(y) for y in years]     # numeric x: string years + annotations break plotly's category axis
    fig = go.Figure()
    for tc in TC:
        fig.add_bar(x=xs, y=piv[tc] / 1000, name=tc, marker=dict(color=TC_GREY[tc], line=dict(color="white", width=1.5),
                    pattern=dict(shape=["/" if y in part else "" for y in years], fgcolor="white" if tc != "blitz" else "#7a7977", size=7, solidity=0.25)),
                    customdata=piv[tc], hovertemplate=f"%{{x}} {tc}: %{{customdata:,.0f}} games<extra></extra>", showlegend=False)
    tot = piv.sum(axis=1)
    for x, t in zip(xs, tot):
        fig.add_annotation(x=x, y=t / 1000, text=f"{t / 1000:.0f}k" if t >= 1000 else f"{t:.0f}", showarrow=False, yanchor="bottom", font=dict(size=12, color=INK))
    # direct labels on the last complete year
    lab_year = max(y for y in years if y not in part)
    cum = 0
    for tc in TC:
        v = piv.loc[lab_year, tc] / 1000
        fig.add_annotation(x=lab_year, y=cum + v / 2, text=tc, showarrow=False, font=dict(size=12, color="white" if tc != "blitz" else INK))
        cum += v
    if py:
        fig.add_annotation(x=py, y=tot.loc[py] / 1000, yshift=22, text=f"partial year: {span}", showarrow=False, yanchor="bottom",
                           font=dict(size=12, color=INK2))
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_yaxes(title_text="Games (thousands)", range=[0, tot.max() / 1000 * 1.18])
    fig.update_xaxes(title_text="Year (game date)", tickvals=xs, ticks="")
    share24 = tot.loc[[y for y in years if y >= 2024]].sum() / tot.sum()
    style(fig, 380, title=f"Games per year in the analysis set: {share24:.0%} of the data is from 2024 onwards")
    extra = months[months < 100]
    cap_txt = (f"Analysis games per calendar year of the game date ({int(tot.sum()):,} in total), stacked by time control. "
               f"{_pct(share24, 0)} of the games are from 2024 or later; classical is {_pct(piv['classical'].sum() / tot.sum(), 0)} of all games. "
               f"Limitation: {py} is partial (hatched; data cover {span} {py}"
               + (f", plus {int(extra.sum())} games dated later" if len(extra) else "") +
               f"), so its bar is not comparable with full years, and growth reflects how many events were broadcast on Lichess, not how much chess was played.")
    return fig, cap_txt


FIGURES = [fig_gap_distribution, fig_outcomes_by_gap, fig_heatmap, fig_simpson, fig_upset_points, fig_set_compare, fig_games_per_year]


# ==================================================================================================== PART 2
# Model + rule layer. Tables: make_viz_data.py (model block) plus the small model outputs of notebook 03
# (reports/model_outputs/). Use load2() instead of load() for these functions:
#
#     T = viz.load2()
#     fig, cap = viz.fig_model_vs_data(T)
#     df, md = viz.table_critical_gaps(T)
STAGE2_TABLES = ["model_bins", "model_bins_by_verified", "data_support", "critical_gap_by_alpha", "critical_gap_by_elo",
                 "avg_elo_hist", "expected_upsets", "expected_upsets_interaction", "checks"]
PROCESSED_TABLES = ["critical_gap_empirical", "critical_gap_empirical_verified", "critical_gap_empirical_sens",
                    "critical_gap_isotonic", "critical_gap_isotonic_verified", "critical_gap_isotonic_sens",
                    "critical_gap_diff_empirical", "calibration_empirical", "fitted_curves",
                    "critical_gap_empirical_interaction", "critical_gap_main_vs_interaction", "model_fit_by_band", "ordering_check"]
PROC_DIR = os.path.abspath(os.environ.get("CHESS_MODEL_OUTPUTS", os.path.join(HERE, "..", "reports", "model_outputs")))
COLS2 = ["white", "black"]
# underdog colour is encoded by grey level + marker shape + line style (never by hue)
UC_STYLE = {"white": dict(color=INK, symbol="circle", dash="solid", label="White underdog"),
            "black": dict(color="#8a8987", symbol="diamond-open", dash="dash", label="Black underdog")}
SPARSE = "#c9c8c5"                       # bins with fewer than MIN_N games
REF_ELO2 = {"<2200": 2100, "2200-2499": 2350, "2500+": 2600}
ALPHA0 = 0.10                            # the project's default alpha


def load2(data_dir=os.path.join(HERE, "viz_data"), proc_dir=PROC_DIR):
    """load() plus the model-layer tables (viz_data) and the small model outputs of notebook 03 (reports/model_outputs)."""
    T = load(data_dir)
    for t in STAGE2_TABLES:
        T[t] = pd.read_csv(os.path.join(data_dir, t + ".csv"))
    for t in PROCESSED_TABLES:
        T[t] = pd.read_csv(os.path.join(proc_dir, t + ".csv"))
    for t in PROCESSED_TABLES:
        T[t] = T[t].rename(columns={"underdog_colour": "underdog_color"})
    return T


def _wilson(k, n, z=1.959964):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def _gx(gap_bin):
    """x of a 50-point gap bin of model_bins (900 = pooled 900+, drawn at 925)."""
    return np.asarray(gap_bin, float) + 25


GAP_TICKS2 = dict(tickvals=[0, 200, 400, 600, 800, 925], ticktext=["0", "200", "400", "600", "800", "900+"])


# ---------------------------------------------------------------------------------------------------- S2.1
def model_fit_stats(T, min_n=MIN_N):
    """Per time control x band x underdog colour (and both colours, 'both'): over the gap bins with >= min_n games,
    the share of bins whose like-for-like model prediction lies inside the Wilson 95% interval of the observed win rate,
    and the mean signed error observed - predicted in percentage points, weighted by n and unweighted."""
    mb = T["model_bins"]
    mb = mb[mb.n >= min_n].copy()
    mb["inside"] = (mb.pred_win >= mb.obs_lo95) & (mb.pred_win <= mb.obs_hi95)
    mb["err"] = (mb.obs_win - mb.pred_win) * 100
    mb["err_ref"] = (mb.obs_win - mb.pred_win_ref) * 100
    rows = []
    for (tc, band), d0 in mb.groupby(["tc_class", "band"]):
        for col in COLS2 + ["both"]:
            d = d0 if col == "both" else d0[d0.underdog_color == col]
            rows.append(dict(tc_class=tc, band=band, underdog_color=col, bins=len(d), games=int(d.n.sum()),
                             inside=int(d.inside.sum()), share_inside=d.inside.mean(),
                             err_wtd_pp=np.average(d.err, weights=d.n), err_unwtd_pp=d.err.mean(),
                             abs_err_wtd_pp=np.average(d.err.abs(), weights=d.n), max_abs_err_pp=d.err.abs().max(),
                             err_ref_wtd_pp=np.average(d.err_ref, weights=d.n)))
    out = pd.DataFrame(rows)
    out["tc_class"] = pd.Categorical(out.tc_class, TC)
    out["band"] = pd.Categorical(out.band, BANDS)
    return out.sort_values(["tc_class", "band", "underdog_color"]).reset_index(drop=True)


def fig_model_vs_data(T, min_n=MIN_N, x_max=950):
    """Question: does the fitted model describe the games it is applied to, bin by bin?
    Shows observed underdog win rate (markers, Wilson 95% whiskers) against the model's like-for-like prediction (line):
    the mean of the model's P(win) over the same games, each at its own gap and average rating. Rows = rating band,
    columns = time control; White vs Black underdog by marker shape, grey level and line style. Bins with fewer than
    min_n games are light grey without whiskers. Each panel states, for the bins with >= min_n games, how many contain
    the model inside the observed interval and the n-weighted mean error (observed - predicted).
    Hides: the model's own parameter uncertainty (the whiskers are sampling intervals of the observed rate only) and
    the draws."""
    mb = T["model_bins"].copy()
    st = model_fit_stats(T, min_n).set_index(["tc_class", "band", "underdog_color"])
    fig = make_subplots(3, 3, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.025, vertical_spacing=0.06,
                        column_titles=[f"<b>{tc}</b>" for tc in TC], row_titles=[f"avg Elo {b}" for b in BANDS])
    ymax = 50
    for r, band in enumerate(BANDS, 1):
        for c, tc in enumerate(TC, 1):
            for col in COLS2:
                s = UC_STYLE[col]
                d = mb[(mb.tc_class == tc) & (mb.band == band) & (mb.underdog_color == col)].sort_values("gap_bin")
                off = -6 if col == "white" else 6
                ok, sp = d[d.n >= min_n], d[d.n < min_n]
                first = (r == 1 and c == 1)
                # model: like-for-like mean prediction, drawn over the well-populated bins; thin dotted beyond
                fig.add_scatter(x=_gx(ok.gap_bin), y=ok.pred_win * 100, mode="lines", line=dict(color=s["color"], dash=s["dash"], width=2),
                                name=f"model, {s['label']}", legendgroup=f"m{col}", showlegend=first,
                                hovertemplate=f"model ({col} underdog): %{{y:.1f}}%<extra></extra>", row=r, col=c)
                fig.add_scatter(x=_gx(d[d.gap_bin >= ok.gap_bin.max()].gap_bin) if len(ok) else _gx(d.gap_bin),
                                y=d[d.gap_bin >= (ok.gap_bin.max() if len(ok) else 0)].pred_win * 100, mode="lines",
                                line=dict(color=SPARSE, dash="dot", width=1.5), showlegend=False, hoverinfo="skip", row=r, col=c)
                fig.add_scatter(x=_gx(ok.gap_bin) + off, y=ok.obs_win * 100, mode="markers",
                                marker=dict(symbol=s["symbol"], size=6.5, color=s["color"], line=dict(color=s["color"], width=1.5)),
                                error_y=dict(type="data", symmetric=False, array=(ok.obs_hi95 - ok.obs_win) * 100,
                                             arrayminus=(ok.obs_win - ok.obs_lo95) * 100, color=s["color"], thickness=1.1, width=0),
                                name=f"observed, {s['label']} (95% interval)", legendgroup=f"o{col}", showlegend=first,
                                customdata=np.c_[ok.n, ok.pred_win * 100, ok.mean_avg_elo],
                                hovertemplate=(f"{col} underdog, gap bin %{{x:.0f}}: observed %{{y:.1f}}% vs model %{{customdata[1]:.1f}}% "
                                               f"(n=%{{customdata[0]:,}}, mean avg Elo %{{customdata[2]:.0f}})<extra>{tc}, {band}</extra>"), row=r, col=c)
                fig.add_scatter(x=_gx(sp.gap_bin) + off, y=sp.obs_win * 100, mode="markers",
                                marker=dict(symbol=s["symbol"].replace("-open", "") + "-open", size=5.5, color=SPARSE, line=dict(color=SPARSE, width=1.2)),
                                showlegend=False,
                                customdata=sp.n, hovertemplate=f"{col}: %{{y:.1f}}% (only %{{customdata}} games)<extra></extra>", row=r, col=c)
            fig.add_scatter(x=[0, x_max], y=[ALPHA0 * 100] * 2, mode="lines", line=dict(color=HI, width=1.5), showlegend=False,
                            hoverinfo="skip", row=r, col=c)
            b = st.loc[(tc, band, "both")]
            fig.add_annotation(x=x_max, y=ymax * 0.97, xanchor="right", yanchor="top", showarrow=False, align="right",
                               text=f"model inside interval: {int(b.inside)}/{int(b.bins)} bins<br>mean error {b.err_wtd_pp:+.1f} pp (n-weighted)",
                               font=dict(size=11, color=INK2), row=r, col=c)
    fig.add_annotation(x=x_max, y=ALPHA0 * 100, xanchor="right", yanchor="bottom", showarrow=False, text="10%",
                       font=dict(size=11, color=HI), row=1, col=3)
    fig.add_scatter(x=[None], y=[None], mode="markers+lines", marker=dict(symbol="circle-open", size=6, color=SPARSE),
                    line=dict(color=SPARSE, dash="dot", width=1.5), name=f"bins with fewer than {min_n} games (and the model there)")
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=HI, width=1.5), name="10% (default upset threshold)")
    fig.update_xaxes(range=[0, x_max], **GAP_TICKS2)
    fig.update_xaxes(title_text=GAP_TITLE, row=3, col=2)
    fig.update_yaxes(range=[0, ymax], ticksuffix="%", dtick=10)
    fig.update_yaxes(title_text="Underdog win rate", row=2, col=1)
    fig.for_each_annotation(lambda a: a.update(font=dict(size=13)) if a.text.startswith("avg Elo") else None)
    both = st.xs("both", level="underdog_color")
    tot_in, tot_b = int(both.inside.sum()), int(both.bins.sum())
    worst = st.reset_index()
    worst = worst[worst.underdog_color != "both"].sort_values("err_wtd_pp", key=np.abs, ascending=False).iloc[0]
    w25 = st.loc[("classical", "2500+", "both")]
    fbi = T["model_fit_by_band"].set_index(["model", "tc_class", "band"]).loc[("interaction", "classical", "2500+")]
    rest = both.drop(index=("classical", "2500+"))
    worst_rest = float(rest.err_wtd_pp.abs().max())
    style(fig, 780, legend_y=-0.1, title=(f"Applied game by game, the model's mean error (n-weighted) stays within {worst_rest:.1f} points in every panel "
                           f"except strong classical games (2500+: {w25.err_wtd_pp:+.1f} pp)"), margin=dict(r=60, b=110))
    mbw = T["model_bins"]
    mbw = mbw[mbw.n >= min_n]
    shift = (mbw.pred_win - mbw.pred_win_ref).abs()
    shift2 = shift[mbw.band == "2200-2499"]
    cap_txt = (f"Observed underdog win rate (markers, Wilson 95% intervals) per 50-point gap bin against the model's like-for-like prediction "
               f"(line: mean of each game's own predicted P(win), using its own gap and average rating), by rating band and time control. "
               f"In the {tot_b} bins with at least {min_n} games the prediction lies inside the observed interval in {tot_in} ({tot_in / tot_b:.0%}; "
               f"about 95% would be expected if the model were exact), and the n-weighted mean error per panel and colour ranges from "
               f"{st.err_wtd_pp.min():+.1f} to {st.err_wtd_pp.max():+.1f} percentage points (largest: {worst.tc_class}, {worst.band}, {worst.underdog_color} underdog, "
               f"{worst.err_wtd_pp:+.1f} pp). Limitation: the prediction is in-sample (the model was fitted to these games; the held-out check is in the "
               f"calibration figure), the intervals ignore the clustering of games in tournaments, and evaluating the curve at one reference rating instead "
               f"would shift the prediction by at most {shift2.max() * 100:.2f} pp in any well-filled bin of the 2200-2499 band "
               f"(up to {shift.max() * 100:.1f} pp in the wide <2200 band). The clearest misfit is classical 2500+, where underdogs win less often than predicted "
               f"(model inside the interval in only {int(w25.inside)} of {int(w25.bins)} bins); the interaction model (rating x gap terms) does not "
               f"remove it ({fbi.inside:.0f} of {fbi.bins:.0f} bins, {fbi.mean_err_wtd_pp:+.1f} pp; see fig_model_misfit).")
    return fig, cap_txt


def fig_verified_split(T, band="2200-2499", color="white", min_n=MIN_N, x_max=700):
    """Question: why did the old notebook plot show observed win rates far above the fitted curve in rapid and blitz?
    Shows, for one band and underdog colour, the curve at the band's reference rating (as in the notebook) with its 95%
    band, the observed win rate of ALL games, and the same split into games whose two ratings are confirmed by the FIDE
    list (verified) and those that are not; the old plot used data/dashboard/curve_bins.csv filtered to verified == False.
    Hides: the other bands and colours (pass band=, color=)."""
    fc = T["fitted_curves"]
    mb, mv = T["model_bins"], T["model_bins_by_verified"]
    fig = make_subplots(1, 3, shared_yaxes=True, horizontal_spacing=0.03, subplot_titles=[f"<b>{tc}</b>" for tc in TC])
    spec = [(None, "all games", INK, "circle", 7), (False, "unverified ratings only (what the old plot showed)", HI, "diamond", 7),
            (True, "verified ratings only", GREY, "square-open", 6)]
    nums = {}
    for c, tc in enumerate(TC, 1):
        f = fc[(fc.tc_class == tc) & (fc.underdog_color == color) & (fc.band == band) & (fc.gap <= x_max)]
        fig.add_scatter(x=list(f.gap) + list(f.gap[::-1]), y=list(f.p_win_hi95 * 100) + list(f.p_win_lo95[::-1] * 100), fill="toself",
                        fillcolor="rgba(120,120,120,0.18)", line=dict(width=0), hoverinfo="skip", showlegend=False, row=1, col=c)
        fig.add_scatter(x=f.gap, y=f.p_win * 100, mode="lines", line=dict(color=INK2, width=2), row=1, col=c, showlegend=(c == 1),
                        name=f"model curve at one reference rating ({REF_ELO2[band]}), 95% band")
        for k, (ver, lab, colr, sym, size) in enumerate(spec):
            if ver is None:
                d = mb[(mb.tc_class == tc) & (mb.band == band) & (mb.underdog_color == color)]
            else:
                d = mv[(mv.tc_class == tc) & (mv.band == band) & (mv.underdog_color == color) & (mv.verified == ver)]
            d = d[(d.n >= min_n) & (d.gap_bin < x_max)].sort_values("gap_bin")
            lo, hi = _wilson(d.wins, d.n)
            fig.add_scatter(x=d.gap_bin + 25 + (k - 1) * 7, y=d.wins / d.n * 100, mode="markers",
                            marker=dict(symbol=sym, size=size, color=colr, line=dict(color=colr, width=1.4)),
                            error_y=dict(type="data", symmetric=False, array=(hi - d.wins / d.n) * 100, arrayminus=(d.wins / d.n - lo) * 100,
                                         color=colr, thickness=1, width=0),
                            name=lab, showlegend=(c == 1), customdata=d.n,
                            hovertemplate=f"{lab}: %{{y:.1f}}% (n=%{{customdata:,}})<extra>{tc}</extra>", row=1, col=c)
            nums[(tc, ver)] = d.set_index("gap_bin")
        fig.add_scatter(x=[0, x_max], y=[10, 10], mode="lines", line=dict(color=GREY, dash="dash", width=1), hoverinfo="skip",
                        showlegend=False, row=1, col=c)
    fig.update_xaxes(range=[0, x_max], title_text=GAP_TITLE, tickvals=list(range(0, x_max + 1, 100)))
    fig.update_yaxes(range=[0, 45], ticksuffix="%", dtick=10)
    fig.update_yaxes(title_text=f"Win rate of the {color.capitalize()} underdog", col=1)
    vshare = T["model_bins_by_verified"]
    vsh = vshare[vshare.verified].n.sum() / vshare.n.sum()
    r250 = (nums[("rapid", False)].loc[250], nums[("rapid", None)].loc[250], nums[("rapid", True)].loc[250])
    b300 = (nums[("blitz", False)].loc[300], nums[("blitz", None)].loc[300])
    curve = lambda tc, g: float(fc[(fc.tc_class == tc) & (fc.underdog_color == color) & (fc.band == band) & (fc.gap == g)].p_win.iloc[0])
    dv = (r250[0].wins / r250[0].n - curve("rapid", 275)) * 100
    da = (r250[1].wins / r250[1].n - curve("rapid", 275)) * 100
    style(fig, 430, title=(f"The old plot's dots were only the games with unverified ratings ({band}, {color.capitalize()} underdog): "
                           f"in rapid at gap 250-299 they lie {dv:+.1f} points from the curve, all games {da:+.1f}"))
    cap_txt = (f"Win rate of the {color.capitalize()} underdog, average rating {band}: the model curve at the band's reference rating with its 95% band "
               f"(as drawn in the notebook) against observed win rates of all games and of the two subsets split by whether both ratings are confirmed by the "
               f"FIDE list within 100 points ({_pct(vsh, 0)} of all games are verified). The notebook's dots were built from data/dashboard/curve_bins.csv filtered to "
               f"verified == False, i.e. only the unverified games: in rapid at gap 250-299 they show {_pct(r250[0].wins / r250[0].n)} "
               f"(n={int(r250[0].n)}), whereas all games show {_pct(r250[1].wins / r250[1].n)} (n={int(r250[1].n):,}) and the verified games "
               f"{_pct(r250[2].wins / r250[2].n)}, against {_pct(curve('rapid', 275))} for the curve at gap 275; blitz at 300-349: {_pct(b300[0].wins / b300[0].n)} "
               f"(unverified) vs {_pct(b300[1].wins / b300[1].n)} (all games). Limitation: the 95% band reflects only parameter uncertainty, so observed bins "
               f"of a few hundred games will often lie outside it even when the model is right; dots for bins with fewer than {min_n} games are not drawn.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- S2.2
def _support(T):
    return T["data_support"].set_index(["tc_class", "underdog_color", "band"])


TC_LINE = {"classical": "#2b2b2b", "rapid": "#6f6e6c", "blitz": "#aaa9a6"}     # time control by grey level (greyscale-safe)
TC_WIDTH = {"classical": 2.8, "rapid": 2.8, "blitz": 3.2}


def fig_critical_gap_alpha(T, a_lo=0.01, a_hi=0.30):
    """Question: does the time-control ordering of the critical gap depend on the choice of alpha, and where does the
    rule rest on extrapolation?
    Shows the model's critical gap (first 5-point gap where predicted P(underdog wins) < alpha) as a function of alpha,
    per rating band (columns, at the band's reference rating) and underdog colour (rows), one line per time control
    (grey level, labelled directly). A line is solid where the critical gap lies within the observed data (gaps up to the
    end of the last 50-point bin with >= 100 games in that time control x colour x band slice; thin horizontal lines)
    and dotted beyond (extrapolation). Whiskers: the published 95% intervals at alpha = 0.20 / 0.10 / 0.05 / 0.01.
    alpha = 0.10 is marked in the highlight colour.
    Hides: alphas where the curve never falls below alpha (no line; the model is flat beyond a gap of 900)."""
    cga, cg = T["critical_gap_by_alpha"], T["critical_gap_empirical"]
    sup = _support(T)
    cga = cga[(cga.alpha >= a_lo - 1e-9) & (cga.alpha <= a_hi + 1e-9)]
    fig = make_subplots(2, 3, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.025, vertical_spacing=0.07,
                        column_titles=[f"<b>avg Elo {b}</b> (reference {REF_ELO2[b]})" for b in BANDS],
                        row_titles=["White underdog", "Black underdog"])
    for r, col in enumerate(COLS2, 1):
        for c, b in enumerate(BANDS, 1):
            lims = {}
            for tc in TC:
                d = cga[(cga.tc_class == tc) & (cga.underdog_color == col) & (cga.band == b)].sort_values("alpha")
                lim = float(sup.loc[(tc, col, b), "support_limit"])
                lims.setdefault(lim, []).append(tc)
                ins = d.critical_gap <= lim
                fig.add_scatter(x=d.alpha, y=d.critical_gap, mode="lines", line=dict(color=TC_LINE[tc], dash="dot", width=2),
                                showlegend=False, hovertemplate=f"{tc}: alpha %{{x:.3f}} -> %{{y:.0f}}<extra></extra>", row=r, col=c)
                fig.add_scatter(x=d.alpha.where(ins), y=d.critical_gap.where(ins), mode="lines", line=dict(color=TC_LINE[tc], width=TC_WIDTH[tc]),
                                showlegend=False, hoverinfo="skip", row=r, col=c)
                p = cg[(cg.tc_class == tc) & (cg.underdog_color == col) & (cg.band == b) & cg.critical_gap.notna()]
                fig.add_scatter(x=p.alpha, y=p.critical_gap, mode="markers", marker=dict(size=5, color=TC_LINE[tc]),
                                error_y=dict(type="data", symmetric=False, array=p.hi95 - p.critical_gap, arrayminus=p.critical_gap - p.lo95,
                                             color=TC_LINE[tc], thickness=1.4, width=4),
                                showlegend=False, customdata=np.c_[p.lo95, p.hi95],
                                hovertemplate=f"{tc}, alpha %{{x}}: %{{y:.0f}} (95%: %{{customdata[0]:.0f}}-%{{customdata[1]:.0f}})<extra>published</extra>", row=r, col=c)
                if r == 1 and c == 1:
                    y = float(d.loc[np.isclose(d.alpha, 0.27), "critical_gap"].iloc[0])
                    fig.add_annotation(x=0.27, y=y, text=tc, showarrow=False, yanchor="bottom" if tc == "blitz" else "top",
                                       yshift=2 if tc == "blitz" else -2, xanchor="center",
                                       font=dict(size=12, color=TC_LINE[tc] if tc != "blitz" else "#8a8987"), row=r, col=c)
            for lim, tcs in lims.items():
                fig.add_scatter(x=[a_lo, a_hi], y=[lim, lim], mode="lines", line=dict(color=LIGHT, width=1), hoverinfo="skip", showlegend=False, row=r, col=c)
                fig.add_annotation(x=a_hi, y=lim, text="data end: " + ", ".join(tcs), showarrow=False, xanchor="right", yanchor="bottom",
                                   font=dict(size=10, color=INK2), row=r, col=c)
            fig.add_scatter(x=[ALPHA0, ALPHA0], y=[0, 990], mode="lines", line=dict(color=HI, width=1.5), showlegend=False, hoverinfo="skip", row=r, col=c)
    fig.add_annotation(x=ALPHA0, y=985, text="alpha = 0.10", showarrow=False, xanchor="left", yanchor="top", xshift=3, font=dict(size=11, color=HI), row=1, col=2)
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=GREY, width=2.8), name="critical gap inside the observed gaps (>= 100 games per 50-pt bin)")
    fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=GREY, width=2, dash="dot"), name="extrapolation beyond the data")
    fig.add_scatter(x=[None], y=[None], mode="markers", marker=dict(color=GREY, size=5), name="published value with 95% interval")
    fig.update_xaxes(range=[0, a_hi + 0.005], dtick=0.05, tickformat=".2f")
    fig.update_xaxes(title_text="alpha (upset threshold: the model gives the underdog P(win) below alpha)", row=2, col=2)
    fig.update_yaxes(range=[0, 990], dtick=200)
    fig.update_yaxes(title_text="Critical gap (Elo points)", col=1)
    tab, _ = table_critical_gaps(T)
    beyond = tab[~tab.within_data]
    n_ord, n_all, ties, a_tie, tie_cols = _ordering_on_grid(T)
    bb = sorted(set(beyond.band.astype(str)))
    beyond_txt = (f"; {len(beyond)} of the {len(tab)} critical gaps at alpha = 0.10 ({', '.join(bb)}) lie beyond the data" if len(beyond) else
                  "; all alpha = 0.10 critical gaps lie inside the data")
    ord_txt = "Classical < rapid < blitz at every alpha (point estimates)" if n_ord + ties == n_all else f"Classical < rapid < blitz at {n_ord} of {n_all} grid points"
    style(fig, 640, title=ord_txt + beyond_txt)
    mr10 = misfit_ratios(T, (0.10,)).set_index(["model", "band"]).ratio
    cmi = T["critical_gap_main_vs_interaction"]
    c05 = cmi[np.isclose(cmi.alpha, 0.05)]
    ch05, in05 = c05.change.dropna(), int(c05.inside_main_interval.astype(bool).sum())
    cap_txt = (f"Critical gap (smallest gap at which the model's P(underdog wins) falls below alpha) for alpha from {a_lo:.2f} to {a_hi:.2f}, at each band's reference "
               f"average rating, with the published 95% intervals at the four reported alphas. On the 0.005-step grid from 0.05 to 0.30 the ordering classical < rapid < blitz "
               f"holds at {n_ord} of {n_all} points; the {ties} exceptions are {tie_cols}-underdog ties at gap 0 for alpha of {a_tie:.3f} or more (even an equal-rated "
               f"{tie_cols} player has P(win) below alpha there, so classical and rapid both give 0). At alpha = 0.10, {len(beyond)} of {len(tab)} critical gaps lie beyond the observed gaps of their slice"
               + (f" ({'; '.join(f'{r.tc_class}, {r.underdog_color}, {r.band}: {r.critical_gap:.0f} vs data to {r.support_limit:.0f}' for r in beyond.itertuples())})" if len(beyond) else "") +
               ". Limitation: 'inside the data' only means that every 50-point bin up to that gap holds at least 100 games, so the tail still rests on a few hundred games "
               "per bin; beyond a gap of 900 the model is flat by construction, so the smallest alphas have no critical gap at all. "
               f"Two model caveats: strong underdogs (2500+) win only {mr10.loc[('main', '2500+')]:.2f} times the expected number in the model's tail at alpha = 0.10 "
               f"(interaction model: {mr10.loc[('interaction', '2500+')]:.2f}; pooled over time controls, mostly classical games), so the 2500+ critical gaps, at least in "
               f"classical, are probably too large; and at alpha = 0.05 the interaction model "
               f"gives critical gaps {-ch05.mean():.0f} points smaller on average ({int(in05)} of {len(ch05)} inside the main interval), so the main model's "
               f"alpha = 0.05 values should only be quoted with that caveat (fig_model_misfit).")
    return fig, cap_txt


def _ordering_on_grid(T, a_lo=0.05, a_hi=0.30):
    """(points where classical < rapid < blitz, points with all three defined, exceptions that are ties at gap 0, first alpha of a tie,
    underdog colour(s) of the ties)."""
    cga = T["critical_gap_by_alpha"]
    cga = cga[(cga.alpha >= a_lo - 1e-9) & (cga.alpha <= a_hi + 1e-9)]
    p = cga.pivot_table(index=["underdog_color", "band", "alpha"], columns="tc_class", values="critical_gap").dropna()
    ok = (p.classical < p.rapid) & (p.rapid < p.blitz)
    ties0 = (~ok) & (p.classical == 0) & (p.rapid == 0)
    assert int((~ok).sum()) == int(ties0.sum()), "ordering exception that is not a tie at gap 0"
    a_tie = float(p[ties0].index.get_level_values("alpha").min()) if ties0.any() else np.nan
    tie_cols = "/".join(c.capitalize() for c in sorted(set(p[ties0].index.get_level_values("underdog_color")))) or "no"
    return int(ok.sum()), len(p), int(ties0.sum()), a_tie, tie_cols


# ---------------------------------------------------------------------------------------------------- S2.3
def fig_upset_rule(T, alpha=ALPHA0, min_row_games=500, elo_range=(1600, 2800)):
    """Question: what is the upset rule for any pair of players, not just three reference ratings?
    Shows the model's critical gap at alpha = 0.10 as a continuous function of the average rating (line), drawn solid only
    where (a) the 100-point rating row has at least min_row_games games and (b) the critical gap lies inside the gaps
    observed in that row (grey area: gaps up to the end of the last 50-point bin with >= 100 games, from heat.csv; the
    600+ bin is open-ended); dotted elsewhere. Dots with whiskers: the published critical gaps at the band reference
    ratings. The strip below each panel counts games by average rating (25-point bins).
    Hides: the interval around the continuous line (only the published points have one)."""
    ce, cg, heat, eh = T["critical_gap_by_elo"], T["critical_gap_empirical"], T["heat"], T["avg_elo_hist"]
    ce = ce[np.isclose(ce.alpha, alpha)]
    fig = make_subplots(4, 3, shared_xaxes=True, shared_yaxes=False, row_heights=[0.37, 0.11, 0.37, 0.11], vertical_spacing=0.035,
                        horizontal_spacing=0.03, column_titles=[f"<b>{tc}</b>" for tc in TC],
                        row_titles=["White underdog", "", "Black underdog", ""])
    sup_rows = {}
    for k, col in enumerate(COLS2):
        r = 1 + 2 * k
        for c, tc in enumerate(TC, 1):
            h = heat[(heat.tc_class == tc) & (heat.underdog_color == col)]
            rows = []
            for e0 in range(elo_range[0], elo_range[1], 100):
                d = h[h.elo_lo == e0].set_index("gap_bin").n
                lim = 0
                for b in range(0, 650, 50):
                    if d.get(b, 0) >= 100:
                        lim = b + 50
                    else:
                        break
                rows.append((e0, lim, int(d.sum())))
            sr = pd.DataFrame(rows, columns=["elo_lo", "lim", "n"])
            sup_rows[(tc, col)] = sr
            xs, ys = [], []
            for e0, lim, n in rows:
                xs += [e0, e0 + 100]; ys += [lim if n >= min_row_games else 0] * 2
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color="#e4e3e0", width=0), fill="tozeroy", fillcolor="#ebeae7",
                            hoverinfo="skip", showlegend=(r == 1 and c == 1), name="gaps observed at that rating (>= 100 games per 50-pt bin)", row=r, col=c)
            d = ce[(ce.tc_class == tc) & (ce.underdog_color == col)].sort_values("avg_elo")
            row_of = ((d.avg_elo.clip(upper=elo_range[1] - 1) - elo_range[0]) // 100).astype(int)
            lim_e = sr.lim.to_numpy()[row_of]
            n_e = sr.n.to_numpy()[row_of]
            ok = (d.critical_gap <= lim_e) & (n_e >= min_row_games)
            fig.add_scatter(x=d.avg_elo, y=d.critical_gap, mode="lines", line=dict(color=INK, width=1.6, dash="dot"),
                            showlegend=(r == 1 and c == 1), name="model, outside the data", hovertemplate="avg Elo %{x}: %{y:.0f}<extra></extra>", row=r, col=c)
            fig.add_scatter(x=d.avg_elo, y=d.critical_gap.where(ok), mode="lines", line=dict(color=INK, width=3),
                            showlegend=(r == 1 and c == 1), name=f"model critical gap at alpha = {alpha:.2f}, within the data",
                            hovertemplate="avg Elo %{x}: %{y:.0f}<extra></extra>", row=r, col=c)
            p = cg[(cg.tc_class == tc) & (cg.underdog_color == col) & np.isclose(cg.alpha, alpha)]
            fig.add_scatter(x=p.ref_avg_elo, y=p.critical_gap, mode="markers", marker=dict(color=HI, size=8, line=dict(color="white", width=1)),
                            error_y=dict(type="data", symmetric=False, array=p.hi95 - p.critical_gap, arrayminus=p.critical_gap - p.lo95, color=HI, thickness=1.5, width=4),
                            showlegend=(r == 1 and c == 1), name="published value at the band reference rating (95% interval)",
                            customdata=p.band, hovertemplate="%{customdata} (ref %{x}): %{y:.0f}<extra></extra>", row=r, col=c)
            e = eh[(eh.tc_class == tc) & (eh.underdog_color == col) & (eh.elo_lo >= elo_range[0]) & (eh.elo_lo < elo_range[1])]
            fig.add_bar(x=e.elo_lo + 12.5, y=e.n / 1000, width=22, marker=dict(color=GREY, line_width=0), showlegend=False,
                        customdata=e.n, hovertemplate="avg Elo %{x:.0f}: %{customdata:,} games<extra></extra>", row=r + 1, col=c)
    fig.update_xaxes(range=list(elo_range), dtick=200)
    fig.update_xaxes(title_text="Average rating of the two players (Elo)", row=4, col=2)
    for r in (1, 3):
        fig.update_yaxes(range=[0, 760], dtick=200, row=r)
        fig.update_yaxes(title_text="Critical gap (Elo)", row=r, col=1)
    for r in (2, 4):
        fig.update_yaxes(rangemode="tozero", nticks=2, row=r)
        fig.update_yaxes(title_text="games (k)", title_font=dict(size=11), row=r, col=1)
    fig.update_yaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, col=3)
    fig.update_yaxes(matches="y", row=1)
    style(fig, 760, legend_y=-0.105, margin=dict(b=120), title=f"The upset rule as a function of rating level (alpha = {alpha:.2f}): the stronger the players, the smaller the gap that counts as an upset")
    # numbers for the caption
    eu = T["expected_upsets"]
    eb = eu[(eu.tc_class == "all") & np.isclose(eu.alpha, alpha) & (eu.band != "all")].set_index("band")
    eb = eb.obs_wins / eb.exp_wins
    ei_ = T["expected_upsets_interaction"]
    ebi = ei_[(ei_.tc_class == "all") & np.isclose(ei_.alpha, alpha) & (ei_.band != "all")].set_index("band")
    ebi = ebi.obs_wins / ebi.exp_wins
    def at(tc, col, e):
        return float(ce[(ce.tc_class == tc) & (ce.underdog_color == col) & (ce.avg_elo == e)].critical_gap.iloc[0])
    cap_txt = (f"Critical gap at alpha = {alpha:.2f} computed from the model for every average rating (line), with the three published band values (dots with 95% "
               f"intervals) and the number of games by average rating (strips). For a White underdog in classical the critical gap falls from {at('classical', 'white', 1800):.0f} "
               f"points at an average rating of 1800 to {at('classical', 'white', 2600):.0f} at 2600 (blitz: {at('blitz', 'white', 1800):.0f} to {at('blitz', 'white', 2600):.0f}). "
               f"The line is solid only where the rating row holds at least {min_row_games} games and the critical gap lies within the gaps observed at that rating "
               f"(grey area). Limitation: the rating effect is a single linear term in the model, so the smooth decline is an assumption, and the data say it is too "
               f"simple: among games the model gives P(win) < {alpha:.2f}, underdogs win {eb.loc['<2200']:.2f} times the expected number at average ratings below 2200 but "
               f"{eb.loc['2200-2499']:.2f} times at 2200-2499 and {eb.loc['2500+']:.2f} times at 2500+, so the line is probably too high for strong players. "
               f"A model with rating x gap terms (interaction model) brings 2200-2499 to {ebi.loc['2200-2499']:.2f} but leaves 2500+ at {ebi.loc['2500+']:.2f}, "
               f"so this strong-player misfit remains a limitation (fig_model_misfit); the open-ended 600+ bin of the heat table caps the grey area at 650.")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- S2.4 tables
def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def table_critical_gaps(T, alpha=ALPHA0):
    """Critical gap by time control x underdog colour x band at alpha with its 95% interval and a data-support flag
    (within_data: the critical gap is <= the end of the last 50-point gap bin with >= 100 games in that slice).
    Returns (DataFrame, markdown)."""
    cg = T["critical_gap_empirical"]
    d = cg[np.isclose(cg.alpha, alpha)].merge(T["data_support"], on=["tc_class", "underdog_color", "band"], how="left")
    d["within_data"] = d.critical_gap <= d.support_limit
    d["hi95_within_data"] = d.hi95 <= d.support_limit
    d["tc_class"] = pd.Categorical(d.tc_class, TC)
    d["band"] = pd.Categorical(d.band, BANDS)
    d["underdog_color"] = pd.Categorical(d.underdog_color, COLS2)
    d = d.sort_values(["underdog_color", "band", "tc_class"]).reset_index(drop=True)
    out = d[["underdog_color", "band", "tc_class", "ref_avg_elo", "critical_gap", "lo95", "hi95", "support_limit", "within_data", "hi95_within_data"]]
    show = pd.DataFrame({"underdog": out.underdog_color.astype(str), "band (ref Elo)": [f"{b} ({e})" for b, e in zip(out.band, out.ref_avg_elo)],
                         "time control": out.tc_class.astype(str),
                         "critical gap (95% CI)": [f"{g:.0f} ({l:.0f}-{h:.0f})" for g, l, h in zip(out.critical_gap, out.lo95, out.hi95)],
                         "data reach gap": out.support_limit.astype(int),
                         "data support": np.where(out.within_data, np.where(out.hi95_within_data, "inside", "inside (upper CI beyond)"), "BEYOND the data")})
    return out, _md(show)


def table_differences(T, alpha=ALPHA0):
    """Time-control differences of the critical gap at alpha (published critical_gap_diff_empirical.csv) next to the
    plain difference of the two published point estimates. The published 'diff' is the MEDIAN of 1000 parameter-draw
    differences (notebook 03), not point(a) - point(b), so the two can differ by a grid step or more.
    Returns (DataFrame, markdown)."""
    df, cg = T["critical_gap_diff_empirical"], T["critical_gap_empirical"]
    df = df[np.isclose(df.alpha, alpha)].copy()
    pt = cg[np.isclose(cg.alpha, alpha)].set_index(["tc_class", "underdog_color", "band"]).critical_gap
    df["a"], df["b"] = df.pair.str.split(" - ").str[0], df.pair.str.split(" - ").str[1]
    df["point_diff"] = [pt[(a, c, b)] - pt[(bb, c, b)] for a, bb, c, b in zip(df.a, df.b, df.underdog_color, df.band)]
    df["excludes_0"] = (df.hi95 < 0) | (df.lo95 > 0)
    df["band"] = pd.Categorical(df.band, BANDS)
    df["underdog_color"] = pd.Categorical(df.underdog_color, COLS2)
    df = df.sort_values(["underdog_color", "band"], kind="stable").reset_index(drop=True)
    out = df[["underdog_color", "band", "pair", "diff", "lo95", "hi95", "point_diff", "excludes_0"]]
    show = pd.DataFrame({"underdog": out.underdog_color.astype(str), "band": out.band.astype(str), "pair": out.pair,
                         "difference (median of draws, 95% interval)": [f"{d:.0f} ({l:.0f} to {h:.0f})" for d, l, h in zip(out["diff"], out.lo95, out.hi95)],
                         "point(a) - point(b)": out.point_diff.map(lambda v: f"{v:.0f}"),
                         "CI excludes 0": np.where(out.excludes_0, "yes", "NO")})
    return out, _md(show)


def table_robustness(T, alpha=ALPHA0, band="2200-2499"):
    """Critical gap at alpha (band reference rating) in the main run, the verified-ratings run, the sensitivity run, the
    model-free isotonic cross-check (pooled over ALL bands, so not strictly comparable) and the interaction model
    (rating x gap terms), with the spread (max - min) per time control x colour over all five and over the four model runs.
    Returns (DataFrame, markdown)."""
    def pick(name, col_name):
        t = T[name]
        t = t[np.isclose(t.alpha, alpha)]
        if "band" in t:
            t = t[t.band == band]
        v = "critical_gap" if "critical_gap" in t else "critical_gap_isotonic"
        return t.set_index(["tc_class", "underdog_color"])[v].rename(col_name)
    runs = ["main", "verified ratings", "sensitivity run", "isotonic (all bands)", "interaction model"]
    parts = [pick("critical_gap_empirical", runs[0]), pick("critical_gap_empirical_verified", runs[1]),
             pick("critical_gap_empirical_sens", runs[2]), pick("critical_gap_isotonic", runs[3]),
             pick("critical_gap_empirical_interaction", runs[4])]
    d = pd.concat(parts, axis=1).reset_index()
    models = [r for r in runs if not r.startswith("isotonic")]
    d["spread (all 5)"] = d[runs].max(axis=1) - d[runs].min(axis=1)
    d["spread (4 model runs)"] = d[models].max(axis=1) - d[models].min(axis=1)
    d["tc_class"] = pd.Categorical(d.tc_class, TC)
    d["underdog_color"] = pd.Categorical(d.underdog_color, COLS2)
    d = d.sort_values(["underdog_color", "tc_class"]).reset_index(drop=True)
    show = d.copy()
    for c in show.columns[2:]:
        show[c] = show[c].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    show = show.rename(columns={"tc_class": "time control", "underdog_color": "underdog"})
    show["underdog"], show["time control"] = show["underdog"].astype(str), show["time control"].astype(str)
    return d, _md(show)


def misfit_ratios(T, alphas=(0.10, 0.05)):
    """Observed / expected underdog WINS among the games the model gives P(win) < alpha, pooled over time controls, by rating
    band, for the main and the interaction model, with an approximate 95% interval (ratio +- 1.96 sd/expected; sd = binomial-sum
    standard deviation, which ignores parameter uncertainty and tournament clustering, so the interval is too narrow)."""
    rows = []
    for model, key in (("main", "expected_upsets"), ("interaction", "expected_upsets_interaction")):
        e = T[key]
        for al in alphas:
            for b in BANDS + ["all"]:
                r = e[(e.tc_class == "all") & (e.band == b) & np.isclose(e.alpha, al)].iloc[0]
                ratio = r.obs_wins / r.exp_wins
                h = 1.959964 * r.sd_wins / r.exp_wins
                rows.append(dict(model=model, alpha=al, band=b, games=int(r.games_pwin_below), exp_wins=r.exp_wins, obs_wins=int(r.obs_wins),
                                 ratio=ratio, lo95=ratio - h, hi95=ratio + h))
    return pd.DataFrame(rows)


def table_model_misfit(T, alphas=(0.10, 0.05)):
    """Expected vs observed underdog wins in the model's tail by rating band, main vs interaction model, plus the model-vs-data
    fit in 50-point bins by band (model_fit_by_band.csv of notebook 03). Returns (DataFrame, markdown, note)."""
    r = misfit_ratios(T, alphas)
    show = pd.DataFrame({"alpha": r.alpha.map(lambda v: f"{v:.2f}"), "band": r.band, "model": r.model,
                         "games with P(win) < alpha": r.games.map(lambda v: f"{v:,}"),
                         "underdog wins: expected vs observed": [f"{a:,.0f} vs {b:,}" for a, b in zip(r.exp_wins, r.obs_wins)],
                         "observed / expected (approx. 95%)": [f"{v:.2f} ({l:.2f}-{h:.2f})" for v, l, h in zip(r.ratio, r.lo95, r.hi95)]})
    order = {**{b: i for i, b in enumerate(BANDS + ["all"])}, "main": 0, "interaction": 1}
    show = show.sort_values(["alpha", "band", "model"], ascending=[False, True, True], key=lambda c: c.map(order) if c.name in ("band", "model") else c)
    fb = T["model_fit_by_band"].copy()
    fb["tc_class"] = pd.Categorical(fb.tc_class, TC)
    fb["band"] = pd.Categorical(fb.band, BANDS)
    fb = fb.sort_values(["tc_class", "band", "model"], ascending=[True, True, False])
    show2 = pd.DataFrame({"time control": fb.tc_class.astype(str), "band": fb.band.astype(str), "model": fb.model,
                          "bins inside the 95% interval": [f"{int(i)}/{int(b)}" for i, b in zip(fb.inside, fb.bins)],
                          "mean error, n-weighted (pp)": fb.mean_err_wtd_pp.map(lambda v: f"{v:+.2f}")})
    m10 = r[np.isclose(r.alpha, 0.10)].set_index(["model", "band"]).ratio
    note = ("Model vs data in 50-point bins (n >= 100 games), observed - predicted:\n\n" + _md(show2) + "\n\n"
            f"Reading: in the tail (P(win) < 0.10) strong underdogs (2500+) win only {m10[('main', '2500+')]:.2f} times the number the main model "
            f"expects and {m10[('interaction', '2500+')]:.2f} times with the interaction model, so adding the rating x gap terms does not fix the "
            f"strong-player misfit; it does fix the 2200-2499 tail ({m10[('main', '2200-2499')]:.2f} -> {m10[('interaction', '2200-2499')]:.2f}). "
            "The intervals ignore tournament clustering and parameter uncertainty, so they are too narrow.")
    return r, _md(show), note


def table_alpha005_caveat(T, band=None):
    """Critical gaps of the main and the interaction model at alpha 0.20 / 0.10 / 0.05 (all bands), with the change
    (interaction - main) and whether the interaction estimate lies inside the main model's 95% interval
    (critical_gap_main_vs_interaction.csv). Returns (DataFrame, markdown, note: summary per alpha)."""
    c = T["critical_gap_main_vs_interaction"].copy()
    c = c[c.alpha.isin([0.20, 0.10, 0.05])]
    if band:
        c = c[c.band == band]
    c["tc_class"] = pd.Categorical(c.tc_class, TC)
    c["band"] = pd.Categorical(c.band, BANDS)
    c["underdog_color"] = pd.Categorical(c.underdog_color, COLS2)
    c = c.sort_values(["alpha", "underdog_color", "band", "tc_class"], ascending=[False, True, True, True]).reset_index(drop=True)
    f = lambda v: "never" if pd.isna(v) else f"{v:.0f}"
    show = pd.DataFrame({"alpha": c.alpha.map(lambda v: f"{v:.2f}"), "underdog": c.underdog_color.astype(str), "band": c.band.astype(str),
                         "time control": c.tc_class.astype(str),
                         "main (95% CI)": [f"{f(g)} ({f(l)}-{f(h)})" for g, l, h in zip(c.critical_gap_main, c.lo95_main, c.hi95_main)],
                         "interaction (95% CI)": [f"{f(g)} ({f(l)}-{f(h)})" for g, l, h in zip(c.critical_gap_inter, c.lo95_inter, c.hi95_inter)],
                         "change": c.change.map(lambda v: "-" if pd.isna(v) else f"{v:+.0f}"),
                         "inside main CI": np.where(c.inside_main_interval.astype(bool), "yes", "NO")})
    summ = c.groupby("alpha").agg(cells=("change", "size"), min_change=("change", "min"), max_change=("change", "max"),
                                  mean_change=("change", "mean"), inside=("inside_main_interval", "sum")).sort_index(ascending=False)
    note = "Change interaction - main by alpha (all bands): " + "; ".join(
        f"alpha {a:.2f}: {r.min_change:+.0f} to {r.max_change:+.0f} (mean {r.mean_change:+.0f}), {int(r.inside)} of {int(r.cells)} inside the main 95% interval"
        for a, r in summ.iterrows()) + ("."
        " Because the interaction model's tail is better calibrated at alpha 0.05 (see table_model_misfit), the main model's alpha 0.05 "
        "critical gaps are probably too large; quote them only with this caveat.")
    return c, _md(show), note


# ---------------------------------------------------------------------------------------------------- S2.7
MODEL_STYLE = {"main": dict(color=INK, symbol="circle", dash="solid", label="main model"),
               "interaction": dict(color="#8a8987", symbol="diamond-open", dash="dash", label="interaction model (rating x gap)")}


def fig_model_misfit(T):
    """Question: is the model's tail right for strong players, and does adding a rating x gap interaction fix it?
    Shows (left) observed / expected underdog wins among the games the model gives P(win) < alpha, by rating band, at
    alpha 0.10 and 0.05, for the main and the interaction model (1 = calibrated; approximate 95% intervals); (right) the
    change of every critical gap when the interaction model replaces the main model (interaction - main, all 18 cells
    per alpha) at alpha 0.20 / 0.10 / 0.05, with the cells whose interaction value lies outside the main 95% interval marked.
    Hides: which time control a band's ratio comes from (pooled; see expected_upsets*.csv) and the clustering of games."""
    r = misfit_ratios(T)
    c, _, _ = table_alpha005_caveat(T)
    fig = make_subplots(1, 2, column_widths=[0.55, 0.45], horizontal_spacing=0.12,
                        subplot_titles=["<b>(a)</b> observed / expected underdog wins in the tail (P(win) < alpha)",
                                        "<b>(b)</b> critical gap, interaction - main (Elo points)"])
    cats = [(al, b) for al in (0.10, 0.05) for b in BANDS]
    xlab = {(al, b): i + (0.6 if al == 0.05 else 0) for i, (al, b) in enumerate(cats)}
    for model, st in MODEL_STYLE.items():
        d = r[(r.model == model) & (r.band != "all")]
        off = -0.13 if model == "main" else 0.13
        x = [xlab[(al, b)] + off for al, b in zip(d.alpha.round(2), d.band)]
        fig.add_scatter(x=x, y=d.ratio, mode="markers", name=st["label"], marker=dict(symbol=st["symbol"], size=10, color=st["color"], line=dict(color=st["color"], width=2)),
                        error_y=dict(type="data", symmetric=False, array=d.hi95 - d.ratio, arrayminus=d.ratio - d.lo95, color=st["color"], thickness=1.4, width=0),
                        customdata=np.c_[d.games, d.exp_wins, d.obs_wins, d.alpha],
                        hovertemplate=f"{model}, alpha %{{customdata[3]:.2f}}: %{{y:.2f}} (expected %{{customdata[1]:,.0f}}, observed %{{customdata[2]:,}}, "
                                      f"%{{customdata[0]:,}} games)<extra></extra>", row=1, col=1)
    fig.add_scatter(x=[-0.5, max(xlab.values()) + 0.5], y=[1, 1], mode="lines", line=dict(color=HI, width=1.5), showlegend=False, hoverinfo="skip", row=1, col=1)
    xsep = (xlab[(0.10, "2500+")] + xlab[(0.05, "<2200")]) / 2
    fig.add_annotation(x=xsep, y=1, text="calibrated", showarrow=False, xanchor="center", yanchor="bottom", bgcolor="white",
                       font=dict(size=11, color=HI), row=1, col=1)
    fig.add_scatter(x=[xsep, xsep], y=[0, 10], mode="lines", line=dict(color=LIGHT, width=1), showlegend=False, hoverinfo="skip", row=1, col=1)
    fig.update_xaxes(tickvals=list(xlab.values()), ticktext=[f"{b}" for al, b in cats], range=[-0.5, max(xlab.values()) + 0.5], row=1, col=1)
    lo = float(np.floor(min(r.lo95.min(), 0.7) * 10) / 10)
    hi = float(np.ceil(max(r.hi95.max(), 1.15) * 10) / 10) + 0.1
    for al in (0.10, 0.05):
        fig.add_annotation(x=xlab[(al, "2200-2499")], y=hi, text=f"<b>alpha = {al:.2f}</b>", showarrow=False, yanchor="top",
                           font=dict(size=12, color=INK2), row=1, col=1)
    fig.update_yaxes(range=[lo, hi], title_text="Observed / expected wins", row=1, col=1)
    # (b) change in critical gaps
    alx = {0.20: 0, 0.10: 1, 0.05: 2}
    rng = np.random.default_rng(3)
    for inside, sym, name in ((True, "circle", "inside the main model's 95% interval"), (False, "x", "outside the main model's 95% interval")):
        d = c[(c.inside_main_interval.astype(bool) == inside) & c.change.notna()]
        x = d.alpha.map(alx).to_numpy() + rng.uniform(-0.18, 0.18, len(d))
        fig.add_scatter(x=x, y=d.change, mode="markers", name=f"critical gap {name}",
                        marker=dict(symbol=sym, size=8, color=INK2 if inside else INK, opacity=0.75 if inside else 1,
                                    line=dict(width=1.5 if not inside else 0, color=INK)),
                        customdata=np.c_[d.tc_class.astype(str), d.underdog_color.astype(str), d.band.astype(str), d.critical_gap_main, d.critical_gap_inter],
                        hovertemplate="%{customdata[0]}, %{customdata[1]}, %{customdata[2]}: %{customdata[3]:.0f} -> %{customdata[4]:.0f}<extra></extra>", row=1, col=2)
    summ = c.groupby("alpha").change.mean()
    for al, x0 in alx.items():
        fig.add_scatter(x=[x0 - 0.28, x0 + 0.28], y=[summ[al]] * 2, mode="lines", line=dict(color=HI, width=3), showlegend=(al == 0.20),
                        name="mean change", hoverinfo="skip", row=1, col=2)
    fig.add_scatter(x=[-0.5, 2.5], y=[0, 0], mode="lines", line=dict(color=LIGHT, width=1.5), showlegend=False, hoverinfo="skip", row=1, col=2)
    fig.update_xaxes(tickvals=list(alx.values()), ticktext=[f"alpha = {a:.2f}" for a in alx], range=[-0.5, 2.5], row=1, col=2)
    fig.update_yaxes(title_text="Interaction - main (Elo points)", row=1, col=2)
    m10 = r[np.isclose(r.alpha, 0.10)].set_index(["model", "band"]).ratio
    m05 = r[np.isclose(r.alpha, 0.05)].set_index(["model", "band"]).ratio
    s = c.groupby("alpha").agg(mn=("change", "min"), mx=("change", "max"), mean=("change", "mean"), inside=("inside_main_interval", "sum"), n=("change", "size"))
    fixes_strong = m10[("interaction", "2500+")] > 0.9
    style(fig, 470, legend_y=-0.24, margin=dict(b=130),
          title=("The interaction model " + ("fixes" if fixes_strong else "does not fix") + f" the strong-player misfit "
                 f"(2500+: {m10[('main', '2500+')]:.2f} vs {m10[('interaction', '2500+')]:.2f} at alpha 0.10), and at alpha 0.05 it moves the critical gaps "
                 f"by {s.loc[0.05, 'mean']:+.0f} points on average"))
    cap_txt = (f"(a) Among the games in which the model gives the underdog P(win) < alpha, the number of underdog wins observed divided by the number "
               f"the model expects (sum of its own probabilities), pooled over time controls, by rating band. At alpha = 0.10 the main model gives "
               f"{m10[('main', '<2200')]:.2f} / {m10[('main', '2200-2499')]:.2f} / {m10[('main', '2500+')]:.2f} for <2200 / 2200-2499 / 2500+ and the "
               f"interaction model {m10[('interaction', '<2200')]:.2f} / {m10[('interaction', '2200-2499')]:.2f} / {m10[('interaction', '2500+')]:.2f}: "
               f"the extra rating x gap terms repair the 2200-2499 tail but not the 2500+ tail, where underdogs still win about "
               f"{(1 - m10[('interaction', '2500+')]) * 100:.0f}% less often than predicted; at alpha = 0.05 the pooled ratio is "
               f"{m05[('main', 'all')]:.2f} (main) vs {m05[('interaction', 'all')]:.2f} (interaction), and the 2500+ values at alpha = 0.05 rest on only "
               f"{int(r[(r.model == 'main') & np.isclose(r.alpha, 0.05) & (r.band == '2500+')].games.iloc[0]):,} games (main). "
               f"(b) Change of each critical gap (interaction - main, 18 cells per alpha): alpha 0.20 {s.loc[0.20, 'mn']:+.0f} to {s.loc[0.20, 'mx']:+.0f}, "
               f"0.10 {s.loc[0.10, 'mn']:+.0f} to {s.loc[0.10, 'mx']:+.0f} (mean {s.loc[0.10, 'mean']:+.0f}; {int(s.loc[0.10, 'inside'])} of {int(s.loc[0.10, 'n'])} inside the main interval), "
               f"0.05 {s.loc[0.05, 'mn']:+.0f} to {s.loc[0.05, 'mx']:+.0f} (mean {s.loc[0.05, 'mean']:+.0f}; only {int(s.loc[0.05, 'inside'])} of {int(s.loc[0.05, 'n'])} inside). "
               f"So the alpha = 0.10 results are robust to the model form, but the main model's alpha = 0.05 critical gaps are probably too large and should be "
               f"quoted with this caveat. Limitation: the intervals in (a) ignore tournament clustering and parameter uncertainty (too narrow); "
               f"the cause of the 2500+ misfit is untested (the 2500+ band mixes 2500 and 2700+ players; elite events have many short draws).")
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- S2.5
def calibration_stats(T, min_n=MIN_N):
    c = T["calibration_empirical"].copy()
    c["err"] = (c.obs_win - c.pred_win) * 100
    w = float(np.average(c.err.abs(), weights=c.n))
    u = float(c.err.abs().mean())
    big = c.loc[c.err.abs().idxmax()]
    return dict(mae_wtd_pp=w, mae_unwtd_pp=u, max_pp=float(abs(big.err)), max_signed_pp=float(big.err), max_n=int(big.n), max_tc=big.tc_class,
                max_col=big.underdog_color, max_gap=int(big.gap100), bins=len(c), bins_small=int((c.n < min_n).sum()),
                mean_signed_wtd_pp=float(np.average(c.err, weights=c.n)))


def fig_calibration_check(T, min_n=MIN_N):
    """Question: on tournaments the model never saw, do predicted and observed underdog win rates agree?
    Shows the held-out calibration bins (time control x underdog colour x 100-point gap bin, the last pooling 700+):
    top, observed vs predicted win rate with the diagonal of perfect calibration; bottom, the error observed - predicted
    (percentage points) by gap with Wilson 95% intervals. Colour by marker shape / grey level; bins under min_n games grey.
    Hides: the not-lose model (also in calibration_empirical.csv) and the rating level within a bin."""
    c = T["calibration_empirical"].copy()
    c["wins"] = np.round(c.obs_win * c.n)
    lo, hi = _wilson(c.wins, c.n)
    c["lo"], c["hi"] = lo, hi
    c["err"] = (c.obs_win - c.pred_win) * 100
    s = calibration_stats(T, min_n)
    fig = make_subplots(2, 3, row_heights=[0.55, 0.45], vertical_spacing=0.19, horizontal_spacing=0.04,
                        subplot_titles=[f"<b>{tc}</b>: observed vs predicted" for tc in TC] + [f"{tc}: error by gap" for tc in TC])
    top = float(np.ceil(max(c.obs_win.max(), c.pred_win.max()) * 20) / 20) * 100
    for i, tc in enumerate(TC, 1):
        fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=LIGHT, width=2), hoverinfo="skip", showlegend=False, row=1, col=i)
        fig.add_scatter(x=[-30, 760], y=[0, 0], mode="lines", line=dict(color=LIGHT, width=2), hoverinfo="skip", showlegend=False, row=2, col=i)
        for col in COLS2:
            st_ = UC_STYLE[col]
            d = c[(c.tc_class == tc) & (c.underdog_color == col)].sort_values("gap100")
            mcol = np.where(d.n >= min_n, st_["color"], SPARSE)
            fig.add_scatter(x=d.pred_win * 100, y=d.obs_win * 100, mode="markers", name=st_["label"], showlegend=(i == 1),
                            marker=dict(symbol=st_["symbol"], size=np.clip(np.sqrt(d.n) / 12, 5, 16), color=mcol, line=dict(color=mcol, width=1.5)),
                            customdata=np.c_[d.gap100, d.n], hovertemplate="gap %{customdata[0]}+: predicted %{x:.1f}%, observed %{y:.1f}% (n=%{customdata[1]:,})<extra></extra>",
                            row=1, col=i)
            off = -12 if col == "white" else 12
            fig.add_scatter(x=d.gap100 + 50 + off, y=d.err, mode="markers", showlegend=False,
                            marker=dict(symbol=st_["symbol"], size=6.5, color=mcol, line=dict(color=mcol, width=1.5)),
                            error_y=dict(type="data", symmetric=False, array=(d.hi - d.obs_win) * 100, arrayminus=(d.obs_win - d.lo) * 100,
                                         color=st_["color"], thickness=1, width=0),
                            customdata=d.n, hovertemplate="%{y:+.2f} pp (n=%{customdata:,})<extra></extra>", row=2, col=i)
    big = c.loc[c.err.abs().idxmax()]
    bi = TC.index(big.tc_class) + 1
    fig.add_annotation(x=big.gap100 + 50 + (12 if big.underdog_color == "black" else -12), y=big.err, text=f"largest: {big.err:+.1f} pp, n={int(big.n)}",
                       showarrow=True, arrowhead=0, ax=-75, ay=28, font=dict(size=11, color=HI), arrowcolor=HI, row=2, col=bi)
    fig.add_annotation(x=top, y=top, text="perfect calibration", showarrow=False, xanchor="right", yanchor="top", textangle=0, yshift=-4,
                       font=dict(size=11, color=GREY), row=1, col=1)
    fig.update_xaxes(range=[0, top], ticksuffix="%", row=1)
    fig.update_yaxes(range=[0, top], ticksuffix="%", row=1)
    fig.update_xaxes(title_text="Predicted underdog win rate (model fitted on the other tournaments)", row=1, col=2)
    fig.update_yaxes(title_text="Observed", row=1, col=1)
    fig.update_xaxes(range=[-10, 810], tickvals=[50, 150, 250, 350, 450, 550, 650, 750],
                     ticktext=["0", "100", "200", "300", "400", "500", "600", "700+"], row=2)
    fig.update_xaxes(title_text="Rating gap bin (lower edge, Elo points)", row=2, col=2)
    yl = max(abs(c.lo * 100 - c.pred_win * 100).max(), abs(c.hi * 100 - c.pred_win * 100).max())
    yl = min(yl * 1.05, 5)
    fig.update_yaxes(range=[-yl, yl], ticksuffix=" pp", row=2)
    fig.update_yaxes(title_text="Observed - predicted", row=2, col=1)
    fast = c[c.tc_class.isin(["rapid", "blitz"]) & c.gap100.between(200, 500)]
    fast_txt = ("fast-chess underdogs win slightly less than predicted at mid gaps" if fast.err.mean() < 0 and (fast.err < 0).mean() > 0.75
                else f"rapid/blitz at mid gaps: mean error {fast.err.mean():+.1f} pp")
    style(fig, 640, legend_y=-0.12, margin=dict(b=70), title=f"On unseen tournaments the model is off by {s['mae_wtd_pp']:.2f} points on average (n-weighted); {fast_txt}")
    cap_txt = (f"Held-out check: the model was refitted on half of the tournaments and evaluated on the other half ({int(c.n.sum()):,} games in {s['bins']} bins of "
               f"time control x underdog colour x 100-point gap, the last pooling 700+). Mean absolute error {s['mae_wtd_pp']:.2f} percentage points n-weighted and "
               f"{s['mae_unwtd_pp']:.2f} unweighted; the largest bin error is {s['max_signed_pp']:+.1f} pp ({s['max_tc']}, {s['max_col']} underdog, gap {s['max_gap']}"
               f"{'+' if s['max_gap'] == 700 else '-' + str(s['max_gap'] + 99)}, n = {s['max_n']}). In rapid and blitz at gaps 200-599 the observed rate is below the prediction "
               f"in {int((fast.err < 0).sum())} of {len(fast)} bins (mean {fast.err.mean():+.1f} pp), so on new tournaments the fast-chess curves are, if anything, slightly too "
               f"generous to the underdog. Limitation: one random split of tournaments; bins pool all rating levels, which hides the strong-player misfit "
               f"(fig_model_misfit)"
               + (f", and {s['bins_small']} bins have fewer than {min_n} games (grey)." if s['bins_small'] else f"; every bin has at least {min_n} games, so none is greyed."))
    return fig, cap_txt


# ---------------------------------------------------------------------------------------------------- S2.6
def table_expected_upsets(T, by_band=False, alphas=(0.20, 0.10, 0.05)):
    """Expected number of upsets if the model were calibrated (sum of the model's own P(win) over the games with
    P(win) < alpha, plus P(not lose) - P(win) over the games with P(not lose) < alpha for the draw upsets) vs the observed
    count. z_wins uses the binomial-sum standard deviation, which ignores parameter uncertainty and the clustering of
    games in tournaments, so it overstates significance. by_band=True splits by rating band. Returns (DataFrame, markdown)."""
    e = T["expected_upsets"]
    if "band" not in e:
        e = e.assign(band="all")
    e = e[e.alpha.isin(alphas) & (e.games_pwin_below > 0)]
    e = (e[e.band != "all"] if by_band else e[e.band == "all"]).copy()
    e["obs_minus_exp"] = e.obs_total - e.exp_total
    e["ratio"] = e.obs_total / e.exp_total
    e["ratio_wins"] = e.obs_wins / e.exp_wins
    e["z_wins"] = (e.obs_wins - e.exp_wins) / e.sd_wins
    show = pd.DataFrame({"time control": e.tc_class, **({"band": e.band} if by_band else {}), "alpha": e.alpha.map(lambda v: f"{v:.2f}"),
                         "games with P(win) < alpha": e.games_pwin_below.map(lambda v: f"{v:,}"),
                         "expected upsets": e.exp_total.map(lambda v: f"{v:,.0f}"), "observed upsets": e.obs_total.map(lambda v: f"{v:,}"),
                         "observed / expected": e.ratio.map(lambda v: f"{v:.2f}"),
                         "wins: exp vs obs (z)": [f"{a:,.0f} vs {b:,} ({z:+.1f})" for a, b, z in zip(e.exp_wins, e.obs_wins, e.z_wins)],
                         "draws: exp vs obs": [f"{a:,.0f} vs {b:,}" for a, b in zip(e.exp_draws, e.obs_draws)]})
    return e.reset_index(drop=True), _md(show)


def ordering_check(T, alphas=(0.20, 0.10, 0.05)):
    """Is classical < rapid < blitz at every published alpha, colour and band? Returns (DataFrame of cells, list of exceptions)."""
    cg = T["critical_gap_empirical"]
    p = cg[cg.alpha.isin(alphas)].pivot_table(index=["underdog_color", "band", "alpha"], columns="tc_class", values="critical_gap")
    p["holds"] = (p.classical < p.rapid) & (p.rapid < p.blitz)
    exc = [f"{i}: " + ", ".join(f"{t} {p.loc[i, t]:.0f}" for t in TC) for i in p.index[~p.holds]]
    return p.reset_index(), exc


FIGURES2 = [fig_model_vs_data, fig_verified_split, fig_critical_gap_alpha, fig_upset_rule, fig_calibration_check, fig_model_misfit]
TABLES2 = [("table_critical_gaps", table_critical_gaps), ("table_differences", table_differences),
           ("table_robustness", table_robustness), ("table_model_misfit", table_model_misfit),
           ("table_alpha005_caveat", table_alpha005_caveat), ("table_expected_upsets", table_expected_upsets),
           ("table_expected_upsets(by_band=True)", lambda T: table_expected_upsets(T, by_band=True, alphas=(0.10, 0.05)))]


def render_all2(data_dir=os.path.join(HERE, "viz_data"), proc_dir=PROC_DIR, out_dir=os.path.join(HERE, "renders"), plotlyjs="cdn"):
    """Write every figure (part 1 and part 2) as HTML (plotly.js from the CDN) + caption (<name>.md), all captions in
    captions.md and every table as markdown in tables.md. PNGs: render_png.py (needs playwright)."""
    os.makedirs(out_dir, exist_ok=True)
    T = load2(data_dir, proc_dir)
    caps = {}
    for f in FIGURES + FIGURES2:
        fig, cap = f(T)
        name = f.__name__
        fig.write_html(os.path.join(out_dir, name + ".html"), include_plotlyjs=plotlyjs)
        caps[name] = cap
        with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as fh:
            fh.write(cap + "\n")
    with open(os.path.join(out_dir, "captions.md"), "w", encoding="utf-8") as fh:
        for k, v in caps.items():
            fh.write(f"## {k}\n\n{v}\n\n")
    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as fh:
        for name, t in TABLES2:
            res = t(T)
            fh.write(f"## {name}\n\n{res[1]}\n\n" + (f"{res[2]}\n\n" if len(res) > 2 else ""))
    return caps


render_all = render_all2          # one entry point for everything


# ==================================================================================================== SHORT CAPTIONS
# The rubric asks for two to three sentences under each plot. short_captions(T) gives, per figure (and the main tables),
# at most three sentences: (1) what the plot shows / the question it answers, (2) the key finding with numbers,
# (3) the single most important caveat. Every number is computed from T with the same computations as the long
# captions (the long captions are unchanged and are returned as "details" by captions_all).
REPORT_ORDER = ["fig_games_per_year", "fig_gap_distribution", "fig_outcomes_by_gap", "fig_simpson", "fig_heatmap", "fig_set_compare",
                "fig_upset_points", "fig_model_vs_data", "fig_verified_split", "fig_critical_gap_alpha", "table_critical_gaps",
                "table_differences", "table_robustness", "fig_upset_rule", "fig_calibration_check", "table_expected_upsets",
                "fig_model_misfit", "table_model_misfit", "table_alpha005_caveat"]


def _sc_games_per_year(T):
    """Numbers of fig_games_per_year's caption: (total, share 2024+, classical share, partial year, month span, games dated later)."""
    yc, mc = T["yearly_counts"], T["monthly_counts"]
    years = sorted(yc.year.unique())
    piv = yc.pivot(index="year", columns="tc_class", values="n").reindex(years).fillna(0)
    tot = piv.sum(axis=1)
    part = set(yc.loc[yc.partial, "year"])
    py = max(part) if part else None
    months = mc[mc.period.str.startswith(str(py))].groupby("period").n.sum() if py else pd.Series(dtype=float)
    full = months[months >= 100]
    mname = lambda p: pd.Timestamp(p + "-01").strftime("%b")
    span = f"{mname(full.index.min())}-{mname(full.index.max())}" if len(full) else ""
    share24 = tot.loc[[y for y in years if y >= 2024]].sum() / tot.sum()
    return int(tot.sum()), share24, piv["classical"].sum() / tot.sum(), py, span, int(months[months < 100].sum())


def _sc_heat_cross(T, min_n=MIN_N, color="all", threshold=0.10, elo_range=(1200, 2800)):
    """fig_heatmap's step: per time control, {row elo_lo: first gap bin (n >= min_n) with win rate < threshold}; plus the share of games in shown cells."""
    h = T["heat"]
    h = h[(h.underdog_color == color) & (h.elo_lo >= elo_range[0]) & (h.elo_lo < elo_range[1])].copy()
    h["rate"] = h.wins / h.n
    ok = h[(h.n >= min_n) & (h.rate < threshold)]
    cross = {tc: ok[ok.tc_class == tc].groupby("elo_lo").gap_bin.min().astype(float).to_dict() for tc in TC}
    return cross, h[h.n >= min_n].n.sum() / h.n.sum()


def _sc_med_cross(cross, tc, lo, hi):
    v = [x for y, x in cross[tc].items() if lo <= y < hi]
    return np.median(v) if v else np.nan


def _sc_verified(T, band="2200-2499", color="white", min_n=MIN_N, x_max=700, tc="rapid", gap_bin=250):
    """fig_verified_split's numbers in one gap bin: (unverified row, all-games row, verified row, curve at the bin centre)."""
    mb, mv, fc = T["model_bins"], T["model_bins_by_verified"], T["fitted_curves"]
    def pick(d):
        d = d[(d.tc_class == tc) & (d.band == band) & (d.underdog_color == color)]
        return d[(d.n >= min_n) & (d.gap_bin < x_max)].set_index("gap_bin").loc[gap_bin]
    curve = float(fc[(fc.tc_class == tc) & (fc.underdog_color == color) & (fc.band == band) & (fc.gap == gap_bin + 25)].p_win.iloc[0])
    return pick(mv[mv.verified == False]), pick(mb), pick(mv[mv.verified == True]), curve


def _sc_cg(T, alpha=ALPHA0, band="2200-2499"):
    """Published critical gaps at alpha for one band: {(colour, tc): gap}."""
    cg = T["critical_gap_empirical"]
    d = cg[np.isclose(cg.alpha, alpha) & (cg.band == band)]
    return {(r.underdog_color, r.tc_class): r.critical_gap for r in d.itertuples()}


def _sc_cg_txt(g, col):
    return " / ".join(f"{g[(col, tc)]:.0f}" for tc in TC)


def _sc_caveat_summary(T):
    c, _, _ = table_alpha005_caveat(T)
    return c.groupby("alpha").agg(mn=("change", "min"), mx=("change", "max"), mean=("change", "mean"),
                                  inside=("inside_main_interval", "sum"), n=("change", "size"))


def _short_games_per_year(T):
    n, s24, scl, py, span, extra = _sc_games_per_year(T)
    return (f"Shows how many analysis games each calendar year contributes, stacked by time control. "
            f"{_pct(s24, 0)} of the {n:,} games are from 2024 or later, and classical makes up {_pct(scl, 0)} of all games. "
            f"Caveat: {py} is a partial year (data cover {span} {py}), and the growth reflects how many events were broadcast on Lichess, "
            f"not how much chess was played.")


def _short_gap_distribution(T):
    gs = T["gap_summary"].set_index("tc_class")
    a = gs.loc["all"]
    return (f"Shows how far apart in rating the two players are, as the share of each time control's games per 25-point gap bin. "
            f"The median gap is {gs.loc['classical', 'median']:.0f} points in classical, {gs.loc['rapid', 'median']:.0f} in rapid and "
            f"{gs.loc['blitz', 'median']:.0f} in blitz, and only {_pct(a.share_ge_400)} of the {int(a.n):,} games with an underdog have a gap of 400 or more. "
            f"Caveat: the gaps are between published FIDE ratings, which lag a player's current strength, especially for fast-improving juniors.")


def _short_outcomes_by_gap(T):
    w = _rates(T["wdl_by_gap"], ["tc_class", "gap_bin"]).set_index(["tc_class", "gap_bin"])
    return (f"Shows how often the lower-rated player wins, draws and loses as the rating gap grows, per time control. "
            f"In classical the underdog wins {_pct(w.loc[('classical', 0), 'win'])} of games in the 0-49 gap bin but only "
            f"{_pct(w.loc[('classical', 400), 'win'])} at 400-449 (blitz: {_pct(w.loc[('blitz', 0), 'win'])} vs {_pct(w.loc[('blitz', 400), 'win'])}), "
            f"and draws take up much of the slack in classical ({_pct(w.loc[('classical', 0), 'draw'])} draws at 0-49). "
            f"Caveat: the curves pool both colours and all rating levels, which hides large subgroup differences (see the Simpson figure).")


def _short_simpson(T, tc="classical"):
    w = T["wdl_by_gap"]
    w = w[w.tc_class == tc]
    sc = _rates(w[w.gap_bin.isin([50, 100, 150])], ["underdog_color"]).set_index("underdog_color")
    ne = T["near_equal"].set_index(["tc_class", "band"])
    dr = [ne.loc[(tc, b), "draw_rate"] for b in BANDS]
    return (f"Asks whether one pooled {tc} curve hides subgroups, by splitting the underdog's results by its colour and by the rating band. "
            f"At gaps of 50-199 the underdog scores {sc.loc['white', 'score']:.3f} points per game with White but {sc.loc['black', 'score']:.3f} with Black, "
            f"and between near-equal players the draw rate rises from {_pct(dr[0], 0)} (<2200) to {_pct(dr[2], 0)} (2500+). "
            f"Caveat: the band is defined by the average rating, so bands also differ in their gap mix and in who plays, and colour and band are shown one at a time.")


def _short_heatmap(T, min_n=MIN_N, threshold=0.10):
    cross, cov = _sc_heat_cross(T, min_n, threshold=threshold)
    lo, hi = _sc_med_cross(cross, "classical", 1600, 2000), _sc_med_cross(cross, "classical", 2200, 2600)
    return (f"Shows the underdog's observed win rate in each cell of rating gap x average rating, per time control, to test whether the chance "
            f"depends on the rating level as well as on the gap. "
            f"In classical the win rate first falls below {threshold:.0%} at a median gap of {lo:.0f} points for average ratings 1600-1999, "
            f"but already at {hi:.0f} for 2200-2599. "
            f"Caveat: the green step is the first crossing of a noisy observed rate, not a fitted boundary, and draws count as not winning "
            f"(cells with fewer than {min_n} games are hatched; the coloured cells hold {_pct(cov)} of the games).")


def _short_set_compare(T):
    sc, ds = T["set_compare"], T["drop_steps"].set_index("step")
    n_clean, n_an = int(ds.removed.sum() + ds.remaining.iloc[-1]), int(ds.remaining.iloc[-1])
    tit = sc[sc.dimension == "titled"].pivot(index="category", columns="set", values="share").loc["at least one FIDE title"]
    return (f"Compares the larger cleaned set ({n_clean:,} games) with the analysis set ({n_an:,}) to show whom the filters removed. "
            f"Most of the {int(ds.removed.sum()):,} dropped games lack a FIDE rating for a player "
            f"({int(ds.loc['missing Elo for at least one player', 'removed']):,}), and only {_pct(tit.dropped, 0)} of the dropped games have a FIDE-titled player, "
            f"so the share of games with a titled player rises from {_pct(tit.clean, 0)} to {_pct(tit.analysis, 0)}. "
            f"Caveat: the analysis set therefore over-represents titled, rated players, so the results describe rated tournament chess, not every broadcast game.")


def _short_upset_points(T, top_k=20):
    u = T["upset_points"]
    top = u.nsmallest(top_k, "q_emp")
    n = lambda tc: f"{(u.tc_class == tc).sum():,}"
    tt = top.tc_class.value_counts()
    cls = "all" if tt.get("classical", 0) == top_k else f"{tt.get('classical', 0)} of them"
    res = "wins" if (top.underdog_score == 1).all() else "results"
    return (f"Shows where the upsets at the 10% level (the underdog won, or drew, although the model gave it less than a 10% chance of doing so) "
            f"fall in rating gap and average rating, with the {top_k} rarest numbered and listed. "
            f"There are {len(u):,} upsets ({n('classical')} classical, {n('rapid')} rapid, {n('blitz')} blitz; {int((u.underdog_score == 0.5).sum()):,} are draws), "
            f"and the {top_k} rarest are {cls} classical {res} with gaps of {int(top.gap.min())}-{int(top.gap.max())} points. "
            f"Caveat: the plot shows where upsets occur, not how likely they are (dense regions are dense mainly because many games are played there), "
            f"and a very large gap often means an outdated rating rather than a sensational game.")


def _short_model_vs_data(T, min_n=MIN_N):
    st = model_fit_stats(T, min_n).set_index(["tc_class", "band", "underdog_color"])
    both = st.xs("both", level="underdog_color")
    tot_in, tot_b = int(both.inside.sum()), int(both.bins.sum())
    w25 = st.loc[("classical", "2500+", "both")]
    fbi = T["model_fit_by_band"].set_index(["model", "tc_class", "band"]).loc[("interaction", "classical", "2500+")]
    return (f"Checks bin by bin whether the model's predicted underdog win rate matches the observed rate, by rating band and time control. "
            f"The prediction lies inside the observed 95% interval in {tot_in} of {tot_b} bins with at least {min_n} games ({tot_in / tot_b:.0%}; "
            f"about 95% would be expected for an exact model). "
            f"Caveat: the clearest misfit is classical 2500+, where underdogs win less often than predicted (only {int(w25.inside)} of {int(w25.bins)} bins inside, "
            f"mean error {w25.err_wtd_pp:+.1f} percentage points), and the interaction model does not remove it ({fbi.inside:.0f} of {fbi.bins:.0f} bins).")


def _short_verified_split(T, band="2200-2499", color="white"):
    unv, al, ver, curve = _sc_verified(T, band, color)
    return (f"Explains why the old notebook plot showed observed win rates far above the model curve ({color.capitalize()} underdog, {band}): "
            f"its dots used only the games whose ratings could not be verified against the FIDE list. "
            f"In rapid at gap 250-299 those games show {_pct(unv.wins / unv.n)} (n={int(unv.n):,}), whereas all games show {_pct(al.wins / al.n)} "
            f"and the verified games {_pct(ver.wins / ver.n)}, against {_pct(curve)} for the curve. "
            f"Caveat: the grey 95% band reflects only the model's parameter uncertainty, so bins of a few hundred games often lie outside it even when the model is right.")


def _short_critical_gap_alpha(T):
    n_ord, n_all, ties, a_tie, tie_cols = _ordering_on_grid(T)
    g = _sc_cg(T)
    s = _sc_caveat_summary(T)
    exc = (f" (the {ties} exceptions are {tie_cols}-underdog ties at gap 0 for alpha of {a_tie:.3f} or more)" if ties else "")
    return (f"Shows the critical gap (the smallest gap at which the model gives the underdog P(win) below alpha) for every alpha from 0.01 to 0.30, "
            f"to test whether the ordering classical < rapid < blitz depends on the choice of alpha. "
            f"The ordering holds at {n_ord} of {n_all} grid points from 0.05 to 0.30{exc}; at alpha = 0.10 and average rating 2200-2499 the critical gaps "
            f"are {_sc_cg_txt(g, 'white')} points (classical / rapid / blitz) for a White and {_sc_cg_txt(g, 'black')} for a Black underdog. "
            f"Caveat: at alpha = 0.05 the interaction model gives critical gaps {-s.loc[0.05, 'mean']:.0f} points smaller on average "
            f"(only {int(s.loc[0.05, 'inside'])} of {int(s.loc[0.05, 'n'])} inside the main interval), so the main model's alpha = 0.05 values should only be quoted with that caveat.")


def _short_table_critical_gaps(T, alpha=ALPHA0):
    tab, _ = table_critical_gaps(T, alpha)
    g = _sc_cg(T, alpha)
    beyond = tab[~tab.within_data]
    where = "; ".join(f"{r.tc_class}, {r.underdog_color}, {r.band}" for r in beyond.itertuples())
    return (f"Lists the critical gap at alpha = {alpha:.2f} for every time control, underdog colour and rating band, with its 95% interval and "
            f"whether it lies inside the observed data. "
            f"In the 2200-2499 band an underdog's win counts as an upset from a gap of {_sc_cg_txt(g, 'white')} points (classical / rapid / blitz) with White "
            f"and {_sc_cg_txt(g, 'black')} with Black. "
            + (f"Caveat: {len(beyond)} of the {len(tab)} values ({where}) lie beyond the gaps observed in their slice, so they are extrapolations."
               if len(beyond) else f"Caveat: all {len(tab)} values lie inside the observed data, but the tail rests on a few hundred games per bin."))


def _short_table_differences(T, alpha=ALPHA0):
    d, _ = table_differences(T, alpha)
    k = d.set_index(["underdog_color", "band", "pair"])
    r = k.loc[("white", "2200-2499", "classical - blitz")]
    rng = d["diff"].abs()
    gap = (d["diff"] - d.point_diff).abs().max()
    return (f"Gives the difference of the critical gap between time controls at alpha = {alpha:.2f} (median of 1,000 parameter draws, with 95% interval), "
            f"per underdog colour and rating band. "
            f"All {len(d)} intervals exclude 0, with differences from {rng.min():.0f} to {rng.max():.0f} points in size; "
            f"for a White underdog at 2200-2499, classical minus blitz is {r['diff']:.0f} ({r.lo95:.0f} to {r.hi95:.0f}). "
            f"Caveat: the published difference is the median of the per-draw differences, so it can differ from the difference of the two point estimates "
            f"(here by up to {gap:.0f} points).") if int(d.excludes_0.sum()) == len(d) else (
            f"Gives the difference of the critical gap between time controls at alpha = {alpha:.2f} (median of 1,000 parameter draws, with 95% interval). "
            f"{int(d.excludes_0.sum())} of {len(d)} intervals exclude 0; for a White underdog at 2200-2499, classical minus blitz is {r['diff']:.0f} "
            f"({r.lo95:.0f} to {r.hi95:.0f}). "
            f"Caveat: the published difference is the median of the per-draw differences and can differ from the difference of the point estimates by up to {gap:.0f} points.")


def _short_table_robustness(T, alpha=ALPHA0, band="2200-2499"):
    d, _ = table_robustness(T, alpha, band)
    return (f"Compares the critical gaps at alpha = {alpha:.2f} (band {band}) across the main run, the verified-ratings run, the sensitivity run, "
            f"a model-free isotonic cross-check and the interaction model. "
            f"Across the four model runs each critical gap moves by at most {d['spread (4 model runs)'].max():.0f} points "
            f"(classical: {d.loc[d.tc_class == 'classical', 'spread (4 model runs)'].max():.0f}), and the ordering classical < rapid < blitz is the same in every run. "
            f"Caveat: the isotonic values are pooled over all rating bands, so they are not strictly comparable (with them the spread reaches "
            f"{d['spread (all 5)'].max():.0f} points).") if _sc_ordering_all_runs(d) else (
            f"Compares the critical gaps at alpha = {alpha:.2f} (band {band}) across the main, verified-ratings, sensitivity, isotonic and interaction runs. "
            f"Across the four model runs each critical gap moves by at most {d['spread (4 model runs)'].max():.0f} points. "
            f"Caveat: the isotonic values are pooled over all rating bands (spread with them up to {d['spread (all 5)'].max():.0f} points).")


def _sc_ordering_all_runs(d):
    runs = ["main", "verified ratings", "sensitivity run", "isotonic (all bands)", "interaction model"]
    for col in COLS2:
        x = d[d.underdog_color == col].set_index("tc_class")
        for r in runs:
            if not (x.loc["classical", r] < x.loc["rapid", r] < x.loc["blitz", r]):
                return False
    return True


def _short_upset_rule(T, alpha=ALPHA0):
    ce = T["critical_gap_by_elo"]
    ce = ce[np.isclose(ce.alpha, alpha)]
    at = lambda tc, e: float(ce[(ce.tc_class == tc) & (ce.underdog_color == "white") & (ce.avg_elo == e)].critical_gap.iloc[0])
    m = misfit_ratios(T, (alpha,)).set_index(["model", "band"]).ratio
    return (f"Shows the critical gap at alpha = {alpha:.2f} as a continuous function of the average rating, per time control and underdog colour, "
            f"i.e. the upset rule for any pair of players. "
            f"For a White underdog in classical it falls from {at('classical', 1800):.0f} points at an average rating of 1800 to {at('classical', 2600):.0f} at 2600 "
            f"(blitz: {at('blitz', 1800):.0f} to {at('blitz', 2600):.0f}). "
            f"Caveat: the decline comes from a single linear rating term, and in the model's tail 2500+ underdogs win only {m.loc[('main', '2500+')]:.2f} times "
            f"the expected number ({m.loc[('interaction', '2500+')]:.2f} with the interaction model), so the line is probably too high for strong players.")


def _short_calibration_check(T):
    s = calibration_stats(T)
    n = int(T["calibration_empirical"].n.sum())
    g = f"{s['max_gap']}+" if s["max_gap"] == 700 else f"{s['max_gap']}-{s['max_gap'] + 99}"
    return (f"Tests on tournaments the model never saw (a held-out half of the tournaments, {n:,} games) whether predicted and observed underdog win rates agree. "
            f"The mean absolute error is {s['mae_wtd_pp']:.2f} percentage points (n-weighted, over {s['bins']} bins), and the largest bin error is "
            f"{s['max_signed_pp']:+.1f} percentage points ({s['max_tc']}, {s['max_col']} underdog, gap {g}, n = {s['max_n']}). "
            f"Caveat: the bins pool all rating levels, which hides the strong-player misfit (fig_model_misfit), and this is a single random split.")


def _short_table_expected_upsets(T, alpha=ALPHA0):
    e, _ = table_expected_upsets(T, alphas=(alpha, 0.05))
    a = e[(e.tc_class == "all") & np.isclose(e.alpha, alpha)].iloc[0]
    b = e[(e.tc_class == "all") & np.isclose(e.alpha, 0.05)].iloc[0]
    return (f"Compares the number of upsets a calibrated model would expect (the sum of its own probabilities) with the number observed. "
            f"At alpha = {alpha:.2f} the model expects {a.exp_total:,.0f} upsets and {int(a.obs_total):,} are observed (ratio {a.ratio:.2f}); "
            f"at alpha = 0.05 it is {b.exp_total:,.0f} vs {int(b.obs_total):,} ({b.ratio:.2f}). "
            f"Caveat: the z-values use a binomial standard deviation that ignores tournament clustering and parameter uncertainty, so they overstate significance.")


def _short_model_misfit(T):
    m = misfit_ratios(T, (0.10,)).set_index(["model", "band"]).ratio
    s = _sc_caveat_summary(T)
    return (f"Asks whether the model's tail is right for strong players, and whether adding rating x gap terms (the interaction model) fixes it "
            f"and moves the critical gaps. "
            f"Among games the model gives P(win) < 0.10, underdogs at 2500+ win only {m.loc[('main', '2500+')]:.2f} times the expected number "
            f"(interaction model {m.loc[('interaction', '2500+')]:.2f}), while the 2200-2499 tail is repaired ({m.loc[('main', '2200-2499')]:.2f} to "
            f"{m.loc[('interaction', '2200-2499')]:.2f}). "
            f"Caveat: at alpha = 0.05 the interaction model makes the critical gaps {-s.loc[0.05, 'mean']:.0f} points smaller on average (only "
            f"{int(s.loc[0.05, 'inside'])} of {int(s.loc[0.05, 'n'])} inside the main interval, vs {int(s.loc[0.10, 'inside'])} of {int(s.loc[0.10, 'n'])} at 0.10), "
            f"so the main model's alpha = 0.05 critical gaps are probably too large.")


def _short_table_model_misfit(T):
    r = misfit_ratios(T, (0.10,))
    x = r[(r.model == "main") & (r.band == "2500+")].iloc[0]
    xi = r[(r.model == "interaction") & (r.band == "2500+")].iloc[0]
    return (f"Lists, by rating band, the underdog wins expected and observed among the games each model gives P(win) < alpha, plus the bin-by-bin fit by band. "
            f"At alpha = 0.10 and 2500+ the main model expects {x.exp_wins:,.0f} wins and {x.obs_wins:,} are observed (ratio {x.ratio:.2f}); "
            f"the interaction model gives {xi.ratio:.2f}. "
            f"Caveat: the approximate 95% intervals ignore tournament clustering and parameter uncertainty, so they are too narrow.")


def _short_table_alpha005_caveat(T):
    s = _sc_caveat_summary(T)
    f = lambda a: (f"{s.loc[a, 'mn']:+.0f} to {s.loc[a, 'mx']:+.0f} (mean {s.loc[a, 'mean']:+.0f}; {int(s.loc[a, 'inside'])} of {int(s.loc[a, 'n'])} "
                   f"inside the main 95% interval)")
    return (f"Lists the critical gaps of the main and the interaction model at alpha 0.20 / 0.10 / 0.05 for every cell, with the change interaction - main. "
            f"At alpha = 0.10 the changes range {f(0.10)}, at alpha = 0.05 {f(0.05)}. "
            f"Caveat: at alpha = 0.05 most interaction values fall outside the main model's interval, so the main model's alpha = 0.05 critical gaps "
            f"are probably too large and should only be quoted with this caveat (the alpha = 0.10 results are robust to the model form).")


_SHORT = {"fig_games_per_year": (_short_games_per_year, ["yearly_counts", "monthly_counts"]),
          "fig_gap_distribution": (_short_gap_distribution, ["gap_summary"]),
          "fig_outcomes_by_gap": (_short_outcomes_by_gap, ["wdl_by_gap"]),
          "fig_simpson": (_short_simpson, ["wdl_by_gap", "near_equal"]),
          "fig_heatmap": (_short_heatmap, ["heat"]),
          "fig_set_compare": (_short_set_compare, ["set_compare", "drop_steps"]),
          "fig_upset_points": (_short_upset_points, ["upset_points"]),
          "fig_model_vs_data": (_short_model_vs_data, ["model_bins", "model_fit_by_band"]),
          "fig_verified_split": (_short_verified_split, ["model_bins", "model_bins_by_verified", "fitted_curves"]),
          "fig_critical_gap_alpha": (_short_critical_gap_alpha, ["critical_gap_by_alpha", "critical_gap_empirical", "critical_gap_main_vs_interaction"]),
          "table_critical_gaps": (_short_table_critical_gaps, ["critical_gap_empirical", "data_support"]),
          "table_differences": (_short_table_differences, ["critical_gap_diff_empirical", "critical_gap_empirical"]),
          "table_robustness": (_short_table_robustness, ["critical_gap_empirical", "critical_gap_empirical_verified", "critical_gap_empirical_sens",
                                                         "critical_gap_isotonic", "critical_gap_empirical_interaction"]),
          "fig_upset_rule": (_short_upset_rule, ["critical_gap_by_elo", "expected_upsets", "expected_upsets_interaction"]),
          "fig_calibration_check": (_short_calibration_check, ["calibration_empirical"]),
          "table_expected_upsets": (_short_table_expected_upsets, ["expected_upsets"]),
          "fig_model_misfit": (_short_model_misfit, ["expected_upsets", "expected_upsets_interaction", "critical_gap_main_vs_interaction"]),
          "table_model_misfit": (_short_table_model_misfit, ["expected_upsets", "expected_upsets_interaction"]),
          "table_alpha005_caveat": (_short_table_alpha005_caveat, ["critical_gap_main_vs_interaction"])}


def short_captions(T):
    """{name: short caption} in report order (REPORT_ORDER): at most three sentences each (what the plot shows, the key finding
    with numbers, one caveat), all numbers computed from T. With T = load() only the data-layer figures are returned (the model
    figures need load2()); entries whose tables are missing from T are skipped."""
    return {k: _SHORT[k][0](T) for k in REPORT_ORDER if all(t in T for t in _SHORT[k][1])}


def _details(T, name):
    if name.startswith("fig_"):
        return globals()[name](T)[1]
    res = dict(TABLES2)[name](T)
    return res[1] + ("\n\n" + res[2] if len(res) > 2 else "")


def captions_all(T):
    """{name: {"short": short caption, "details": long caption (fig_*) or markdown table (+ note) (table_*)}} in report order."""
    return {k: {"short": v, "details": _details(T, k)} for k, v in short_captions(T).items()}


def write_short_captions(T=None, path=os.path.join(HERE, "renders", "captions_short.md")):
    """Write renders/captions_short.md (name + short caption). Returns the dict."""
    T = load2() if T is None else T
    sc = short_captions(T)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Short captions (two to three sentences per figure / table; computed by viz.short_captions)\n\n")
        for k, v in sc.items():
            fh.write(f"## {k}\n\n{v}\n\n")
    return sc



if __name__ == "__main__":
    for k, v in render_all2().items():
        print(f"## {k}\n{v}\n")
