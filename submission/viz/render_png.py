"""Optional helper: render every figure of viz.py to PNG (colour + greyscale) with a headless Chromium.

Needs plotly, pandas, pillow and playwright (`pip install playwright && playwright install chromium`); kaleido is not
needed. Writes renders/<fig_name>.png, renders/<fig_name>_grey.png, the HTML + caption files of viz.render_all2()
and renders/captions.md.

    python render_png.py                 # all figures
    python render_png.py fig_simpson     # only the named ones
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import viz                                   # noqa: E402
from PIL import Image                        # noqa: E402
from playwright.sync_api import sync_playwright   # noqa: E402

WIDTH, SCALE = 1300, 1.5


def main(names=None, out_dir=os.path.join(HERE, "renders")):
    os.makedirs(out_dir, exist_ok=True)
    caps = viz.render_all2(out_dir=out_dir)                # HTML (plotly.js from CDN) + captions + tables
    T = viz.load2()
    figs = [f for f in viz.FIGURES + viz.FIGURES2 if names is None or f.__name__ in names]
    tmp = tempfile.mkdtemp()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": WIDTH + 40, "height": 900}, device_scale_factor=SCALE)
        for f in figs:
            fig, _ = f(T)
            fig.update_layout(width=WIDTH)
            path = os.path.join(tmp, f.__name__ + ".html")
            fig.write_html(path, include_plotlyjs=True, full_html=True, config={"staticPlot": True})
            pg.goto("file://" + path)
            pg.wait_for_selector(".main-svg")
            pg.wait_for_timeout(400)
            png = os.path.join(out_dir, f.__name__ + ".png")
            pg.locator(".plotly-graph-div").first.screenshot(path=png)
            Image.open(png).convert("L").save(os.path.join(out_dir, f.__name__ + "_grey.png"))
            print("rendered", f.__name__)
        b.close()
    return caps


if __name__ == "__main__":
    main(sys.argv[1:] or None)
