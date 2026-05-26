"""Centralised matplotlib styling for publication-quality figures.

This module exposes a single entry point ``setup_publication_style`` that
configures matplotlib's ``rcParams`` for a consistent, journal-ready look
across all figures in the manuscript and the supplementary material. It also
defines the canonical class-color map (so that, e.g., ``agave`` is the same
colour everywhere it is plotted) and a per-region palette.

Helpers for map ornamentation (scale bar, north arrow, locator inset) are
also provided. They are written to be tolerant of missing optional geo
dependencies: if ``cartopy`` / ``contextily`` are not installed, the helpers
degrade gracefully by drawing a non-georeferenced placeholder.

Style decisions (kept in one place by design):

* Base font: ``DejaVu Sans`` (always available with matplotlib) at 9 pt,
  axis labels 9 pt, tick labels 8 pt, titles 10 pt.
* Colour palette: viridis-derived discrete steps for sequential variables;
  ColorBrewer Dark2 for nominal classes (chosen for colourblind safety and
  print fidelity).
* DPI: 300 for raster outputs; SVG is preferred for vector outputs.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple


# ---------------------------------------------------------------------------
# Canonical colour maps
# ---------------------------------------------------------------------------

#: Mapping from land-cover class label to a hex colour.
#: These colours are referenced by every figure that plots class membership;
#: changing them here updates the entire figure set at once.
CLASS_COLORS: dict[str, str] = {
    "agave": "#d95f02",          # ColorBrewer Dark2 orange — the headline class
    "agave_mature": "#d95f02",
    "agave_young": "#fdb863",
    "milpa": "#e6ab02",
    "milpa_or_annual": "#e6ab02",
    "forest": "#1b9e77",
    "secondary_veg": "#66a61e",
    "scrubland": "#a6761d",
    "bare_or_urban": "#7570b3",
    "bare": "#7570b3",
    "water": "#386cb0",
    "other_perennial": "#e7298a",
    "other": "#999999",
    # Change-class palette
    "stable_agave": "#1b9e77",
    "agave_loss": "#fdae6b",
    "agave_gain": "#d95f02",
    "stable_not_agave": "#f0f0f0",
}

#: Mapping from Oaxaca statistical region (snake-case) to a hex colour.
REGION_COLORS: dict[str, str] = {
    "canada": "#7fc97f",
    "costa": "#beaed4",
    "istmo": "#fdc086",
    "mixteca": "#ffff99",
    "papaloapan": "#386cb0",
    "sierra_norte": "#f0027f",
    "sierra_sur": "#bf5b17",
    "valles_centrales": "#666666",
}

#: Display names for the regions (used in legends, captions).
REGION_DISPLAY: dict[str, str] = {
    "canada": "Cañada",
    "costa": "Costa",
    "istmo": "Istmo",
    "mixteca": "Mixteca",
    "papaloapan": "Papaloapan",
    "sierra_norte": "Sierra Norte",
    "sierra_sur": "Sierra Sur",
    "valles_centrales": "Valles Centrales",
}

#: Headline cities to be annotated on the study-area map.
KEY_CITIES: list[dict[str, Any]] = [
    {"name": "Oaxaca de Juárez", "lon": -96.7266, "lat": 17.0732},
    {"name": "Tlacolula", "lon": -96.4783, "lat": 16.9558},
    {"name": "Miahuatlán", "lon": -96.5942, "lat": 16.3322},
    {"name": "Sola de Vega", "lon": -96.9742, "lat": 16.5167},
]


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------


def setup_publication_style(base_font_size: float = 9.0) -> None:
    """Configure ``matplotlib.rcParams`` for publication figures.

    Sets typography, line widths, default DPI, tight savefig padding, and
    the default colour cycle. Idempotent — calling it more than once is a
    no-op for downstream callers.

    Args:
        base_font_size: Base font size in pt (default 9 pt for two-column
            journal layouts).
    """
    import matplotlib as mpl

    rc: dict[str, Any] = {
        # Typography
        "font.family": "DejaVu Sans",
        "font.size": base_font_size,
        "axes.titlesize": base_font_size + 1,
        "axes.labelsize": base_font_size,
        "xtick.labelsize": base_font_size - 1,
        "ytick.labelsize": base_font_size - 1,
        "legend.fontsize": base_font_size - 1,
        "figure.titlesize": base_font_size + 2,
        # Lines / markers
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Grid and spines
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Resolution
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Colour cycle (Dark2 for categorical work)
        "axes.prop_cycle": mpl.cycler(
            color=[
                "#1b9e77",
                "#d95f02",
                "#7570b3",
                "#e7298a",
                "#66a61e",
                "#e6ab02",
                "#a6761d",
                "#666666",
            ]
        ),
        # Output
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    mpl.rcParams.update(rc)


# ---------------------------------------------------------------------------
# Map ornamentation helpers
# ---------------------------------------------------------------------------


def add_scalebar(
    ax: Any,
    length_km: float = 50.0,
    location: str = "lower right",
    *,
    units: str = "km",
    color: str = "black",
) -> None:
    """Add a simple horizontal scale bar to an axis.

    The bar is positioned in axes-fraction coordinates, so it is robust to
    any data-coordinate units. The label is drawn directly above the bar.

    Args:
        ax: A matplotlib Axes (any projection).
        length_km: Numeric length of the bar in kilometres (used for the
            label only; we approximate the rendered length as 15% of the
            axes width regardless).
        location: One of ``"lower right"``, ``"lower left"``,
            ``"upper right"``, ``"upper left"``.
        units: Units string for the label.
        color: Bar / text colour.
    """
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    import matplotlib.transforms as mtransforms

    loc_map = {
        "lower right": 4,
        "lower left": 3,
        "upper right": 1,
        "upper left": 2,
    }
    loc = loc_map.get(location, 4)

    # Build a small composite artist in axes-fraction units.
    width_frac = 0.15
    height_frac = 0.012

    fig = ax.get_figure()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in = bbox.width * width_frac
    height_in = bbox.height * height_frac

    transform = mtransforms.Affine2D().scale(fig.dpi)

    box = AuxTransformBox(transform)
    rect = Rectangle((0, 0), width_in, height_in, fc=color, ec=color)
    box.add_artist(rect)
    label = Text(
        width_in / 2,
        height_in * 1.6,
        f"{length_km:g} {units}",
        ha="center",
        va="bottom",
        color=color,
        fontsize=7,
    )
    box.add_artist(label)

    anchored = AnchoredOffsetbox(
        loc=loc,
        child=box,
        pad=0.3,
        borderpad=0.4,
        frameon=False,
    )
    ax.add_artist(anchored)


def add_north_arrow(
    ax: Any,
    location: str = "upper left",
    *,
    color: str = "black",
) -> None:
    """Draw a north arrow on a map axis.

    A minimalist triangle + ``N`` label, positioned in axes-fraction
    coordinates so it is independent of the data CRS.
    """
    loc_map = {
        "upper left": (0.04, 0.92),
        "upper right": (0.94, 0.92),
        "lower left": (0.04, 0.10),
        "lower right": (0.94, 0.10),
    }
    x, y = loc_map.get(location, loc_map["upper left"])
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.06),
        textcoords="axes fraction",
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=color,
        arrowprops=dict(
            facecolor=color,
            edgecolor=color,
            width=3.5,
            headwidth=8,
            headlength=8,
        ),
    )


def inset_locator_map(
    ax: Any,
    aoi_gdf: Any | None = None,
    *,
    bounds: Tuple[float, float, float, float] = (0.02, 0.65, 0.30, 0.32),
) -> Any:
    """Create a small locator inset showing the AOI within Mexico.

    The inset is added to ``ax`` in axes-fraction coordinates (`bounds` is
    ``(x, y, width, height)`` in 0–1). If a ``geopandas`` GeoDataFrame is
    supplied, its envelope is drawn as a red rectangle on top of a coarse
    Mexico outline (rendered as a simple bounding box if no Mexico polygon
    is available).
    """
    fig = ax.get_figure()
    inset = fig.add_axes(
        [
            ax.get_position().x0 + bounds[0] * ax.get_position().width,
            ax.get_position().y0 + bounds[1] * ax.get_position().height,
            bounds[2] * ax.get_position().width,
            bounds[3] * ax.get_position().height,
        ]
    )
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("#f7f8fa")
    for spine in inset.spines.values():
        spine.set_edgecolor("#888")
        spine.set_linewidth(0.6)

    # Approximate Mexico bounding box in lon/lat
    mx_minx, mx_miny, mx_maxx, mx_maxy = -118.4, 14.4, -86.7, 32.7
    inset.set_xlim(mx_minx, mx_maxx)
    inset.set_ylim(mx_miny, mx_maxy)

    # Try plotting a real Mexico outline if cartopy or natural-earth file is
    # available; otherwise fall back to a labelled box.
    try:  # pragma: no cover - optional dep
        import cartopy.io.shapereader as shpreader  # type: ignore
        from shapely.geometry import shape  # type: ignore

        shp = shpreader.natural_earth(
            resolution="50m", category="cultural", name="admin_0_countries"
        )
        reader = shpreader.Reader(shp)
        for rec in reader.records():
            if rec.attributes.get("ADMIN") == "Mexico":
                geom = rec.geometry
                if geom.geom_type == "MultiPolygon":
                    for g in geom.geoms:
                        xs, ys = g.exterior.xy
                        inset.fill(xs, ys, color="#e0e3e8", ec="#888", lw=0.4)
                else:
                    xs, ys = geom.exterior.xy
                    inset.fill(xs, ys, color="#e0e3e8", ec="#888", lw=0.4)
                break
    except Exception:  # noqa: BLE001
        # Coarse rectangle as a Mexico stand-in.
        from matplotlib.patches import Rectangle

        inset.add_patch(
            Rectangle(
                (mx_minx, mx_miny),
                mx_maxx - mx_minx,
                mx_maxy - mx_miny,
                facecolor="#e0e3e8",
                edgecolor="#888",
                linewidth=0.4,
            )
        )

    # Draw AOI envelope if provided.
    if aoi_gdf is not None:
        try:
            gdf = aoi_gdf
            if hasattr(gdf, "to_crs"):
                gdf = gdf.to_crs(4326)
            minx, miny, maxx, maxy = gdf.total_bounds
            from matplotlib.patches import Rectangle

            inset.add_patch(
                Rectangle(
                    (minx, miny),
                    maxx - minx,
                    maxy - miny,
                    facecolor="none",
                    edgecolor="#d62728",
                    linewidth=1.0,
                )
            )
        except Exception:  # noqa: BLE001
            # Fallback: highlight the Oaxaca approximate envelope.
            from matplotlib.patches import Rectangle

            inset.add_patch(
                Rectangle(
                    (-98.6, 15.6),
                    3.2,
                    2.6,
                    facecolor="none",
                    edgecolor="#d62728",
                    linewidth=1.0,
                )
            )

    inset.set_title("Mexico", fontsize=7, pad=2)
    return inset


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def discrete_viridis(n: int) -> list[str]:
    """Return ``n`` evenly spaced colours from the viridis colormap as hex."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmap = cm.get_cmap("viridis", n)
    return [mcolors.to_hex(cmap(i)) for i in range(n)]


def class_palette(classes: Iterable[str]) -> Mapping[str, str]:
    """Return a colour mapping for the given classes, falling back to grey."""
    return {c: CLASS_COLORS.get(c, "#999999") for c in classes}
