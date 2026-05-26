"""v11 cover-change analysis with INEGI Serie VII (2018) baseline.

Why INEGI instead of (or in addition to) ESA WorldCover 2020:

  - INEGI Serie VII is the authoritative Mexican land-cover/land-use
    dataset, 1:250,000 vector polygons published by INEGI in 2021 with
    a 2018 baseline. It carries Mexican-specific vocabulary
    ('agricultura de temporal anual', 'vegetación secundaria arbustiva',
    'bosque de encino', 'selva baja caducifolia', etc.) that ESA's
    global taxonomy does not.
  - The 2018 baseline is closer to the boom mid-window than ESA's 2020.
  - INEGI explicitly separates primary from secondary vegetation, which
    is the policy-relevant distinction for the tree-cover replacement
    claim (re-clearing 30-year-old fallow is qualitatively different
    from clearing primary cloud forest).
  - INEGI Serie VII is the legal reference layer used by the Mexican
    federal authorities (CONAFOR, SEMARNAT) for land-use policy.

Trade-offs:
  - INEGI is coarser (1:250k vector vs ESA's 10 m raster); near-polygon-
    edge pixels may be mis-attributed. We rasterise to the v11 grid
    (~20 m) and accept this resolution mismatch.
  - INEGI is a single 2018 snapshot, like ESA's 2020 -- same caveat.

Workflow:
  1. Open the INEGI Serie VII shapefile.
  2. Rasterise DESCRIPCIO (via the coarse-class mapping reused from
     tools/intersect_inegi_v9.py) onto the v11 grid (6764 x 8524,
     EPSG:4326, v11 bbox), clipped to the 69-muni union mask.
  3. Intersect with the 2017->2025 new-agave mask + per-year transitions.
  4. Write replaced_covers_v11_inegi_phase1.csv (overall) and
     replaced_covers_v11_inegi_per_year.csv (per-year).

Run:
  conda activate agave-oaxaca
  PYTHONPATH=. python -m tools.intersect_inegi_v11
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import unary_union
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Reuse the DESCRIPCIO -> coarse-class mapping from the v9 script.
from tools.intersect_inegi_v9 import descripcio_to_coarse, list_inegi_field

DATA = Path("dashboard/data")
INTERIM = Path("data/interim")
INEGI_SHP = Path("data/external/inegi_serie_vii/usv250s7gw.shp")
AOI_GEO = Path("data/reference/v11_aoi.geojson")
RASTER_OUT = INTERIM / "inegi_serie_vii_v11_aoi.tif"

V11_BBOX = (-97.0904, 15.9375, -95.5560, 17.1550)

# Coarse class -> stable integer code + hex color (used by the v11 figures)
# and the WORLDCOVER-style palette so fig8/fig9 v11 INEGI re-uses the same
# bar styling.
COARSE_CODE = {
    "Cropland":          {"code": 40, "color": "#F096FF"},
    "Grassland":         {"code": 30, "color": "#FFFF4C"},
    "Shrubland":         {"code": 20, "color": "#FFBB22"},
    "Tree cover":        {"code": 10, "color": "#006400"},
    "Secondary tree":    {"code": 11, "color": "#3FA34D"},
    "Secondary shrub":   {"code": 21, "color": "#D9A521"},
    "Secondary veg.":    {"code": 22, "color": "#A6BB4D"},
    "Bare / sparse veg.": {"code": 60, "color": "#B4B4B4"},
    "Built-up":          {"code": 50, "color": "#FA0000"},
    "Permanent water":   {"code": 80, "color": "#0064C8"},
    "Wetland":           {"code": 90, "color": "#0096A0"},
    "Mangrove":          {"code": 95, "color": "#00CF75"},
    "Other":             {"code": 99, "color": "#888888"},
}
CODE_TO_LABEL = {v["code"]: k for k, v in COARSE_CODE.items()}


def rasterise_inegi_to_v11():
    """Rasterise INEGI Serie VII DESCRIPCIO -> coarse code onto the v11 grid."""
    # Use the v11 grid resolution from pixels_v11_2025.json
    j = json.loads((DATA / "pixels_v11_2025.json").read_text())
    H, W = j["grid"]
    (s, w), (n, e) = j["bounds"]
    transform = from_bounds(w, s, e, n, W, H)
    print(f"  v11 grid: {H} x {W}  bbox=({w:.4f},{s:.4f},{e:.4f},{n:.4f})")

    print(f"  reading INEGI shapefile: {INEGI_SHP}")
    shapes = []
    n_feats = 0
    with fiona.open(INEGI_SHP) as src:
        print(f"    layer crs: {src.crs}  total features: {len(src)}")
        for f in src:
            geom = f["geometry"]
            if geom is None:
                continue
            descripcio = list_inegi_field(f["properties"])
            coarse = descripcio_to_coarse(descripcio)
            code = COARSE_CODE.get(coarse, COARSE_CODE["Other"])["code"]
            # Quick bbox prefilter: skip features whose bbox doesn't overlap v11
            try:
                gx_min, gy_min, gx_max, gy_max = shape(geom).bounds
            except Exception:
                continue
            if gx_max < w or gx_min > e or gy_max < s or gy_min > n:
                continue
            shapes.append((shape(geom), code))
            n_feats += 1
    print(f"    features overlapping v11 bbox: {n_feats}")

    print(f"  rasterising onto v11 grid ...")
    arr = rasterize(
        ((g, c) for g, c in shapes),
        out_shape=(H, W), transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    print(f"    unique codes: {sorted(np.unique(arr).tolist())}")

    RASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        RASTER_OUT, "w",
        driver="GTiff", width=W, height=H, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform,
        compress="deflate", nodata=0,
    ) as dst:
        dst.write(arr, 1)
    print(f"  wrote {RASTER_OUT} ({RASTER_OUT.stat().st_size/1e6:.1f} MB)")
    return arr, transform, (H, W), (w, s, e, n)


def _load_or_build():
    if RASTER_OUT.exists():
        with rasterio.open(RASTER_OUT) as src:
            arr = src.read(1)
            b = src.bounds
            return arr, src.transform, src.shape, (b.left, b.bottom, b.right, b.top)
    return rasterise_inegi_to_v11()


def main():
    arr, transform, (H, W), (w, s, e, n) = _load_or_build()

    # Load v11 binary masks + AOI muni-union mask
    YEARS = list(range(2017, 2026))
    print(f"  loading {len(YEARS)} v11 binary masks ...")
    masks = {y: np.array(Image.open(DATA / f"pixels_v11_{y}.png"))[:, :, 3] > 0
             for y in YEARS}
    assert masks[2017].shape == (H, W), (masks[2017].shape, (H, W))

    # Rasterise the 69-muni union to crop the analysis to the v11 AOI
    aoi_fc = json.loads(AOI_GEO.read_text())
    geoms = [shape(f["geometry"]) for f in aoi_fc["features"]
             if f["properties"].get("cve_mun") != "__union__"]
    aoi_polygon = unary_union(geoms)
    aoi_mask = rasterize(
        [(aoi_polygon, 1)],
        out_shape=(H, W), transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    ) == 1

    # Per-pixel area
    deg_lon = (e - w) / W
    deg_lat = (n - s) / H
    lat_mean = (s + n) / 2
    cell_m_lon = deg_lon * 111_320 * np.cos(np.deg2rad(lat_mean))
    cell_m_lat = deg_lat * 111_320
    PIX_AREA_HA = (cell_m_lon * cell_m_lat) / 10_000
    print(f"  per-pixel area: {PIX_AREA_HA:.4f} ha")

    # ----- Phase 1: overall 2017 -> 2025 -----
    a17 = masks[2017]; a25 = masks[2025]
    persistent = a17 & a25 & aoi_mask
    new_agave  = (~a17) & a25 & aoi_mask
    lost_agave = a17 & (~a25) & aoi_mask
    print(f"\n  Phase 1 transitions 2017 -> 2025 (inside v11 AOI):")
    print(f"    Persistent: {persistent.sum():>10,} px = "
          f"{persistent.sum()*PIX_AREA_HA:>9.0f} ha")
    print(f"    New (boom): {new_agave.sum():>10,} px = "
          f"{new_agave.sum()*PIX_AREA_HA:>9.0f} ha")
    print(f"    Lost:       {lost_agave.sum():>10,} px = "
          f"{lost_agave.sum()*PIX_AREA_HA:>9.0f} ha")

    def _counts(mask):
        codes = arr[mask]
        c = Counter(codes.tolist())
        c.pop(0, None)
        return c, sum(c.values())

    c_p, n_p = _counts(persistent)
    c_n, n_n = _counts(new_agave)
    c_l, n_l = _counts(lost_agave)

    print(f"\n=== INEGI Serie VII composition of NEW agave (2017 -> 2025) ===")
    for code, count in c_n.most_common():
        label = CODE_TO_LABEL.get(code, f"code {code}")
        pct = 100 * count / max(n_n, 1)
        print(f"    {code:>3}  {label:<25}  {count:>10,} px  ({pct:5.1f}%)"
              f"  = {count*PIX_AREA_HA:>8.0f} ha")

    rows_p1 = []
    for label, info in COARSE_CODE.items():
        code = info["code"]
        rows_p1.append({
            "inegi_code": code,
            "inegi_class": label,
            "color": info["color"],
            "persistent_pixels": c_p.get(code, 0),
            "new_agave_pixels": c_n.get(code, 0),
            "lost_agave_pixels": c_l.get(code, 0),
            "persistent_ha": round(c_p.get(code, 0) * PIX_AREA_HA, 1),
            "new_agave_ha": round(c_n.get(code, 0) * PIX_AREA_HA, 1),
            "lost_agave_ha": round(c_l.get(code, 0) * PIX_AREA_HA, 1),
            "pct_of_new": round(100 * c_n.get(code, 0) / max(n_n, 1), 2),
        })
    rows_p1.sort(key=lambda r: -r["new_agave_pixels"])
    out1 = DATA / "replaced_covers_v11_inegi_phase1.csv"
    with out1.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_p1[0].keys()))
        wtr.writeheader(); wtr.writerows(rows_p1)
    print(f"  wrote {out1}")

    # ----- Per-year -----
    rows_year = []
    print(f"\n=== INEGI per-year new agave ===")
    for i in range(1, len(YEARS)):
        prev_y, this_y = YEARS[i - 1], YEARS[i]
        new_this_year = (~masks[prev_y]) & masks[this_y] & aoi_mask
        c = Counter(arr[new_this_year].tolist())
        c.pop(0, None)
        total = sum(c.values())
        print(f"  {prev_y} -> {this_y}: {total:,} new px "
              f"({total*PIX_AREA_HA:.0f} ha)")
        for code, count in c.most_common(5):
            label = CODE_TO_LABEL.get(code, f"code {code}")
            pct = 100 * count / max(total, 1)
            print(f"    {code:>3}  {label:<25}  {count:>10,} px ({pct:5.1f}%)")
        for label, info in COARSE_CODE.items():
            code = info["code"]
            rows_year.append({
                "transition": f"{prev_y}->{this_y}",
                "target_year": this_y,
                "inegi_code": code,
                "inegi_class": label,
                "color": info["color"],
                "n_pixels": c.get(code, 0),
                "area_ha": round(c.get(code, 0) * PIX_AREA_HA, 1),
                "pct_of_year": round(100 * c.get(code, 0) / max(total, 1), 2),
            })

    out2 = DATA / "replaced_covers_v11_inegi_per_year.csv"
    with out2.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_year[0].keys()))
        wtr.writeheader(); wtr.writerows(rows_year)
    print(f"  wrote {out2}")


if __name__ == "__main__":
    main()
