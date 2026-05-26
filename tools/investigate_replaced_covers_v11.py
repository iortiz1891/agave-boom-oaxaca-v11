"""v11 cover-change analysis: which land covers does the agave boom replace?

Restricted to the v11 AOI (union of 69 municipios across Tlacolula + Yautepec
+ Miahuatlán). Uses ESA WorldCover 2020 as the LC baseline.

The v11 binary PNGs are already muni-masked (pixels outside the 69-muni
union are transparent), so all transition counts automatically exclude
out-of-AOI land -- no extra masking step is needed here.

Outputs:
  - dashboard/data/replaced_covers_v11_phase1.csv  (overall 2017->2025)
  - dashboard/data/replaced_covers_v11_per_year.csv (year-by-year)

Run:
  conda activate agave-oaxaca
  PYTHONPATH=. python -m tools.investigate_replaced_covers_v11
"""
from __future__ import annotations
import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path

DATA = Path("dashboard/data")
WC_DIR = Path("data/interim")
WC_LOCAL = WC_DIR / "worldcover_v11_aoi.tif"

# v11 bbox = bbox of the 69-muni union (data/reference/v11_aoi.geojson).
# WorldCover tile coverage: N15W099 (15-18 N, 96-99 W) + N15W096 (15-18 N,
# 93-96 W) span this footprint.
V11_BBOX = (-97.0904, 15.9375, -95.5560, 17.1550)
WC_TILES = [
    "ESA_WorldCover_10m_2020_v100_N15W099_Map.tif",
    "ESA_WorldCover_10m_2020_v100_N15W096_Map.tif",
]

# ESA WorldCover v100 (2020) class codes
WORLDCOVER = {
    10:  {"name": "Tree cover",              "color": "#006400"},
    20:  {"name": "Shrubland",               "color": "#FFBB22"},
    30:  {"name": "Grassland",               "color": "#FFFF4C"},
    40:  {"name": "Cropland",                "color": "#F096FF"},
    50:  {"name": "Built-up",                "color": "#FA0000"},
    60:  {"name": "Bare / sparse veg.",      "color": "#B4B4B4"},
    70:  {"name": "Snow / ice",              "color": "#F0F0F0"},
    80:  {"name": "Permanent water",         "color": "#0064C8"},
    90:  {"name": "Herbaceous wetland",      "color": "#0096A0"},
    95:  {"name": "Mangroves",               "color": "#00CF75"},
    100: {"name": "Moss and lichen",         "color": "#FAE6A0"},
}


def ensure_worldcover_v11_aoi():
    """Merge available WorldCover tiles + clip to the v11 bbox."""
    import rasterio
    from rasterio.merge import merge

    if WC_LOCAL.exists():
        print(f"  WC v11 AOI already exists: {WC_LOCAL} ({WC_LOCAL.stat().st_size / 1e6:.1f} MB)")
        return

    WC_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    base = "https://esa-worldcover.s3.amazonaws.com/v100/2020/map"
    local_tiles = []
    for t in WC_TILES:
        local = WC_DIR / t
        if not local.exists():
            url = f"{base}/{t}"
            print(f"  downloading {url} ...")
            urllib.request.urlretrieve(url, str(local))
            print(f"    {local.stat().st_size / 1e6:.0f} MB")
        local_tiles.append(local)

    print(f"  merging WC tiles + clipping to v11 AOI ...")
    srcs = [rasterio.open(t) for t in local_tiles]
    w, s, e, n = V11_BBOX
    mosaic, out_trans = merge(srcs, bounds=(w, s, e, n))
    prof = srcs[0].profile.copy()
    prof.update(height=mosaic.shape[1], width=mosaic.shape[2],
                transform=out_trans, count=1, compress="deflate")
    with rasterio.open(WC_LOCAL, "w", **prof) as dst:
        dst.write(mosaic[0], 1)
    for src in srcs:
        src.close()
    print(f"  saved {WC_LOCAL} ({WC_LOCAL.stat().st_size / 1e6:.1f} MB)")


def main():
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    import PIL.Image as Image
    Image.MAX_IMAGE_PIXELS = None

    ensure_worldcover_v11_aoi()

    YEARS = list(range(2017, 2026))
    pngs = {y: DATA / f"pixels_v11_{y}.png" for y in YEARS}
    j25 = DATA / "pixels_v11_2025.json"
    for y, p in pngs.items():
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if not j25.exists():
        raise SystemExit(f"missing {j25}")

    # Load all 9 binary masks (alpha > 0 = agave-in-muni-union at that year)
    print(f"  loading {len(YEARS)} v11 binary masks ...")
    masks = {y: np.array(Image.open(pngs[y]))[:, :, 3] > 0 for y in YEARS}
    H, W = masks[YEARS[0]].shape
    print(f"  v11 grid: {H} x {W}")
    meta = json.loads(j25.read_text())
    (s, w), (n, e) = meta["bounds"]

    # Per-pixel area from the 2025 metadata (n_agave_cells / area_ha)
    # actually we'll compute from the grid geometry: at lat (s+n)/2,
    # 1 deg lat ~ 111,320 m and 1 deg lon ~ 111,320*cos(lat) m
    deg_lon = (e - w) / W
    deg_lat = (n - s) / H
    lat_mean = (s + n) / 2
    cell_m_lon = deg_lon * 111_320 * np.cos(np.deg2rad(lat_mean))
    cell_m_lat = deg_lat * 111_320
    PIX_AREA_HA = (cell_m_lon * cell_m_lat) / 10_000
    print(f"  per-pixel area: {cell_m_lon:.1f} m x {cell_m_lat:.1f} m = {PIX_AREA_HA:.4f} ha")

    # Reproject WorldCover to the v11 grid
    print(f"  reprojecting WorldCover to v11 grid ...")
    wc_aligned = np.zeros((H, W), dtype="uint8")
    dst_transform = from_bounds(w, s, e, n, W, H)
    with rasterio.open(WC_LOCAL) as wc_src:
        reproject(
            source=rasterio.band(wc_src, 1),
            destination=wc_aligned,
            src_transform=wc_src.transform, src_crs=wc_src.crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
            src_nodata=0, dst_nodata=0,
        )
    print(f"  reprojection done. classes seen: {sorted(np.unique(wc_aligned).tolist())}")

    # AOI mask = union of all years' alpha (any pixel that was ever agave OR
    # was potentially-agave inside the muni mask). To get "anything inside
    # the 69-muni union", we use the probability PNG which has alpha=0 only
    # for out-of-muni pixels. Simpler: any year where the muni-mask said
    # "this pixel is inside the AOI" -- but our binary PNG masks set alpha
    # = 0 for both out-of-AOI AND below-threshold-inside-AOI. So we can't
    # recover AOI extent from the binary PNGs alone.
    # Approach: read pixels_v11_prob_2025.png alpha (which is non-zero
    # for any in-AOI pixel with prob>=PROB_MIN_SHOW=0.40). This still
    # misses some in-AOI low-prob pixels. The simplest exact recovery is
    # to rasterize the muni-union polygon at the v11 grid.
    print(f"  rasterizing 69-muni union mask at v11 grid for AOI denominator ...")
    from rasterio.features import rasterize
    from shapely.geometry import shape
    from shapely.ops import unary_union
    aoi_fc = json.loads(Path("data/reference/v11_aoi.geojson").read_text())
    geoms = [shape(f["geometry"]) for f in aoi_fc["features"]
             if f["properties"].get("cve_mun") != "__union__"]
    aoi_polygon = unary_union(geoms)
    aoi_mask = rasterize(
        [(aoi_polygon, 1)],
        out_shape=(H, W), transform=dst_transform,
        fill=0, dtype="uint8", all_touched=False,
    ) == 1
    print(f"  AOI mask: {aoi_mask.sum():,} pixels inside the 69-muni union "
          f"({aoi_mask.sum()*PIX_AREA_HA:.0f} ha)")

    # =====================================================================
    # PHASE 1: overall 2017 -> 2025 composition
    # =====================================================================
    a17 = masks[2017]; a25 = masks[2025]
    persistent = a17 & a25 & aoi_mask
    new_agave  = (~a17) & a25 & aoi_mask
    lost_agave = a17 & (~a25) & aoi_mask
    print(f"\n  Phase 1 transitions 2017 -> 2025 (inside AOI):")
    print(f"    Persistent: {persistent.sum():>10,} px = {persistent.sum()*PIX_AREA_HA:>9.0f} ha")
    print(f"    New (boom): {new_agave.sum():>10,} px = {new_agave.sum()*PIX_AREA_HA:>9.0f} ha")
    print(f"    Lost:       {lost_agave.sum():>10,} px = {lost_agave.sum()*PIX_AREA_HA:>9.0f} ha")

    def _counts(mask, label):
        codes = wc_aligned[mask]
        c = Counter(codes.tolist())
        c.pop(0, None)
        total = sum(c.values())
        print(f"\n  {label} (n={total:,} px = {total*PIX_AREA_HA:.0f} ha):")
        for code, n in c.most_common():
            name = WORLDCOVER.get(code, {}).get("name", f"class {code}")
            pct = 100 * n / max(total, 1)
            print(f"    {code:>3}  {name:<25}  {n:>10,} px  ({pct:5.1f}%)  = {n*PIX_AREA_HA:>8.0f} ha")
        return c, total

    print("\n=== v11 Phase 1 composition (WorldCover 2020 baseline) ===")
    c_persist, n_persist = _counts(persistent, "Persistent agave (2017 & 2025)")
    c_new,     n_new     = _counts(new_agave,  "NEW agave (the boom, 2017->2025)")
    c_lost,    n_lost    = _counts(lost_agave, "Lost agave (2017 only)")

    rows_phase1 = []
    for code, meta_cls in WORLDCOVER.items():
        rows_phase1.append({
            "wc_class": code,
            "wc_name": meta_cls["name"],
            "wc_color": meta_cls["color"],
            "persistent_pixels": c_persist.get(code, 0),
            "new_agave_pixels": c_new.get(code, 0),
            "lost_agave_pixels": c_lost.get(code, 0),
            "persistent_ha": round(c_persist.get(code, 0) * PIX_AREA_HA, 1),
            "new_agave_ha": round(c_new.get(code, 0) * PIX_AREA_HA, 1),
            "lost_agave_ha": round(c_lost.get(code, 0) * PIX_AREA_HA, 1),
            "pct_of_new": round(100 * c_new.get(code, 0) / max(n_new, 1), 2),
        })
    rows_phase1.sort(key=lambda r: -r["new_agave_pixels"])
    out1 = DATA / "replaced_covers_v11_phase1.csv"
    with out1.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_phase1[0].keys()))
        wtr.writeheader(); wtr.writerows(rows_phase1)
    print(f"\n  wrote {out1}")

    # =====================================================================
    # Per-year breakdown: new agave per transition Y-1 -> Y
    # =====================================================================
    print(f"\n=== v11 per-year new agave by WorldCover 2020 class ===")
    rows_year = []
    for i in range(1, len(YEARS)):
        prev_y, this_y = YEARS[i - 1], YEARS[i]
        new_this_year = (~masks[prev_y]) & masks[this_y] & aoi_mask
        codes = wc_aligned[new_this_year]
        c = Counter(codes.tolist()); c.pop(0, None)
        total = sum(c.values())
        print(f"\n  {prev_y} -> {this_y}: {total:,} new px ({total*PIX_AREA_HA:.0f} ha)")
        for code, n in c.most_common(5):
            name = WORLDCOVER.get(code, {}).get("name", f"class {code}")
            pct = 100 * n / max(total, 1)
            print(f"    {code:>3}  {name:<25}  {n:>10,} px  ({pct:5.1f}%)")
        for code in WORLDCOVER:
            rows_year.append({
                "transition": f"{prev_y}->{this_y}",
                "target_year": this_y,
                "wc_class": code,
                "wc_name": WORLDCOVER[code]["name"],
                "wc_color": WORLDCOVER[code]["color"],
                "n_pixels": c.get(code, 0),
                "area_ha": round(c.get(code, 0) * PIX_AREA_HA, 1),
                "pct_of_year": round(100 * c.get(code, 0) / max(total, 1), 2),
            })

    out2 = DATA / "replaced_covers_v11_per_year.csv"
    fields = list(rows_year[0].keys())
    with out2.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=fields)
        wtr.writeheader(); wtr.writerows(rows_year)
    print(f"\n  wrote {out2}")


if __name__ == "__main__":
    main()
