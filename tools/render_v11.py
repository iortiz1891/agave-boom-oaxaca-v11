"""Render v11 overlays from existing v10 predictions, masked to the 69-muni AOI.

v11 = same trained classifier as v9/v10, same data-driven threshold 0.60,
but restricted to the union of 69 municipios across the Yautepec,
Tlacolula and Miahuatlan districts (data/reference/v11_aoi.geojson).

This script does *not* re-run prediction. It:
  1. Pulls the v10 prediction GeoTIFF shards that intersect the v11 bbox
     from gs://ee-ivanortiz-ccdc86-agave/predictions_ae_v10/ to a local cache
  2. Reprojects + mosaics them to a single PNG over the v11 bbox at native
     ~10 m resolution
  3. Applies an alpha mask: pixels outside the 69-muni union are zeroed
  4. Writes dashboard/data/pixels_v11_<year>.{png,json} and pixels_v11_prob_<year>.png
  5. Computes state_totals_v11.csv (agave hectares inside the muni union, per year)

Run:
  PYTHONPATH=. python -m tools.render_v11 --year 2025
  PYTHONPATH=. python -m tools.render_v11 --all
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import unary_union
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
AOI_GEO = ROOT / "data/reference/v11_aoi.geojson"
DASH_DATA = ROOT / "dashboard/data"
CACHE = ROOT / "data/interim/predictions_ae_v10"

GCS_BUCKET = "gs://ee-ivanortiz-ccdc86-agave"
GCS_PREFIX = "predictions_ae_v10"

THRESHOLD = 0.60
AGAVE_HEX = "E91E63"
ALPHA = 230
PROB_MIN_SHOW = 0.40
PROB_STOPS = [
    (0.40, (0x2E, 0xCC, 0x40)),  # green
    (0.598, (0xFF, 0xDC, 0x00)),  # yellow
    (0.802, (0xFF, 0x85, 0x00)),  # orange
    (1.00, (0xDC, 0x14, 0x00)),  # red
]


def load_aoi():
    """Return (union_polygon, [(west, south, east, north)])."""
    fc = json.loads(AOI_GEO.read_text())
    geoms = []
    for f in fc["features"]:
        if f["properties"].get("cve_mun") == "__union__":
            continue
        geoms.append(shape(f["geometry"]))
    union = unary_union(geoms)
    bx = union.bounds
    return union, bx


def list_v10_shards(year: int) -> list[str]:
    """List v10 prediction shards in GCS for the given year."""
    cmd = ["gsutil", "ls",
           f"{GCS_BUCKET}/{GCS_PREFIX}/agave_mask_ae_{year}*"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return [u.strip() for u in out.splitlines() if u.strip().endswith(".tif")]


def fetch_shards(urls: list[str], year: int) -> list[Path]:
    """Cache shards locally with parallel fetch (gsutil -m cp). Skip existing."""
    cache_year = CACHE / str(year)
    cache_year.mkdir(parents=True, exist_ok=True)
    local = []
    to_fetch = []
    for u in urls:
        name = Path(u).name
        p = cache_year / name
        local.append(p)
        if not p.exists():
            to_fetch.append(u)
    if to_fetch:
        print(f"  fetching {len(to_fetch)} shards for {year} -> {cache_year}  "
              f"(parallel via gsutil -m)")
        # gsutil -m cp <url1> <url2> ... <dst_dir>/
        # Splitting into chunks of 64 to stay well under any argv limits.
        for i in range(0, len(to_fetch), 64):
            chunk = to_fetch[i:i + 64]
            subprocess.run(
                ["gsutil", "-m", "cp", *chunk, str(cache_year) + "/"],
                check=True,
            )
    return local


def shard_intersects(shard_path: Path, bbox: tuple[float, float, float, float]) -> bool:
    """Return True if the shard's footprint intersects the AOI bbox."""
    with rasterio.open(shard_path) as src:
        sb = src.bounds  # (left, bottom, right, top) in source CRS (likely 4326 EPSG)
        if src.crs and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            sb = transform_bounds(src.crs, "EPSG:4326", *sb, densify_pts=21)
        w, s, e, n = bbox
        return not (sb[2] < w or sb[0] > e or sb[3] < s or sb[1] > n)


def reproject_to_grid(shard_path: Path, dst_array: np.ndarray,
                       dst_transform, dst_crs="EPSG:4326"):
    """Reproject a shard's P(agave) onto the destination grid using max-aggregation.

    v10 shards carry two bands: band 1 = class_id (0 or 1), band 2 = max_prob
    (probability of the *predicted* class, always >= 0.5). To recover the
    continuous P(agave):
        P(agave) = max_prob       if class_id == 1
                 = 1 - max_prob   if class_id == 0
    """
    with rasterio.open(shard_path) as src:
        # Reproject both bands onto a tmp grid
        tmp_cls = np.zeros_like(dst_array, dtype="float32")
        tmp_prob = np.zeros_like(dst_array, dtype="float32")
        reproject(
            source=rasterio.band(src, 1), destination=tmp_cls,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        reproject(
            source=rasterio.band(src, 2), destination=tmp_prob,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        agave_prob = np.where(tmp_cls == 1, tmp_prob, 1.0 - tmp_prob)
        # any NaN -> 0 (no info for this pixel from this shard)
        agave_prob = np.where(np.isfinite(agave_prob), agave_prob, 0.0)
        # max-aggregate across overlapping shards
        np.fmax(dst_array, agave_prob, out=dst_array)


def rasterize_mask(union, dst_transform, shape_hw) -> np.ndarray:
    """Rasterize the union polygon onto the destination grid; 1 inside, 0 outside."""
    mask = rasterize(
        [(union, 1)],
        out_shape=shape_hw,
        transform=dst_transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    return mask


def prob_to_rgba(prob: np.ndarray, mask_in_aoi: np.ndarray) -> np.ndarray:
    """Probability gradient (green->yellow->orange->red), alpha=0 outside AOI
    or below PROB_MIN_SHOW."""
    H, W = prob.shape
    rgba = np.zeros((H, W, 4), dtype="uint8")
    show = (prob >= PROB_MIN_SHOW) & (mask_in_aoi == 1)
    # piecewise linear on the 4 stops
    stops_p = np.array([s[0] for s in PROB_STOPS])
    stops_c = np.array([s[1] for s in PROB_STOPS])
    for c_idx in range(3):
        rgba[..., c_idx] = np.interp(prob, stops_p, stops_c[:, c_idx]).astype("uint8")
    rgba[..., 3] = np.where(show, ALPHA, 0).astype("uint8")
    return rgba


def binary_rgba(prob: np.ndarray, mask_in_aoi: np.ndarray) -> np.ndarray:
    """Magenta solid where prob >= THRESHOLD and inside AOI; transparent elsewhere."""
    H, W = prob.shape
    r = int(AGAVE_HEX[0:2], 16)
    g = int(AGAVE_HEX[2:4], 16)
    b = int(AGAVE_HEX[4:6], 16)
    show = (prob >= THRESHOLD) & (mask_in_aoi == 1)
    rgba = np.zeros((H, W, 4), dtype="uint8")
    rgba[..., 0] = r; rgba[..., 1] = g; rgba[..., 2] = b
    rgba[..., 3] = np.where(show, ALPHA, 0).astype("uint8")
    return rgba


def per_pixel_ha(transform, lat_mean: float) -> float:
    """Approximate per-cell area (ha) in WGS84 at given mean latitude."""
    deg_lat = abs(transform.e)  # |dy| in degrees per pixel
    deg_lon = abs(transform.a)
    m_lat = deg_lat * 111_320
    m_lon = deg_lon * 111_320 * np.cos(np.deg2rad(lat_mean))
    return (m_lat * m_lon) / 10_000  # m^2 -> ha


def render_year(year: int, union, bbox, target_res_deg: float = 0.00018):
    """Render v11 PNGs + JSON for one year. Returns dict with stats."""
    w, s, e, n = bbox
    W = int(round((e - w) / target_res_deg))
    H = int(round((n - s) / target_res_deg))
    dst_transform = from_bounds(w, s, e, n, W, H)
    print(f"  year {year}: target grid {W} x {H}")

    # Fetch shards from GCS that intersect the bbox
    all_urls = list_v10_shards(year)
    local = fetch_shards(all_urls, year)
    relevant = [p for p in local if shard_intersects(p, (w, s, e, n))]
    print(f"  year {year}: {len(relevant)} shards intersect AOI bbox")

    prob = np.zeros((H, W), dtype="float32")
    for i, shp in enumerate(relevant, 1):
        reproject_to_grid(shp, prob, dst_transform)
        if i % 20 == 0:
            print(f"    shard {i}/{len(relevant)} reprojected")

    # Rasterize muni mask
    mask = rasterize_mask(union, dst_transform, (H, W))
    n_in_aoi = int(mask.sum())
    print(f"  year {year}: {n_in_aoi} pixels inside 69-muni union")

    binary = binary_rgba(prob, mask)
    prob_rgba = prob_to_rgba(prob, mask)

    # Stats
    n_agave = int(((prob >= THRESHOLD) & (mask == 1)).sum())
    lat_mean = (s + n) / 2
    cell_ha = per_pixel_ha(dst_transform, lat_mean)
    area_ha = n_agave * cell_ha
    aoi_ha = n_in_aoi * cell_ha
    print(f"  year {year}: agave pixels={n_agave}  area={area_ha:,.0f} ha "
          f"({100*area_ha/max(aoi_ha,1):.2f}% of AOI)")

    # Write PNGs
    out_bin = DASH_DATA / f"pixels_v11_{year}.png"
    out_prob = DASH_DATA / f"pixels_v11_prob_{year}.png"
    Image.fromarray(binary, mode="RGBA").save(out_bin, optimize=True)
    Image.fromarray(prob_rgba, mode="RGBA").save(out_prob, optimize=True)
    # JSON metadata (Leaflet expects [[south, west], [north, east]])
    meta = {
        "year": year,
        "bounds": [[s, w], [n, e]],
        "agave_color": f"#{AGAVE_HEX}",
        "threshold": THRESHOLD,
        "n_agave_cells": n_agave,
        "n_aoi_cells": n_in_aoi,
        "grid": [H, W],
        "n_shards": len(relevant),
        "method": "v10 predictions reprojected + 69-muni mask",
        "prob_overlay": {
            "png": f"pixels_v11_prob_{year}.png",
            "min_show": PROB_MIN_SHOW,
            "colormap": "traffic-4stop",
            "stops": [{"prob": p, "rgb": "{:02X}{:02X}{:02X}".format(*c)}
                      for p, c in PROB_STOPS],
        },
    }
    (DASH_DATA / f"pixels_v11_{year}.json").write_text(json.dumps(meta, indent=2))
    print(f"  year {year}: wrote {out_bin.name}, {out_prob.name}, json")

    return {
        "year": year,
        "n_agave_cells": n_agave,
        "n_aoi_cells": n_in_aoi,
        "area_ha": round(area_ha, 1),
        "aoi_ha": round(aoi_ha, 1),
        "cell_ha": cell_ha,
    }


def write_state_totals(stats: list[dict]):
    """Write dashboard/data/state_totals_v11.csv with raw area + bootstrap CI."""
    out = DASH_DATA / "state_totals_v11.csv"
    # Bootstrap CI: simple binomial Wilson-like band on the agave proportion
    # inside the AOI mask. Using sqrt(N * p * (1-p)) approximation; for our
    # purposes the band is dominated by spatial autocorrelation, not pixel
    # binomial variance, so this is a lower bound -- consistent with v10's
    # bootstrap that resamples per-shard counts.
    rows = sorted(stats, key=lambda r: r["year"])
    prev = None
    with out.open("w") as f:
        w = csv.writer(f)
        w.writerow(["year", "area_ha", "area_ha_lo95",
                    "area_ha_hi95", "expansion_rate_pct"])
        for r in rows:
            n = r["n_agave_cells"]
            tot = max(r["n_aoi_cells"], 1)
            p = n / tot
            se = (p * (1 - p) / tot) ** 0.5
            lo = max(0, (p - 1.96 * se) * tot) * r["cell_ha"]
            hi = ((p + 1.96 * se) * tot) * r["cell_ha"]
            yoy = 0.0 if prev is None else (r["area_ha"] / max(prev, 1) - 1) * 100
            w.writerow([r["year"], r["area_ha"],
                        round(lo, 1), round(hi, 1), round(yoy, 2)])
            prev = r["area_ha"]
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--years", help="comma-separated list, e.g. 2017,2018,2019")
    ap.add_argument("--target-res-deg", type=float, default=0.00018,
                    help="output PNG resolution (deg/pixel); default 0.00018 = ~20m")
    args = ap.parse_args()

    if args.all:
        years = list(range(2017, 2026))
    elif args.years:
        years = [int(y) for y in args.years.split(",")]
    elif args.year:
        years = [args.year]
    else:
        ap.error("specify --year YYYY, --years 2017,2018, or --all")

    union, bx = load_aoi()
    print(f"v11 AOI: bbox {bx},  union area covers union polygon")

    stats = []
    for y in years:
        try:
            stats.append(render_year(y, union, bx, args.target_res_deg))
        except Exception as e:
            print(f"  ERROR for year {y}: {e}", file=sys.stderr)
            raise

    if len(stats) >= 2:
        write_state_totals(stats)
    else:
        print("(skip state_totals: rendered fewer than 2 years; rerun with --all)")


if __name__ == "__main__":
    main()
