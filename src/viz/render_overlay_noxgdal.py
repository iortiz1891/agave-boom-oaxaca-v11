"""Same as render_overlay.py but bypasses gdalbuildvrt by reprojecting each shard
individually into a shared destination grid (max-aggregation across shards).

Designed to run in environments without GDAL CLI tools (e.g., a minimal Python
sandbox where only rasterio/numpy are available).

Outputs:
    dashboard/data/pixels_<year>.png   RGBA, transparent except agave pixels
    dashboard/data/pixels_<year>.json  bounds + meta

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json \\
    PYTHONPATH=. python -m src.viz.render_overlay_noxgdal \\
        --bucket ee-ivanortiz-ccdc86-agave \\
        --years 2017,2018,2019 --prob-threshold 0.90 \\
        --dashboard-data dashboard/data
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--years")
    p.add_argument("--pred-prefix", default="predictions_ae/agave_mask_ae_")
    p.add_argument("--local-pred-dir", default=None,
                   help="Local directory holding the prediction shards. "
                        "If set, shards are read from disk instead of /vsigs/.")
    p.add_argument("--dashboard-data", type=Path, default=Path("dashboard/data"))
    p.add_argument("--prob-threshold", type=float, default=0.90)
    p.add_argument("--target-res-deg", type=float, default=0.002)
    p.add_argument("--agave-rgb", default="C29F5E")
    p.add_argument("--alpha", type=int, default=210)
    p.add_argument("--bounds", default="-98.6,15.5,-93.5,18.8",
                   help="WGS84 extent west,south,east,north")
    p.add_argument("--prob-colormap", choices=["traffic", "magenta"],
                   default="traffic",
                   help="Probability gradient. 'traffic' = green→yellow→orange→red "
                        "(more legible at a glance). 'magenta' = single-hue ramp "
                        "matching the binary brand color.")
    return p.parse_args(argv)


def _setup_log():
    log = logging.getLogger("render_overlay_noxgdal")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def _list_shards(bucket, prefix, log):
    from google.cloud import storage
    client = storage.Client()
    pat = re.compile(rf"{re.escape(prefix.rsplit('/', 1)[-1])}(\.tif|\d+-\d+\.tif)$")
    shards = sorted(b.name for b in client.list_blobs(bucket, prefix=prefix)
                    if pat.search(b.name))
    return shards


def _render_year(args, year, log):
    import numpy as np
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    from PIL import Image

    prefix = f"{args.pred_prefix}{year}"
    shards = _list_shards(args.bucket, prefix, log)
    if not shards:
        log.warning("year %d: no shards in GCS, skip", year)
        return
    log.info("year %d: %d shards", year, len(shards))

    w, s, e, n = map(float, args.bounds.split(","))
    res = args.target_res_deg
    out_w = int(round((e - w) / res))
    out_h = int(round((n - s) / res))
    log.info("year %d: target grid %dx%d (~%dm)", year, out_w, out_h, int(res * 111000))

    dst_transform = from_bounds(w, s, e, n, out_w, out_h)
    dst_cls = np.zeros((out_h, out_w), dtype="float32")   # 0 = nodata, 1 = agave
    dst_prob = np.zeros((out_h, out_w), dtype="float32")

    t0 = time.time()
    for i, key in enumerate(shards, 1):
        if args.local_pred_dir:
            uri = str(Path(args.local_pred_dir) / Path(key).name)
        else:
            uri = f"/vsigs/{args.bucket}/{key}"
        try:
            with rasterio.open(uri) as src:
                if src.count < 2:
                    log.warning("  shard %s has <2 bands, skip", key); continue
                # reproject class
                tmp_cls = np.zeros((out_h, out_w), dtype="float32")
                reproject(
                    source=rasterio.band(src, 1),
                    destination=tmp_cls,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs="EPSG:4326",
                    resampling=Resampling.max,
                    src_nodata=float("nan"), dst_nodata=0,
                )
                tmp_prob = np.zeros((out_h, out_w), dtype="float32")
                reproject(
                    source=rasterio.band(src, 2),
                    destination=tmp_prob,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=dst_transform, dst_crs="EPSG:4326",
                    resampling=Resampling.max,
                    src_nodata=float("nan"), dst_nodata=0,
                )
            # Where this shard contributed agave (class==1 & high prob), update
            mask = (tmp_cls == 1) & (tmp_prob >= args.prob_threshold)
            dst_cls[mask] = 1.0
            # Convert from "max prob across both classes" (what predict writes)
            # to true P(agave) before accumulating across shards. Otherwise a
            # non-agave pixel where the model is 0.95 confident gets stored as
            # 0.95 in the prob overlay, indistinguishable from a 0.95-confident
            # agave pixel. For a binary classifier:
            #   if predicted class == 1 (agave):     P(agave) = max_prob
            #   if predicted class == 0 (not_agave): P(agave) = 1 - max_prob
            agave_prob = np.where(tmp_cls == 1, tmp_prob, 1.0 - tmp_prob)
            # IMPORTANT: use np.fmax (NaN-safe) instead of np.maximum, which propagates NaN
            # from shard tiles where the source had no data.
            dst_prob = np.fmax(dst_prob, agave_prob)
        except Exception as ex:  # noqa: BLE001
            log.error("  shard %s FAILED: %s", key, ex)
            continue

        if i % 20 == 0 or i == len(shards):
            elapsed = time.time() - t0
            eta = elapsed * (len(shards) - i) / max(i, 1)
            log.info("  shard %3d / %3d | elapsed %3.0fs | ETA %3.0fs | agave cells so far=%d",
                     i, len(shards), elapsed, eta, int(dst_cls.sum()))

    n_agave = int(dst_cls.sum())
    log.info("year %d: %d agave cells / %d total (%.2f%%)",
             year, n_agave, out_w * out_h, 100.0 * n_agave / (out_w * out_h))

    # ------- RGBA #1: binary (threshold-filtered) -------
    r = int(args.agave_rgb[0:2], 16)
    g = int(args.agave_rgb[2:4], 16)
    b = int(args.agave_rgb[4:6], 16)
    rgba = np.zeros((out_h, out_w, 4), dtype="uint8")
    rgba[..., 0] = r; rgba[..., 1] = g; rgba[..., 2] = b
    rgba[dst_cls == 1, 3] = args.alpha

    args.dashboard_data.mkdir(parents=True, exist_ok=True)
    out_png = args.dashboard_data / f"pixels_{year}.png"
    Image.fromarray(rgba, mode="RGBA").save(out_png, optimize=True)
    log.info("year %d: wrote %s (%.0f KB)",
             year, out_png, out_png.stat().st_size / 1024)

    # ------- RGBA #2: probability gradient (continuous) -------
    # Pixels with prob < min_show are fully transparent. Both hue and alpha
    # vary with probability — high prob is visually distinguishable from low
    # prob even when both are rendered.
    min_show = max(0.3, args.prob_threshold - 0.2)
    prob_safe = np.nan_to_num(dst_prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob_clip = np.clip(prob_safe, 0, 1)
    norm = np.clip((prob_clip - min_show) / max(1e-6, 1.0 - min_show), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)

    if args.prob_colormap == "traffic":
        # Traffic-light gradient: green (low) → yellow → orange → red (high).
        # More legible than single-hue ramps because hue itself communicates
        # magnitude (people read 'red = high' intuitively).
        stops = np.array([
            [ 46, 204,  64],   # green   (low prob, "no concern")
            [255, 220,   0],   # yellow  (mid-low)
            [255, 133,   0],   # orange  (mid-high)
            [220,  20,   0],   # red     (high prob, "definitely agave")
        ], dtype="float32")
        # Alpha ramp: low prob fairly translucent (basemap shows through),
        # high prob opaque (red pops).
        alpha_stops = np.array([90, 170, 220, 245], dtype="float32")
    else:
        # 4-stop magenta gradient (legacy): brand-consistent with binary.
        stops = np.array([
            [255, 220, 235],   # very pale pink (low prob)
            [245, 130, 175],   # light magenta
            [233,  30,  99],   # primary magenta (Binary brand color)
            [140,   0,  60],   # deep crimson (high prob)
        ], dtype="float32")
        alpha_stops = np.array([60, 130, 200, 245], dtype="float32")
    idx = norm * (len(stops) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, len(stops) - 1)
    frac = (idx - i0)[..., None]
    col = stops[i0] * (1 - frac) + stops[i1] * frac
    alpha = alpha_stops[i0] * (1 - frac[..., 0]) + alpha_stops[i1] * frac[..., 0]
    prob_rgba = np.zeros((out_h, out_w, 4), dtype="uint8")
    prob_rgba[..., 0] = col[..., 0]
    prob_rgba[..., 1] = col[..., 1]
    prob_rgba[..., 2] = col[..., 2]
    # Zero alpha for cells below min_show (norm == 0 exactly)
    final_alpha = np.where(norm > 0, alpha, 0.0)
    prob_rgba[..., 3] = final_alpha.astype("uint8")

    out_prob = args.dashboard_data / f"pixels_prob_{year}.png"
    Image.fromarray(prob_rgba, mode="RGBA").save(out_prob, optimize=True)
    log.info("year %d: wrote %s (%.0f KB)",
             year, out_prob, out_prob.stat().st_size / 1024)

    sidecar = {
        "year": year,
        "bounds": [[s, w], [n, e]],
        "agave_color": "#" + args.agave_rgb,
        "threshold": args.prob_threshold,
        "n_agave_cells": n_agave,
        "grid": [out_h, out_w],
        "n_shards": len(shards),
        "method": "per-shard reproject + max-aggregation",
        "prob_overlay": {
            "png": f"pixels_prob_{year}.png",
            "min_show": min_show,
            "colormap": ("traffic-4stop" if args.prob_colormap == "traffic"
                         else "magenta-4stop"),
            "stops": ([
                {"prob": min_show + 0.00, "rgb": "2ECC40"},
                {"prob": min_show + (1.0 - min_show) * 0.33, "rgb": "FFDC00"},
                {"prob": min_show + (1.0 - min_show) * 0.67, "rgb": "FF8500"},
                {"prob": 1.0, "rgb": "DC1400"},
            ] if args.prob_colormap == "traffic" else [
                {"prob": min_show + 0.00, "rgb": "FFDCEB"},
                {"prob": min_show + (1.0 - min_show) * 0.33, "rgb": "F582AF"},
                {"prob": min_show + (1.0 - min_show) * 0.67, "rgb": "E91E63"},
                {"prob": 1.0, "rgb": "8C003C"},
            ]),
        },
    }
    (args.dashboard_data / f"pixels_{year}.json").write_text(json.dumps(sidecar, indent=2))


def main(argv=None):
    args = _parse_args(argv)
    log = _setup_log()
    years = ([int(y.strip()) for y in args.years.split(",")] if args.years
             else [args.year] if args.year else [])
    if not years:
        log.error("Pass --year or --years"); return 1
    t0 = time.time()
    for y in years:
        try:
            _render_year(args, y, log)
        except Exception as ex:  # noqa: BLE001
            log.error("year %d FAILED: %s", y, ex)
    log.info("Done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
