"""Predict from AlphaEarth shards per year, COG to GCS.

Mirror of ``predict_features.py`` but consumes the AlphaEarth-trained model
and AlphaEarth feature shards.

CLI:
    python -m src.models.predict_alphaearth \\
        --year 2024 --bucket "$GCS_BUCKET" \\
        --model data/processed/model_rf_alphaearth.joblib \\
        --tmp-dir data/interim/predict_ae
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--features-prefix", default="alphaearth/alphaearth_")
    p.add_argument("--out-prefix", default="predictions_ae/agave_mask_ae_")
    p.add_argument("--tmp-dir", type=Path, default=Path("data/interim/predict_ae"))
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--force", action="store_true")
    p.add_argument("--threads", type=int, default=1,
                   help="Parallel shard workers. I/O-bound, so 4-8 is a good sweet spot on a 2-core runner.")
    p.add_argument("--shard-cache-dir", type=Path, default=None,
                   help="Persist downloaded input shards in this dir (also reads cache hits "
                        "from it). Pair with actions/cache to skip GCS downloads on repeat runs.")
    p.add_argument("--aoi-bbox", default=None,
                   help="WGS84 'west,south,east,north' — skip shards that don't intersect this bbox. Huge speedup for small AOIs.")
    p.add_argument("--dem-uri", default=None,
                   help="Path or /vsigs/ URI to a single-band DEM raster. Reprojected to "
                        "each shard's grid and concatenated as one extra feature.")
    p.add_argument("--terrain-uri", default=None,
                   help="Path or /vsigs/ URI to a multi-band terrain raster (e.g. v8's "
                        "terrain_5band.tif with elevation/slope/northness/eastness/TRI). "
                        "All bands are reprojected to each shard's grid and concatenated "
                        "in order. Mutually exclusive with --dem-uri.")
    return p.parse_args(argv)


def _shard_intersects_bbox(shard_uri, aoi_w, aoi_s, aoi_e, aoi_n):
    """Cheap header-only check: does this shard's footprint touch the AOI bbox?"""
    import rasterio
    from rasterio.warp import transform_bounds
    try:
        with rasterio.open(shard_uri) as src:
            sb = src.bounds   # in src.crs
            w, s, e, n = transform_bounds(src.crs, "EPSG:4326",
                                          sb.left, sb.bottom, sb.right, sb.top,
                                          densify_pts=21)
            # Standard bbox intersection
            return not (e < aoi_w or w > aoi_e or n < aoi_s or s > aoi_n)
    except Exception:
        # If we can't read header, default to including (safer than skipping)
        return True


def _list_shards(bucket, prefix):
    from google.cloud import storage
    client = storage.Client()
    pat = re.compile(rf"{re.escape(prefix.rsplit('/', 1)[-1])}(\.tif|\d+-\d+\.tif)$")
    return sorted(b.name for b in client.list_blobs(bucket, prefix=prefix)
                  if pat.search(b.name))


def _predict_one(feat_uri, out_local, est, classes, chunk, log, local_in=None,
                 dem_uri=None, terrain_uri=None):
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import reproject, Resampling
    src_path = str(local_in) if local_in is not None else feat_uri.replace("gs://", "/vsigs/", 1)
    out_local.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        H, W = src.height, src.width
        prof = src.profile.copy()
        prof.update(count=2, dtype="float32", compress="deflate",
                    predictor=2, tiled=True, blockxsize=512, blockysize=512,
                    BIGTIFF="IF_SAFER", nodata=float("nan"))
        # If a DEM (single-band) or terrain stack (multi-band) is given,
        # reproject each band to this shard's exact grid so we can concat
        # per-window. ~30 m resampled via bilinear to the shard's 10 m grid.
        # dem_uri  → 1 extra band (elevation)
        # terrain_uri → N extra bands (e.g. elevation, slope, northness,
        #               eastness, TRI for v8)
        extra_aligned = None  # shape (n_bands, H, W) when set
        if terrain_uri is not None:
            with rasterio.open(terrain_uri) as t_src:
                n_bands = t_src.count
                extra_aligned = np.zeros((n_bands, H, W), dtype="float32")
                for bi in range(n_bands):
                    reproject(
                        source=rasterio.band(t_src, bi + 1),
                        destination=extra_aligned[bi],
                        src_transform=t_src.transform, src_crs=t_src.crs,
                        dst_transform=src.transform, dst_crs=src.crs,
                        resampling=Resampling.bilinear,
                        src_nodata=t_src.nodata, dst_nodata=float("nan"),
                    )
            log.info("  terrain stack (%d bands) reprojected to %d × %d",
                     n_bands, H, W)
        elif dem_uri is not None:
            extra_aligned = np.zeros((1, H, W), dtype="float32")
            with rasterio.open(dem_uri) as dem_src:
                reproject(
                    source=rasterio.band(dem_src, 1),
                    destination=extra_aligned[0],
                    src_transform=dem_src.transform, src_crs=dem_src.crs,
                    dst_transform=src.transform, dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=dem_src.nodata, dst_nodata=float("nan"),
                )
            log.info("  DEM (1 band) reprojected to %d × %d", H, W)
        with rasterio.open(out_local, "w", **prof) as dst:
            dst.set_band_description(1, "class_id")
            dst.set_band_description(2, "max_prob")
            for row in range(0, H, chunk):
                for col in range(0, W, chunk):
                    h = min(chunk, H - row); w = min(chunk, W - col)
                    win = Window(col, row, w, h)
                    feats = src.read(window=win).astype("float32")
                    pix = feats.reshape(src.count, h * w).T
                    if extra_aligned is not None:
                        extras = extra_aligned[:, row:row+h, col:col+w]
                        extras = extras.reshape(extras.shape[0], h * w).T
                        pix = np.hstack([pix, extras])
                    valid = np.isfinite(pix).all(axis=1)
                    cls = np.full(h * w, np.nan, dtype="float32")
                    prob = np.full(h * w, np.nan, dtype="float32")
                    if valid.any():
                        proba = est.predict_proba(pix[valid])
                        cls[valid] = np.array(classes)[proba.argmax(axis=1)]
                        prob[valid] = proba.max(axis=1)
                    dst.write(cls.reshape(h, w), 1, window=win)
                    dst.write(prob.reshape(h, w), 2, window=win)


def predict(args):
    import joblib
    from google.cloud import storage
    from src.utils.logging import get_logger
    log = get_logger("predict_alphaearth")
    b = joblib.load(args.model)
    est = b["estimator"] if isinstance(b, dict) else b
    classes = list(getattr(est, "classes_", [0, 1]))
    log.info("Loaded model | classes=%s", classes)

    prefix = f"{args.features_prefix}{args.year}"
    shards = _list_shards(args.bucket, prefix)
    if not shards:
        log.error("No shards at gs://%s/%s*", args.bucket, prefix); return 1
    log.info("Year %d: %d AlphaEarth shards (before AOI filter)", args.year, len(shards))

    # Optional: filter shards to those intersecting an AOI bbox (huge speedup for small AOIs)
    if args.aoi_bbox:
        aoi_w, aoi_s, aoi_e, aoi_n = (float(x) for x in args.aoi_bbox.split(","))
        log.info("Filtering shards to AOI bbox: %.4f,%.4f,%.4f,%.4f",
                 aoi_w, aoi_s, aoi_e, aoi_n)
        kept = []
        for key in shards:
            if _shard_intersects_bbox(f"/vsigs/{args.bucket}/{key}",
                                      aoi_w, aoi_s, aoi_e, aoi_n):
                kept.append(key)
        log.info("Year %d: kept %d / %d shards intersecting AOI", args.year, len(kept), len(shards))
        shards = kept
        if not shards:
            log.error("No shards intersect AOI"); return 1

    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    gcs_bucket = client.bucket(args.bucket)
    feat_base = args.features_prefix.rsplit("/", 1)[-1]
    out_base = args.out_prefix.rsplit("/", 1)[-1]

    def _process(idx_key):
        idx, feat_key = idx_key
        name = Path(feat_key).name
        suffix = name[len(f"{feat_base}{args.year}"):].rsplit(".tif", 1)[0]
        out_key = f"{args.out_prefix.rsplit('/', 1)[0]}/{out_base}{args.year}{suffix}.tif"
        if not args.force and gcs_bucket.blob(out_key).exists():
            log.info("[%d/%d] %s exists; skip", idx, len(shards), out_key)
            return ("skip", out_key)
        out_local = args.tmp_dir / f"{out_base}{args.year}{suffix}.tif"
        # Input shards live in shard_cache_dir if set (persists across runs via
        # actions/cache); otherwise in tmp_dir (deleted after each task).
        if args.shard_cache_dir is not None:
            args.shard_cache_dir.mkdir(parents=True, exist_ok=True)
            in_local = args.shard_cache_dir / name
            keep_in = True
        else:
            in_local = args.tmp_dir / f"_in_{name}"
            keep_in = False
        feat_uri = f"gs://{args.bucket}/{feat_key}"
        log.info("[%d/%d] predict %s", idx, len(shards), feat_uri)
        try:
            # Prefetch shard locally; rasterio reads from disk are 10–50× faster
            # than /vsigs/ for the windowed-read loop in _predict_one.
            if not in_local.exists() or in_local.stat().st_size == 0:
                gcs_bucket.blob(feat_key).download_to_filename(str(in_local))
            _predict_one(feat_uri, out_local, est, classes, args.chunk, log,
                         local_in=in_local, dem_uri=args.dem_uri,
                         terrain_uri=args.terrain_uri)
            gcs_bucket.blob(out_key).upload_from_filename(str(out_local))
            log.info("  uploaded gs://%s/%s", args.bucket, out_key)
            return ("ok", out_key)
        except Exception as e:  # noqa: BLE001
            log.error("[%d/%d] FAILED %s -> %s", idx, len(shards), feat_uri, e)
            return ("err", out_key)
        finally:
            if out_local.exists():
                try: out_local.unlink()
                except OSError: pass
            if not keep_in and in_local.exists():
                try: in_local.unlink()
                except OSError: pass

    indexed = list(enumerate(shards, 1))
    if args.threads <= 1:
        for it in indexed:
            _process(it)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        ok = err = skip = 0
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(_process, it) for it in indexed]
            for f in as_completed(futs):
                status, _ = f.result()
                if status == "ok": ok += 1
                elif status == "err": err += 1
                else: skip += 1
        log.info("Year %d summary: ok=%d skip=%d err=%d (threads=%d)",
                 args.year, ok, skip, err, args.threads)
    log.info("Year %d done.", args.year)
    return 0


def main(argv=None):
    args = _parse_args(argv)
    try: return predict(args)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr); return 1


if __name__ == "__main__":
    sys.exit(main())
