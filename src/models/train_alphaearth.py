"""Train a Random Forest classifier on AlphaEarth (64-dim) embeddings.

Equivalent to ``train_features.py`` but reads the AlphaEarth annual embedding
shards instead of our custom 92-band feature stack. Output model is a separate
file so the two pipelines can be compared head-to-head on 2024.

CLI:
    python -m src.models.train_alphaearth \\
        --ref data/reference/reference_points_labeled.csv \\
        --bucket "$GCS_BUCKET" \\
        --out-model data/processed/model_rf_alphaearth.joblib \\
        --out-report reports/training_report_alphaearth.md \\
        [--years 2024]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_AGAVE_LABELS = ("agave_mature", "agave_young", "yes", "agave")
_NOT_AGAVE_LABELS = (
    "milpa_or_annual", "forest", "forest_primary", "secondary_veg",
    "bare_or_urban", "water", "other_perennial",
    "not_agave", "no", "non_agave",
)
DEFAULT_YEARS = list(range(2017, 2026))   # AlphaEarth coverage starts ~2017


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--out-model", type=Path, required=True)
    p.add_argument("--out-report", type=Path, required=True)
    p.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    p.add_argument("--prefix", default="alphaearth/alphaearth_",
                   help="GCS prefix for the AlphaEarth shards.")
    p.add_argument("--tmp-dir", type=Path, default=Path("/tmp/agave_train_ae"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--aoi-bbox", default=None,
                   help="WGS84 'west,south,east,north' — drop reference points outside this bbox before sampling. Cheap; sampling already self-limits to relevant shards.")
    p.add_argument("--clf", choices=["rf", "hgb"], default="rf",
                   help="rf: RandomForestClassifier (legacy default). "
                        "hgb: HistGradientBoostingClassifier (5–10× faster, often equal or better accuracy).")
    p.add_argument("--sample-workers", type=int, default=1,
                   help="ThreadPool size for per-year sampling. Default 1 = sequential (legacy). "
                        "Set to 9 to sample all years concurrently — peak ~1.4GB RAM.")
    p.add_argument("--extra-features", default="",
                   help="Comma-separated CSV columns to append to AlphaEarth features "
                        "(e.g. 'elevation' or 'elevation,slope,aspect'). The columns must "
                        "already exist in --ref. The model is then trained on a "
                        "(64 + len(extras))-dim feature vector. predict_alphaearth.py must "
                        "pass --dem-uri (etc.) to reconstruct the same features per pixel.")
    p.add_argument("--cleanup-shards-per-year", action="store_true",
                   help="Delete each year's prefetched shard directory immediately after "
                        "sampling. Trades cross-year cache reuse for ~9× lower peak disk "
                        "(~3 GB vs ~27 GB). Required on GHA free runners when training on "
                        "the hybrid panel without --aoi-bbox.")
    p.add_argument("--no-prefetch", action="store_true",
                   help="Skip the local shard download and sample directly from /vsigs/. "
                        "For sparse-point training (typical n=5k vs 200M+ pixels per shard) "
                        "the prefetch downloads ~22 GB to read 128 KB — pure waste. "
                        "/vsigs/ + GDAL block cache reads only the tiles containing the "
                        "points (~150 MB total), ~6-10× faster end-to-end and zero disk "
                        "footprint.")
    return p.parse_args(argv)


def _parse_bbox(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--aoi-bbox expects 4 comma-separated floats, got {len(parts)}")
    w, s_, e, n = (float(x) for x in parts)
    return w, s_, e, n


# Tile-id regex: AlphaEarth shards are named alphaearth_<YYYY><tile>.tif
# where <tile> is a row-col offset string identical across years (same EE grid).
_TILE_RE = re.compile(r"alphaearth_\d{4}(.+)\.tif$")


def _tile_id(shard_path):
    m = _TILE_RE.search(shard_path)
    return m.group(1) if m else shard_path


def _shard_intersects_bbox(shard_uri, aoi_w, aoi_s, aoi_e, aoi_n):
    import rasterio
    from rasterio.warp import transform_bounds
    try:
        with rasterio.open(shard_uri) as src:
            sb = src.bounds
            w, s_, e, n = transform_bounds(src.crs, "EPSG:4326",
                                           sb.left, sb.bottom, sb.right, sb.top,
                                           densify_pts=21)
            return not (e < aoi_w or w > aoi_e or n < aoi_s or s_ > aoi_n)
    except Exception:
        return True  # default include on error


# Cross-year tile cache: tile_id -> bool (intersects AOI). The EE export grid is
# identical across years, so we only pay the bounds-check cost once per tile.
_TILE_AOI_CACHE = {}

# Cross-year cache of (w, s, e, n) lat/lon bounds per tile_id. Used by both
# the bbox filter and the points filter so we only open each tile once.
_TILE_BOUNDS_CACHE = {}


def _shard_bounds_latlon(bucket, shard_path):
    """Return (w, s, e, n) lat/lon bounds, cached by tile_id."""
    tid = _tile_id(shard_path)
    if tid in _TILE_BOUNDS_CACHE:
        return _TILE_BOUNDS_CACHE[tid]
    import rasterio
    from rasterio.warp import transform_bounds
    try:
        with rasterio.open(f"/vsigs/{bucket}/{shard_path}") as src:
            sb = src.bounds
            b = transform_bounds(src.crs, "EPSG:4326",
                                 sb.left, sb.bottom, sb.right, sb.top,
                                 densify_pts=21)
        _TILE_BOUNDS_CACHE[tid] = b
        return b
    except Exception:
        _TILE_BOUNDS_CACHE[tid] = None
        return None


def _filter_shards_by_points(bucket, shards, points_lonlat, log=None, max_workers=32):
    """Keep only shards whose (lat/lon) bounds contain at least one ref point.

    Sweet spot when the panel mixes AOI-dense polygons with sparse state-wide
    points (v5/v6 hybrid panels): a bbox filter would either drop the
    state-wide points or have to span all of Oaxaca (pulling all 204 shards).
    This filter drops shards with zero relevant points, typically going from
    204 → 60-100 shards/year.
    """
    from concurrent.futures import ThreadPoolExecutor
    points = list(points_lonlat)
    # Bulk-fetch bounds (parallel, cached)
    todo = [s for s in shards if _tile_id(s) not in _TILE_BOUNDS_CACHE]
    if todo:
        if log: log.info("  fetching bounds for %d new tiles ...", len(todo))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(lambda s: _shard_bounds_latlon(bucket, s), todo))
    kept = []
    for s in shards:
        b = _TILE_BOUNDS_CACHE.get(_tile_id(s))
        if b is None:
            kept.append(s)  # default include if bounds unknown
            continue
        w, s_, e, n = b
        for lon, lat in points:
            if w <= lon <= e and s_ <= lat <= n:
                kept.append(s)
                break
    if log:
        log.info("  point-based shard filter: kept %d / %d shards (%d ref points)",
                 len(kept), len(shards), len(points))
    if not kept:
        raise RuntimeError("No shards contain any reference points")
    return kept


def _filter_shards_by_aoi(bucket, shards, bbox, log=None, max_workers=32):
    """Parallel + cross-year-cached AOI filter."""
    from concurrent.futures import ThreadPoolExecutor
    aoi_w, aoi_s, aoi_e, aoi_n = bbox
    to_check, kept_cached, dropped_cached = [], [], 0
    for s in shards:
        tid = _tile_id(s)
        if tid in _TILE_AOI_CACHE:
            if _TILE_AOI_CACHE[tid]:
                kept_cached.append(s)
            else:
                dropped_cached += 1
        else:
            to_check.append((s, tid))
    if log:
        log.info("  AOI filter: %d shards (cache hits: %d kept, %d drop; %d to check)",
                 len(shards), len(kept_cached), dropped_cached, len(to_check))
    new_kept = []
    if to_check:
        def _check(item):
            s, tid = item
            return tid, _shard_intersects_bbox(f"/vsigs/{bucket}/{s}",
                                               aoi_w, aoi_s, aoi_e, aoi_n)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for tid, keep in ex.map(_check, to_check):
                _TILE_AOI_CACHE[tid] = keep
        new_kept = [s for s, tid in to_check if _TILE_AOI_CACHE[tid]]
    kept = kept_cached + new_kept
    if log: log.info("  AOI filter: kept %d / %d shards", len(kept), len(shards))
    if not kept:
        raise RuntimeError("No shards intersect the AOI bbox")
    return kept


def _binary(c):
    if c in _AGAVE_LABELS: return 1
    if c in _NOT_AGAVE_LABELS: return 0
    return None


def _load_ref(p):
    import pandas as pd
    df = pd.read_csv(p)
    if "split" not in df.columns: df["split"] = "train"
    if "region" not in df.columns: df["region"] = "unknown"
    df["y"] = df["class"].map(_binary)
    df = df.dropna(subset=["y", "lon", "lat", "year"]).copy()
    df["y"] = df["y"].astype(int)
    df["year"] = df["year"].astype(int)
    return df


def _list_shards(bucket, prefix):
    from google.cloud import storage
    client = storage.Client()
    base = prefix.rsplit("/", 1)[-1]
    pat = re.compile(rf"{re.escape(base)}(\.tif|\d+-\d+\.tif)$")
    return sorted(b.name for b in client.list_blobs(bucket, prefix=prefix)
                  if pat.search(b.name))


def _prefetch_shards(bucket, shard_names, local_dir, log=None, max_workers=16):
    """Download GCS shards in parallel to local_dir. Returns list of local Paths.

    AlphaEarth shards are ~5MB each; ~30/year × 9 years = ~1.4GB across a run,
    fits in /tmp on a GHA runner (~14GB free). Local rasterio.sample is 10–50×
    faster than /vsigs/ HTTP range reads for sparse point sampling.
    """
    from concurrent.futures import ThreadPoolExecutor
    from google.cloud import storage
    local_dir.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    bucket_obj = client.bucket(bucket)

    def _dl(name):
        local = local_dir / name.rsplit("/", 1)[-1]
        if local.exists() and local.stat().st_size > 0:
            return local  # cache hit
        bucket_obj.blob(name).download_to_filename(str(local))
        return local

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        local_paths = list(ex.map(_dl, shard_names))
    if log:
        total_mb = sum(p.stat().st_size for p in local_paths) / 1e6
        log.info("  prefetched %d shards (%.1f MB) to %s",
                 len(local_paths), total_mb, local_dir)
    return local_paths


def _build_vrt(bucket, prefix, vrt_path, bbox=None, points=None, log=None, local_dir=None):
    import shutil, subprocess
    if not shutil.which("gdalbuildvrt"):
        raise RuntimeError("gdalbuildvrt not on PATH")
    shards = _list_shards(bucket, prefix)
    if not shards:
        raise RuntimeError(f"No shards at gs://{bucket}/{prefix}*.tif")
    if bbox is not None:
        shards = _filter_shards_by_aoi(bucket, shards, bbox, log=log)
    elif points is not None:
        shards = _filter_shards_by_points(bucket, shards, points, log=log)
    if local_dir is not None:
        local_paths = _prefetch_shards(bucket, shards, local_dir, log=log)
        uris = [str(p) for p in local_paths]
    else:
        uris = [f"/vsigs/{bucket}/{s}" for s in shards]
    lst = vrt_path.with_suffix(".inputs.txt")
    lst.write_text("\n".join(uris) + "\n")
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["gdalbuildvrt", "-overwrite", "-input_file_list", str(lst), str(vrt_path)],
        capture_output=True, text=True,
    )
    lst.unlink(missing_ok=True)
    if res.returncode != 0:
        raise RuntimeError(f"gdalbuildvrt failed: {res.stderr.strip()}")
    return vrt_path, len(shards)


def _sample(year, sub, bucket, prefix, tmp, log, bbox=None, cleanup_shards=False,
            no_prefetch=False):
    import numpy as np
    import shutil
    import rasterio
    from rasterio.warp import transform as rio_t
    vrt_path = tmp / f"alphaearth_{year}.vrt"
    # When no explicit AOI bbox is given, filter shards by the actual ref
    # points they need to cover. Avoids downloading all 204 shards/year for
    # hybrid panels that have a few sparse state-wide points.
    points = None if bbox is not None else list(zip(sub["lon"].tolist(), sub["lat"].tolist()))
    if no_prefetch:
        # Skip the local download — sample directly via /vsigs/. GDAL's block
        # cache (VSI_CACHE) keeps the few tiles we hit warm, so per-point reads
        # are fast after the first one in each tile. Ideal for sparse train
        # sampling: ~150 MB transferred instead of ~22 GB.
        local_dir = None
        vrt, n = _build_vrt(bucket, f"{prefix}{year}", vrt_path, bbox=bbox,
                            points=points, log=log, local_dir=None)
        log.info("year %d: VRT over %d shards (/vsigs/ no-prefetch)", year, n)
    else:
        local_dir = tmp / f"shards_{year}"
        vrt, n = _build_vrt(bucket, f"{prefix}{year}", vrt_path, bbox=bbox,
                            points=points, log=log, local_dir=local_dir)
        log.info("year %d: VRT over %d shards (local prefetch)", year, n)
    with rasterio.open(vrt) as src:
        names = [src.descriptions[i] or f"AE_{i:02d}" for i in range(src.count)]
        xs, ys = rio_t("EPSG:4326", src.crs, sub["lon"].tolist(), sub["lat"].tolist())
        X = np.array(list(src.sample(zip(xs, ys))), dtype="float32")
    vrt.unlink(missing_ok=True)
    log.info("year %d: %d × %d sampled", year, X.shape[0], X.shape[1])
    # Free the per-year shard dir to keep cumulative disk modest. Without this,
    # 9 years × ~3 GB of shards stack to ~27 GB before the model.fit() runs,
    # which has killed v5/v6 on the GHA free runner ('No space left on device').
    if cleanup_shards and local_dir is not None and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
        log.info("year %d: cleaned up local shard dir", year)
    return X, names


def train(args):
    import numpy as np, pandas as pd, joblib
    from src.utils.logging import get_logger
    from src.utils.config import seed as cfg_seed
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

    log = get_logger("train_alphaearth")
    if args.out_model.exists() and not args.force:
        log.info("Model exists; --force to overwrite."); return 0
    seed = cfg_seed("master_seed"); np.random.seed(seed)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    ref = _load_ref(args.ref)
    log.info("Loaded %d reference rows", len(ref))

    bbox = None
    if args.aoi_bbox:
        bbox = _parse_bbox(args.aoi_bbox)
        aoi_w, aoi_s, aoi_e, aoi_n = bbox
        n0 = len(ref)
        m = ref["lon"].between(aoi_w, aoi_e) & ref["lat"].between(aoi_s, aoi_n)
        ref = ref.loc[m].reset_index(drop=True)
        log.info("AOI bbox [%.4f,%.4f,%.4f,%.4f]: kept %d / %d reference rows",
                 aoi_w, aoi_s, aoi_e, aoi_n, len(ref), n0)
        if ref.empty:
            raise RuntimeError("No reference points inside --aoi-bbox")

    # Build per-year work list once
    year_subs = []
    for year in args.years:
        sub = ref[ref["year"] == year]
        if sub.empty:
            log.warning("year %d: no rows; skip", year); continue
        year_subs.append((int(year), sub))

    def _sample_one(item):
        year, sub = item
        try:
            X, ns = _sample(year, sub, args.bucket, args.prefix, args.tmp_dir, log,
                            bbox=bbox,
                            cleanup_shards=args.cleanup_shards_per_year,
                            no_prefetch=args.no_prefetch)
            return year, sub, X, ns, None
        except RuntimeError as e:
            return year, sub, None, None, str(e)

    Xs, ys_, metas, names = [], [], [], []
    if args.sample_workers <= 1:
        results = [_sample_one(it) for it in year_subs]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.sample_workers) as ex:
            results = list(ex.map(_sample_one, year_subs))

    extra_cols = ([c.strip() for c in args.extra_features.split(",") if c.strip()]
                  if args.extra_features else [])
    if extra_cols:
        for c in extra_cols:
            if c not in ref.columns:
                raise RuntimeError(f"--extra-features asked for '{c}' but it's not a column "
                                    f"in {args.ref}. Available: {list(ref.columns)}")

    for year, sub, X, ns, err in sorted(results, key=lambda r: r[0]):
        if err is not None:
            log.warning("year %d skipped: %s", year, err); continue
        Xs.append(X); ys_.append(sub["y"].to_numpy())
        meta_cols = ["region", "split", "year", "plot_id"] + extra_cols
        metas.append(sub[meta_cols].copy())
        names = ns

    if not Xs:
        raise RuntimeError("No samples; AlphaEarth shards not in GCS?")
    X = np.vstack(Xs); y = np.concatenate(ys_)
    meta = pd.concat(metas, ignore_index=True)

    # Append extra features (e.g. DEM elevation) to X. They live in meta but
    # need to ride alongside the AlphaEarth bands so the model treats them as
    # input dimensions. Done BEFORE the finite filter so NaN extras get caught.
    if extra_cols:
        extras = []
        for c in extra_cols:
            col = pd.to_numeric(meta[c], errors="coerce").to_numpy(dtype="float32")
            extras.append(col[:, None])
            names.append(c)
        X_extra = np.concatenate(extras, axis=1)
        X = np.hstack([X, X_extra])
        log.info("Appended %d extra feature(s) %s; X shape now %s",
                 len(extra_cols), extra_cols, X.shape)

    # Filter non-finite AND >50% zero-bands (coverage gaps).
    finite = np.isfinite(X).all(axis=1)
    X, y, meta = X[finite], y[finite], meta.loc[finite].reset_index(drop=True)
    zero_frac = (X == 0).sum(axis=1) / max(X.shape[1], 1)
    keep = zero_frac < 0.5
    n_drop = int((~keep).sum())
    if n_drop:
        log.warning("Dropping %d samples with >50%% zero-bands (coverage gaps)", n_drop)
    X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
    log.info("Training matrix: %s | agave_share=%.3f", X.shape, y.mean())

    tr = meta["split"].isin(["train", "val"])
    te = meta["split"] == "test"
    n_trees = 100 if args.quick else 500
    if args.clf == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        # class_weight='balanced' supported since sklearn 1.4
        clf = HistGradientBoostingClassifier(
            max_iter=n_trees if args.quick else 300,
            max_depth=None, learning_rate=0.1,
            class_weight="balanced", random_state=seed,
        )
        log.info("Classifier: HistGradientBoostingClassifier max_iter=%d", clf.max_iter)
    else:
        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=25, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )
        log.info("Classifier: RandomForestClassifier n_estimators=%d", n_trees)
    clf.fit(X[tr], y[tr])
    pred_te = clf.predict(X[te]) if te.any() else None
    acc = float((pred_te == y[te]).mean()) if pred_te is not None else None
    kappa = float(cohen_kappa_score(y[te], pred_te)) if pred_te is not None else None

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": clf, "feature_names": names,
                 "classes": list(clf.classes_),
                 "source": "alphaearth"}, args.out_model)
    log.info("Saved %s", args.out_model)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w") as f:
        f.write("# Training report — AlphaEarth (features-mode)\n\n")
        f.write(f"- Years used: {sorted(set(meta['year'].tolist()))}\n")
        f.write(f"- Training samples: {int(tr.sum())}\n")
        f.write(f"- Test samples: {int(te.sum())}\n")
        f.write(f"- Agave share (train): {y[tr].mean():.3f}\n")
        f.write(f"- N features: {X.shape[1]}\n")
        f.write(f"- Samples dropped (coverage): {n_drop}\n")
        f.write(f"- Trees: {n_trees}\n")
        if pred_te is not None:
            f.write(f"- Test accuracy: {acc:.4f}\n")
            f.write(f"- Test Cohen's kappa: {kappa:.4f}\n\n")
            cm = confusion_matrix(y[te], pred_te)
            f.write(f"## Confusion matrix\n\n```\n{cm}\n```\n\n")
            f.write("## Classification report\n\n```\n")
            f.write(classification_report(y[te], pred_te,
                                          target_names=["not_agave","agave"]))
            f.write("```\n\n")
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            order = np.argsort(imp)[::-1][:20]
            f.write("## Top 20 MDI importances\n\n```\n")
            for i in order:
                f.write(f"{names[i]:25s} {imp[i]:.4f}\n")
            f.write("```\n")
        else:
            f.write("## Feature importances\n\n_Not available for this estimator "
                    "(HistGradientBoosting < sklearn 1.4 or no MDI exposed)._\n")
    log.info("Wrote %s", args.out_report)
    return 0


def main(argv=None):
    args = _parse_args(argv)
    try: return train(args)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr); return 1


if __name__ == "__main__":
    sys.exit(main())
