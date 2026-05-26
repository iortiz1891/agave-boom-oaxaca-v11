"""MVP pilot for v11 — sample monthly S2 NDVI at v9 reference points
and compute 8 phenology features per (lat, lon, year).

Workflow:
  1. Initialize Earth Engine with service account
  2. Build S2 SR HARMONIZED monthly NDVI medians per year (cloud<30%)
  3. For each year in 2017..2025, batch-sample at all reference points of
     that year via reduceRegions (server-side, fast)
  4. Compute 8 derived features per row:
       phen_mean, phen_std, phen_amplitude, phen_cv,
       phen_peak_month, phen_trough_month, phen_q90_q10, phen_n_peaks
  5. Save augmented CSV for v11 training

Outputs:
  data/reference/reference_points_v9_phen.csv
    (= reference_points_v9.csv + 12 monthly NDVI columns + 8 derived features)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REF_CSV = Path("data/reference/reference_points_v9.csv")
OUT_CSV = Path("data/reference/reference_points_v9_phen.csv")
INTERIM = Path("data/interim/phenology")
INTERIM.mkdir(parents=True, exist_ok=True)


def init_ee():
    import ee
    cred = Path(".creds/ee-sa.json").resolve()
    if not cred.exists():
        raise SystemExit(f"missing {cred}")
    sa_email = "ee-ivanortiz@ee-ivanortiz.iam.gserviceaccount.com"
    creds = ee.ServiceAccountCredentials(sa_email, str(cred))
    ee.Initialize(credentials=creds)
    print(f"  EE initialized as {sa_email}")


def monthly_ndvi_image(year: int, month: int):
    """Median NDVI for one (year, month), masked by SCL clouds."""
    import ee
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))
    # SCL classes 4 (vegetation), 5 (bare), 6 (water), 7 (low prob cloud) are kept
    def mask_scl(img):
        scl = img.select('SCL')
        good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return img.updateMask(good)
    s2 = s2.map(mask_scl)
    ndvi = s2.median().normalizedDifference(['B8', 'B4']).rename(
        f'ndvi_y{year}_m{month:02d}')
    return ndvi


def sample_year(year: int, points_features):
    """Sample 12 monthly NDVI at all points for a given year. Returns list of dicts."""
    import ee
    fc = ee.FeatureCollection(points_features)
    out = {}
    for m in range(1, 13):
        ndvi = monthly_ndvi_image(year, m)
        sampled = ndvi.reduceRegions(
            collection=fc, reducer=ee.Reducer.median(), scale=10)
        # Extract just (plot_id, ndvi value)
        info = sampled.select(['plot_id', 'median'], retainGeometry=False).getInfo()
        for f in info['features']:
            pid = f['properties']['plot_id']
            val = f['properties'].get('median')
            out.setdefault(pid, {})[f'm{m:02d}'] = val
        print(f"    {year}-{m:02d}: sampled {len(info['features'])} pts")
    return out


def derive_phenology_features(ndvi_12):
    """Compute 8 phenology metrics from a 12-month NDVI array."""
    import numpy as np
    from scipy.signal import find_peaks
    arr = np.array([v if v is not None else np.nan for v in ndvi_12], dtype=float)
    # If too many missing, return nans
    if np.isfinite(arr).sum() < 4:
        return {k: float('nan') for k in
                ('phen_mean', 'phen_std', 'phen_amplitude', 'phen_cv',
                 'phen_peak_month', 'phen_trough_month', 'phen_q90_q10', 'phen_n_peaks')}
    valid = arr[np.isfinite(arr)]
    mean = float(np.mean(valid))
    std  = float(np.std(valid))
    amp  = float(np.nanmax(arr) - np.nanmin(arr))
    cv   = std / max(abs(mean), 1e-3)
    peak_m   = int(np.nanargmax(arr)) + 1
    trough_m = int(np.nanargmin(arr)) + 1
    q90 = float(np.nanpercentile(valid, 90))
    q10 = float(np.nanpercentile(valid, 10))
    # n peaks via find_peaks on the linearly-imputed series
    series = arr.copy()
    if np.isnan(series).any():
        idx = np.arange(len(series))
        good = np.isfinite(series)
        series = np.interp(idx, idx[good], series[good])
    peaks, _ = find_peaks(series, prominence=0.05)
    return {
        'phen_mean': round(mean, 4),
        'phen_std': round(std, 4),
        'phen_amplitude': round(amp, 4),
        'phen_cv': round(cv, 4),
        'phen_peak_month': peak_m,
        'phen_trough_month': trough_m,
        'phen_q90_q10': round(q90 - q10, 4),
        'phen_n_peaks': int(len(peaks)),
    }


def main():
    import pandas as pd
    df = pd.read_csv(REF_CSV)
    df = df[df['year'] >= 2017].copy().reset_index(drop=True)  # AE/S2 coverage
    print(f"  {len(df):,} reference points (years 2017-2025)")

    init_ee()
    import ee

    all_features = {}  # plot_id -> {m01..m12 ndvi}
    cache_dir = INTERIM
    for year, sub in df.groupby('year'):
        cache = cache_dir / f"ndvi_{year}.csv"
        if cache.exists():
            print(f"  [cache] year {year}: loading from {cache}")
            cdf = pd.read_csv(cache)
            for _, r in cdf.iterrows():
                pid = r['plot_id']
                all_features[pid] = {f'm{m:02d}': r[f'm{m:02d}'] for m in range(1, 13)}
            continue
        print(f"  year {year}: building point FC ({len(sub)} points) ...")
        feats = []
        for _, r in sub.iterrows():
            feats.append(ee.Feature(
                ee.Geometry.Point([r['lon'], r['lat']]),
                {'plot_id': r['plot_id']}))
        year_data = sample_year(int(year), feats)
        # Save year cache
        rows = []
        for pid, mdict in year_data.items():
            row = {'plot_id': pid}
            row.update({f'm{m:02d}': mdict.get(f'm{m:02d}') for m in range(1, 13)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(cache, index=False)
        print(f"  year {year}: cached {len(rows)} rows -> {cache}")
        all_features.update({r['plot_id']: {k: r.get(k) for k in
                                              [f'm{m:02d}' for m in range(1, 13)]}
                              for r in rows})

    # Augment original df with monthly NDVI + 8 derived features
    print(f"\n  computing 8 phenology features per row ...")
    monthly_cols = [f'm{m:02d}' for m in range(1, 13)]
    for col in monthly_cols:
        df[f'ndvi_{col}'] = df['plot_id'].map(
            lambda pid: all_features.get(pid, {}).get(col))
    feat_rows = []
    for _, r in df.iterrows():
        ndvi_12 = [r[f'ndvi_{col}'] for col in monthly_cols]
        feat_rows.append(derive_phenology_features(ndvi_12))
    feat_df = pd.DataFrame(feat_rows)
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    out.to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV} ({len(out)} rows × {len(out.columns)} cols)")
    print(f"  phenology coverage: {feat_df['phen_mean'].notna().sum()} / {len(feat_df)} "
          f"({100 * feat_df['phen_mean'].notna().mean():.1f}%)")


if __name__ == "__main__":
    main()
