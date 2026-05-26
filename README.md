# Agave Boom Oaxaca — v11 production release

> **Live dashboard:** <https://iortiz1891.github.io/agave-boom-oaxaca-v11/>

Decadal (2017–2025), 10-meter mapping of *Agave* cultivation across the
three core mezcal-producing INEGI districts of Oaxaca (Tlacolula,
Yautepec, Miahuatlán; **69 municipios, ~11,850 km²**).

This repository is the production-ready snapshot of the v11 classifier,
its training panel, the validated 2017–2025 annual masks, and the
interactive dashboard. The accompanying manuscript (Ortiz, in
preparation) is maintained in a separate repository.

## Headline numbers (v11, threshold τ = 0.60)

| Metric | Value |
| --- | --- |
| Held-out overall accuracy | **0.9425** |
| Cohen's κ | **0.8018** |
| Agave-class precision / recall / F1 | 0.853 / 0.821 / 0.837 |
| Held-out confusion (TN, FP, FN, TP) | 692, 22, 28, 128 |
| Test panel size | 870 labels |
| Mapped agave extent 2017 → 2025 | 4,289 ha → 18,694 ha (**+336 %**) |
| Largest single-year jump | 2024 (+107 %) |

The full per-year area table with 95 % bootstrap CIs lives in
[`dashboard/data/state_totals_v11.csv`](dashboard/data/state_totals_v11.csv).

## Model summary

- **Predictors**: 64-dim Google *AlphaEarth Foundations* annual embeddings
  (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) stacked with a 5-band terrain
  layer (DEM elevation, slope, northness, eastness, TRI from Copernicus
  DEM GLO-30).
- **Classifier**: scikit-learn
  `HistGradientBoostingClassifier(class_weight='balanced')`, defaults
  (`max_iter=100`, `learning_rate=0.1`), seed 42.
- **Training labels**: 4,849 photo-interpreted points (500 unique 80 m
  centroids × multi-year 2016–2025; iaor labelling team).
- **Decision threshold**: τ = 0.60, selected as the global F1 optimum on a
  per-year threshold sweep over the held-out test panel (see
  `tools/threshold_sweep_v10.py`).
- **AOI**: union of 69 municipios across the three mezcal districts. The
  AOI polygon is in `data/reference/v11_aoi.geojson`; the municipio list
  is in `data/reference/v11_aoi_munis.csv`. Predictions outside the union
  are masked to zero.
- **Outputs**: per-year 10 m binary masks + posterior-probability rasters
  for 2017–2025, plus aggregated tables for the dashboard.

Compute runs on GitHub Actions free-tier matrix workflows (one runner per
year). The trained classifier weights, AlphaEarth shards, and per-year
prediction GeoTIFFs are hosted on Google Cloud Storage under
`gs://ee-ivanortiz-ccdc86-agave/`.

## Repository layout

```
agave-boom-oaxaca-v11/
├── README.md                    # You are here
├── LICENSE                      # MIT — code
├── LICENSE-DATA                 # CC-BY 4.0 — derived data products
├── CITATION.cff                 # How to cite
├── environment.yml              # Conda environment (reproducible)
├── requirements.txt             # pip mirror of environment
│
├── configs/                     # Run configs (AOI, dates, seeds)
├── data/
│   └── reference/               # v11 AOI polygon + municipio list
│       ├── v11_aoi.geojson      # 69-municipio union polygon
│       └── v11_aoi_munis.csv    # (district, cve_mun, municipio) table
│
├── src/
│   ├── models/
│   │   ├── train_alphaearth.py   # HGB training (read training CSV from GCS)
│   │   └── predict_alphaearth.py # Per-shard inference
│   ├── viz/
│   │   ├── render_overlay_noxgdal.py  # Binary + probability PNG renderer
│   │   └── style.py                   # Brand colours / probability ramp
│   └── utils/                    # Logging, config loader
│
├── tools/
│   ├── render_v11.py             # Re-render v10 predictions clipped to v11 AOI
│   ├── intersect_inegi_v11.py    # INEGI Serie VII land-cover intersection
│   ├── sample_phenology_v11.py   # Sentinel-2 phenology sampler
│   ├── investigate_replaced_covers_v11.py  # What does agave replace?
│   ├── state_totals_from_pixels.py
│   └── threshold_sweep_v10.py    # Per-year F1 sweep that fixed τ = 0.60
│
├── .github/workflows/
│   └── pipeline.yml              # Train → predict → render matrix workflow
│
├── docs/
│   ├── modelcard_v11.md          # Model card (Mitchell et al. 2019)
│   ├── datasheet_reference_v11.md  # Dataset datasheet (Gebru et al. 2021)
│   ├── METHODS.md                # Pipeline architecture and design choices
│   ├── DATA_INVENTORY.md
│   ├── REFERENCE_SAMPLING.md
│   └── literature_review.md
│
├── dashboard/
│   ├── index.html                # Leaflet dashboard (v11-only)
│   └── data/
│       ├── pixels_v11_<year>.png         # Binary masks (2017–2025)
│       ├── pixels_v11_prob_<year>.png    # Probability rasters
│       ├── pixels_v11_<year>.json        # Bbox / scale metadata
│       ├── state_totals_v11.csv          # Per-year ha + 95 % CI
│       └── replaced_covers_v11_*.csv     # ESA WorldCover + INEGI intersections
│
└── tests/
```

## Quick start

### 1 — Create the environment

```bash
conda env create -f environment.yml
conda activate agave-oaxaca
```

### 2 — Set up credentials

Earth Engine + GCS access requires a service-account key. **Never commit
this file.** Place it at `.creds/ee-sa.json` and export:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/.creds/ee-sa.json"
export GEE_SERVICE_ACCOUNT_KEY="$PWD/.creds/ee-sa.json"
```

### 3 — Browse the live dashboard

```bash
cd dashboard
python -m http.server 8765 &
open http://localhost:8765/
```

### 4 — Re-render the v11 overlays from cached v10 predictions

```bash
PYTHONPATH=. python -m tools.render_v11 --all
```

This pulls v10 prediction shards from
`gs://ee-ivanortiz-ccdc86-agave/predictions_ae_v10/`, mosaics them inside
the 69-municipio AOI, and writes `dashboard/data/pixels_v11_*.{png,json}`
together with `state_totals_v11.csv`.

### 5 — Re-run the full training + inference pipeline on GitHub Actions

```bash
gh workflow run pipeline.yml
```

The matrix runs one job per year, reads the AlphaEarth + terrain shards
from GCS, trains the HGB classifier on the 4,849-label panel, predicts
across the four-region bounding box, and uploads predictions back to
GCS under `predictions_ae_v10/`. v11 then masks these to the 69-muni
AOI via `tools/render_v11.py`.

## Reproducibility

- **Code**: pinned in `environment.yml` and `requirements.txt`.
- **Seeds**: 42 throughout (training, bootstrap CIs).
- **Reference labels**: hashed and versioned in `data/reference/`.
- **AOI definition**: `data/reference/v11_aoi.geojson` (69-muni union).
- **GHA workflow**: `.github/workflows/pipeline.yml` is the authoritative
  recipe; everything else is convenience.
- **Threshold provenance**: `tools/threshold_sweep_v10.py` produces the
  global F1 sweep that fixes τ = 0.60.

## Data availability

| Asset | Location |
| --- | --- |
| Source remote-sensing data | Free public archives (Sentinel-2 / Landsat / Copernicus DEM via Earth Engine) |
| AlphaEarth shards | `gs://ee-ivanortiz-ccdc86-agave/alphaearth/` |
| Trained classifier | `gs://ee-ivanortiz-ccdc86-agave/models/model_rf_alphaearth_v9.joblib` (used for v10/v11) |
| Per-shard predictions | `gs://ee-ivanortiz-ccdc86-agave/predictions_ae_v10/` |
| Reference panel | `gs://ee-ivanortiz-ccdc86-agave/reference/reference_points_v9.csv` |
| Annual rendered overlays | `dashboard/data/pixels_v11_*.png` (this repo) |
| Per-year ha + CIs | `dashboard/data/state_totals_v11.csv` (this repo) |

A Zenodo DOI will be issued at the time of release; the placeholder is
in `CITATION.cff`.

## Licensing

- **Code**: MIT — see [`LICENSE`](LICENSE).
- **Derived data products & figures**: CC-BY 4.0 — see
  [`LICENSE-DATA`](LICENSE-DATA).
- **Source data** (Sentinel-2, Landsat, Copernicus DEM, ESA WorldCover,
  INEGI Serie VII) is governed by the issuing agency's licence.

## Citing

See [`CITATION.cff`](CITATION.cff). A short citation:

> Ortiz, I. (2026). *Decadal mapping of agave cultivation across the
> three core mezcal-producing districts of Oaxaca (Tlacolula, Yautepec,
> Miahuatlán; 2017–2025) using AlphaEarth Foundations embeddings,
> terrain stacks, and a histogram-gradient-boosting classifier with a
> data-driven decision threshold.* GitHub release v11.0.0.
> https://github.com/iortiz1891/agave-boom-oaxaca-v11

## Acknowledgments

- Google DeepMind / Google Earth — *AlphaEarth Foundations* embeddings.
- ESA — Copernicus Sentinel-2 and DEM GLO-30.
- INEGI — Serie VII land-cover and municipio boundaries.
- ESA WorldCover team — 2020 / 2021 global 10 m land cover.
- The *iaor / Iniciativa Agave Oaxaca Research* labelling team for the
  4,849-point reference panel underlying v9/v10/v11.

## Contact

Iván Ortiz — <iortiz1891@gmail.com>
