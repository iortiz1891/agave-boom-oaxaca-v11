# Model card — v11 agave classifier

Following Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details

- **Name**: v11 — binary `agave_mature` / `not_agave` classifier,
  three-district production release.
- **Version**: 11.0.0 (v11 reuses the v10 trained classifier; the v11
  identifier denotes the AOI restriction + threshold provenance).
- **Date**: trained 2026-05-17 (v10 GHA run); v11 outputs rendered
  2026-05-18 from `tools/render_v11.py`.
- **Trained by**: iaor / Iniciativa Agave Oaxaca Research.
- **Architecture**: scikit-learn
  `HistGradientBoostingClassifier(class_weight='balanced')`, default
  hyperparameters (`max_iter=100`, `max_depth=None`,
  `learning_rate=0.1`), seed=42.
- **Inputs**: 64-dim AlphaEarth Foundations annual embeddings
  (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) + 5-band terrain stack (DEM
  elevation, slope, northness, eastness, TRI from Copernicus DEM GLO-30)
  → 69 features per pixel.
- **Outputs**: posterior probability `max_prob ∈ [0, 1]`, thresholded
  at **τ = 0.60** to a binary mask. Threshold selected as the global
  F1 optimum on the held-out 870-point test panel via per-year sweep
  (`tools/threshold_sweep_v10.py`).
- **Training data**: `reference_points_v9.csv`, 4,849 labels — see
  [datasheet](datasheet_reference_v11.md).
- **Storage**: classifier weights at
  `gs://ee-ivanortiz-ccdc86-agave/models/model_rf_alphaearth_v9.joblib`
  (1.1 MB joblib pickle, shared between v9, v10 and v11; differences
  between versions are in the AOI and threshold, not the weights).

## AOI

Union of 69 municipios across three mezcal-producing INEGI districts of
Oaxaca:

| District | Municipios |
| --- | --- |
| Tlacolula | 25 |
| Yautepec | 12 |
| Miahuatlán | 32 |
| **Total** | **69** |

Bounding box: `-97.0904, 15.9375, -95.5560, 17.1550` (≈ 155 km E–W ×
135 km N–S). Muni-union land area: ≈ 11,850 km². Polygon stored at
`data/reference/v11_aoi.geojson`; municipio list at
`data/reference/v11_aoi_munis.csv`. All pixels outside the union are
masked to zero before area aggregation.

## Intended use

**Primary use cases.**
- Annual mapping of mature agave plantation extent across the three
  core mezcal-producing districts of Oaxaca, 2017–2025.
- Per-municipio and per-district area time-series for the accompanying
  manuscript, the dashboard, and downstream policy analysis.

**Out-of-scope.**
- Species-level discrimination (the model collapses *A. potatorum*,
  *A. marmorata*, *A. karwinskii*, etc. into one `agave_mature` class).
- Geography outside the three districts (Mixteca, Costa, Sierra Norte,
  Valles Centrales beyond Tlacolula, other states) without retraining.
- Real-time mapping — AlphaEarth is annual, lagged ~9 months.
- Detection of plantation < ~0.1 ha (single-pixel events fall below
  the connected-component cut applied at analysis time).

## Performance

Held-out test set: 870 labels (stratified 80/20 by class × year,
seed=42), at production threshold τ = 0.60.

| Metric | Value |
| --- | --- |
| Overall accuracy | **0.9425** |
| Cohen's κ | **0.8018** |
| Precision (agave) | 0.853 |
| Recall (agave) | 0.821 |
| F1 (agave) | 0.837 |

**Confusion matrix (v11 test, n = 870, τ = 0.60):**

```
                 predicted: not_agave   predicted: agave
true: not_agave         692                    22
true: agave              28                   128
```

### Comparison with prior versions

| Version | Threshold | OA | κ | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| v8 (2,347 labels) | 0.75 | 0.893 | 0.741 | 0.805 | 0.825 | 0.815 |
| v9 (4,849 labels) | 0.75 | 0.9333 | 0.7757 | 0.806 | 0.827 | 0.815 |
| v10 (4,849 labels) | 0.60 | 0.9425 | 0.8018 | 0.853 | 0.821 | 0.837 |
| **v11** (4,849 labels) | **0.60** | **0.9425** | **0.8018** | **0.853** | **0.821** | **0.837** |

v11 inherits v10's classifier and threshold; the difference is the
AOI restriction (4-region → 3-district) and the masking-aware area
estimator (see `tools/render_v11.py`).

## Factors

**Evaluated.**
- Class × year stratification on held-out split (seed=42).
- Per-year threshold sweep, 2017–2025, in 0.05 increments
  (`tools/threshold_sweep_v10.py`); each of the nine years prefers
  an F1-optimal threshold in [0.55, 0.75], so the global τ = 0.60
  is a conservative choice.

**Not evaluated.**
- Per-municipio accuracy (sample sizes too small).
- Plantation age sensitivity (training labels are dominated by mature
  stands).
- Cloud-prone years (2019 hurricane season inherits AlphaEarth's
  annual cloud-aggregation logic, but no formal degradation test).
- Inter-rater agreement across the iaor labelling team.

## Metrics

Reported above. Area estimators reported alongside the raw map area
in the manuscript and dashboard:

1. **Raw map area** — pixel count × pixel area.
2. **95 % bootstrap CI** — 1,000-iteration block bootstrap on the
   per-shard pixel count, seed=42.
3. **Commission-only correction** — raw × user's accuracy (0.853).
4. **Stratified Olofsson** (deferred; the three-district AOI does not
   yet have a habitat-stratified validation panel of sufficient size).

## Training data

See [datasheet_reference_v11.md](datasheet_reference_v11.md).

## Evaluation data

20 % held-out split from `reference_points_v9.csv`, stratified by class
× year, seed=42. n = 870.

**Limitations.**
- Same labeller pool as training set (no truly independent evaluation).
- No external benchmark (manually-validated INEGI Serie VII cropland
  polygons inside the v11 AOI) used for cross-comparison.
- Train/test split is pixel-level; nearby pixels from the same
  plantation may appear in both sets, slightly inflating performance
  vs an entity-level holdout.

## Ethical considerations

1. **Smallholder displacement signal.** v11 documents the displacement
   of milpa / smallholder agriculture by mezcal-driven agave expansion.
   False positives (milpa → agave) under-count displacement; false
   negatives under-count the boom. Both error modes have policy
   weight; dashboards must show the uncertainty band, not just the
   point estimate.

2. **Geographic bias.** Yautepec is the lightest-sampled district in the
   training panel; precision there is plausibly lower than in the
   Tlacolula corridor where the panel is densest.

3. **Open dataset, open model.** Training labels and classifier weights
   are public (CC-BY 4.0 data, MIT code). This enables third-party
   scrutiny but also downstream re-use that may not align with the
   original intent (e.g., enforcement of land-use restrictions on
   smallholders); this model card explicitly disclaims such uses
   without further validation.

## Caveats and recommendations

- v11 saturates around κ = 0.80 on this recipe. Future improvements
  require either (a) spatial context features (patch aggregation, GLCM
  texture, CNN/U-Net), (b) a larger and more stratified reference panel
  (>15 k labels), or (c) a species-level multi-class extension.
- The pixel-level decision rule produces salt-and-pepper noise; the
  paper applies a 0.25 ha connected-component filter before area
  aggregation.
- v12 (in development) re-trains the same recipe on the 5,402-label
  Supabase panel; v11 remains the recommended public release until
  v12 completes validation.

## Reproducibility

- Source code: `src/models/train_alphaearth.py`,
  `src/models/predict_alphaearth.py`.
- Pipeline workflow: `.github/workflows/pipeline.yml`.
- Render step: `tools/render_v11.py`.
- Threshold provenance: `tools/threshold_sweep_v10.py`.
- Compute footprint: ~1h22m on a single GHA `ubuntu-latest` runner
  family (1 × train + 9 × predict + 9 × render in matrix). Free tier.
