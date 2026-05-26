# Reference Data Protocol — DO Mezcal Oaxaca Agave Mapping (2015–2025)

| Field | Value |
| --- | --- |
| Document | `docs/REFERENCE_SAMPLING.md` |
| Companion code | `src/data/reference_sample.py` |
| AOI | DO Mezcal polygon intersected with state of Oaxaca (`configs/aoi.yml`) |
| Years labeled per point | 11 (2015–2025) |
| Target sample size | n = 1,500 (with ≥ 400 candidate-agave points) |
| Status | Authoritative — any deviation requires plan amendment |

This document specifies the formal protocol for collecting the reference data used to (i) train the per-pixel agave classifier and (ii) compute area-adjusted accuracy estimates following Olofsson et al. (2014). The protocol is designed to be statistically defensible, reproducible, and inter-operable with Collect Earth Online (CEO).

---

## 1. Sampling design

### 1.1 Two-stage stratified probability sample

The design is a **two-stage stratified probability sample**, with strata constructed from the cross of (a) the eight statistical regions of Oaxaca and (b) quartiles of a coarse rule-based agave-likelihood mask computed across the AOI.

- **Stage 1 — Stratification.** The AOI is partitioned into mutually exclusive, exhaustive strata defined by `region × likelihood_quartile`. The likelihood mask is a coarse rule-based prior built from (i) climatic envelope (CHIRPS annual precipitation in [400, 1200] mm), (ii) elevation (Copernicus DEM in [800, 2400] m), (iii) slope (in [0°, 35°]) and (iv) NDVI dry-season median in a plausible agave range. The continuous likelihood surface is binned by quartile *within each region*, yielding `8 regions × 4 quartiles = 32` strata.
- **Stage 2 — Within-stratum simple random sample.** Inside each stratum the sample units (point locations) are drawn by simple random sample without replacement, with inclusion probability proportional to allocation / stratum area.

### 1.2 Justification

This design satisfies the GOFC-GOLD / Olofsson area-estimation framework requirements: each AOI pixel has a known, non-zero inclusion probability, the design is probability-based, and the strata are explicit and reproducible. Stratifying by a coarse likelihood prior is the recommended practice for **rare-class** mapping (here: agave is a small fraction of the AOI) — it allows oversampling of likely-agave areas to reduce the variance of producer's accuracy and area estimates for the minority class without biasing the area estimate, because each point retains its design weight (`stratum_area / stratum_n`) in all subsequent estimators (Olofsson et al. 2014, Eq. 1–10; Stehman & Foody 2019, Sections 3 and 5).

Stratifying additionally by region addresses the geographic non-stationarity discussed in H2 of the project plan: the spectral and topographic signature of agave varies between the Valles Centrales / Tlacolula core, the Sierra Sur, and the Mixteca, and reporting per-region accuracy requires adequate sample in every region. Stehman & Foody (2019) explicitly endorse "post-stratification by region" for transparent, region-disaggregated accuracy reporting; we make it part of the *a priori* design rather than post hoc.

References:
- Olofsson, P. *et al.* (2014). Good practices for estimating area and assessing accuracy of land change. *Remote Sensing of Environment* 148: 42–57.
- Stehman, S. V. & Foody, G. M. (2019). Key issues in rigorous accuracy assessment of land cover products. *Remote Sensing of Environment* 231: 111199.

---

## 2. Sample size and allocation

### 2.1 Target

- **Total n = 1,500.**
- **Minimum allocation per stratum: 150** (we have 32 strata, so a uniform allocation would be 47/stratum; we instead enforce a regional floor of 150 and distribute residual sample disproportionately to the high-likelihood quartile).
- **Candidate-agave floor: ≥ 400 points** in the highest likelihood quartile (quartile 4) summed across all regions.

We allocate to strata using a *region-balanced, likelihood-disproportionate* scheme:

1. Split the budget 60% / 40% between the within-region floor (`8 regions × 150 = 1200` from the floor; we use 8×150 = 1,200 distributed proportionally inside each region) and an oversample pot for high-likelihood strata (300 points reserved for quartile-4 strata).
2. Inside each region the 150-point floor is split across the 4 likelihood quartiles as `(15, 25, 45, 65)` to push mass toward the top quartile while keeping the minimum non-trivial.
3. The oversample pot (300 points) is divided across the 8 region × Q4 strata pro rata to estimated agave area, so that the global Q4 total reaches ≥ 400.

### 2.2 Allocation table (target; subject to small rounding)

| Region | Q1 (low) | Q2 | Q3 | Q4 (high) | **Region total** | of which oversample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cañada            | 15 | 25 | 45 | 65 + 25 = 90  | 175 | 25 |
| Costa             | 15 | 25 | 45 | 65 + 20 = 85  | 170 | 20 |
| Istmo             | 15 | 25 | 45 | 65 + 35 = 100 | 185 | 35 |
| Mixteca           | 15 | 25 | 45 | 65 + 40 = 105 | 190 | 40 |
| Papaloapan        | 15 | 25 | 45 | 65 + 15 = 80  | 165 | 15 |
| Sierra Norte      | 15 | 25 | 45 | 65 + 25 = 90  | 175 | 25 |
| Sierra Sur        | 15 | 25 | 45 | 65 + 70 = 135 | 220 | 70 |
| Valles Centrales  | 15 | 25 | 45 | 65 + 70 = 135 | 220 | 70 |
| **Column total**  | 120 | 200 | 360 | 820          | **1,500** | **300** |

- All region totals are ≥ 150 (floor satisfied).
- Q4 total = 820 ≥ 400 (candidate-agave floor satisfied with margin to absorb commission errors in the prior).
- The Q4 oversample is concentrated in Sierra Sur and Valles Centrales, the two regions with the largest known commercial espadín belts. The actual oversample-pot distribution will be re-weighted from a pilot run of 100 points before the full draw.

### 2.3 Inclusion-probability weights

For each drawn point `i` in stratum `s` with allocation `n_s` and area `A_s` (in ha), the design weight is
`w_i = A_s / n_s`,
recorded as `allocation_weight` in the output CSV. Olofsson estimators consume these weights directly; do not discard them.

---

## 3. Class scheme (eight classes)

Each point is labeled per year (2015–2025) into exactly one of the following classes. Visual definitions describe the appearance in (a) Sentinel-2 10 m false-color composites (typical SWIR-NIR-Red, e.g. B12-B8-B4) and (b) sub-meter Google Earth / Planet NICFI imagery.

1. **`agave_mature`** — *Fully established agave plantation.*
   - **Sub-meter (Google Earth / NICFI):** Clear individual rosette structures, typical inter-plant spacing 1.5–3 m, regular row geometry across a contiguous parcel; gray-blue to blue-green tone of the leaves; rows usually contoured along slope. Bare soil between rosettes is generally visible as light buff inter-row strips.
   - **Sentinel-2:** Smooth, low-amplitude NDVI seasonality (intra-annual range typically < 0.2), moderate dry-season NDVI (0.25–0.45), bright SWIR1/SWIR2 due to inter-row bare soil; weak harmonic phase. GLCM contrast elevated where rows are still resolvable.

2. **`agave_young`** — *Recently planted agave, rosettes not yet closed.*
   - **Sub-meter:** Plants visible as discrete dots only at sub-meter resolution; inter-plant bare soil dominant; row geometry visible but rosette canopies do not yet touch.
   - **Sentinel-2:** Spectrum dominated by bare soil; very low NDVI (< 0.20), elevated BSI; can be confused with `bare_or_urban` and with fallow land. Distinguished from fallow only with sub-meter co-interpretation and/or with later-year imagery showing the parcel maturing into a closed rosette plantation.

3. **`milpa_or_annual`** — *Annual cropping system (corn / bean / squash; milpa) or other annual.*
   - **Sub-meter:** Small, irregular parcels, often on terraces or near settlements; visible plowing texture; in dry season completely bare and stubble-textured.
   - **Sentinel-2:** Strong NDVI seasonality with a narrow peak in the rainy season (typically Jul–Sep), large intra-annual amplitude (> 0.4 typical); harmonic-regression phase aligned with monsoon onset; low dry-season NDVI.

4. **`forest`** — *Closed-canopy native vegetation.*
   - **Sub-meter:** Continuous green canopy, no visible soil, irregular crown texture; oak-pine in the highlands, tropical dry deciduous forest at lower elevations on the Pacific slope; canopy color shifts with elevation/biome.
   - **Sentinel-2:** High NDVI year-round in evergreen oak-pine (> 0.6 even in dry season); strong seasonal NDVI in tropical dry forest (high in wet, drops in late dry season but still above bare-soil baseline); low SWIR in evergreen cases; smooth GLCM texture at 10 m.

5. **`secondary_veg`** — *Disturbed shrubland / acahual / matorral / regeneration.*
   - **Sub-meter:** Patchy mix of shrubs, isolated trees, and bare patches; no row geometry; often on abandoned or rotated agricultural land.
   - **Sentinel-2:** Intermediate dry-season NDVI (0.25–0.45), moderate seasonality (less than milpa, more than mature forest); irregular harmonic phase; commonly the most ambiguous class versus `agave_young` and is the primary commission risk for the agave class.

6. **`bare_or_urban`** — *Bare soil, rock outcrops, settlements, paved surfaces, roads.*
   - **Sub-meter:** No vegetation cover; rooftops, streets, quarries, river bars, eroded slopes.
   - **Sentinel-2:** NDVI persistently < 0.15; elevated BSI; in built-up areas, high NDBI and texture from buildings; thermal anomaly possible.

7. **`water`** — *Surface water (perennial or seasonal).*
   - **Sub-meter:** Open water, dam reservoirs, river channels.
   - **Sentinel-2:** Strongly negative NDWI (McFeeters), very low NIR; rare in this AOI but included for completeness so that the eight-class scheme is collectively exhaustive over land + water within the AOI.

8. **`other_perennial`** — *Other permanent woody / perennial crops: coffee, agroforestry, nopal, fruit orchards, sugarcane.*
   - **Sub-meter:** Row geometry like agave but with broader-leaf canopies, taller stature, or evidently different planting pattern (e.g., shade-coffee under tree canopy; nopal as low rectangular paddles); sugarcane tall and monsoon-driven.
   - **Sentinel-2:** Higher NDVI than agave in most months; in coffee, low seasonality and high evergreen NDVI; nopal can mimic young agave at 10 m and requires sub-meter inspection.

The scheme is exhaustive: every land pixel in the AOI is assignable to exactly one class. Mixed pixels are labeled by majority cover within the 10 m × 10 m plot footprint centered on the point.

---

## 4. Interpretation tools

### 4.1 Collect Earth Online (CEO)

- Platform URL: https://app.collect.earth/
- Project setup: a CEO institution and project will be created for `agave-boom-oaxaca`. The plot/sample CSV uploaded to CEO has the schema:

  ```
  lon, lat, plot_id, stratum, year
  ```

  where `year` denotes the labeling year (the CSV is replicated 11 times — once per analysis year — and stacked, so the CEO project surfaces one labeling card per (point, year)). The auto-generated upload file is at `data/reference/ceo_upload.csv`.

- Plot geometry: 10 m × 10 m square (matches the analysis pixel grid in EPSG:6372). Sample-point geometry: a single point at the plot centroid. Both are rendered in CEO over the configured imagery.

### 4.2 Imagery sources used inside CEO

- **Planet NICFI mosaics 2017+** as the primary high-resolution time series (4–5 m, biannual then monthly). NICFI is the workhorse for trajectory questions: when did this parcel become agave?
- **Google Earth historical imagery** 2015–2017 (sub-meter where available) for the pre-NICFI period and specifically for "was it already agave before 2015?" questions; pre-2015 imagery (back to ~2005 in the Valles Centrales) is consulted to resolve ambiguous mature-agave points.
- **Sentinel-2 quarterly composites** (CEO built-in or via custom WMTS) for spectral context.
- **Bing imagery** as a tertiary high-resolution source.

### 4.3 Ancillary time series for ambiguous points

For points where sub-meter interpretation is ambiguous (notably `agave_young` versus `secondary_veg` versus fallow), the interpreter consults:

- **Sentinel-2 NDVI time series** at the point for 2017–2025, plotted from a GEE app linked from the CEO question card.
- **CHIRPS** monthly precipitation for that point's pixel, to disambiguate seasonal vs structural NDVI dynamics.
- **DEM-derived slope and elevation** from Copernicus GLO-30, displayed in the CEO sidebar.

If after consulting the ancillary time series the interpreter still cannot decide, the point is labeled with the best-guess class and `confidence = 1`; the QA review (Section 6) prioritizes such low-confidence points for adjudication.

---

## 5. Per-point form

Each point is captured as one row in `data/reference/labels_raw.csv` with the following columns:

| Column | Type | Description |
| --- | --- | --- |
| `plot_id` | int | Stable identifier of the sample point. |
| `lon` | float | Longitude, EPSG:4326. |
| `lat` | float | Latitude, EPSG:4326. |
| `year_label_2015` | str | Class label for 2015 (one of the eight class names, or `unknown` if imagery insufficient). |
| `year_label_2016` | str | Class label for 2016. |
| `year_label_2017` | str | Class label for 2017. |
| `year_label_2018` | str | Class label for 2018. |
| `year_label_2019` | str | Class label for 2019. |
| `year_label_2020` | str | Class label for 2020. |
| `year_label_2021` | str | Class label for 2021. |
| `year_label_2022` | str | Class label for 2022. |
| `year_label_2023` | str | Class label for 2023. |
| `year_label_2024` | str | Class label for 2024. |
| `year_label_2025` | str | Class label for 2025. |
| `confidence_2015` | int | Ordinal 1 (low) / 2 (medium) / 3 (high). |
| `confidence_2016` | int | Ordinal 1–3. |
| `confidence_2017` | int | Ordinal 1–3. |
| `confidence_2018` | int | Ordinal 1–3. |
| `confidence_2019` | int | Ordinal 1–3. |
| `confidence_2020` | int | Ordinal 1–3. |
| `confidence_2021` | int | Ordinal 1–3. |
| `confidence_2022` | int | Ordinal 1–3. |
| `confidence_2023` | int | Ordinal 1–3. |
| `confidence_2024` | int | Ordinal 1–3. |
| `confidence_2025` | int | Ordinal 1–3. |
| `notes` | str | Free-text. Required when any `confidence_*` ≤ 1 or when the trajectory contains a class transition. |
| `interpreter_id` | str | Initials or HR ID of the labeler. |
| `interpretation_date` | date | ISO date of the labeling session (UTC). |

The 11 per-year columns enable full **trajectory labeling** (e.g. `secondary_veg` → `agave_young` → `agave_mature`), which is essential for the change-detection and HMM stages of the methodology.

---

## 6. QA — second-interpreter agreement

- A **10% random subsample** (n = 150) of points is independently labeled by a second interpreter blind to the first labels.
- Agreement is computed as **Cohen's κ**, both pooled and per-class, on the per-year labels (so 11 κ values per point), and overall on the trajectory class (mode across years).
- Disagreements are resolved by **consensus adjudication** in a moderated session with both interpreters present; the agreed label is recorded with `interpreter_id = "consensus"` and `confidence = 3`.
- Reporting: pooled κ ≥ 0.75 is the acceptance threshold for the reference dataset to proceed to model training; κ between 0.6 and 0.75 triggers a class-scheme refresher and re-labeling of the worst-agreeing class; κ < 0.6 triggers a full re-design of the class definitions and re-labeling.

---

## 7. Train / validation / test split

- A **60 / 20 / 20** split is applied to the cleaned reference set after QA.
- The split is **region-stratified random**: within each of the 8 regions, points are shuffled and partitioned 60/20/20, so each region is represented in each split in proportion to its sample.
- Seed: `configs/seed.yml#splits.test` (= `20251111`) is used for the deterministic split. The training and validation seeds are also drawn from `configs/seed.yml`.
- The **test set is held out from all model selection**: hyperparameter tuning and feature selection use only train + validation; the test set is consulted exactly once, at final-model evaluation, and is the input to the Olofsson area-adjusted estimators.

---

## 8. Output schema (consumed by `src/models/train_rf.py`)

The post-QA reference table is written to `data/reference/labels_clean.csv` with the following exact columns, in this order:

```
plot_id,
lon,
lat,
region,
stratum,
allocation_weight,
year,
class,
confidence,
split,
interpreter_id,
interpretation_date,
qa_status,
notes
```

Notes on the schema:

- The clean table is the **long-form** reshape of the per-point form: one row per (`plot_id`, `year`), so each plot contributes 11 rows.
- `region` ∈ {`canada`, `costa`, `istmo`, `mixteca`, `papaloapan`, `sierra_norte`, `sierra_sur`, `valles_centrales`}.
- `stratum` is the `region × likelihood_quartile` integer in `[0, 31]` matching the stratification in Section 1.1.
- `allocation_weight` is the design weight `A_s / n_s` (ha per point) and is required by the Olofsson estimators.
- `class` ∈ the eight classes of Section 3, plus `unknown`.
- `confidence` ∈ {1, 2, 3}.
- `split` ∈ {`train`, `val`, `test`} (deterministic, see Section 7).
- `qa_status` ∈ {`single`, `double`, `consensus`} indicating whether the point was labeled by one interpreter, double-labeled, or adjudicated.
- `interpretation_date` is ISO 8601.

`src/models/train_rf.py` reads this CSV and joins it on (lon, lat, year) with the per-year feature stack to assemble the model training matrix.

---

## Appendix A — Pilot run

A pilot of 100 points (proportional within strata) is interpreted before the full draw to (i) calibrate the likelihood-prior parameters, (ii) confirm the realism of the Q4 oversample weights, and (iii) validate the CEO project ergonomics. Pilot points are drawn with a separate seed (`sampling.reference_points + 1`) and are *not* re-used in the main sample.

## Appendix B — Reproducibility

The full sample is re-derivable from `configs/aoi.yml`, `configs/seed.yml`, and `src/data/reference_sample.py`. The CEO export is re-derivable from the same script. The QA, split, and clean-table assembly are deterministic given the master seed and the raw labels CSV.
