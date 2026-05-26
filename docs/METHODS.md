# METHODS

**Project:** *Documenting the Agave Boom in Oaxaca's Mezcal Denominación de Origen Region (2015–2025)*
**Document:** `docs/METHODS.md`
**Version:** v0.1
**Last updated:** 2026-05-09
**Owner:** Methods agent (WS-C)

This document specifies the full processing chain that produces annual *Agave* spp. plantation maps over the Oaxaca portion of the Mezcal Denominación de Origen (DO Mezcal) at 10 m spatial resolution from 2015 through 2025. It is written to be reproducible by an independent team. Section numbering and equation labels are stable and are referenced from the main manuscript and the supplementary information. Where a step is owned by another agent, the cross-reference is given inline.

---

## 1. Area of Interest (AOI) construction

### 1.1 Source polygons

1. **DO Mezcal polygon.** The official Mezcal Denominación de Origen polygon as published by the *Diario Oficial de la Federación* (DOF, 2018 update). The polygon is a single multipart feature digitized from the DOF municipal listing and the corresponding INEGI municipal geometries. We use the DOF municipal listing to assemble the polygon, not the schematic map that accompanies the decree.
2. **INEGI Oaxaca state boundary.** Marco Geoestadístico Integrado (MGI), 2024 release, layer `00ent.shp` filtered to `CVE_ENT == "20"`.

### 1.2 Intersection logic and CRS handling

All vector operations are executed in **EPSG:6372** (Mexico ITRF2008 Lambert Conformal Conic), the official geodesic CRS for cartographic operations over the Mexican territory. Web display layers are reprojected to **EPSG:4326** (WGS84) as the final delivery step only.

Let $D$ be the DO Mezcal polygon and $S$ the Oaxaca state polygon. The AOI is

$$
\mathrm{AOI} = D \cap S \qquad (\text{both reprojected to EPSG:6372 prior to the operation}).
$$

Both inputs are first **buffered by 0 m** (`ST_Buffer(geom, 0)` / `shapely.make_valid`) to repair self-intersections and degenerate rings, then **snapped to a 1 m grid** using `shapely.set_precision(1.0)` to suppress sliver artifacts at the intersection boundary. The intersection is computed with `geopandas.overlay(how="intersection")`, dissolved to a single multipolygon, and exported as `data/processed/aoi.gpkg` (layer `aoi`, EPSG:6372) and `aoi_4326.gpkg` for web display.

### 1.3 Tile / grid scheme

Heavy raster computations are tiled to control Earth Engine memory and to allow incremental re-runs.

- **Sentinel-2 / Sentinel-1 / Landsat tiling.** A 10 km × 10 km regular grid aligned to **EPSG:32614 (UTM 14N)**, which covers the bulk of Oaxaca (Oaxaca spans UTM 13N and 14N; we choose 14N as the dominant zone and accept the small reprojection error in the western Mixteca). The grid origin is snapped to `(easting=300000, northing=1700000)`. Each tile carries a unique key `T{row:02d}{col:02d}`. Tiles that intersect the AOI by less than 1% are dropped.
- **Aggregation grid.** For statistical summaries we additionally use **H3** (Uber hexagonal hierarchical index) at resolution 8 (mean edge ~460 m, area ~0.74 km²) for fine-grained landscape statistics, and resolution 6 (~36 km²) for regional roll-ups.
- **Output rasters.** Final products are written on a single 10 m grid in EPSG:6372 with origin snapped to a multiple of 10 m, BlockxSize = BlockySize = 512, COG-compliant (GDAL ≥ 3.5, `COMPRESS=ZSTD`, `PREDICTOR=2` for floats and `PREDICTOR=1` for byte masks), with internal overviews 2 / 4 / 8 / 16 / 32.

---

## 2. Year-window definition

We define an analysis "year" $y$ as the dry-season window

$$
W_y = [\,\text{1 Nov of year } y-1,\ \text{30 Apr of year } y\,].
$$

This window is the **anchor** for per-pixel feature aggregation. Rationale:

1. **Phenology of *Agave angustifolia*.** Espadín retains rosette structure year-round but its surrounding non-agave matrix (milpa, weedy fallow, deciduous tropical scrub) senesces through the dry season. The contrast between the persistent succulent NDVI and the senescing matrix maximizes class separability between November and April. Wet-season composites suffer from spectral confusion between agave and any healthy herbaceous cover.
2. **Cloud climatology.** Mean MODIS MOD09GA cloud frequency over Oaxaca shows a sharp seasonal minimum from late November through early May (CHIRPS dry-season fraction > 0.85 across most of the AOI). Restricting feature aggregation to this window roughly doubles the number of clear S2 observations per pixel relative to a calendar-year window.
3. **Industry alignment.** SIAP and CRT statistics report on a calendar-year basis, so the year label $y$ corresponds to the agronomic year that closes within calendar year $y$.

For the harmonic regression and intra-annual phenology features (Section 4.3) we use the full calendar year $[\,\text{1 Jan } y,\ \text{31 Dec } y\,]$ to preserve the seasonal cycle; for percentile summaries we use $W_y$.

---

## 3. Sensor preprocessing

### 3.1 Sentinel-2 L2A (2017–2025)

Source collection: `COPERNICUS/S2_SR_HARMONIZED` (Earth Engine), which applies the post-2022 reflectance offset adjustment so that scenes before and after processing baseline 04.00 are on a common scale.

**Cloud and shadow masking.** We use the `s2cloudless` probability product (`COPERNICUS/S2_CLOUD_PROBABILITY`) with the following protocol:

1. Threshold the cloud probability $p_{\mathrm{cld}}$ at $\tau_{\mathrm{cld}} = 40$ to obtain a binary cloud mask $M_{\mathrm{cld}}$.
2. **Dilate** $M_{\mathrm{cld}}$ by a 50 m square structuring element to capture cloud-edge contamination.
3. Project cloud shadows following the Hagolle et al. (2010) geometry: for each cloud pixel at projected position $\mathbf{x}$, the candidate shadow at ground level is $\mathbf{x}' = \mathbf{x} + h \cdot (\tan\theta_s\cos\phi_s,\ \tan\theta_s\sin\phi_s)$, where $\theta_s$ is the solar zenith, $\phi_s$ the solar azimuth, and $h$ the assumed cloud-base height. We sweep $h \in [\,500, 4000\,]$ m at 250 m intervals and intersect the projected shadows with the dark-NIR mask defined by $\mathrm{B8} < 0.15$ and $\mathrm{NDVI} < 0.25$ to retain plausible shadow pixels.
4. Combine cloud + shadow masks; dilate the joint mask by 30 m; mark masked pixels as `NA`.

**Reflectance scaling.** L2A bands 1–12 are rescaled to surface reflectance via $\rho = \mathrm{DN} / 10000$ (post-harmonization), and the BOA offset (subtract 1000 DN before division for scenes with `PROCESSING_BASELINE >= 04.00` not already corrected by the harmonized collection) is verified per scene. Bands B1, B9, B10 are retained for QA only.

**S2A vs S2B harmonization.** Although `S2_SR_HARMONIZED` partly addresses cross-platform drift, we apply the additional band-wise linear adjustment of Saunier et al. (2022) where the regression coefficients are non-trivial (B05, B06, B07, B8A, B11, B12; absolute drift > 0.5%). Coefficients are stored in `configs/s2_harmonization.yml`.

### 3.2 Sentinel-2 L1C for 2015–2016

L2A is not available in Earth Engine for scenes before 2017-03-28. For 2015 and 2016 we **prefer Landsat 8** (Section 3.3), which has full coverage and similar 30 m resolution to a downsampled S2 stack. We did experiment with on-the-fly Sen2Cor on L1C tiles but found the cost / quality trade-off unfavorable for a decadal mosaic and dropped this branch. The final 2015 and 2016 products are therefore **Landsat-only**, and this is flagged in the per-pixel QA layer (`source = LANDSAT_ONLY`) and reflected in the validation strata (Section 10).

### 3.3 Landsat 8 / 9 Collection 2 Level-2 SR (2013–2025)

Source collections: `LANDSAT/LC08/C02/T1_L2` and `LANDSAT/LC09/C02/T1_L2`.

**Cloud masking.** We use the `QA_PIXEL` bit field (CFMask). The mask is the bitwise OR of bits 1 (dilated cloud), 2 (cirrus), 3 (cloud), and 4 (cloud shadow). The mask is dilated by one 30 m pixel.

**Reflectance scaling.** Collection 2 surface reflectance is recovered with the published scaling:

$$
\rho_{\mathrm{SR}} = 2.75\times10^{-5}\cdot \mathrm{DN} - 0.2,
$$

clipped to $[\,0,\,1\,]$. Surface temperature ($\mathrm{ST\_B10}$) is scaled $T_s = 0.00341802\cdot\mathrm{DN} + 149.0$ K and converted to Celsius.

**Cross-sensor harmonization to Sentinel-2.** Landsat 8/9 OLI reflectance is harmonized to S2 MSI using the band-pair linear coefficients of Roy et al. (2016) (`roy2016sensorharm`):

$$
\rho_{\mathrm{S2}} = a_b\,\rho_{\mathrm{OLI}} + b_b, \qquad b \in \{\mathrm{Blue},\mathrm{Green},\mathrm{Red},\mathrm{NIR},\mathrm{SWIR1},\mathrm{SWIR2}\}.
$$

Coefficients $(a_b,b_b)$ are stored in `configs/oli_to_msi.yml`. Indices (NDVI, EVI, etc.) are computed **after** harmonization. Spatial resampling to the 10 m product grid is performed by bilinear interpolation for spectral bands and nearest neighbour for QA layers.

### 3.4 Sentinel-1 GRD

Source: `COPERNICUS/S1_GRD` (IW mode, dual polarization VV+VH, 10 m, ascending and descending). Earth Engine returns GRD products that have already had (1) thermal noise removal, (2) radiometric calibration to $\sigma^0$, and (3) Range-Doppler terrain correction with the SRTM DEM applied.

We layer the following additional steps:

1. **Orbit selection.** Per AOI tile we use the orbit (ascending or descending) with the higher long-term observation count, then pool ascending + descending only if separation is required (we do not, in the baseline).
2. **Convert to $\gamma^0$.** Multiply $\sigma^0$ by $\cos\theta_i / \cos\theta_{\mathrm{ref}}$, with $\theta_{\mathrm{ref}} = 38^\circ$, where $\theta_i$ is the local incidence angle, to obtain a flat-terrain-equivalent backscatter $\gamma^0$.
3. **Conversion to dB.** $\gamma^0_{\mathrm{dB}} = 10\log_{10}\gamma^0$.
4. **Speckle filtering (optional).** A 3×3 Lee filter is offered as a configurable step (`configs/s1.yml: lee_3x3 = false` by default). For texture features (Section 4.5) we use the unfiltered backscatter, since the Lee filter biases GLCM statistics.

---

## 4. Feature engineering

All features below are computed per pixel on the unified 10 m EPSG:6372 grid and stacked into a per-year feature vector $\mathbf{f}_{p,y} \in \mathbb{R}^F$ where $p$ indexes pixels and $y$ indexes years. The feature list and its dimension $F$ are pinned in `configs/features.yml`. As of v0.1 the baseline stack has $F = 96$.

### 4.1 Spectral bands

Sentinel-2 (post-harmonization, after cloud masking and percentile reduction over $W_y$): B2 (Blue, 490 nm), B3 (Green, 560), B4 (Red, 665), B5 (RE1, 705), B6 (RE2, 740), B7 (RE3, 783), B8 (NIR, 842), B8A (NIRn, 865), B11 (SWIR1, 1610), B12 (SWIR2, 2190). Bands B1, B9, B10 are excluded.

### 4.2 Spectral indices

Computed at the per-observation level prior to any temporal reduction:

- **NDVI** (Rouse et al. 1974):

$$
\mathrm{NDVI} = \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}}}.
\tag{1}
$$

- **EVI** (Huete et al. 2002):

$$
\mathrm{EVI} = G\cdot\frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}{\rho_{\mathrm{NIR}} + C_1\rho_{\mathrm{Red}} - C_2\rho_{\mathrm{Blue}} + L},\quad G=2.5,\ C_1=6,\ C_2=7.5,\ L=1.
\tag{2}
$$

- **SAVI** (Huete 1988) with $L=0.5$:

$$
\mathrm{SAVI} = \frac{(\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}})(1+L)}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + L}.
\tag{3}
$$

- **NDMI** (Gao 1996, normalized-difference moisture):

$$
\mathrm{NDMI} = \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR1}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR1}}}.
\tag{4}
$$

- **NDWI** (McFeeters 1996, water index, `mcfeeters1996ndwi`):

$$
\mathrm{NDWI}_{\mathrm{McF}} = \frac{\rho_{\mathrm{Green}} - \rho_{\mathrm{NIR}}}{\rho_{\mathrm{Green}} + \rho_{\mathrm{NIR}}}.
\tag{5}
$$

- **NDWI** (Gao 1996, vegetation water content, `gao1996ndwi`) — identical to NDMI as defined above; we report the McFeeters variant as `NDWI` and the Gao variant as `NDMI` to keep the names disjoint.

- **NBR** (Key & Benson 2006, normalized burn ratio):

$$
\mathrm{NBR} = \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{SWIR2}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{SWIR2}}}.
\tag{6}
$$

- **BSI** (Bare Soil Index, Rikimaru et al. 2002):

$$
\mathrm{BSI} = \frac{(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) - (\rho_{\mathrm{NIR}} + \rho_{\mathrm{Blue}})}{(\rho_{\mathrm{SWIR1}} + \rho_{\mathrm{Red}}) + (\rho_{\mathrm{NIR}} + \rho_{\mathrm{Blue}})}.
\tag{7}
$$

- **kNDVI** (Camps-Valls et al. 2021):

$$
\mathrm{kNDVI} = \tanh\!\left(\left(\frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}}}{2\sigma}\right)^{\!2}\right),\qquad \sigma = 0.5(\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}}).
\tag{8}
$$

### 4.3 Per-year per-pixel statistics

For each index $X \in \{\mathrm{NDVI},\mathrm{EVI},\mathrm{SAVI},\mathrm{NDMI},\mathrm{NDWI},\mathrm{NBR},\mathrm{BSI},\mathrm{kNDVI}\}$ and each spectral band, over the dry-season window $W_y$:

$$
\{P_{10},P_{25},P_{50},P_{75},P_{90}\}(X),\quad \mathrm{IQR}(X) = P_{75}(X) - P_{25}(X),\quad \mu(X),\ \sigma(X).
$$

### 4.4 Harmonic regression on NDVI

On the **calendar-year** NDVI time series at pixel $p$, we fit a 1-cycle and 2-cycle harmonic model (Verbesselt et al. 2010, `verbesselt2010bfast`; Zhu et al. 2015):

$$
\mathrm{NDVI}(t) = \beta_0 + \beta_1 t + \sum_{k=1}^{K}\!\left[a_k\cos(2\pi k t / T) + b_k\sin(2\pi k t / T)\right] + \epsilon(t),\qquad K\in\{1,2\},\ T=365.25\ \text{d}.
\tag{9}
$$

For each $k$ we derive amplitude and phase:

$$
A_k = \sqrt{a_k^2 + b_k^2},\qquad \varphi_k = \mathrm{atan2}(b_k, a_k).
\tag{10}
$$

We retain $\{A_1,\varphi_1,A_2,\varphi_2,\ \beta_1,\ \mathrm{RMSE}_y\}$, where $\mathrm{RMSE}_y$ is the residual standard deviation in window year $y$. Phases are encoded as $(\sin\varphi_k,\cos\varphi_k)$ to avoid the wrap-around discontinuity.

### 4.5 GLCM textures

On a Nov–Mar dry-season median composite of B8 (rescaled to 8-bit per-tile by linear stretch from $[\,P_{2},P_{98}\,]$) and on the same composite of NDVI (rescaled to 8-bit), we compute the Haralick (1973) GLCM statistics over a 5×5 neighbourhood with offsets $\{(1,0),(0,1),(1,1),(1,-1)\}$ averaged. Retained features: contrast, entropy, angular second moment (ASM), homogeneity, correlation. Computed via `ee.Image.glcmTexture(size=5)`.

### 4.6 SAR features

Over $W_y$:

- $\widetilde{\gamma^0_{\mathrm{VV}}}$ (median, dB).
- $\widetilde{\gamma^0_{\mathrm{VH}}}$ (median, dB).
- $\Delta = \widetilde{\gamma^0_{\mathrm{VV}}} - \widetilde{\gamma^0_{\mathrm{VH}}}$ (dB difference; agave row/canopy structure tends to push this difference relative to bare or grassland).
- Std-dev of $\gamma^0_{\mathrm{VV}}$ (linear units, then converted) as a temporal-roughness proxy.
- GLCM contrast on the dry-season VV median.

### 4.7 Topography (time-invariant)

From Copernicus DEM GLO-30 resampled to 10 m by cubic interpolation and reprojected to EPSG:6372:

- Elevation $z$ (m).
- Slope $\alpha$ (degrees) by Horn's method.
- Aspect $\beta$ encoded as $(\sin\beta,\cos\beta)$ to remove the 0/360 discontinuity.
- Topographic Position Index, $\mathrm{TPI}_{30} = z(p) - \overline{z}_{30\text{m}}(p)$ over an annulus of inner radius 30 m, outer radius 300 m.
- Topographic Wetness Index,

$$
\mathrm{TWI} = \ln\!\left(\frac{a}{\tan\alpha}\right),
\tag{11}
$$

where $a$ is the specific upslope contributing area per unit contour length (D-infinity flow accumulation).

### 4.8 Climate

- **CHIRPS v2** monthly precipitation (UCSB) reduced to: annual sum $P_y$, dry-season (Nov–Apr) fraction $f_{\mathrm{dry}} = P_{W_y}/P_y$, anomaly $\Delta P_y = P_y - \overline{P}_{2010\text{–}2020}$.
- **ERA5-Land** monthly air temperature (2 m) reduced to: annual mean $T_y$, anomaly $\Delta T_y = T_y - \overline{T}_{2010\text{–}2020}$.

Climate features are bilinearly resampled to 10 m for inclusion in the per-pixel feature stack; they are spatially smooth and act effectively as regional priors.

---

## 5. Reference data protocol

The full reference-data sampling, labeling, and QA protocol is owned by the reference-sampling agent and lives in `docs/REFERENCE_SAMPLING.md`. This Methods document depends on that protocol but does not replicate it. The integration contract is fixed here:

### 5.1 Class scheme

The reference-sampling protocol assigns one of the following nine **fine classes** to each interpreted point-year:

1. `agave_mature` — visually identifiable agave rows, plants > 2 yr old, spacing characteristic of plantation.
2. `agave_young` — recent planting, rosettes < 2 yr old, often visible bare interrow.
3. `milpa_or_annual` — maize / bean / squash or other annual cropping.
4. `forest_primary` — dense closed canopy, INEGI-style "selva" or "bosque".
5. `secondary_veg` — acahual / disturbed scrub / fallow regrowth.
6. `bare_or_urban` — bare soil, rock, settlement, road.
7. `water` — open water.
8. `other_perennial` — orchard, coffee, sugarcane, other woody crop.
9. `unknown` — interpreter declines to label; never enters training.

### 5.2 Collapse to the binary detector

For the per-pixel binary classifier the labels collapse as

$$
y_{\mathrm{bin}} = \begin{cases} 1\ (\text{agave}) & \text{if label} \in \{\texttt{agave\_mature},\,\texttt{agave\_young}\}, \\ 0\ (\text{not-agave}) & \text{if label} \in \{\texttt{milpa\_or\_annual},\,\texttt{forest\_primary},\,\texttt{secondary\_veg},\,\texttt{bare\_or\_urban},\,\texttt{water},\,\texttt{other\_perennial}\}, \\ \text{drop} & \text{if label} = \texttt{unknown}. \end{cases}
$$

The fine labels are also retained for diagnostic confusion matrices and per-class error-budget reporting.

---

## 6. Classifier

### 6.1 Random Forest baseline

We train a Random Forest (Breiman 2001; Belgiu & Drăguţ 2016, `belgiu2016random`) on the per-pixel feature vector $\mathbf{f}_{p,y}$ jointly with the binary label $y_{\mathrm{bin}}$. Configuration:

- 500 trees.
- Splitting criterion: **Gini impurity**.
- `max_features = sqrt(F)`.
- `min_samples_leaf = 5`.
- Class weights: `balanced` (inverse class frequency).
- Bootstrap: yes.
- Random seed: `seed.yml::rf_seed`.

**Cross-validation.** Region-stratified 5-fold CV, with the strata being the eight Oaxaca statistical regions intersected with the AOI. Each fold leaves one block of regions out (folds re-balanced so that each fold sees ≥ 8% of `agave` points). Mean and standard deviation across folds are reported for accuracy, F1, and AUC; the final operational model is retrained on all training data and held-out test points are not used in CV.

**Feature importance.** Two complementary rankings are reported:

1. **Mean Decrease in Impurity (MDI).** Built-in to scikit-learn `RandomForestClassifier.feature_importances_`.
2. **Permutation importance.** Computed on the held-out test set per Strobl et al. (2007); `sklearn.inspection.permutation_importance`, `n_repeats = 30`, `scoring = "balanced_accuracy"`. Permutation importance is the headline metric reported in the manuscript.

### 6.2 Optional advanced model: TempCNN

As a stretch model we implement TempCNN (Pelletier et al. 2019, `pelletier2019tempcnn`) on monthly Sentinel-2 composites. For each pixel we build a tensor $\mathbf{X}_{p,y}\in\mathbb{R}^{12\times 10}$ of 12 monthly observations × 10 channels (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12). Months without a clear observation are imputed by linear interpolation along the time axis; the imputation mask is concatenated as an 11th channel.

**Architecture:**

- Three **1-D convolutional blocks** along the temporal axis: `Conv1D(64, kernel=5, padding="same")` → `BatchNorm` → `ReLU` → `Dropout(0.3)`, repeated three times with output channels (64, 64, 64).
- **Global Average Pooling** over the temporal axis.
- **Dense(256)** → `ReLU` → `Dropout(0.5)` → **Dense(2)** → `Softmax`.

**Training:**

- Optimizer: Adam, learning rate $10^{-3}$, weight decay $10^{-4}$.
- Loss: weighted cross-entropy with class weights matching the RF baseline.
- Batch size: 256.
- Epochs: 100 with early stopping on validation loss (patience 10).
- Data augmentation: random temporal shift ±1 month, random Gaussian channel noise $\mathcal{N}(0,0.005^2)$.
- Hardware: single A100 (cloud).
- Random seed: `seed.yml::tempcnn_seed`.

The TempCNN output is a per-pixel class probability that is fed into the same temporal post-processing pipeline as the RF.

---

## 7. Per-year prediction

For each year $y\in\{2015,\dots,2025\}$:

1. Build $\mathbf{f}_{p,y}$ for every pixel in the AOI.
2. Apply the trained classifier.
3. Export two rasters:
   - `prob_agave_{y}.tif` — single-band Float32 in $[\,0,\,1\,]$, the predicted probability of class `agave`.
   - `argmax_{y}.tif` — single-band UInt8 with $\{0,1\}$ encoding the argmax class (1 = agave).

Tiles are exported through GEE → Cloud Storage → local mosaic → COG.

---

## 8. Temporal post-processing

The raw per-year argmax is noisy. We apply two complementary regularizations.

### 8.1 Hidden Markov smoothing on the probability sequence

For each pixel $p$, let $\mathbf{o}_p = (o_{p,1},\dots,o_{p,Y}) \in [\,0,1\,]^Y$ be the per-year `agave` probability. Define a hidden state $z_{p,y}\in\{0,1\}$ for the true class. Using a binary HMM with:

- Emission: $P(o_{p,y}\,|\,z_{p,y}=1) = \mathrm{Beta}(o; \alpha_1,\beta_1)$, $P(o_{p,y}\,|\,z_{p,y}=0) = \mathrm{Beta}(o; \alpha_0,\beta_0)$ with parameters fit on the held-out reference set.
- Transition prior favoring **monotonic establishment within ~7 years** and tolerated harvest-replant transitions:

$$
A = \begin{pmatrix} P(0\to0) & P(0\to1) \\ P(1\to0) & P(1\to1) \end{pmatrix} = \begin{pmatrix} 0.92 & 0.08 \\ 0.02 & 0.98 \end{pmatrix}.
$$

The strong $P(1\to1) = 0.98$ encodes plant longevity. The non-zero $P(1\to0) = 0.02$ permits harvest events. Initial distribution $\pi = (0.97, 0.03)$ matches the prior agave prevalence.

We solve for the most probable state sequence with the Viterbi algorithm and, separately, for posterior marginals with the forward-backward algorithm. The smoothed argmax $\hat{z}_{p,y}$ is used downstream; the marginal posteriors are exported for uncertainty maps.

### 8.2 Morphological sieve

After temporal smoothing, each annual binary mask is sieved with a connected-components filter at 8-connectivity, removing connected components of fewer than 4 pixels (= 400 m²), which is the **minimum mapping unit (MMU)**. This MMU is consistent with the typical smallest commercial agave parcel observed in the reference data and with cartographic conventions for 10 m products.

---

## 9. Change detection

### 9.1 Adjacent-year transitions

For each pair $(y, y+1)$ we compute the per-pixel transition

$$
\delta_{p,y\to y+1} = 2\hat{z}_{p,y+1} + \hat{z}_{p,y}\ \in\ \{0,1,2,3\},
$$

mapping to {`stable not-agave`, `loss`, `gain`, `stable agave`}. Per-municipality summaries of gain, loss, and net change are produced as GeoPackage attribute tables.

### 9.2 Establishment year

The establishment year for pixel $p$ is

$$
y^{*}_p = \min\{\,y\,:\ \hat{z}_{p,y}=1\ \text{and}\ \hat{z}_{p,y'}=1\ \forall\ y'\in[\,y, y+1\,]\,\},
$$

i.e. the first year of a sustained `agave` run (≥ 2 consecutive years), which suppresses single-year false positives.

### 9.3 LandTrendr cross-validation

Independently, we run a LandTrendr-like break detection (Kennedy et al. 2010, `kennedy2010landtrendr`) on the annual NDVI percentile-90 trajectory, configured with `maxSegments = 6`, `recoveryThreshold = 0.5`, `pvalThreshold = 0.05`, `bestModelProportion = 0.75`, `minObservationsNeeded = 6`. We extract, for each pixel where LandTrendr identifies a positive-magnitude break, the year of break onset $y^{\mathrm{LT}}_p$. Agreement between $y^{*}_p$ and $y^{\mathrm{LT}}_p$ within ±1 year is reported per region as a cross-validation diagnostic; we do **not** use LandTrendr to override the HMM result.

---

## 10. Accuracy assessment

We follow the **Olofsson et al. (2014) good-practices protocol** (`olofsson2014good`) for stratified-random sampling, unbiased area estimation, and confidence-interval reporting.

### 10.1 Sampling design

Strata $h = 1,\dots,H$ are the cells of the cross-tabulation of the **final binary map class** (agave vs not-agave) with the **eight statistical regions** of Oaxaca, giving $H = 16$ strata. A minimum of $n_h \geq 50$ is enforced for the rare `agave` strata, and the total reference sample is **at least $n=1{,}000$** per evaluated year, with a final pooled total ≥ 1,500 across all years; specifically the sample is allocated by Cochran's optimal allocation modified to oversample the `agave` strata, following Olofsson et al. (2014, eq. 25):

$$
n_h \approx n \cdot \frac{W_h\,S_h}{\sum_{h'} W_{h'} S_{h'}},
$$

with $W_h$ the area proportion of stratum $h$ and $S_h$ a target standard deviation; $S_h$ is set higher for the `agave` strata to push the sample there.

### 10.2 Estimators

Let $p_{ij}$ be the unbiased estimator of the proportion of area mapped as $i$ and referenced as $j$. With strata $h$ corresponding to map class $i$:

$$
\hat{p}_{ij} = W_i\,\frac{n_{ij}}{n_{i\cdot}}.
\tag{12}
$$

**Overall accuracy.**

$$
\hat{O} = \sum_{j} \hat{p}_{jj}.
\tag{13}
$$

**User's accuracy** (commission complement) for class $i$:

$$
\hat{U}_i = \frac{\hat{p}_{ii}}{\sum_j \hat{p}_{ij}} = \frac{n_{ii}}{n_{i\cdot}}.
\tag{14}
$$

**Producer's accuracy** (omission complement) for class $j$:

$$
\hat{P}_j = \frac{\hat{p}_{jj}}{\sum_i \hat{p}_{ij}}.
\tag{15}
$$

**Adjusted area** of class $j$ (Olofsson et al. 2014, eq. 9):

$$
\hat{A}_j = A_{\mathrm{tot}}\sum_i \hat{p}_{ij} = A_{\mathrm{tot}}\sum_i W_i\,\frac{n_{ij}}{n_{i\cdot}}.
\tag{16}
$$

**Standard error of adjusted area** (Olofsson et al. 2014, eq. 10):

$$
S(\hat{A}_j) = A_{\mathrm{tot}}\sqrt{\sum_i W_i^2\,\frac{\hat{u}_{ij}(1-\hat{u}_{ij})}{n_{i\cdot}-1}},\qquad \hat{u}_{ij} = \frac{n_{ij}}{n_{i\cdot}}.
\tag{17}
$$

The 95% confidence interval on adjusted area is

$$
\hat{A}_j \pm 1.96\cdot S(\hat{A}_j).
\tag{18}
$$

The standard errors of overall, user's, and producer's accuracy follow Olofsson et al. (2014), eqs. 5–8 and 11.

### 10.3 Reporting

We report, per year and per region:

- The 2×2 (agave vs not-agave) confusion matrix in counts.
- $\hat{O}$, $\hat{U}_{\mathrm{agave}}$, $\hat{U}_{\mathrm{not}}$, $\hat{P}_{\mathrm{agave}}$, $\hat{P}_{\mathrm{not}}$ with 95% CIs.
- Adjusted area $\hat{A}_{\mathrm{agave}}$ with 95% CI.
- Comparison plot of $\hat{A}_{\mathrm{agave}}$ vs SIAP municipal `agave` planted-area statistic (descriptive only; SIAP is not treated as truth).

The independent test set referred to in the project plan is reserved within the reference sample as a region-stratified 20% holdout that is not used in any model selection step.

---

## 11. Spatial analysis

Spatial analyses are performed at the **municipality** level on annual panels of (i) the adjusted agave area, (ii) the agave-area share of total municipal area, and (iii) the year-on-year expansion rate.

### 11.1 Getis-Ord Gi*

For municipality $i$ with attribute $x_i$ and spatial-weights matrix $\mathbf{W}=(w_{ij})$ (queen contiguity, row-standardized):

$$
G_i^{*} = \frac{\sum_j w_{ij} x_j - \bar{x}\sum_j w_{ij}}{S\sqrt{\dfrac{n\sum_j w_{ij}^2 - \big(\sum_j w_{ij}\big)^2}{n-1}}},
\tag{19}
$$

where $\bar{x} = n^{-1}\sum_i x_i$, $S^2 = n^{-1}\sum_i x_i^2 - \bar{x}^2$, and $n$ is the number of municipalities. $G_i^{*}$ is approximately standard normal under the null of complete spatial randomness; municipalities with $|G_i^{*}|>1.96$ are flagged as 95% hot or cold spots, with FDR correction (Benjamini–Hochberg) for multiple testing across 570 municipalities.

### 11.2 Local Moran's I

For municipality $i$:

$$
I_i = \frac{x_i - \bar{x}}{m_2}\sum_j w_{ij}(x_j - \bar{x}),\qquad m_2 = \frac{1}{n}\sum_i (x_i-\bar{x})^2.
\tag{20}
$$

We classify municipalities into the standard High-High, Low-Low, High-Low, Low-High quadrants and report significance at $p<0.05$ from a 999-permutation test.

### 11.3 Transition matrix

For 2015 → 2025, we cross-tabulate the predicted agave class against the INEGI Series VI (2014) baseline land-cover class on the AOI grid, producing a $K\times2$ transition matrix where $K$ is the number of Series VI classes. Row-normalized values give the share of each prior land-cover class converted to agave by 2025.

### 11.4 Topographic envelope

For each year $y$ we compute kernel density estimates on the joint distribution of $(z, \alpha)$ — elevation in meters, slope in degrees — restricted to **newly classified** agave pixels (i.e., $\hat{z}_{p,y}=1\wedge \hat{z}_{p,y-1}=0$). KDE bandwidth follows Scott's rule. The yearly envelopes are overlaid as a 2-D heatmap in the manuscript.

### 11.5 Driver model

We fit a panel fixed-effects regression with **municipality-year** as the unit:

$$
\Delta A^{\mathrm{agave}}_{i,y} = \alpha_i + \gamma_y + \beta_1 \mathrm{ExportsCRT}_{y-1} + \beta_2\,\mathrm{PalenqueDensity}_{i,y} + \beta_3\,\mathrm{TravelTime}_i + \beta_4\,\mathrm{LULCMix}_{i,y-1} + \beta_5\,\mathrm{PopDensity}_{i,y} + \epsilon_{i,y},
\tag{21}
$$

where $\alpha_i$ are municipality fixed effects, $\gamma_y$ year fixed effects, and standard errors are clustered at the municipality level. `TravelTime` is taken from the Weiss et al. (2018) Oxford global accessibility raster; `LULCMix` is a vector of prior-year land-cover proportions (forest, secondary, milpa, bare, other) from INEGI Series VI/VII updated by our annual `not-agave` predictions. As a complementary specification, a Random Forest regressor is fit on the same panel and SHAP values are reported for cross-method robustness.

---

## 12. Reproducibility

### 12.1 Configuration files

All non-trivial parameters are externalized to YAML in `configs/`:

- `configs/aoi.yml` — DOF / INEGI source paths, intersection options.
- `configs/year_window.yml` — start, end month-day for $W_y$.
- `configs/s2.yml` — cloud threshold, dilation, shadow projection heights.
- `configs/s2_harmonization.yml`, `configs/oli_to_msi.yml` — harmonization coefficients.
- `configs/s1.yml` — orbit selection, optional Lee filter.
- `configs/features.yml` — full feature list with formulas and dependencies.
- `configs/rf.yml`, `configs/tempcnn.yml` — model hyperparameters.
- `configs/hmm.yml` — emission Beta parameters, transition matrix.
- `configs/seed.yml` — single source of truth for all PRNG seeds (`global_seed=20260509`, derived per-stage seeds via `numpy.random.SeedSequence`).

### 12.2 Software versions

Pinned in `environment.yml`:

- Python 3.11
- `earthengine-api` 0.1.401
- `geemap` 0.32.x
- `geopandas` 0.14.x
- `rioxarray` 0.15.x
- `xarray` 2024.x
- `rasterio` 1.3.x
- `pyproj` 3.6.x
- `scikit-learn` 1.4.x
- `xgboost` 2.0.x
- `lightgbm` 4.x
- `pysal` (libpysal 4.9.x, esda 2.5.x)
- `statsmodels` 0.14.x
- `pymannkendall` 1.4.x
- `pytorch` 2.2.x (TempCNN)
- `gdal` 3.8.x

### 12.3 Earth Engine asset IDs

- AOI: `projects/{ee-project}/assets/agave-boom-oaxaca/aoi`
- DEM: `COPERNICUS/DEM/GLO30`
- S2 SR: `COPERNICUS/S2_SR_HARMONIZED`
- S2 cloud probability: `COPERNICUS/S2_CLOUD_PROBABILITY`
- Landsat 8 / 9 SR: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`
- Sentinel-1 GRD: `COPERNICUS/S1_GRD`
- CHIRPS: `UCSB-CHG/CHIRPS/DAILY`
- ERA5-Land: `ECMWF/ERA5_LAND/MONTHLY_AGGR`
- Hansen GFC: `UMD/hansen/global_forest_change_2023_v1_11`

Project assets (per-year feature stacks, predictions, posterior probabilities) are exported under `projects/{ee-project}/assets/agave-boom-oaxaca/{stage}/{year}` with hash-suffixed names.

### 12.4 Expected runtime (reference machine)

Reference machine: 16-core Intel Xeon, 64 GB RAM, 1 TB NVMe; cloud compute via GEE free tier + 1 × A100 hour for TempCNN.

| Stage | Wall time (one full re-run) |
| --- | --- |
| AOI build | < 1 min local |
| S2/S1/L8 ingestion to GEE assets (per year) | 30–90 min GEE wall (asynchronous) |
| Feature stack export per year | 1–3 GEE hours; 4–8 GB GeoTIFF |
| RF training (pooled all years) | 10–20 min local |
| TempCNN training | 1–2 A100 hours |
| Per-year prediction (all 11 years) | 2–4 GEE hours + local mosaic |
| HMM smoothing | 30–60 min local |
| Olofsson accuracy assessment | < 5 min local |
| Spatial analysis | 5–15 min local |

### 12.5 Pipeline orchestration

Stages are wired with **Snakemake**; the DAG is in `Snakefile`. Each rule writes a hashed artifact under `data/processed/`, and a stage is re-run only when its inputs or parameters change. A `make all` convenience wrapper invokes `snakemake --cores 4 --use-conda`.

---

## 13. Equations (consolidated reference)

| # | Quantity | Equation |
| --- | --- | --- |
| (1) | NDVI | $\mathrm{NDVI} = (\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}})/(\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}})$ |
| (2) | EVI | $\mathrm{EVI} = G(\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}})/(\rho_{\mathrm{NIR}} + 6\rho_{\mathrm{Red}} - 7.5\rho_{\mathrm{Blue}} + 1)$ |
| (3) | SAVI | $\mathrm{SAVI} = (\rho_{\mathrm{NIR}} - \rho_{\mathrm{Red}})(1+L)/(\rho_{\mathrm{NIR}} + \rho_{\mathrm{Red}} + L),\ L=0.5$ |
| (8) | kNDVI | $\mathrm{kNDVI} = \tanh\!\big(((\rho_{\mathrm{NIR}}-\rho_{\mathrm{Red}})/(2\sigma))^{2}\big)$ |
| (9) | Harmonic regression | $X(t) = \beta_0 + \beta_1 t + \sum_{k=1}^{K}[a_k\cos(2\pi k t/T) + b_k\sin(2\pi k t/T)] + \epsilon(t)$ |
| (16) | Olofsson adjusted area | $\hat{A}_j = A_{\mathrm{tot}}\sum_i W_i\,n_{ij}/n_{i\cdot}$ |
| (17) | Olofsson SE of adjusted area | $S(\hat{A}_j) = A_{\mathrm{tot}}\sqrt{\sum_i W_i^2\,\hat{u}_{ij}(1-\hat{u}_{ij})/(n_{i\cdot}-1)}$ |
| (19) | Getis-Ord $G_i^{*}$ | $G_i^{*} = \big(\sum_j w_{ij}x_j - \bar{x}\sum_j w_{ij}\big)/\big(S\sqrt{[n\sum_j w_{ij}^2 - (\sum_j w_{ij})^2]/(n-1)}\big)$ |
| (20) | Local Moran's I | $I_i = (x_i-\bar{x})\,m_2^{-1}\sum_j w_{ij}(x_j-\bar{x})$ |

---

## 14. Cross-references

| Topic | Owning document |
| --- | --- |
| Reference data (sampling, labeling, QA) | `docs/REFERENCE_SAMPLING.md` |
| Data inventory and licenses | `docs/DATA_INVENTORY.md` |
| Literature review and full BibTeX | `paper/bibliography/refs.bib` |
| Pipeline DAG | `Snakefile` |
| Parameter pinning | `configs/*.yml` |
| Acceptance criteria | `PROJECT_PLAN.md` §15 |

---

## 15. Selected methodological references (inline keys)

- `olofsson2014good` Olofsson et al. (2014). Good practices for estimating area and assessing accuracy of land change. *RSE* 148, 42–57.
- `roy2016sensorharm` Roy et al. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity. *RSE* 185, 57–70 (and OLI ↔ MSI extension).
- `pelletier2019tempcnn` Pelletier, Webb & Petitjean (2019). Temporal Convolutional Neural Network for the classification of satellite image time series. *Remote Sensing* 11(5), 523.
- `kennedy2010landtrendr` Kennedy et al. (2010). Detecting trends in forest disturbance and recovery using yearly Landsat time series: 1. LandTrendr. *RSE* 114(12), 2897–2910.
- `verbesselt2010bfast` Verbesselt et al. (2010). Detecting trend and seasonal changes in satellite image time series. *RSE* 114(1), 106–115.
- `mcfeeters1996ndwi` McFeeters (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features. *IJRS* 17(7), 1425–1432.
- `gao1996ndwi` Gao (1996). NDWI — a normalized difference water index for remote sensing of vegetation liquid water from space. *RSE* 58(3), 257–266.
- `belgiu2016random` Belgiu & Drăguţ (2016). Random forest in remote sensing: a review of applications and future directions. *ISPRS J. Photogramm. Remote Sens.* 114, 24–31.

The full bibliography is maintained by the lit-review agent under `paper/bibliography/refs.bib` using the same citation keys.
