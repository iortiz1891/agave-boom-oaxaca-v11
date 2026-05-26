# Datasheet — v11 reference panel (`reference_points_v9.csv`, 4,849 labels)

Following Gebru et al. (2021), *Datasheets for Datasets*. The v11 release
uses the same training panel as v9/v10; the v11 identifier denotes the
AOI restriction (three mezcal districts) and the data-driven decision
threshold (τ = 0.60), not a re-labelling.

## Motivation

**For what purpose was the dataset created?**
To train and validate the v9/v10/v11 binary classifier of mature *Agave*
plantations at 10 m resolution across the mezcal-producing districts of
eastern Oaxaca. The panel underpins the manuscript *Decadal mapping of
agave cultivation across the three core mezcal-producing districts of
Oaxaca (2017–2025)*.

**Who created the dataset and on behalf of which entity?**
The iaor (*Iniciativa Agave Oaxaca Research*) labelling team, with PI
Iván Ortiz. No institutional funder.

## Composition

- **Total labels**: 4,849, after dropping rows with missing
  AlphaEarth-embedding or terrain values.
- **Source structure**: 500 unique 80 m centroids, each visually
  interpreted in up to 10 annual mosaics (2016–2025) — yielding up to
  5,000 (centroid × year) rows.
- **Class balance**: 17.0 % positive (`agave_mature`).
- **Spatial coverage**: four INEGI biocultural regions of Oaxaca —
  valles_centrales, sierra_norte, sierra_sur, costa.  The v11 AOI
  restricts inference to the three core mezcal districts (Tlacolula,
  Yautepec, Miahuatlán) but the labelling panel is broader by design,
  to ensure the model sees enough non-agave variety to generalise.
- **Temporal coverage**: 2016–2025.
- **Features per row**: 64-dim AlphaEarth embedding + 5-band terrain
  (elevation, slope, northness, eastness, TRI) = 69 columns + label.

## Collection process

- **Interpretation reference**: Google Satellite high-resolution
  basemap (multi-date), Sentinel-2 false-colour composites pulled
  from Earth Engine, and field-knowledge cross-checks for the
  Tlacolula corridor.
- **Centroid sampling**: stratified random across the four regions,
  with a habitat-stratified over-sample of cropland and shrubland
  to balance the agave class.
- **Annotation tool**: custom Leaflet UI hosted on Supabase
  (`gs://ee-ivanortiz-ccdc86-agave/reference/`).
- **Inter-rater agreement**: not formally measured for v9/v10/v11;
  a small consensus-review subset (5 % of centroids) was
  cross-labelled and showed κ ≈ 0.91 between two labellers.

## Preprocessing / cleaning / labelling

- Drop rows where AlphaEarth or terrain values are missing
  (cloud-saturated pixels, no-data over inland water).
- Coerce label set to two values: `agave_mature` / `not_agave`. Any
  ambiguous label (young agave < 2 yrs, fallow former agave, etc.)
  is removed before training.
- Train/test split: 80/20 stratified by class × year, seed=42.

## Uses

- **Used for**: training and evaluation of v9, v10 and v11 of the
  agave classifier.
- **Could be used for**: training new classifiers (CNNs, time-series
  models), benchmarking against independent panels (e.g.,
  manually-validated INEGI Serie VII), studying inter-rater agreement
  among Mexico-trained labellers.
- **Should not be used for**: enforcement decisions affecting
  smallholders without independent ground-truthing; species-level
  discrimination; geographies outside Oaxaca without re-labelling.

## Distribution

- **License**: CC-BY 4.0 (see `LICENSE-DATA`).
- **Location**: `gs://ee-ivanortiz-ccdc86-agave/reference/reference_points_v9.csv`.
- **Format**: CSV; 4,849 rows × ~75 columns including
  `(lon, lat, year, plot_id, region, label, B01..B64, elev, slope,
  northness, eastness, TRI)`.

## Maintenance

- **Maintainer**: Iván Ortiz (<iortiz1891@gmail.com>).
- **Update cadence**: re-labelling every ~6 months as the iaor team
  adds new centroids. v12 (in preparation) refreshes the panel to
  5,402 labels via the Supabase pipeline.
- **Errata**: filed as GitHub issues in this repository.
