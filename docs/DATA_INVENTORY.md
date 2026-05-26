# Data Inventory

| Field | Value |
| --- | --- |
| Project ID | `agave-boom-oaxaca` |
| Document | `docs/DATA_INVENTORY.md` |
| Version | v0.1 |
| Last updated | 2026-05-09 |
| Maintainer | Ivan Ortiz (PI) |
| Companion documents | `PROJECT_PLAN.md`, `docs/METHODS.md`, `docs/REFERENCE_SAMPLING.md` |
| Coordinate reference (analysis) | EPSG:6372 (México ITRF2008 LCC) |
| Native catalog reference (most rasters) | EPSG:4326 |
| Temporal scope | 2015-01-01 to 2025-12-31 |
| Spatial scope | DO Mezcal polygon clipped to Oaxaca state (≈ 95,364 km²) |

This inventory is a journal-supplementary-quality enumeration of every dataset
ingested by the project. Each entry lists custodian, coverage, native
resolution, projection, format, access method, license, known limitations,
intended use, and recommended citation. URLs are given inline as plain text;
Earth Engine asset IDs are quoted verbatim where applicable. Symbols in the
year-availability table at the end:

- ✓ : Full-coverage, analysis-ready data available for the AOI for that year.
- ~ : Partial coverage, degraded quality, or non-native source (e.g., Landsat
  fallback before Sentinel-2 reaches usable density).
- ✗ : Not available.

---

## A. Optical satellite

### A.1 Sentinel-2 L2A Surface Reflectance (Harmonized)

| Field | Value |
| --- | --- |
| Custodian | ESA Copernicus / processed by Google for the Earth Engine Harmonized collection |
| Spatial coverage | Global; full coverage of Oaxaca AOI |
| Temporal coverage | 2017-03-28 to present (sparse in 2017, dense from 2018) |
| Native resolution | 10 m (B2, B3, B4, B8); 20 m (B5–B7, B8A, B11, B12); 60 m (B1, B9, B10) |
| Projection | UTM zones 14N (EPSG:32614) and 15N (EPSG:32615) over Oaxaca |
| Format | JPEG2000 tiles in SAFE; COG-equivalent in GEE |
| Access | GEE: `COPERNICUS/S2_SR_HARMONIZED`; Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu); AWS Open Data `s3://sentinel-s2-l2a` |
| License | Copernicus Sentinel data — free and open use under Copernicus terms (https://sentinels.copernicus.eu/web/sentinel/terms-conditions) |
| Known limitations | L2A SR not available before 28-Mar-2017 (use L1C + Sen2Cor or use Landsat for 2015–2016); persistent cloud cover in Sierra Sur during Jun–Oct; striping after S2A/S2B disagreement events; baseline change in Jan-2022 (offset 1000) — Harmonized collection corrects this |
| Intended use | Primary 10 m feature source: spectral indices (NDVI, EVI, SAVI, NDWI, NDMI, BSI, NBR), per-year percentile composites, harmonic regression amplitude/phase, GLCM texture |
| Citation | Copernicus Sentinel-2 mission, European Space Agency. Drusch, M. et al. (2012) *Remote Sensing of Environment* 120, 25–36. |

### A.2 Sentinel-2 L1C TOA (2015–2016 backfill)

| Field | Value |
| --- | --- |
| Custodian | ESA Copernicus |
| Spatial coverage | Global; coverage over Oaxaca begins 2015-06-23 (after S2A commissioning) |
| Temporal coverage | 2015-06-23 to present |
| Native resolution | Same as A.1 |
| Projection | UTM 14N / 15N (EPSG:32614 / 32615) |
| Format | JPEG2000 tiles in SAFE |
| Access | GEE: `COPERNICUS/S2_HARMONIZED`; Copernicus Data Space; AWS `s3://sentinel-s2-l1c` |
| License | Copernicus terms (free and open) |
| Known limitations | TOA only — atmospheric correction must be applied locally (Sen2Cor or Py6S) before use as SR; only S2A available 2015 → 2016, so revisit ≥ 10 days; very thin temporal coverage in cloudy season |
| Intended use | Bridges the 2015–2016 gap when L2A is unavailable; used for Landsat–S2 cross-calibration only, not as primary feature source |
| Citation | Same as A.1 |

### A.3 Landsat 8 / 9 Collection 2 Level-2 Surface Reflectance

| Field | Value |
| --- | --- |
| Custodian | USGS / NASA |
| Spatial coverage | Global; AOI fully covered by WRS-2 paths 023–026, rows 047–050 |
| Temporal coverage | L8: 2013-04-11 to present; L9: 2021-10-31 to present |
| Native resolution | 30 m optical (B1–B7); 100 m thermal (B10) resampled to 30 m; 15 m panchromatic |
| Projection | UTM 14N / 15N |
| Format | GeoTIFF (USGS Cloud-Optimized) |
| Access | GEE: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`; USGS EarthExplorer (https://earthexplorer.usgs.gov); STAC at https://landsatlook.usgs.gov |
| License | Public domain (USGS) |
| Known limitations | 30 m resolution loses small smallholder parcels (< 0.5 ha); 16-day revisit means 2015 dry-season median based on ≤ 5 scenes per tile after cloud masking; thermal band required for ET-style features — used as auxiliary only |
| Intended use | Primary feature source for 2015 and 2016; thermal (LST) used as secondary feature in all years; Landsat–S2 harmonization following Roy et al. (2016) |
| Citation | Wulder, M. A. et al. (2019) *Remote Sensing of Environment* 225, 127–147. |

### A.4 MODIS Terra/Aqua Surface Reflectance and NDVI

| Field | Value |
| --- | --- |
| Custodian | NASA LP DAAC |
| Spatial coverage | Global |
| Temporal coverage | 2000-02-24 to present |
| Native resolution | 250 m (MOD09GQ, MOD13Q1); 500 m (MOD09GA); 1 km (MOD11A1 LST) |
| Projection | Sinusoidal |
| Format | HDF-EOS / GeoTIFF |
| Access | GEE: `MODIS/061/MOD13Q1` (NDVI 16-day 250 m), `MODIS/061/MOD11A1` (LST), `MODIS/061/MOD09GA`; LP DAAC (https://lpdaac.usgs.gov) |
| License | Public domain (NASA) |
| Known limitations | 250–1000 m resolution too coarse for parcel-level classification; useful only as a temporally dense reference signal |
| Intended use | Cross-comparison of cloud-free annual composites and as a long-record phenological reference (NDVI seasonal amplitude) for the Valles Centrales / Sierra Sur transition; NOT used as a feature in the per-pixel classifier |
| Citation | Didan, K. (2021) MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V061. |

---

## B. SAR

### B.1 Sentinel-1 GRD (Ground Range Detected)

| Field | Value |
| --- | --- |
| Custodian | ESA Copernicus |
| Spatial coverage | Global; AOI covered by S1A (and S1B until Dec-2021) descending and ascending tracks |
| Temporal coverage | 2014-10-03 to present (single-satellite revisit ≈ 12 days after S1B failure) |
| Native resolution | 10 m (IW mode, GRD High Resolution) |
| Projection | UTM 14N / 15N |
| Format | GeoTIFF (after preprocessing); native is GRD SAFE |
| Access | GEE: `COPERNICUS/S1_GRD` (preprocessed: thermal noise removal, radiometric calibration, terrain correction); Copernicus Data Space; AWS Sentinel-1 Public Dataset |
| License | Copernicus terms (free and open) |
| Known limitations | S1B failure 23-Dec-2021 reduced revisit to 12 days from 6 days; speckle requires multi-temporal averaging or speckle filter (Refined Lee, 5×5 box); GRD is logarithmic (dB), not amplitude — apply `linear()` then `log10()` consistently; terrain shadow on steep north-facing Sierra Sur slopes |
| Intended use | VV, VH, VV/VH ratio annual median + percentiles; row-pattern detection complementary to optical GLCM; cloud-immune signal critical for Jun–Oct in the Sierra Sur |
| Citation | Torres, R. et al. (2012) *Remote Sensing of Environment* 120, 9–24. |

---

## C. Topography and soils

### C.1 Copernicus DEM GLO-30

| Field | Value |
| --- | --- |
| Custodian | European Space Agency |
| Spatial coverage | Near-global (60°S–60°N) |
| Temporal coverage | Static (acquired ≈ 2011–2015 from TanDEM-X) |
| Native resolution | 30 m (1 arc-second) |
| Projection | EPSG:4326 |
| Format | GeoTIFF DTED-equivalent |
| Access | GEE: `COPERNICUS/DEM/GLO30`; AWS Open Data `s3://copernicus-dem-30m`; ESA portal (https://spacedata.copernicus.eu) |
| License | Free for any use under Copernicus DEM End-User License Agreement (https://spacedata.copernicus.eu/documents/20126/0/CSCDA_ESA_Mission-specific+Annex.pdf) |
| Known limitations | Vegetation bias (TanDEM-X measures top-of-canopy); residual voids over water; mosaic seam artifacts in Sierra Sur valleys |
| Intended use | Slope, aspect, TPI, TWI, hillshade for visualization; preferred over SRTM where coverage exists |
| Citation | European Space Agency, Sinergise (2021). Copernicus Global Digital Elevation Model. |

### C.2 SRTM 1-arc-second

| Field | Value |
| --- | --- |
| Custodian | NASA / USGS |
| Spatial coverage | 60°N–56°S |
| Temporal coverage | Static (Feb-2000) |
| Native resolution | 30 m (1 arc-second), void-filled |
| Projection | EPSG:4326 |
| Format | GeoTIFF |
| Access | GEE: `USGS/SRTMGL1_003`; USGS EarthExplorer |
| License | Public domain |
| Known limitations | C-band penetration causes ≈ 2 m bias in dense forest; 25-year-old elevations in active landslide / gully zones |
| Intended use | Cross-validation against Copernicus DEM; fallback for tile edges with GLO-30 voids |
| Citation | Farr, T. G. et al. (2007) *Reviews of Geophysics* 45, RG2004. |

### C.3 SoilGrids v2.0

| Field | Value |
| --- | --- |
| Custodian | ISRIC — World Soil Information |
| Spatial coverage | Global |
| Temporal coverage | Static (2020 release; based on legacy profiles) |
| Native resolution | 250 m |
| Projection | Homolosine (native); EPSG:4326 in derived COGs |
| Format | COG GeoTIFF |
| Access | https://soilgrids.org; WCS and STAC at https://maps.isric.org; GEE community: `projects/soilgrids-isric/...` (multiple property assets) |
| License | CC-BY 4.0 |
| Known limitations | Predictions, not measurements; sparse calibration profiles in Sierra Sur Oaxaca; 250 m too coarse for parcel-level inference |
| Intended use | Clay (%), sand (%), organic carbon (g/kg), pH (H₂O), depth-to-bedrock, all at 0–5 cm and 5–15 cm; covariates for the topographic-edaphic envelope analysis (RQ4) |
| Citation | Poggio, L. et al. (2021) *SOIL* 7, 217–240. |

### C.4 INEGI Edafología (Series II / III)

| Field | Value |
| --- | --- |
| Custodian | Instituto Nacional de Estadística y Geografía (INEGI), Mexico |
| Spatial coverage | National (Mexico) |
| Temporal coverage | Series II (1980s, 1:250,000); Series III (most recent edaphology, 1:250,000) |
| Native scale | 1:250,000 vector |
| Projection | EPSG:6372 (LCC ITRF2008) or EPSG:4326 |
| Format | Shapefile / KMZ |
| Access | https://www.inegi.org.mx/temas/edafologia/ (free download, registration optional) |
| License | "Términos de Libre Uso de la Información del INEGI" (free use with attribution) |
| Known limitations | 1:250,000 generalization; mapping units, not pixel-wise; some polygons unchanged since 1980s |
| Intended use | Cross-validation of SoilGrids and qualitative soil-class context (Leptosols, Vertisols, Regosols dominate the espadín belt) |
| Citation | INEGI (2014) Conjunto de datos vectoriales edafológicos, Serie II/III, escala 1:250,000. |

---

## D. Climate

### D.1 CHIRPS v2.0 (precipitation)

| Field | Value |
| --- | --- |
| Custodian | UCSB Climate Hazards Center / USGS |
| Spatial coverage | 50°S–50°N |
| Temporal coverage | 1981-01-01 to present (≈ 2-month latency for final) |
| Native resolution | 0.05° (≈ 5.5 km) |
| Projection | EPSG:4326 |
| Format | NetCDF / GeoTIFF |
| Access | GEE: `UCSB-CHG/CHIRPS/PENTAD`, `UCSB-CHG/CHIRPS/DAILY`; FTP at https://data.chc.ucsb.edu/products/CHIRPS-2.0/ |
| License | Open / public domain (CHC) |
| Known limitations | Blends station + IR satellite; sparse station network in Sierra Sur reduces local accuracy; 5 km resolution too coarse for orographic gradients |
| Intended use | Annual total precipitation, dry-season fraction, year-of-anomaly indicator; covariate for driver model |
| Citation | Funk, C. et al. (2015) *Scientific Data* 2, 150066. |

### D.2 ERA5-Land monthly aggregated

| Field | Value |
| --- | --- |
| Custodian | ECMWF / Copernicus Climate Change Service (C3S) |
| Spatial coverage | Global |
| Temporal coverage | 1950-01 to present (≈ 2–3-month latency) |
| Native resolution | 0.1° (≈ 9 km) |
| Projection | EPSG:4326 |
| Format | NetCDF / GRIB |
| Access | GEE: `ECMWF/ERA5_LAND/MONTHLY_AGGR`; Copernicus CDS (https://cds.climate.copernicus.eu) |
| License | Copernicus Climate Store license (free, attribution) |
| Known limitations | Coarse resolution unsuited for valley-scale climate; reanalysis biases over complex terrain |
| Intended use | Mean monthly 2 m air temperature, dewpoint, soil temperature, surface net radiation; aridity covariates |
| Citation | Muñoz-Sabater, J. et al. (2021) *Earth System Science Data* 13, 4349–4383. |

### D.3 WorldClim v2.1

| Field | Value |
| --- | --- |
| Custodian | WorldClim project (Fick & Hijmans) |
| Spatial coverage | Global |
| Temporal coverage | 1970–2000 climatological normals |
| Native resolution | 30 arc-seconds (≈ 1 km), also 2.5 / 5 / 10 minute |
| Projection | EPSG:4326 |
| Format | GeoTIFF |
| Access | https://worldclim.org/data/worldclim21.html; GEE: `WORLDCLIM/V1/BIO` (v1) / community v2 mirrors |
| License | CC-BY-SA-4.0 (non-commercial encouraged) |
| Known limitations | Climatology only — does not capture 2015–2025 anomalies; 1 km resolution; based on stations, sparse over Sierra Sur |
| Intended use | Bioclimatic variables (BIO1–BIO19) as static climatic envelope predictors |
| Citation | Fick, S. E. & Hijmans, R. J. (2017) *International Journal of Climatology* 37, 4302–4315. |

---

## E. Land cover and forest change

### E.1 INEGI Uso del Suelo y Vegetación, Serie VI (2014) and Serie VII (2018)

| Field | Value |
| --- | --- |
| Custodian | INEGI, Mexico |
| Spatial coverage | National |
| Temporal coverage | Serie VI: nominal year 2014 (publication 2017); Serie VII: nominal year 2018 (publication 2021) |
| Native scale | 1:250,000 vector polygons |
| Projection | EPSG:6372 (LCC ITRF2008) or EPSG:4326 |
| Format | Shapefile / Geodatabase |
| Access | https://www.inegi.org.mx/temas/usosuelo/ ; CONABIO mirror at http://geoportal.conabio.gob.mx |
| License | Términos de Libre Uso INEGI (attribution) |
| Known limitations | Generalized at 1:250,000; agave-specific class is `AGV` (Agricultura de plantaciones agroindustriales) but plantations < 25 ha are subsumed into matrix classes; mapping unit ≈ 25 ha |
| Intended use | Pre-2015 baseline land cover; transition denominator for change analysis; class-conditioned reference sampling stratification |
| Citation | INEGI (2017) Conjunto de datos vectoriales de uso del suelo y vegetación, Serie VI; INEGI (2021) Serie VII. |

### E.2 ESA WorldCover 2020 / 2021

| Field | Value |
| --- | --- |
| Custodian | ESA / VITO consortium |
| Spatial coverage | Global |
| Temporal coverage | v100: nominal 2020; v200: nominal 2021 |
| Native resolution | 10 m |
| Projection | EPSG:4326 |
| Format | COG GeoTIFF |
| Access | GEE: `ESA/WorldCover/v100`, `ESA/WorldCover/v200`; https://esa-worldcover.org |
| License | CC-BY 4.0 |
| Known limitations | 11-class scheme too coarse for agave (collapses into "Cropland"); known confusion between cropland and herbaceous shrubland in semi-arid zones |
| Intended use | Cross-validation, masking water and built-up; not a training label source |
| Citation | Zanaga, D. et al. (2022) ESA WorldCover 10 m 2021 v200. |

### E.3 Hansen Global Forest Change v1.10+

| Field | Value |
| --- | --- |
| Custodian | University of Maryland (Hansen et al.) |
| Spatial coverage | Global |
| Temporal coverage | 2000-baseline tree cover; annual loss 2001–2024 (v1.12, latest at time of writing) |
| Native resolution | 30 m (1 arc-second) |
| Projection | EPSG:4326 |
| Format | GeoTIFF |
| Access | GEE: `UMD/hansen/global_forest_change_2024_v1_12` (update annually); https://earthenginepartners.appspot.com/science-2013-global-forest |
| License | Free for non-commercial / academic use; commercial requires permission |
| Known limitations | "Forest" defined as ≥ 5 m canopy — agave plantations in dry deciduous matrix are not classified as forest loss when removed; 30 m smooths small clearings |
| Intended use | Independent forest-loss layer for cross-validation of agave-on-forest transitions (H3) |
| Citation | Hansen, M. C. et al. (2013) *Science* 342, 850–853. |

### E.4 Dynamic World v1

| Field | Value |
| --- | --- |
| Custodian | Google / World Resources Institute |
| Spatial coverage | Global (where Sentinel-2 acquired) |
| Temporal coverage | 2015-06-23 to present (per S2 acquisition) |
| Native resolution | 10 m |
| Projection | UTM (per S2 tile) |
| Format | GEE near-real-time COG |
| Access | GEE: `GOOGLE/DYNAMICWORLD/V1` |
| License | CC-BY 4.0 |
| Known limitations | 9-class probabilistic — agave again subsumed into "Crops"; moderate confusion crops vs. shrub/scrub; per-image — must aggregate to annual |
| Intended use | Per-pixel class probabilities (especially `crops` and `shrub_and_scrub`) as auxiliary features; consistency check with our model |
| Citation | Brown, C. F. et al. (2022) *Scientific Data* 9, 251. |

---

## F. Administrative

### F.1 INEGI Marco Geoestadístico 2024

| Field | Value |
| --- | --- |
| Custodian | INEGI, Mexico |
| Spatial coverage | National |
| Temporal coverage | 2024 release (boundaries reflect 2020 census + post-census updates) |
| Native scale | 1:250,000 vector |
| Projection | EPSG:6372 |
| Format | Shapefile / Geodatabase |
| Access | https://www.inegi.org.mx/temas/mg/ |
| License | Términos de Libre Uso INEGI (attribution) |
| Known limitations | Some inter-municipal boundary disputes still unresolved (especially Mixteca); slight offsets vs. older Marco editions |
| Intended use | State of Oaxaca outline, eight statistical region boundaries, all 570 municipal polygons; primary AOI definition together with the DO Mezcal polygon |
| Citation | INEGI (2024) Marco Geoestadístico, versión 2024. |

---

## G. Statistics and industry

### G.1 SIAP Cierre Agrícola (anual)

| Field | Value |
| --- | --- |
| Custodian | Servicio de Información Agroalimentaria y Pesquera (SIAP), SADER, Mexico |
| Spatial coverage | National, by municipality |
| Temporal coverage | 1980–latest closed cycle (currently 2023; 2024 cycle closes mid-2025) |
| Native resolution | Municipal aggregate |
| Format | CSV / XLSX |
| Access | https://nube.siap.gob.mx/cierreagricola/ ; legacy https://www.gob.mx/siap/acciones-y-programas/produccion-agricola-33119 |
| License | Información de Libre Uso (gob.mx) |
| Known limitations | Reported agave area is widely acknowledged to under-count smallholder espadín and exclude wild silvestre harvest; categories `Agave mezcalero`, `Agave tequilero` not consistently disambiguated; lag of 12–18 months |
| Intended use | Cross-validation of map-derived municipal area (NOT as ground truth); driver-model covariate (lagged) |
| Citation | SIAP (2024) Cierre de la producción agrícola por municipio. |

### G.2 CRT — Consejo Regulador del Mezcal

| Field | Value |
| --- | --- |
| Custodian | Consejo Regulador del Mezcal A.C. |
| Spatial coverage | DO Mezcal (9 states, here restricted to Oaxaca) |
| Temporal coverage | 2011–present (annual reports); palenque registry updated continuously |
| Native resolution | National and state totals; municipal palenque registry |
| Format | PDF (annual reports), web tables, downloadable CSV from `Estadísticas` page |
| Access | https://www.crm.org.mx/estadisticas.html ; palenque registry at https://www.crm.org.mx/inscritos.html |
| License | CRT publishes for public use; attribution expected; redistribution of registry restricted |
| Known limitations | Captures only certified production; uncertified mezcal (estimated 30–50% of national production) is invisible; palenque addresses geocode imperfectly |
| Intended use | National export volume time series (driver of demand); palenque density per municipality; annual production by category |
| Citation | CRM (2024) Informe estadístico 2024, Consejo Regulador del Mezcal. |

### G.3 INEGI Censo de Población y Vivienda (2010, 2020)

| Field | Value |
| --- | --- |
| Custodian | INEGI |
| Spatial coverage | National |
| Temporal coverage | 2010, 2020 (decennial) |
| Native resolution | Locality, AGEB, manzana (urban); locality only (rural) |
| Format | CSV, ITER tables, shapefiles |
| Access | https://www.inegi.org.mx/programas/ccpv/2020/ |
| License | Términos de Libre Uso INEGI |
| Known limitations | Decennial — interpolation needed for intervening years; rural locality counts approximate |
| Intended use | Population and indigenous-language speaker controls in the driver model; rural / urban classification |
| Citation | INEGI (2021) Censo de Población y Vivienda 2020. |

### G.4 INEGI Censo Agropecuario 2022

| Field | Value |
| --- | --- |
| Custodian | INEGI |
| Spatial coverage | National |
| Temporal coverage | Reference year 2022 (results published 2023–2024) |
| Native resolution | Municipal aggregate; some state-level breakdowns |
| Format | CSV / XLSX |
| Access | https://www.inegi.org.mx/programas/ca/2022/ |
| License | Términos de Libre Uso INEGI |
| Known limitations | First agricultural census since 2007 — comparability issues; questions on agave do not disambiguate species |
| Intended use | Tenure structure (ejido / comunal / privada), parcel-size distribution, mechanization indicators as driver covariates |
| Citation | INEGI (2024) Censo Agropecuario 2022. |

### G.5 INEGI Encuesta Nacional Agropecuaria (ENA)

| Field | Value |
| --- | --- |
| Custodian | INEGI |
| Spatial coverage | National sample survey |
| Temporal coverage | 2012, 2014, 2017, 2019, 2022 |
| Native resolution | State-level estimates with sample weighting |
| Format | CSV / SPSS micro-data |
| Access | https://www.inegi.org.mx/programas/ena/2022/ |
| License | Términos de Libre Uso INEGI |
| Known limitations | Sample-based — sub-state estimates have wide CIs; not all years cover agave |
| Intended use | State-level qualitative cross-check on agricultural practices, inputs, irrigation |
| Citation | INEGI (2023) Encuesta Nacional Agropecuaria 2022. |

---

## H. High-resolution validation imagery

### H.1 Planet NICFI Tropical Mosaics

| Field | Value |
| --- | --- |
| Custodian | Planet Labs PBC under the NICFI (Norway's International Climate and Forest Initiative) program |
| Spatial coverage | Pan-tropical 30°N–30°S — Oaxaca fully included |
| Temporal coverage | Biannual mosaics 2015-12 → 2020-08; monthly mosaics 2020-09 → present |
| Native resolution | ≈ 4.77 m (Planet visual mosaic) |
| Projection | EPSG:3857 (Web Mercator) for the served mosaic |
| Format | XYZ tile service, MBTiles, COG (per quad) |
| Access | GEE: `projects/planet-nicfi/assets/basemaps/americas`; Planet Explorer (https://www.planet.com/nicfi/); requires NICFI registration (free for academic / non-commercial) |
| License | NICFI Public license — free for non-commercial scientific use; redistribution restricted; cannot publish full-resolution imagery in papers without Planet attribution and permission |
| Known limitations | Visual mosaic only (RGB/NIR), not surface reflectance; tile seams and color balancing artifacts; quarterly aggregation hides intra-quarter change |
| Intended use | High-resolution reference imagery for Collect Earth Online interpretation of training and validation points |
| Citation | Planet Labs (2024) NICFI Satellite Data Program. |

### H.2 Google Earth historical imagery

| Field | Value |
| --- | --- |
| Custodian | Various commercial providers via Google |
| Spatial coverage | Global, variable |
| Temporal coverage | 2003–present, sparse and irregular over Sierra Sur |
| Native resolution | 0.3–2 m depending on provider |
| Projection | EPSG:3857 served |
| Format | KML overlay, browser only |
| Access | https://earth.google.com (Pro Desktop has time slider) |
| License | Personal / non-commercial reference use; cannot redistribute imagery |
| Known limitations | Not analysis-ready; cannot programmatically download; time series gaps especially before 2014 |
| Intended use | Manual visual reference during point interpretation in Collect Earth Online (loaded as a basemap layer) |
| Citation | Google Earth (2025), imagery © Maxar / Airbus / CNES / Planet etc. |

### H.3 Bing Maps Aerial / Esri World Imagery

| Field | Value |
| --- | --- |
| Custodian | Microsoft / Esri (third-party imagery aggregators) |
| Spatial coverage | Global |
| Temporal coverage | Single most-recent mosaic; capture date varies by tile |
| Native resolution | 0.3–1 m |
| Projection | EPSG:3857 |
| Format | XYZ tile service |
| Access | Bing: https://dev.virtualearth.net (key required); Esri: https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery |
| License | Bing Maps API ToS; Esri ArcGIS Online ToS — non-commercial reference use permitted with attribution |
| Known limitations | No time series; capture date stamp variable per tile; cannot redistribute |
| Intended use | Secondary basemap for Collect Earth Online and QGIS visual reference |
| Citation | Esri World Imagery (Maxar, GeoEye, Earthstar, CNES/Airbus). |

---

## I. DO Mezcal polygon source

### I.1 Denominación de Origen Mezcal (NOM-070-SCFI-2016)

| Field | Value |
| --- | --- |
| Custodian | Diario Oficial de la Federación (DOF), via IMPI; administered by CRT |
| Spatial coverage | All of Oaxaca + 28 municipalities in 8 other states (Durango, Guanajuato, Guerrero, Michoacán, Puebla, San Luis Potosí, Tamaulipas, Zacatecas) |
| Temporal coverage | DO declared 1994; current scope reflects 2018 expansion (DOF 08-Aug-2018) |
| Native resolution | Defined by named municipalities — derived polygon obtained by dissolving INEGI municipal boundaries |
| Projection | EPSG:6372 after derivation |
| Format | Constructed shapefile from DOF text + INEGI Marco Geoestadístico |
| Access | DOF text: https://dof.gob.mx (search NOM-070-SCFI-2016); IMPI declarations https://www.gob.mx/impi ; CRT operating territory https://www.crm.org.mx |
| License | The DOF text is public; the derived polygon inherits INEGI's Términos de Libre Uso |
| Known limitations | The DO is defined by enumeration of municipalities, not by a polygon — so any boundary refinement (intra-municipal) is not legally meaningful; we use the dissolved-municipal polygon |
| Intended use | Outer AOI mask (intersected with Oaxaca state for this study) |
| Citation | Secretaría de Economía (2018) Modificación a la Declaración General de Protección de la Denominación de Origen Mezcal, DOF 08-Aug-2018. |

---

## J. Year-availability summary

Symbols: ✓ analysis-ready full coverage, ~ partial / degraded / non-native, ✗ unavailable. The right-most column lists the dataset used as the **primary 10 m feature source** for that year.

| Dataset | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Sentinel-2 L2A SR | ✗ | ✗ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sentinel-2 L1C TOA | ~ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Landsat 8/9 C2 SR | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MODIS Terra/Aqua | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sentinel-1 GRD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ~ | ~ | ~ |
| Copernicus DEM GLO-30 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SoilGrids v2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INEGI edaphology | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CHIRPS v2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ |
| ERA5-Land monthly | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ |
| WorldClim v2 (norm.) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INEGI Series VI (2014) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INEGI Series VII (2018) | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ESA WorldCover | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hansen GFC v1.12 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Dynamic World v1 | ~ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Marco Geoestadístico | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SIAP Cierre Agrícola | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ | ✗ |
| CRT statistics | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ~ |
| INEGI Censo Pob. | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| INEGI Censo Agro. 2022 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Planet NICFI mosaics | ~ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Primary feature source** | **L8** | **L8** | **L8 + S2** | **S2** | **S2** | **S2** | **S2** | **S2** | **S2** | **S2** | **S2** |

Notes on the table:

- 2017 is treated as a transition year — S2 SR exists but with thin coverage in the Oct–Mar dry season, so a Landsat 8 + Sentinel-2 fused composite is used.
- 2022–2025 Sentinel-1 is marked `~` because S1B failed in Dec-2021 and revisit dropped from 6 d to 12 d; the time-series density is still adequate for annual medians but is no longer dense enough for weekly compositing.
- 2025 CHIRPS / ERA5-Land / CRT statistics are marked `~` because final products lag 2–3 months at the time of the last data freeze.
- 2025 Hansen GFC ✗ — the 2024-update v1.12 reports loss only through 2024; the 2025 update is expected late 2026.
- INEGI Series VII (2018) ✗ before 2018 — it is used as the *target year* baseline for change with respect to Series VI (2014).

---

## K. Data acquisition checklist

This section enumerates concrete URLs, programmatic endpoints, and approximate volumes. Where Earth Engine is the path, no local storage is required; where local download is needed, the size estimate refers to the AOI clip in EPSG:6372 at native resolution.

### K.1 Earth Engine assets (no local download for processing)

| Dataset | EE asset ID | Notes |
| --- | --- | --- |
| Sentinel-2 SR Harmonized | `COPERNICUS/S2_SR_HARMONIZED` | filter `CLOUDY_PIXEL_PERCENTAGE < 60`, then s2cloudless join |
| Sentinel-2 TOA Harmonized | `COPERNICUS/S2_HARMONIZED` | 2015–2017 backfill only |
| Sentinel-2 cloud probability | `COPERNICUS/S2_CLOUD_PROBABILITY` | for s2cloudless masking |
| Landsat 8 C2 L2 | `LANDSAT/LC08/C02/T1_L2` | apply scaling factors `0.0000275 - 0.2` |
| Landsat 9 C2 L2 | `LANDSAT/LC09/C02/T1_L2` | from 2021-10-31 |
| Landsat 7 C2 L2 (legacy) | `LANDSAT/LE07/C02/T1_L2` | optional, SLC-off issues |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | filter IW, instrumentMode=IW, transmit=VV+VH |
| MODIS NDVI 16-day | `MODIS/061/MOD13Q1` | comparison composite only |
| Copernicus DEM GLO-30 | `COPERNICUS/DEM/GLO30` | mosaic `.mosaic()` before sampling |
| SRTM 1-arc | `USGS/SRTMGL1_003` | fallback |
| CHIRPS pentad | `UCSB-CHG/CHIRPS/PENTAD` | sum to monthly / annual |
| CHIRPS daily | `UCSB-CHG/CHIRPS/DAILY` | for dry-season fraction |
| ERA5-Land monthly | `ECMWF/ERA5_LAND/MONTHLY_AGGR` | bands `temperature_2m`, `total_precipitation_sum` |
| WorldClim v1 BIO | `WORLDCLIM/V1/BIO` | static, 19 bands |
| Hansen GFC v1.12 | `UMD/hansen/global_forest_change_2024_v1_12` | bands `treecover2000`, `loss`, `lossyear` |
| ESA WorldCover 2020 | `ESA/WorldCover/v100` | single-band class |
| ESA WorldCover 2021 | `ESA/WorldCover/v200` | single-band class |
| Dynamic World v1 | `GOOGLE/DYNAMICWORLD/V1` | per-S2 image probabilities |
| Travel time to cities (2015) | `OXFORD/MAP/accessibility_to_cities_2015_v1_0` | road-access proxy for H4 |
| INEGI states (uploaded asset) | `users/<account>/oaxaca_aoi/marco_2024_estados` | upload from local SHP |
| INEGI municipios (uploaded) | `users/<account>/oaxaca_aoi/marco_2024_municipios` | upload from local SHP |
| DO Mezcal polygon (uploaded) | `users/<account>/oaxaca_aoi/do_mezcal_oaxaca` | derived from K.4 |

Approximate Earth Engine compute cost for the AOI (for budgeting):

- Annual S2 percentile composite (10 bands × 7 percentiles): ~ 10 GB internal, < 1 hour wall time per year on default quota.
- 11-year stack export at 10 m: ≈ 60 GB Cloud Storage per band-year — partition by tile.

### K.2 Local-download files

| Dataset | URL (root) | Format | AOI clip size (estimate) |
| --- | --- | --- | --- |
| INEGI Marco Geoestadístico 2024 | https://www.inegi.org.mx/temas/mg/ | SHP zipped | ≈ 80 MB national; 8 MB Oaxaca subset |
| INEGI USV Series VI | https://www.inegi.org.mx/temas/usosuelo/ | SHP / GDB | ≈ 250 MB national; 25 MB Oaxaca |
| INEGI USV Series VII | https://www.inegi.org.mx/temas/usosuelo/ | SHP / GDB | ≈ 280 MB national; 28 MB Oaxaca |
| INEGI Edafología Serie III | https://www.inegi.org.mx/temas/edafologia/ | SHP | ≈ 200 MB; 20 MB Oaxaca |
| SIAP Cierre Agrícola CSV | https://nube.siap.gob.mx/cierreagricola/ | XLSX / CSV | < 50 MB per year, all crops, all states |
| CRT informe estadístico | https://www.crm.org.mx/estadisticas.html | PDF + CSV | < 5 MB per annual report |
| INEGI Censo 2020 ITER Oaxaca | https://www.inegi.org.mx/programas/ccpv/2020/ | CSV + SHP | ≈ 120 MB Oaxaca |
| INEGI Censo Agropecuario 2022 | https://www.inegi.org.mx/programas/ca/2022/ | CSV / XLSX | ≈ 30 MB municipal aggregates |
| SoilGrids v2 (clay, sand, OC, pH × 2 depths) | https://maps.isric.org | COG via WCS | ≈ 4 MB per layer for AOI |
| WorldClim v2 BIO 30s | https://worldclim.org/data/worldclim21.html | GeoTIFF | ≈ 50 MB global per BIO; 2 MB Oaxaca |
| DOF NOM-070-SCFI-2016 text | https://dof.gob.mx | PDF / HTML | < 1 MB |
| Planet NICFI quad index | https://api.planet.com/basemaps/v1/mosaics | JSON / XYZ | tiles streamed, no full download |

### K.3 Programmatic-endpoint authentication notes

- **Earth Engine** — service-account JSON in `secrets/ee-service-account.json`; do not commit. Initialize with `ee.Initialize(credentials, project='<gcp-project>')`.
- **Copernicus Data Space Ecosystem** — OData / OpenSearch APIs require a free account (`https://identity.dataspace.copernicus.eu`); use `cdsetool` or `sentinelsat` Python clients.
- **Planet NICFI** — register at https://www.planet.com/nicfi; the API key gives access to `tiles/v1`, `mosaics/v1`, `quads/v1` endpoints.
- **Copernicus CDS (ERA5-Land)** — `cdsapi` Python client, `~/.cdsapirc` with UID/key from https://cds.climate.copernicus.eu/api-how-to.
- **INEGI / SIAP / DOF** — direct HTTPS download, no authentication required.

### K.4 Derivation steps for the DO Mezcal Oaxaca polygon

1. Download INEGI Marco Geoestadístico 2024 municipal shapefile (`mg_2024_integrado/conjunto_de_datos/00mun.shp`).
2. Filter `CVE_ENT == '20'` (Oaxaca) — 570 municipalities.
3. Confirm DO Mezcal coverage of all 570 municipalities per DOF 08-Aug-2018 enumeration; no exclusions.
4. Dissolve to a single multipolygon → `data/raw/aoi/do_mezcal_oaxaca.gpkg`.
5. Reproject to EPSG:6372; compute and store WGS-84 copy for GEE upload.
6. Upload as Earth Engine asset under `users/<account>/oaxaca_aoi/do_mezcal_oaxaca`.

### K.5 Storage and naming conventions

```
data/
  raw/          # untouched downloads, never edited (read-only)
    sentinel2/...
    landsat/...
    inegi/...
    siap/...
    crt/...
  interim/      # cloud-masked composites, tiled exports
  processed/    # final 10 m rasters EPSG:6372, COG
    agave_mask_2015.tif ... agave_mask_2025.tif
    change_2015_2025.tif
  reference/    # CEO export, training/test splits
  metadata/     # ISO 19139 XML per processed product
```

All derived rasters are Cloud-Optimized GeoTIFFs in EPSG:6372 with `LZW` compression, internal tiling 512×512, and at least one overview level.

---

## L. Provenance and update cadence

| Dataset | Update cadence in this project | Trigger to re-pull |
| --- | --- | --- |
| Sentinel-2 / Landsat / Sentinel-1 | per-year compositing run | new analysis year, or reprocessing |
| CHIRPS, ERA5-Land | monthly | new climate year closes |
| Hansen GFC | annual (typically Apr–May) | new v1.x release |
| ESA WorldCover, Dynamic World | as-released | new version on STAC |
| INEGI Marco / Series | as INEGI publishes | next Series release (Serie VIII expected ≈ 2027) |
| SIAP, CRT | annual | next cierre / informe |
| Planet NICFI | monthly (current); biannual (legacy) | new mosaic |

---

*End of `DATA_INVENTORY.md` v0.1.*
