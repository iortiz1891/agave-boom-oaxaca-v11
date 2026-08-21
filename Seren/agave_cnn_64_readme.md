# Agave Multispectral CNN Patch Classification Workflow

## Short Summary

This script trains and applies a multispectral convolutional neural network to classify **64 × 64 Sentinel-2 image patches** as agave or non-agave.

It:

- Matches Sentinel-2 patches to year-specific training labels
- Supports multiple existing Sentinel-2 filename formats
- Standardizes agave and non-agave labels
- Optionally validates raster dimensions and band counts
- Preserves `base_id` as the spatial grouping variable
- Tracks locations whose labels change across years
- Creates spatially independent train, validation, and test splits
- Supports an optional temporal holdout year
- Computes training-set spectral normalization statistics
- Uses class weighting and optional balanced sampling for class imbalance
- Trains a 9-band multispectral CNN
- Optionally includes acquisition year as an additional model feature
- Automatically selects a classification threshold from the validation set
- Tracks F1, ROC AUC, average precision, balanced accuracy, and other metrics
- Reports test performance separately by year
- Performs descriptive analysis of temporal label transitions
- Applies the trained model to new Sentinel-2 patches
- Exports patch-level agave probabilities and predicted labels to CSV

The model performs **patch-level binary classification**, not pixel-level segmentation.

Each image receives one probability representing whether the 64 × 64 patch is classified as agave. The expected imagery consists of nine Sentinel-2 bands at 64 × 64 pixels.

## Overview

This script provides a spectral-spatial CNN baseline for the agave classification project.

Unlike the U-Net workflow, which requires a pixel-level mask and predicts a class for every pixel, this model treats each Sentinel-2 patch as a single training observation.

The target is therefore:

```text
0 = not_agave
1 = agave
```

The CNN learns from the complete 64 × 64 multispectral patch and outputs one agave probability.

The workflow contains three primary commands:

```text
build-manifest
train
predict
```

The script version is:

```text
1.0-spatiotemporal-patch-cnn
```

## Expected Sentinel-2 Imagery

The model expects each input patch to contain:

```text
9 bands
64 × 64 pixels
```

The expected spectral bands are:

1. B2
2. B3
3. B4
4. B5
5. B6
6. B7
7. B8
8. B11
9. B12

These include visible, red-edge, near-infrared, and shortwave-infrared Sentinel-2 information.

The expected filename suffix is:

```text
_sentinel2_dry_median_64_snapped.tif
```

The core imagery constants are defined as:

```python
EXPECTED_BANDS = 9
EXPECTED_HEIGHT = 64
EXPECTED_WIDTH = 64

BAND_NAMES = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B11",
    "B12",
]
```

## Expected Label Data

The script is designed to work with a geospatial training-label file such as:

```text
data/processed/unet_labels/training_5000_polygons.gpkg
```

The label data may contain fields such as:

- `id`
- `base_id`
- `year`
- `label`
- `standardized_label`

The `id` field is required when building the manifest because imagery is matched to the corresponding year-specific training record.

## Main Commands

| Command | Purpose |
|---|---|
| `build-manifest` | Match Sentinel-2 imagery with year-specific labels |
| `train` | Train and evaluate the grouped multispectral CNN |
| `predict` | Apply a trained CNN to image patches |

## Image Identity Parsing

The script extracts three pieces of identity information from each image:

```text
image_id
base_id
year
```

These values serve different purposes.

### `image_id`

Identifies the specific location-year observation.

For example:

```text
INT_1778232235_2018
```

### `base_id`

Identifies the underlying spatial location independently of year.

For example:

```text
INT_1778232235
```

### `year`

Identifies the observation year.

For example:

```text
2018
```

This separation allows different years from the same physical location to remain connected during model splitting and temporal analysis.

## Supported Image Names

The filename parser supports forms including:

```text
INT_0001_2021_sentinel2_dry_median_64_snapped.tif
```

```text
INT_1778232235_2018_2018_sentinel2_dry_median_64_snapped.tif
```

```text
INT_1778262218_93dd2c_2017_2017_sentinel2_dry_median_64_snapped.tif
```

For a repeated-year filename such as:

```text
INT_1778232235_2018_2018_sentinel2_dry_median_64_snapped.tif
```

the parser produces:

```text
image_id = INT_1778232235_2018
base_id  = INT_1778232235
year     = 2018
```

For location identifiers containing an additional suffix, that suffix remains part of the spatial `base_id`.

## Repeated Year Validation

Some exported Sentinel-2 filenames contain the year twice.

For example:

```text
INT_1778232235_2018_2018_sentinel2_dry_median_64_snapped.tif
```

This format is accepted when the two years match.

A filename such as:

```text
INT_1778232235_2018_2019_sentinel2_dry_median_64_snapped.tif
```

is rejected because the repeated years disagree.

## Image Discovery

The script recursively searches the selected imagery directory for:

```text
*.tif
```

Files containing:

```text
mask
```

in their filename are excluded.

This prevents segmentation-mask GeoTIFFs from accidentally being treated as CNN input imagery.

## Label Standardization

The script converts several source representations into binary targets.

### Agave

The following values become:

```text
1
```

Accepted forms include:

```text
yes
agave
1
true
positive
```

### Non-Agave

The following values become:

```text
0
```

Accepted forms include:

```text
no
not_agave
not agave
0
false
negative
```

### Unknown

Any value that cannot be recognized becomes:

```text
None
```

Unknown labels are excluded from the usable manifest rather than being forced into either class.

## Label Column Selection

A specific source label field can be provided using:

```text
--label-column
```

If no label column is specified, the script searches in this order:

```text
standardized_label
label
class
target
```

The first matching column is used.

If none of these columns exists, manifest construction stops.

## Raster Validation

Raster validation can be enabled during manifest construction using:

```text
--validate-rasters
```

When enabled, each image is opened and checked.

The script verifies:

- Exactly 9 bands
- Exactly 64 × 64 pixels
- At least one valid pixel

The raster inspection also records:

- Band count
- Height
- Width
- Data type
- CRS
- NoData value

A raster containing no valid pixels is rejected.

## `build-manifest`

The first major workflow stage matches image patches to year-specific training labels.

Example:

```bash
python agave_patch_cnn.py build-manifest ^
    --image-dir "C:\data\agave_sentinel2_full_64_dry_median" ^
    --label-file "data\processed\unet_labels\training_5000_polygons.gpkg" ^
    --output-csv "agave_cnn_manifest.csv" ^
    --validate-rasters
```

The exact Python filename should be replaced with the filename used for this script in the repository.

## Manifest Matching Logic

The label file is loaded using GeoPandas.

The script requires:

```text
id
```

because each Sentinel-2 image is matched to a year-specific label record using its parsed `image_id`.

For each image, the script:

1. Parses `image_id`, `base_id`, and `year`.
2. Searches the label lookup for the parsed `image_id`.
3. Excludes images without matching labels.
4. Excludes records with unknown labels.
5. Optionally validates the raster.
6. Uses the label-file `base_id` when available.
7. Uses the label-file year when available.
8. Verifies that the label year agrees with the filename year.
9. Adds the usable observation to the CNN manifest.

## Duplicate Label IDs

Duplicate label IDs are handled conservatively.

If multiple rows share the same:

```text
id
```

the script examines their standardized binary targets.

If the duplicates agree on one target, that consensus can still be used.

If the same `id` contains contradictory targets, the ID is recorded as a duplicate conflict and is not used normally for matching.

This prevents contradictory duplicate records from silently entering the training data.

## CNN Manifest

Each usable record can contain:

- `image_path`
- `image_name`
- `image_id`
- `base_id`
- `year`
- `target`
- `label_raw`

When raster validation is enabled, the manifest also contains:

- `bands`
- `height`
- `width`
- `dtype`
- `crs`
- `nodata`

The completed manifest is sorted by:

1. `base_id`
2. `year`
3. `image_name`

## Manifest Report

Manifest construction automatically creates a JSON report beside the output CSV.

For example:

```text
agave_cnn_manifest.csv
agave_cnn_manifest.report.json
```

The report contains:

- Script version
- Images found
- Usable labeled images
- Class counts
- Year counts
- Missing-label count
- Examples of images without labels
- Number of unknown labels excluded
- Number of duplicate-ID conflicts
- Examples of duplicate conflicts
- Invalid-image count
- Examples of invalid images

It also records temporal properties of the dataset:

- Unique `base_id` count
- Number of `base_id` groups observed in multiple years
- Number of `base_id` groups whose labels change across years



## Temporal Label Transitions

A location may legitimately contain different class labels in different years.

For example:

```text
INT_0042
2019: 0
2020: 0
2021: 1
2022: 1
```

This is treated as a potential temporal transition rather than a contradiction.

The manifest report counts locations where:

```text
target.nunique() > 1
```

across years.

These locations are especially important for studying agave expansion or removal over time.

## CNN Dataset Preparation

During training, each raster is loaded as a nine-band floating-point array.

The dataset also reads the raster's valid-data mask.

Pixels outside the valid raster area are temporarily converted to non-finite values.

Before normalization, residual invalid values are replaced with the corresponding **training-band mean**.

The image is then standardized band-by-band.

## Spectral Normalization

For each band, the transformation is:

```text
x_normalized = (x - mean) / standard deviation
```

The mean and standard deviation are calculated from the **training split only**.

This prevents information from validation or test imagery from affecting model preprocessing.

Normalization values are stored in:

```text
normalization.json
```

## Normalization Sampling

By default, normalization statistics use up to:

```text
1000 training images
```

controlled by:

```text
--normalization-images
```

If the training set contains fewer than this number, all training images are used.

The selected images are sampled reproducibly using the configured random seed.

## Data Augmentation

Training imagery receives random geometric augmentation.

Possible transformations include:

- Horizontal flip
- Vertical flip
- 90° rotations
- 180° rotations
- 270° rotations

Validation and test imagery are not augmented.

These transformations alter spatial orientation while preserving the spectral information and patch-level class.

## Optional Year Feature

By default, the model is **spectral-spatial only**.

The year can optionally be included using:

```text
--include-year-feature
```

When enabled, the year is normalized using:

```text
(year - minimum_year)
─────────────────────────
maximum_year - minimum_year
```

This produces an approximately 0–1 scalar.

The year feature is then appended to the CNN's learned image representation before classification.

This allows the model to learn a direct temporal signal in addition to spectral-spatial features.

## Why the Year Feature Is Optional

Including year may improve performance when agave prevalence or imagery characteristics systematically change over time.

However, it also allows the model to use acquisition year directly when determining the prediction.

For that reason, the default model excludes the year feature and learns from image content alone.

## CNN Architecture

The model is a patch-level convolutional neural network with four convolution blocks.

The feature progression is:

```text
9 bands
↓
32 channels
↓
64 channels
↓
128 channels
↓
256 channels
↓
Adaptive Global Average Pooling
```

Each convolution block contains:

```text
3 × 3 convolution
Batch Normalization
ReLU
3 × 3 convolution
Batch Normalization
ReLU
```

Max pooling is applied between the first three blocks.

The final 256-channel representation is reduced to:

```text
256 × 1 × 1
```

using adaptive average pooling.

## Classifier Head

The pooled image representation is passed through:

```text
Flatten
↓
Linear → 128
↓
ReLU
↓
Dropout
↓
Linear → 1
```

The final value is a binary classification logit.

A sigmoid function converts that value into:

```text
probability_agave
```

If the year feature is enabled, the classifier receives:

```text
256 CNN features + 1 year feature
```

instead of only the 256 CNN features.

## Spatial Grouping

The default model split is based on:

```text
base_id
```

rather than individual image rows.

All years associated with one `base_id` stay in the same model partition.

For example:

```text
INT_0042_2019
INT_0042_2020
INT_0042_2021
INT_0042_2022
```

are all assigned to either:

```text
train
```

or:

```text
validation
```

or:

```text
test
```

They cannot be divided between these sets.

## Why Grouped Splitting Matters

If observations from the same physical location appeared in both training and testing, the model could learn location-specific spatial patterns and then encounter the same location again during evaluation.

This would create spatial and temporal leakage.

The script therefore uses:

```text
GroupShuffleSplit
```

with:

```text
base_id
```

as the grouping variable.

After splitting, an explicit check confirms that no `base_id` occurs in more than one set.

## Default Split

Without a temporal holdout, the default fractions are:

```text
Training:   approximately 70%
Validation: 15%
Test:       15%
```

These are controlled by:

```text
--val-fraction
--test-fraction
```

The resulting split mode is recorded as:

```text
grouped_random
```

## Temporal Holdout Experiment

A stricter temporal experiment can be requested using:

```text
--holdout-year
```

For example:

```bash
--holdout-year 2025
```

In this mode:

```text
2025 → test set
other years → candidates for training/validation
```

However, spatial independence is still enforced.

Any `base_id` appearing in the holdout year is removed from the earlier-year training pool.

This means the test set contains both:

- A held-out year
- Spatial locations unseen during training

The split mode is recorded using a name such as:

```text
spatially_independent_temporal_holdout_2025
```

## Important Temporal Holdout Behavior

Suppose:

```text
INT_0042
2022
2023
2024
2025
```

exists in the dataset and 2025 is the holdout year.

Because `INT_0042` occurs in the 2025 test set, its 2022–2024 observations are also removed from training and validation.

This deliberately sacrifices some training data to prevent location leakage into the temporal test set.

## Split Manifest

The exact train, validation, and test records are saved to:

```text
manifest_with_splits.csv
```

The values written to the split column are:

```text
train
validation
test
```

This file should be preserved with each model run for reproducibility.

## Handling Class Imbalance

The script provides two mechanisms for dealing with unequal agave and non-agave class frequencies:

1. Loss weighting
2. Balanced sampling

These mechanisms can be used independently or together.

## Class-Weighted Loss

Class weighting is enabled by default.

The positive-class weight is calculated as:

```text
number of negative training patches
───────────────────────────────────
number of positive training patches
```

This value is supplied to:

```text
BCEWithLogitsLoss
```

so mistakes on the minority positive class can receive more weight.

Class weighting can be disabled using:

```text
--no-class-weight
```

## Balanced Sampler

Balanced oversampling is optional.

Enable it using:

```text
--balanced-sampler
```

The script calculates an inverse-frequency weight for each training observation and uses:

```text
WeightedRandomSampler
```

with replacement.

This causes minority-class examples to be sampled more frequently during training.

When the balanced sampler is active, normal training-set shuffling is disabled because sampling order is controlled by the sampler.

## Training Device

The script automatically uses CUDA when:

- PyTorch detects an available CUDA device
- `--cpu` has not been specified

Otherwise it trains on the CPU.

CPU operation can be forced with:

```text
--cpu
```

## Loss Function

The CNN uses:

```text
BCEWithLogitsLoss
```

for binary patch classification.

When class weighting is enabled, the estimated positive-class weight is passed to the loss function.

## Optimizer

Training uses:

```text
AdamW
```

with default settings:

```text
learning rate = 0.001
weight decay  = 0.0001
```

## Learning-Rate Scheduler

The script uses:

```text
ReduceLROnPlateau
```

with:

```text
mode = max
factor = 0.5
patience = 3
```

The scheduler monitors the validation model-selection score.

When that score plateaus, the learning rate is reduced by half.

## Validation Model-Selection Metric

When both classes are present in the validation set, model selection uses:

```text
Average Precision
```

If only one class is present, the fallback score is:

```text
-negative validation loss
```

The best model checkpoint is therefore normally selected according to validation average precision rather than raw accuracy.

## Automatic Validation Threshold Selection

Classification is not permanently fixed at:

```text
0.5
```

during validation.

After each epoch, the script tests thresholds from:

```text
0.05
```

through:

```text
0.95
```

in increments of:

```text
0.01
```

For each threshold, validation F1 is calculated.

The threshold with the highest validation F1 becomes that epoch's:

```text
validation_threshold
```

When the best model is saved, its corresponding threshold is saved with it.

## Training Metrics

The script can calculate:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1 score
- Confusion matrix
- ROC AUC
- Average precision

It also records:

- Number of samples
- Number of positive samples
- Classification threshold

ROC AUC and average precision are only calculated when both classes are present.

## Why Balanced Accuracy Is Included

Normal accuracy can be misleading when one class greatly outnumbers the other.

Balanced accuracy gives equal importance to performance on each class and therefore provides a more informative metric for an imbalanced agave/non-agave dataset.

## Training History

For every epoch, the script records:

- Epoch
- Training loss
- Validation loss
- Learning rate
- Validation threshold
- Training F1 at threshold 0.5
- Validation F1
- Validation average precision
- Validation ROC AUC

The complete history is saved after training to:

```text
training_history.csv
```

## Best Model Checkpoint

The best-performing model is saved to:

```text
best_model.pt
```

The checkpoint contains:

- Script version
- Model weights
- Model configuration
- Input-band count
- Dropout value
- Whether the year feature was used
- Band normalization values
- Minimum dataset year
- Maximum dataset year
- Validation-selected threshold
- Split mode

This allows the prediction command to recreate the same network architecture and preprocessing settings automatically.

## Early Stopping

Training supports early stopping.

The default settings are:

```text
patience = 10
min_delta = 0.0001
```

When the validation model-selection score fails to improve by at least `min_delta` for the configured number of epochs, training stops.

The maximum default training length is:

```text
60 epochs
```



## Test Evaluation

After training, the script reloads:

```text
best_model.pt
```

The saved validation threshold is also restored.

The best model is then evaluated on the held-out test set.

Test classification therefore uses the threshold selected from the validation data, not a threshold selected from the test set.

## Test Predictions

Individual test predictions are written to:

```text
test_predictions.csv
```

Each row contains the original manifest information plus:

```text
probability_agave
predicted_label
```

The predicted label is:

```text
1
```

when the predicted probability is greater than or equal to the saved validation threshold.

Otherwise it is:

```text
0
```

## Per-Year Test Metrics

The test predictions are grouped by year.

The script calculates separate metrics for each year represented in the test set.

These results are stored under:

```text
test_by_year
```

in:

```text
metrics.json
```

This makes it possible to identify whether the CNN performs differently across imagery years.

## Temporal Descriptive Analysis

The script also performs a descriptive analysis of locations with multiple test years.

For every test-set `base_id` containing at least two years, it checks whether:

- The true label changes across years
- The model's predicted label changes across years

The summary reports:

```text
test_base_ids_with_2plus_years
test_base_ids_with_true_label_transition
test_base_ids_with_predicted_label_transition
```

This is **descriptive only**.

The CNN still makes each prediction from a single image-year observation. It does not explicitly process an image sequence or learn transitions through a temporal CNN architecture.

## `metrics.json`

The final metrics file contains:

```text
script_version
split_mode
best_validation_threshold
test
test_by_year
temporal_descriptive_summary
```

The `test` section contains the overall held-out test metrics.

The `test_by_year` section contains year-specific test metrics.

The temporal summary describes label-change behavior among multi-year locations in the test set.

## `train`

A typical grouped training run is:

```bash
python agave_patch_cnn.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "cnn_run_01"
```

A more explicit run might use:

```bash
python agave_patch_cnn.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "cnn_run_01" ^
    --epochs 60 ^
    --batch-size 32 ^
    --learning-rate 0.001 ^
    --balanced-sampler
```

## Temporal Holdout Training Example

To use 2025 as a spatially independent temporal test year:

```bash
python agave_patch_cnn.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "cnn_holdout_2025" ^
    --holdout-year 2025
```

To additionally supply year directly to the CNN:

```bash
python agave_patch_cnn.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "cnn_holdout_2025_yearfeature" ^
    --holdout-year 2025 ^
    --include-year-feature
```

These represent different experiments and should be kept in separate output directories.

## Main Training Outputs

A typical training directory contains:

```text
best_model.pt
manifest_with_splits.csv
normalization.json
training_history.csv
test_predictions.csv
metrics.json
```

## `predict`

The prediction command applies a saved CNN to all eligible image patches under an imagery directory.

Example:

```bash
python agave_patch_cnn.py predict ^
    --model "cnn_run_01\best_model.pt" ^
    --image-dir "C:\data\agave_sentinel2_full_64_dry_median" ^
    --output-csv "cnn_predictions.csv"
```

For each image, the script:

1. Parses its identity.
2. Validates the raster.
3. Reads the nine Sentinel-2 bands.
4. Applies the raster validity mask.
5. Replaces remaining invalid pixels using saved band means.
6. Applies the saved band normalization.
7. Reconstructs the trained CNN.
8. Adds the normalized year feature when the model was trained with one.
9. Generates the agave probability.
10. Applies the selected probability threshold.
11. Writes the prediction record to the output CSV.

## Prediction Threshold

By default, prediction uses the validation threshold stored in:

```text
best_model.pt
```

A different threshold can be supplied using:

```text
--threshold
```

For example:

```bash
--threshold 0.60
```

This changes only how probabilities are converted into binary predictions.

It does not retrain the CNN.

## Prediction Output

Successful prediction rows contain:

- `image_path`
- `image_name`
- `image_id`
- `base_id`
- `year`
- `probability_agave`
- `predicted_label`
- `threshold`
- `error`

For example:

```text
probability_agave = 0.843
predicted_label   = 1
threshold         = 0.57
```

If a file cannot be processed, the output still receives a row containing the image information and the corresponding error message.

This allows a large prediction folder to continue processing even when individual files fail.

## Important Training Configuration Options

| Setting | Default | Purpose |
|---|---:|---|
| `--epochs` | `60` | Maximum number of training epochs |
| `--batch-size` | `32` | Training observations per batch |
| `--workers` | `0` | PyTorch DataLoader workers |
| `--learning-rate` | `0.001` | Initial AdamW learning rate |
| `--weight-decay` | `0.0001` | AdamW regularization |
| `--dropout` | `0.30` | Dropout in the classifier head |
| `--patience` | `10` | Early-stopping patience |
| `--min-delta` | `0.0001` | Minimum validation-score improvement |
| `--val-fraction` | `0.15` | Validation fraction |
| `--test-fraction` | `0.15` | Test fraction |
| `--seed` | `42` | Random seed |
| `--normalization-images` | `1000` | Maximum training images used for normalization |
| `--holdout-year` | None | Optional temporally held-out test year |
| `--include-year-feature` | Off | Adds normalized year to the CNN embedding |
| `--balanced-sampler` | Off | Oversamples minority-class training patches |
| `--no-class-weight` | Off | Disables positive-class loss weighting |
| `--cpu` | Off | Forces CPU execution |

These options are defined directly in the command-line interface.

## Reproducibility

The default random seed is:

```text
42
```

The script seeds:

- Python `random`
- NumPy
- PyTorch
- PyTorch CUDA when available

The same seed is also used for grouped splitting and normalization-image sampling.

## Recommended Workflow

### 1. Confirm the imagery

Verify that the Sentinel-2 patches are:

```text
9 bands
64 × 64 pixels
```

and follow the supported naming convention.

### 2. Confirm the label file

Verify that the label dataset contains:

```text
id
```

and preferably:

```text
base_id
year
standardized_label
```

### 3. Build the CNN manifest

```bash
python agave_patch_cnn.py build-manifest ^
    --image-dir "C:\data\agave_sentinel2_full_64_dry_median" ^
    --label-file "data\processed\unet_labels\training_5000_polygons.gpkg" ^
    --output-csv "agave_cnn_manifest.csv" ^
    --validate-rasters
```

### 4. Review the manifest report

Inspect:

```text
agave_cnn_manifest.report.json
```

Pay particular attention to:

- Class counts
- Year counts
- Missing labels
- Unknown labels
- Duplicate conflicts
- Invalid rasters
- Number of multi-year locations
- Number of locations with label transitions

### 5. Train the grouped baseline

```bash
python agave_patch_cnn.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "cnn_grouped_baseline"
```

### 6. Review model performance

Inspect:

```text
training_history.csv
metrics.json
test_predictions.csv
```

Compare:

- Overall test F1
- Balanced accuracy
- ROC AUC
- Average precision
- Per-year performance
- Confusion matrix

### 7. Run alternative experiments

Potential controlled comparisons include:

```text
Grouped baseline
Grouped + balanced sampler
Grouped + year feature
Temporal holdout
Temporal holdout + year feature
```

Each experiment should use its own output directory.

### 8. Apply the selected model

```bash
python agave_patch_cnn.py predict ^
    --model "cnn_grouped_baseline\best_model.pt" ^
    --image-dir "C:\data\agave_sentinel2_full_64_dry_median" ^
    --output-csv "cnn_predictions.csv"
```

## Recommendations

- Use `--validate-rasters` when creating the final training manifest.
- Preserve `base_id` throughout the workflow because it is the key spatial grouping variable.
- Do not randomly split individual image-year rows without grouping by location.
- Preserve locations that change labels across years rather than treating those changes automatically as labeling errors.
- Review duplicate-ID conflicts before model training.
- Keep unknown labels out of the binary training set until they can be resolved.
- Use balanced accuracy, F1, ROC AUC, and average precision alongside ordinary accuracy.
- Preserve `manifest_with_splits.csv` for every experiment.
- Do not select the final probability threshold using the held-out test data.
- Compare the default spectral-spatial model against the optional year-feature model rather than assuming the year feature improves generalization.
- Use the temporal holdout mode when testing whether the model generalizes to a future or otherwise unseen imagery year.
- Remember that temporal holdout mode also removes overlapping spatial locations from training, making it a stricter experiment.
- Review `test_by_year` metrics for evidence of year-dependent model performance.
- Interpret the temporal transition summary descriptively; this CNN is not itself a sequence model.
- Preserve `normalization.json` and `best_model.pt` together with each run.
- Keep experiment directories separate when comparing split strategies, class-balancing methods, or temporal features.
- Use the CNN as a patch-classification baseline for comparison with Random Forest, XGBoost, SVM, temporal CNN, and segmentation-based approaches.

## Current Workflow Summary

In sequence, the script:

1. Defines the expected 9-band, 64 × 64 Sentinel-2 image format.
2. Searches imagery directories recursively for eligible GeoTIFFs.
3. Excludes files containing `mask` from CNN imagery.
4. Parses each filename into an `image_id`, `base_id`, and year.
5. Validates repeated years in filenames when present.
6. Loads the geospatial label dataset.
7. Selects or infers the label column.
8. Converts recognized agave labels to `1`.
9. Converts recognized non-agave labels to `0`.
10. Treats unrecognized labels as unknown.
11. Builds a lookup using year-specific label IDs.
12. Detects conflicting duplicate IDs.
13. Matches each Sentinel-2 image to its year-specific label.
14. Excludes images with missing labels.
15. Excludes observations with unknown binary labels.
16. Optionally validates raster band counts, dimensions, and valid pixels.
17. Confirms that filename year and label year agree.
18. Creates the CNN training manifest.
19. Summarizes class frequencies and year frequencies.
20. Counts unique spatial `base_id` groups.
21. Identifies locations observed across multiple years.
22. Identifies locations whose labels change across years.
23. Loads the completed manifest for training.
24. Removes any rows that do not contain binary targets.
25. Creates a grouped random split by default.
26. Alternatively creates a spatially independent temporal holdout split.
27. Verifies that train, validation, and test `base_id` groups are disjoint.
28. Saves the exact dataset partition to `manifest_with_splits.csv`.
29. Samples training imagery for spectral normalization.
30. Calculates training-band means and standard deviations.
31. Saves normalization values to `normalization.json`.
32. Builds PyTorch datasets for training, validation, and testing.
33. Replaces invalid raster pixels with training-band means.
34. Standardizes each spectral band.
35. Applies random flips and rotations to training imagery.
36. Optionally calculates a normalized year feature.
37. Optionally constructs a weighted training sampler.
38. Calculates a positive-class loss weight by default.
39. Initializes the four-stage multispectral CNN.
40. Trains the model using weighted binary cross-entropy.
41. Optimizes the network using AdamW.
42. Adjusts the learning rate when validation performance plateaus.
43. Calculates training and validation predictions after each epoch.
44. Searches validation thresholds from 0.05 through 0.95.
45. Selects the threshold maximizing validation F1.
46. Uses validation average precision as the primary model-selection score when both classes are available.
47. Saves a new best checkpoint when validation performance sufficiently improves.
48. Stops early when the configured patience is exceeded.
49. Saves the complete training history.
50. Reloads the best model checkpoint.
51. Uses the saved validation threshold for test classification.
52. Calculates overall held-out test metrics.
53. Saves individual test probabilities and predicted labels.
54. Calculates separate test metrics for each year.
55. Descriptively counts true and predicted label transitions among multi-year test locations.
56. Writes final metrics to `metrics.json`.
57. Loads the saved architecture and normalization statistics for later prediction.
58. Validates each prediction raster.
59. Normalizes prediction imagery using the training statistics.
60. Adds the year feature when required by the trained model.
61. Calculates a patch-level agave probability.
62. Converts that probability into a binary prediction using the saved or overridden threshold.
63. Records successful predictions and individual processing errors in the output CSV.

This workflow provides a spatially grouped multispectral CNN baseline for classifying individual Sentinel-2 image patches while preserving the multi-year structure of the agave dataset and supporting stricter temporal generalization experiments.