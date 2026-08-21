# Agave Weakly Supervised U-Net Workflow

## Short Summary

This script trains and applies a **weakly supervised multispectral U-Net** to 64 × 64 Sentinel-2 agave image patches.

Unlike a fully supervised U-Net, this workflow does **not** require pixel-level segmentation masks during training.

Instead, it trains from patch-level labels:

```text
0 = non-agave patch
1 = agave patch
```

while the network itself produces a 64 × 64 pixel probability map.

The script uses multiple-instance learning-style pooling to connect the pixel-level U-Net output to the patch-level training label. The resulting pixel maps are therefore **pseudo-segmentation outputs**, not predictions trained against verified pixel masks.

It:

- Reads an existing single-year agave image manifest
- Accepts several possible label-column formats
- Standardizes labels to binary agave/non-agave targets
- Uses `base_id` to maintain spatially independent data splits
- Reuses supplied train/validation/test splits when available
- Creates grouped geographic splits when they are not supplied
- Optionally validates all Sentinel-2 rasters
- Computes spectral normalization from training imagery only
- Applies geometric augmentation to training patches
- Trains a 9-band U-Net without pixel-level masks
- Uses top-k multiple-instance pooling to convert pixel logits into patch-level predictions
- Penalizes false positive pixels in known non-agave patches
- Encourages spatial smoothness through total-variation regularization
- Encourages sparse positive regions inside positive patches
- Tracks patch-level validation metrics
- Selects the best model using validation average precision
- Selects a classification threshold from validation F1
- Evaluates held-out test performance overall and by year
- Writes pixel-level probability GeoTIFFs
- Thresholds those probability maps into pseudo-masks
- Produces a CSV summary describing each generated pseudo-mask

## Overview

This workflow addresses a specific limitation in the agave segmentation dataset:

```text
Image-level labels exist
but
verified pixel-level field masks do not.
```

A conventional supervised U-Net requires a target mask for every training image.

For example:

```text
Sentinel-2 image
+
64 × 64 true agave mask
→ supervised segmentation training
```

This script instead uses:

```text
Sentinel-2 image
+
single patch-level agave/non-agave label
→ weakly supervised U-Net training
```

The model still produces a pixel grid, but the training loss only has indirect information about where agave occurs inside a positive patch.

The script version is:

```text
1.0-weak-mil-unet
```

## Important Interpretation

The most important limitation of the workflow is:

```text
Pixel outputs are pseudo-segmentation maps.
```

They should **not** be interpreted as equivalent to predictions produced by a U-Net trained against verified field masks.

The script explicitly records:

```text
supervision = weak_patch_level_MIL
```

and:

```text
Pixel outputs are pseudo-segmentation maps; no true pixel masks were used.
```

in the final metrics report.

## Expected Sentinel-2 Imagery

Each image should contain:

```text
9 spectral bands
64 × 64 pixels
```

The expected bands are:

1. B2
2. B3
3. B4
4. B5
5. B6
6. B7
7. B8
8. B11
9. B12

The script defines:

```python
BANDS = 9
SIZE = 64

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

## Main Commands

The script contains two primary commands:

| Command | Purpose |
|---|---|
| `train` | Train and evaluate the weakly supervised U-Net |
| `predict` | Generate pixel probability maps and binary pseudo-masks |

## Required Manifest

The workflow reads a CSV manifest supplied through:

```text
--manifest
```

At minimum, the input must contain:

```text
image_path
```

The script also needs enough information to determine:

```text
target
base_id
year
```

These fields may already exist or may be inferred from alternative columns.

## Manifest Standardization

The script automatically standardizes several manifest formats before training.

The final internal fields are:

```text
image_path
target
base_id
year
```

Only rows with binary targets are retained.

## Target Detection

If the manifest already contains:

```text
target
```

that field is used directly.

Otherwise, the script looks for:

```text
standardized_label
```

and converts:

```text
agave     → 1
not_agave → 0
```

If `standardized_label` is unavailable, it looks for:

```text
label
```

and accepts:

```text
yes       → 1
agave     → 1
1         → 1

no        → 0
not_agave → 0
0         → 0
```

If none of these target sources exists, the script stops.

## `base_id` Detection

If the manifest already contains:

```text
base_id
```

that value is retained.

Otherwise, the script can derive `base_id` from:

```text
id
```

by removing a final four-digit year suffix.

Conceptually:

```text
INT_0042_2023
```

becomes:

```text
INT_0042
```

If neither `base_id` nor `id` is available, the script stops.

## Year Detection

The script uses:

```text
year
```

when available.

If `year` does not exist but:

```text
target_year
```

does, then:

```text
year = target_year
```

If neither is available, the script stops.

These manifest-standardization rules allow the weak U-Net to operate on several existing manifest formats from the agave classification workflows.

## Existing Split Reuse

If the input manifest contains a:

```text
split
```

column, the script attempts to reuse those assignments.

It recognizes:

```text
train
validation
test
```

and converts:

```text
val
```

to:

```text
validation
```

If all three sets contain records, the supplied split is preserved.

## Spatial Leakage Check

Before accepting supplied splits, the script compares the `base_id` groups belonging to:

- Training
- Validation
- Testing

If any `base_id` appears in more than one partition, training stops with:

```text
base_id leakage in supplied splits
```

This prevents different yearly observations from the same spatial location from being distributed across training and evaluation sets.

## Automatic Grouped Splitting

If usable split assignments are not supplied, the script creates new spatially grouped partitions.

The first grouped split assigns approximately:

```text
70% → training
30% → remaining data
```

The remaining data are then split equally:

```text
15% → validation
15% → test
```

using:

```text
GroupShuffleSplit
```

with:

```text
base_id
```

as the grouping variable.

The final approximate distribution is therefore:

```text
Training:   70%
Validation: 15%
Test:       15%
```

The resulting split labels are written directly into the three output manifests.

## Split Outputs

The exact data partitions used during training are saved as:

```text
train_manifest.csv
validation_manifest.csv
test_manifest.csv
```

These files should be preserved with the model run.

They provide the exact geographic partition used to calculate the final performance metrics.

## Optional Raster Validation

Raster validation can be enabled using:

```text
--validate-rasters
```

When enabled, every unique input image is checked.

The expected raster structure is:

```text
9 bands
64 × 64 pixels
```

Any raster with a different band count or image size causes the script to stop.

## Spectral Normalization

Normalization is calculated using the **training split only**.

For each of the nine Sentinel-2 bands, the script calculates:

```text
mean
standard deviation
```

using valid raster pixels.

Each image is then standardized as:

```text
(x - mean) / standard deviation
```

The resulting values are saved to:

```text
normalization.json
```

## Normalization Sample Size

The default maximum number of training images used to estimate normalization statistics is:

```text
2000
```

controlled by:

```text
--normalization-images
```

If fewer than 2000 training images exist, all available training images are used.

## NoData Handling

Raster NoData values are excluded from normalization calculations when the raster defines a NoData value.

During dataset loading, the raster validity mask is also used.

Invalid pixels are first represented as:

```text
NaN
```

and then replaced with the corresponding training-band mean.

After normalization, those mean-filled pixels become approximately:

```text
0
```

in standardized feature space.

## Data Augmentation

Training images receive random geometric transformations.

Possible operations include:

- Horizontal flip
- Vertical flip
- 90° rotation
- 180° rotation
- 270° rotation

Validation and test imagery are not augmented.

## Why This Is a Weakly Supervised U-Net

The neural network itself is structurally a U-Net.

It receives:

```text
9 × 64 × 64
```

Sentinel-2 imagery and produces:

```text
64 × 64
```

pixel logits.

However, the training target is only:

```text
one binary label per image
```

rather than:

```text
one binary label per pixel
```

The script therefore needs a mechanism for turning the pixel-level output into a single patch-level prediction.

That mechanism is **top-k pooling**.

## U-Net Architecture

The model uses four encoder stages plus a bottleneck.

With the default:

```text
base_channels = 32
```

the encoder expands approximately as:

```text
9 input bands
↓
32 channels
↓
64 channels
↓
128 channels
↓
256 channels
↓
512-channel bottleneck
```

The decoder then reconstructs the 64 × 64 output using transposed convolutions and U-Net skip connections.

## Convolution Blocks

Each double-convolution block contains:

```text
3 × 3 convolution
Batch Normalization
ReLU

3 × 3 convolution
Batch Normalization
ReLU
```

## Decoder

The decoder progressively upsamples the bottleneck representation.

Conceptually:

```text
512
↓
256
↓
128
↓
64
↓
32
↓
1 output channel
```

Each decoder stage concatenates the upsampled representation with the corresponding encoder feature map.

## Pixel Output

The final layer produces:

```text
1 × 64 × 64
```

logits.

After removing the single output-channel dimension, the model returns:

```text
64 × 64
```

pixel logits for each patch.

## Top-k Multiple-Instance Pooling

The model cannot directly compare its 64 × 64 output to one patch label.

Instead, the script flattens all pixel logits:

```text
64 × 64 = 4096 pixel logits
```

and selects the highest-scoring fraction.

The default is:

```text
topk_fraction = 0.10
```

or approximately:

```text
10% of pixels
```

For a 64 × 64 patch:

```text
4096 × 0.10 ≈ 410 pixels
```

The selected logits are averaged to produce one patch-level logit.

Conceptually:

```text
U-Net pixel logits
↓
select highest-scoring 10%
↓
average those logits
↓
patch-level logit
↓
compare with patch label
```

## Why Top-k Pooling Is Used

For a positive patch, the weak supervision does not establish that **every** pixel contains agave.

Using the highest-scoring subset allows the network to satisfy a positive patch label when only part of the image appears agave-like.

This is closer to the assumption:

```text
positive patch = some agave should be present
```

than:

```text
positive patch = every pixel is agave
```

## Patch-Level Loss

The pooled patch logit is compared with the true image-level label using:

```text
binary cross-entropy with logits
```

This is the main classification component of the weak supervision.

## Negative-Patch Constraint

The loss also applies a stronger pixel-level constraint to known negative patches.

For samples where:

```text
target = 0
```

every pixel logit is compared against:

```text
0
```

using binary cross-entropy.

This reflects a stronger assumption for negative samples:

```text
non-agave patch
→ pixels should generally not be predicted as agave
```

The negative-pixel loss is multiplied by:

```text
0.5
```

before being added to the full objective.

## Positive-Patch Sparsity Penalty

For positive patches, the script calculates the mean pixel probability.

This becomes a sparsity penalty.

The intention is to discourage the weak model from solving positive patch classification by simply predicting agave across the entire image.

The strength of this penalty is controlled by:

```text
--sparsity-weight
```

with default:

```text
0.02
```

## Total Variation Regularization

The loss also includes spatial total-variation regularization.

The script compares probabilities between adjacent pixels:

- Horizontally
- Vertically

and penalizes large differences.

This encourages smoother spatial probability maps.

The strength is controlled by:

```text
--tv-weight
```

with default:

```text
0.02
```

## Complete Weak-Supervision Loss

Conceptually, the loss is:

```text
patch classification loss
+
0.5 × negative-pixel loss
+
TV weight × spatial smoothness penalty
+
sparsity weight × positive-patch mean probability
```

The exact implementation combines all four terms.

## Important Loss Interpretation

The additional losses are **regularizers**, not replacements for true segmentation supervision.

They encourage the network toward plausible localized regions, but they do not provide actual field boundaries.

For example, the model is never directly told:

```text
these specific 143 pixels are agave
```

It is only told information like:

```text
this overall patch is agave-positive
```

or:

```text
this overall patch is non-agave
```

The pseudo-masks should therefore be interpreted accordingly.

## Handling Class Imbalance

The script supports optional balanced sampling.

Enable it with:

```text
--balanced-sampler
```

Class frequencies are calculated from the training targets.

Each training observation receives an inverse-frequency sampling weight, and training uses:

```text
WeightedRandomSampler
```

with replacement.

This increases the frequency with which minority-class patches are presented during training.

## Important Difference from Other CNN Scripts

This weak U-Net does **not** separately calculate a `pos_weight` for the patch-level loss.

Class balancing is handled through the optional sampler rather than an additional weighted binary classification loss.

## Training Device

The script automatically selects:

```text
CUDA
```

when available.

Otherwise it uses:

```text
CPU
```

A specific PyTorch device can be provided through:

```text
--device
```

## Optimizer

Training uses:

```text
AdamW
```

with defaults:

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

The scheduler monitors validation average precision.

## Patch-Level Evaluation

Although the neural network produces pixel-level maps, model selection and quantitative evaluation are performed at the **patch level**.

The top-k pooled logit is converted to a probability:

```text
probability_agave_patch
```

This patch probability is used to calculate classification metrics.

## Validation Metrics

The workflow calculates:

- Sample count
- Number of positives
- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- Confusion matrix
- ROC AUC
- Average precision

ROC AUC and average precision are only calculated when both target classes are represented.

## Model-Selection Metric

The best model is selected using:

```text
validation average precision
```

The scheduler also monitors this score.

Unlike some of the other agave CNN workflows, there is no fallback metric in the training loop if average precision is unavailable.

The validation set should therefore contain both target classes.

## Training Console Output

Each epoch prints a summary resembling:

```text
Epoch 020 train_loss=0.4821 val_loss=0.5074 val_f1=0.5483 val_AP=0.6021
```

The output includes:

- Epoch
- Training loss
- Validation loss
- Validation F1 at threshold 0.5
- Validation average precision

## Best Model Checkpoint

Whenever validation average precision improves sufficiently, the script saves:

```text
best_model.pt
```

The checkpoint contains:

- U-Net model weights
- Band means
- Band standard deviations
- Base-channel count
- Top-k fraction
- Best epoch
- Script version

## Early Stopping

Early stopping is enabled by default.

The default is:

```text
--patience 10
```

Training stops after ten consecutive epochs without sufficient validation-average-precision improvement.

Setting:

```text
--patience 0
```

disables early stopping and allows all requested epochs to run.

The default maximum training duration is:

```text
60 epochs
```

## Training History

After training, the script writes:

```text
training_history.csv
```

containing:

- Epoch
- Training loss
- Validation loss
- Validation F1 at 0.5
- Validation ROC AUC
- Validation average precision
- Learning rate

The best checkpoint is then reloaded for final evaluation.

## Patch Classification Threshold

After loading the best model, the script recalculates validation patch probabilities.

It tests thresholds from:

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

The threshold maximizing validation F1 is selected.

That threshold is then applied to the test patch probabilities.

## Test Predictions

The held-out test observations are written to:

```text
test_predictions.csv
```

Each record includes the original test-manifest information plus:

```text
probability_agave_patch
prediction
```

where:

```text
prediction = 1
```

when the patch probability exceeds the validation-selected threshold.

## Metrics Report

The final:

```text
metrics.json
```

contains:

```text
script_version
supervision
warning
device
best_model_epoch
selected_patch_threshold
topk_fraction
train_n
validation_n
test_n
test_metrics
test_by_year
```

The supervision type is explicitly recorded as:

```text
weak_patch_level_MIL
```

## Per-Year Test Metrics

Test predictions are grouped by:

```text
year
```

and the patch-level metrics are recalculated independently for each represented year.

The results are stored under:

```text
test_by_year
```

This makes it possible to evaluate whether the weakly supervised model performs differently across observation years.

## `predict`

The prediction command generates pixel-level pseudo-segmentation products.

A typical command is:

```bash
python agave_weak_unet.py predict ^
    --model "weak_unet_run\best_model.pt" ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_predictions"
```

The actual script filename should be replaced with the filename used in the repository.

## Prediction Preprocessing

For every manifest image, the prediction workflow:

1. Opens the Sentinel-2 raster.
2. Loads all nine bands.
3. Reads the raster validity mask.
4. Sets invalid pixels to non-finite values.
5. Replaces those values using the saved training-band means.
6. Applies the saved spectral normalization.
7. Runs the image through the U-Net.
8. Applies sigmoid to obtain pixel probabilities.

## Probability Raster

Each input image produces:

```text
{image_stem}_weak_unet_probability.tif
```

For example:

```text
INT_0042_2023_sentinel2_dry_median_64_snapped_weak_unet_probability.tif
```

The output is:

```text
single-band
float32
64 × 64
```

and retains the source image's geospatial profile.

Each pixel represents the weak U-Net's estimated agave probability.

## Pseudo-Mask Raster

The probability map is thresholded using:

```text
--pixel-threshold
```

The default is:

```text
0.5
```

Pixels with probability greater than or equal to the threshold become:

```text
1
```

and other pixels become:

```text
0
```

The output filename is:

```text
{image_stem}_weak_unet_pseudomask.tif
```

The pseudo-mask is written as:

```text
uint8
```

with:

```text
nodata = 255
```

## Patch Threshold vs. Pixel Threshold

The workflow contains **two different thresholds** that should not be confused.

### Patch Classification Threshold

Selected automatically from validation F1 during training.

It is used for:

```text
agave vs. non-agave patch classification
```

and is saved in the training metrics.

### Pixel Threshold

Specified during prediction through:

```text
--pixel-threshold
```

with default:

```text
0.5
```

It is used only to convert the pixel probability raster into a binary pseudo-mask.

The pixel threshold is not automatically optimized against true segmentation masks because no true pixel masks are available.

## Pseudo-Mask Summary

After prediction, the script writes:

```text
pseudo_mask_summary.csv
```

For every image, it records:

- `image_path`
- `base_id`
- `year`
- `target`
- Mean pixel probability
- Maximum pixel probability
- Fraction of pixels above the selected threshold
- Pixel threshold
- Probability raster path
- Pseudo-mask raster path

This provides a compact way to inspect how localized or widespread each predicted agave region is.

## Interpreting Pseudo-Mask Statistics

The summary statistics can be useful for quality control.

For example:

```text
mean_pixel_probability
```

describes the average agave probability across the patch.

```text
max_pixel_probability
```

shows the strongest localized response.

```text
fraction_pixels_above_threshold
```

shows how much of the image is classified as positive under the selected pixel threshold.

A positive patch with:

```text
fraction_pixels_above_threshold = 1.0
```

would mean the weak U-Net classified the entire patch as agave.

Because no true field mask was used during training, such outputs should be reviewed carefully.

## Main Training Outputs

A standard training directory contains:

```text
best_model.pt
normalization.json
training_history.csv
train_manifest.csv
validation_manifest.csv
test_manifest.csv
test_predictions.csv
metrics.json
```

## Main Prediction Outputs

A prediction directory contains multiple:

```text
*_weak_unet_probability.tif
*_weak_unet_pseudomask.tif
```

files plus:

```text
pseudo_mask_summary.csv
```

## Running the Script

### 1. Activate the project environment

Use a Python or Conda environment containing:

- `torch`
- `numpy`
- `pandas`
- `rasterio`
- `scikit-learn`
- `tqdm`

For example:

```bash
conda activate agave-oaxaca
```

### 2. Confirm the manifest

The manifest must contain:

```text
image_path
```

and sufficient fields to determine:

```text
target
base_id
year
```

### 3. Train the weak U-Net

A basic run is:

```bash
python agave_weak_unet.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_run"
```

### 4. Validate rasters during the final run

```bash
python agave_weak_unet.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_run" ^
    --validate-rasters
```

### 5. Review the geographic splits

Inspect:

```text
train_manifest.csv
validation_manifest.csv
test_manifest.csv
```

Confirm that the expected observations and spatial groups are represented.

### 6. Review training behavior

Inspect:

```text
training_history.csv
```

with particular attention to:

- Validation average precision
- Validation F1
- Validation loss
- Learning-rate changes

### 7. Review patch-level evaluation

Inspect:

```text
metrics.json
test_predictions.csv
```

Remember that these metrics evaluate the **patch-level agave classification**, not pixel-level segmentation accuracy.

### 8. Generate pseudo-masks

```bash
python agave_weak_unet.py predict ^
    --model "weak_unet_run\best_model.pt" ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_predictions"
```

### 9. Review pseudo-mask statistics

Inspect:

```text
pseudo_mask_summary.csv
```

and visually review representative:

```text
*_weak_unet_probability.tif
```

and:

```text
*_weak_unet_pseudomask.tif
```

outputs.

## Example Training Command with Explicit Settings

```bash
python agave_weak_unet.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_run" ^
    --epochs 60 ^
    --batch-size 16 ^
    --learning-rate 0.001 ^
    --weight-decay 0.0001 ^
    --base-channels 32 ^
    --topk-fraction 0.10 ^
    --tv-weight 0.02 ^
    --sparsity-weight 0.02 ^
    --validate-rasters
```

## No-Early-Stopping Example

To force every requested epoch to run:

```bash
python agave_weak_unet.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_full_epochs" ^
    --epochs 60 ^
    --patience 0
```

## Balanced-Sampling Example

```bash
python agave_weak_unet.py train ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_balanced" ^
    --balanced-sampler
```

## Alternate Pixel Threshold Example

To create more conservative pseudo-masks:

```bash
python agave_weak_unet.py predict ^
    --model "weak_unet_run\best_model.pt" ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-dir "weak_unet_predictions_t070" ^
    --pixel-threshold 0.70
```

Changing the pixel threshold does not retrain the network.

## Important Configuration Options

| Setting | Default | Purpose |
|---|---:|---|
| `--manifest` | Required | Input patch-level training manifest |
| `--output-dir` | Required | Training or prediction output directory |
| `--epochs` | `60` | Maximum number of training epochs |
| `--batch-size` | `16` | Image patches per batch |
| `--workers` | `0` | DataLoader worker processes |
| `--seed` | `42` | Reproducibility seed |
| `--learning-rate` | `0.001` | Initial AdamW learning rate |
| `--weight-decay` | `0.0001` | AdamW regularization |
| `--base-channels` | `32` | Initial U-Net channel width |
| `--topk-fraction` | `0.10` | Fraction of highest pixel logits pooled for patch classification |
| `--tv-weight` | `0.02` | Weight of spatial total-variation regularization |
| `--sparsity-weight` | `0.02` | Weight discouraging widespread positive probabilities |
| `--normalization-images` | `2000` | Maximum training images used for normalization |
| `--balanced-sampler` | Off | Enables inverse-frequency patch sampling |
| `--validate-rasters` | Off | Checks 9-band, 64 × 64 raster structure |
| `--patience` | `10` | Early-stopping patience; `0` disables early stopping |
| `--device` | Automatic | Optional PyTorch device override |
| `--pixel-threshold` | `0.5` | Threshold used to create binary pseudo-masks |

These defaults are defined directly in the command-line interface.

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

The seed is also used for the normalization-image sample and grouped splitting.

## PyTorch Checkpoint Loading

The script loads model checkpoints using:

```python
weights_only=False
```

because the checkpoint contains additional metadata such as:

- Band means
- Band standard deviations
- U-Net channel width
- Top-k fraction
- Epoch
- Script version

Only trusted checkpoints should be loaded using this behavior.

## Recommendations

- Treat this workflow as **weakly supervised localization**, not fully supervised segmentation.
- Do not report pseudo-mask pixels as verified agave field boundaries.
- Keep the warning about missing true pixel supervision with any model results derived from this workflow.
- Use a standard single-year CNN as the primary patch-classification baseline when pixel localization is not required.
- Use a fully supervised U-Net when reliable pixel-level agave masks become available.
- Preserve `base_id` throughout the workflow to prevent spatial leakage.
- Prefer supplied geographic splits when comparing this model directly with another experiment using the same samples.
- Use `--validate-rasters` for final training runs.
- Review validation class composition because average precision is the model-selection metric.
- Preserve `train_manifest.csv`, `validation_manifest.csv`, and `test_manifest.csv` with each run.
- Keep normalization statistics with the corresponding model checkpoint.
- Record `topk_fraction`, `tv_weight`, and `sparsity_weight` because they directly affect the weak-localization behavior.
- Treat changes to these weak-supervision parameters as separate experiments.
- Inspect both patch-level performance and pseudo-mask appearance.
- Do not use patch-level accuracy or F1 as evidence that the pixel-level pseudo-masks are spatially accurate.
- Review probability rasters in addition to thresholded pseudo-masks.
- Compare several pixel thresholds before choosing one for visualization or downstream exploratory analysis.
- Do not select a pixel threshold by comparing against the patch target; a patch label does not identify the correct pixel boundary.
- Examine `fraction_pixels_above_threshold` for evidence that the network is producing unrealistically broad positive regions.
- Review negative patches for false positive localization.
- Review positive patches with known field context to determine whether high-probability areas appear visually plausible.
- Use pseudo-masks as exploratory labels, candidate regions, or inputs to later refinement rather than treating them automatically as ground truth.
- Preserve the distinction between the validation-selected **patch threshold** and user-selected **pixel threshold**.
- Compare this weak U-Net against the standard CNN and temporal CNN at the patch-classification level before drawing conclusions about whether the U-Net architecture itself improves classification.
- If verified field masks become available, retrain and evaluate a fully supervised segmentation model rather than treating weak pseudo-masks as equivalent supervision.

## Current Workflow Summary

In sequence, the script:

1. Loads the user-supplied CSV manifest.
2. Confirms that `image_path` exists.
3. Uses `target` directly when available.
4. Otherwise converts `standardized_label` into a binary target.
5. Otherwise converts recognized values from `label`.
6. Requires a usable binary target representation.
7. Uses the existing `base_id` when available.
8. Otherwise derives `base_id` from a year-specific `id`.
9. Uses `year` when available.
10. Otherwise substitutes `target_year`.
11. Removes rows whose targets are not binary.
12. Standardizes target, year, `base_id`, and image-path data types.
13. Checks for supplied train, validation, and test split assignments.
14. Normalizes `val` split names to `validation`.
15. Reuses complete supplied splits when possible.
16. Checks supplied splits for `base_id` leakage.
17. Otherwise creates a grouped 70/15/15 spatial split.
18. Saves the resulting training manifest.
19. Saves the resulting validation manifest.
20. Saves the resulting test manifest.
21. Optionally validates all unique Sentinel-2 rasters.
22. Confirms each validated image contains nine bands.
23. Confirms each validated image is 64 × 64 pixels.
24. Samples training imagery for normalization.
25. Calculates nine-band means and standard deviations.
26. Saves the normalization statistics.
27. Loads each image with its raster validity mask.
28. Converts invalid pixels to non-finite values.
29. Replaces invalid values with training-band means.
30. Standardizes all nine spectral bands.
31. Applies random flips and rotations to training images.
32. Optionally creates an inverse-frequency weighted sampler.
33. Constructs training, validation, and test DataLoaders.
34. Selects CUDA or CPU.
35. Initializes the multispectral U-Net.
36. Encodes the image through four convolutional downsampling stages.
37. Processes the representation through the bottleneck.
38. Reconstructs the spatial representation through the decoder.
39. Combines encoder and decoder information through skip connections.
40. Produces one 64 × 64 pixel-logit map for every patch.
41. Flattens the pixel logits for weak supervision.
42. Selects the highest-scoring configured fraction of pixels.
43. Averages those top pixel logits into one patch-level logit.
44. Calculates patch-level binary cross-entropy.
45. Applies an additional all-negative pixel loss to non-agave patches.
46. Calculates total-variation spatial regularization.
47. Calculates the positive-patch sparsity penalty.
48. Combines the weak-supervision loss terms.
49. Optimizes the model using AdamW.
50. Calculates patch probabilities from the top-k pooled logits.
51. Evaluates validation classification at threshold 0.5 after every epoch.
52. Calculates validation accuracy.
53. Calculates validation balanced accuracy.
54. Calculates validation precision.
55. Calculates validation recall.
56. Calculates validation F1.
57. Calculates validation ROC AUC when possible.
58. Calculates validation average precision when possible.
59. Uses validation average precision for model selection.
60. Adjusts the learning rate when validation average precision plateaus.
61. Saves `best_model.pt` whenever validation performance improves.
62. Tracks consecutive non-improving epochs.
63. Stops early when the configured positive patience is reached.
64. Writes the complete training history.
65. Reloads the best model checkpoint.
66. Recalculates validation patch probabilities.
67. Searches patch-classification thresholds from 0.05 through 0.95.
68. Selects the threshold maximizing validation F1.
69. Generates held-out test patch probabilities.
70. Converts them into binary patch predictions.
71. Writes `test_predictions.csv`.
72. Calculates overall patch-level test metrics.
73. Calculates patch-level metrics separately by year.
74. Records the supervision type as weak patch-level MIL.
75. Records an explicit warning that no true pixel masks were used.
76. Writes the final `metrics.json`.
77. Loads the best U-Net checkpoint for prediction.
78. Restores the training-band normalization values.
79. Standardizes every manifest image using the saved statistics.
80. Generates a full 64 × 64 pixel probability map.
81. Writes the probability map as a georeferenced float32 GeoTIFF.
82. Applies the configured pixel threshold.
83. Creates a binary pseudo-mask.
84. Writes the pseudo-mask as a georeferenced uint8 GeoTIFF.
85. Calculates the mean pixel probability.
86. Calculates the maximum pixel probability.
87. Calculates the fraction of pixels exceeding the threshold.
88. Records the probability and pseudo-mask raster paths.
89. Writes all per-image pseudo-mask statistics to `pseudo_mask_summary.csv`.

This workflow provides an experimental bridge between patch-level classification and full pixel-level segmentation. It allows the existing agave/non-agave patch labels to guide a U-Net toward spatially localized predictions without pretending that the dataset contains verified field masks. The resulting probability surfaces and pseudo-masks are therefore most appropriate for exploratory localization, weak-label generation, and comparison with later fully supervised segmentation results.