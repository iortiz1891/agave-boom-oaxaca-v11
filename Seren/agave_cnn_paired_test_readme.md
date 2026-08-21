# Agave Paired Single-Year CNN Ablation Workflow

## Short Summary

This script trains a **single-year multispectral CNN on the exact same target samples and exact same geographic train/validation/test splits used by a completed temporal CNN run**.

Its purpose is to isolate the value of temporal context.

The temporal CNN receives:

```text
[t-2, t-1, t] → target label at t
```

while this paired single-year CNN receives only:

```text
t → the same target label at t
```

Because both models use the same target observations and the same `base_id` split assignments, differences in performance can be interpreted as evidence about the added value of multi-year temporal imagery rather than differences in sample composition or geographic partitioning.

The script:

- Loads the completed temporal CNN's train, validation, and test sequence files
- Extracts only the final-year image from every temporal sequence
- Preserves the exact temporal model targets
- Preserves the exact temporal model `base_id` partitions
- Checks the inherited splits for spatial leakage
- Optionally validates all final-year Sentinel-2 rasters
- Computes new normalization statistics using paired training images only
- Trains the same general multispectral CNN family used by the single-year baseline
- Supports weighted binary loss or balanced sampling
- Uses validation average precision for model selection when available
- Saves the best model checkpoint
- Selects the final classification threshold using validation F1
- Produces validation and test predictions
- Calculates overall and per-target-year performance
- Automatically compares results with the temporal CNN when its `metrics.json` is available
- Writes direct single-year-minus-temporal metric differences to `paired_comparison.json`

## Overview

This workflow is an **ablation experiment**.

The temporal CNN and paired single-year CNN are deliberately constructed to differ in one central dimension:

```text
Temporal model:
previous years + target year

Paired single-year model:
target year only
```

Everything possible about the evaluation dataset is otherwise kept the same.

For example, if the temporal model evaluates:

```text
2020 image
2021 image
2022 image
→ 2022 target
```

the paired model evaluates:

```text
2022 image
→ 2022 target
```

for the **same sequence record**.

This creates a substantially cleaner comparison than training a separate single-year CNN using a newly generated random split.

The script version is:

```text
1.0-paired-single-year-ablation
```

## Experimental Question

The central question is:

> Does supplying the previous two years of Sentinel-2 imagery improve agave classification compared with supplying only the target-year image?

The paired design attempts to answer this while controlling for:

- Target sample identity
- Target year
- True class label
- Geographic `base_id`
- Train/validation/test assignment

The key experimental contrast is therefore:

```text
single-year target image
vs.
three-year temporal sequence
```

rather than:

```text
different single-year dataset
vs.
different temporal dataset
```

## Required Temporal Run Files

The script expects a completed temporal CNN run directory containing:

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

The temporal run should also contain:

```text
metrics.json
```

when automatic comparison with the temporal model is desired.

The three sequence split files are required.

The temporal `metrics.json` is optional for training, but it is required to automatically produce:

```text
paired_comparison.json
```

## Main Outputs

A standard paired run produces:

```text
best_model.pt
normalization.json
training_history.csv
train_paired.csv
validation_paired.csv
test_paired.csv
validation_predictions.csv
test_predictions.csv
metrics.json
```

When the source temporal run contains a readable `metrics.json`, the script also creates:

```text
paired_comparison.json
```

These expected inputs and outputs are defined directly in the script documentation.

## Expected Sentinel-2 Imagery

The paired model expects:

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

The constants are:

```python
EXPECTED_BANDS = 9
EXPECTED_SIZE = 64
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

## Extracting the Final-Year Image

The temporal sequence files contain image columns such as:

```text
image_path_0
image_path_1
image_path_2
```

The script automatically identifies all columns beginning with:

```text
image_path_
```

extracts their numerical suffixes, sorts them, and selects the highest-numbered image column.

For a standard three-year sequence:

```text
image_path_0 = t-2
image_path_1 = t-1
image_path_2 = t
```

the paired experiment therefore selects:

```text
image_path_2
```

as the single-year CNN input.

This means the model receives the exact final-year image that the temporal CNN also saw as the last frame in its sequence.

## Flexible Sequence Length Support

The final-year selection does not assume that the temporal model always used exactly three years.

For example, if a temporal sequence contains:

```text
image_path_0
image_path_1
image_path_2
image_path_3
image_path_4
```

the paired script selects:

```text
image_path_4
```

because it is the highest indexed image path.

This allows the paired workflow to remain conceptually valid if the temporal sequence length changes.

## Converting Temporal Splits

Each temporal sequence partition is converted into a simpler paired single-year table.

The paired output contains:

```text
image_path
base_id
year
target
sequence_id
source_temporal_image_column
split
```

The values are derived as follows:

```text
image_path = final image_path_N column
base_id    = temporal sequence base_id
year       = temporal target_year
target     = temporal target
```

The original `sequence_id` is preserved when available.

If no `sequence_id` exists, the script creates one from:

```text
{base_id}_{year}
```

The field:

```text
source_temporal_image_column
```

records which temporal image column was used as the paired single-year input.

## Why the Sequence ID Is Preserved

Preserving the original sequence identity provides a direct connection between:

- Temporal prediction
- Paired single-year prediction
- Target sample
- Target year

This makes later sample-level comparison possible without reconstructing the relationship from filenames.

## Exact Split Preservation

The script loads:

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

from the existing temporal run.

It then converts each independently into:

```text
train_paired.csv
validation_paired.csv
test_paired.csv
```

It does **not** run a new random split.

This is the central feature of the experiment.

The paired single-year model must evaluate the same geographic partitions as the temporal model.

## Spatial Leakage Validation

After converting the temporal splits, the script compares their `base_id` sets.

It stops if any location appears in more than one of:

- Training
- Validation
- Testing

The error is:

```text
base_id leakage detected in temporal split files.
```

The workflow therefore inherits the temporal experiment's geographic independence while also explicitly checking that the inherited split remains valid.

## Duplicate Sequence Validation

Each split is also checked for duplicate:

```text
sequence_id
```

values.

Duplicate sequence identifiers cause the script to stop.

This ensures that the same temporal target observation is not unintentionally represented more than once inside one partition.

## Target Validation

All targets must be binary:

```text
0
1
```

Any other target value causes the experiment to stop.

This ensures that the paired model is evaluating the same binary agave classification problem as the temporal CNN.

## Optional Raster Validation

The command-line flag:

```text
--validate-rasters
```

can be used to inspect every unique final-year image across the paired:

- Training split
- Validation split
- Test split

Each raster must contain exactly:

```text
9 bands
64 × 64 pixels
```

A different band count or raster size causes an error.

## Why Only Final-Year Rasters Are Validated

The paired model does not use:

```text
t-2
t-1
```

imagery.

It therefore validates only the selected target-year image.

This is appropriate for the experimental question because the paired model's input is intentionally limited to:

```text
t
```

## Paired Split Outputs

Before training begins, the converted partitions are written to:

```text
train_paired.csv
validation_paired.csv
test_paired.csv
```

These files provide a permanent record of the exact target samples used in the paired experiment.

They should be preserved alongside the temporal run's original:

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

## Spectral Normalization

The paired CNN calculates its own band normalization statistics.

These are derived from:

```text
paired training final-year images only
```

Validation and test imagery are not included.

For each of the nine bands, the script calculates:

```text
mean
standard deviation
```

The normalized value is:

```text
(x - mean) / standard deviation
```

The resulting values are saved to:

```text
normalization.json
```

## Why the Paired Model Recomputes Normalization

The temporal CNN calculates normalization across imagery contained in its training sequences.

The paired CNN instead uses only the final-year images that actually serve as its inputs.

This prevents unused earlier-year imagery from influencing the paired model's preprocessing.

The normalization file explicitly records:

```text
normalization_source:
paired training final-year images only
```



## Normalization Sample Size

By default, normalization uses up to:

```text
2000 images
```

controlled by:

```text
--normalization-images
```

If the training set contains fewer than the configured maximum, all training images are used.

If the training set is larger, a reproducible subset is sampled using the configured random seed.

## Invalid Pixel Handling

Each image is loaded with Rasterio.

The script obtains the raster validity mask and temporarily converts invalid areas to:

```text
NaN
```

Any remaining non-finite values are replaced with the corresponding training-band mean.

The image is then normalized.

This means missing pixels become approximately:

```text
0
```

after standardization because they have been filled with the band mean before normalization.

## Training Data Augmentation

Paired training imagery receives random spatial augmentation.

Possible operations include:

- Horizontal flip
- Vertical flip
- 90° rotation
- 180° rotation
- 270° rotation

Validation and test imagery do not receive augmentation.

## CNN Architecture

The model uses the same general spatial CNN family as the original single-year multispectral baseline.

Its feature hierarchy is:

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
Adaptive Average Pooling
```

The encoder contains four convolution blocks.

Max pooling is applied between the first three stages.

## Convolution Blocks

Each block contains:

```text
3 × 3 convolution
Batch Normalization
ReLU
3 × 3 convolution
Batch Normalization
ReLU
```

The spatial representation is ultimately reduced to:

```text
256 × 1 × 1
```

by adaptive average pooling.

## Classification Head

The final pooled representation passes through:

```text
Flatten
↓
Linear 256 → 128
↓
ReLU
↓
Dropout
↓
Linear 128 → 1
```

The default dropout value is:

```text
0.30
```

The output is one logit representing patch-level agave classification.

A sigmoid converts this into:

```text
probability_agave
```

The architecture matches the same single-year CNN family used for the earlier baseline rather than introducing a substantially different spatial model for the ablation.

## Core Experimental Difference

The spatial CNN sees:

```text
target-year Sentinel-2 patch
```

while the temporal CNN sees:

```text
target-year Sentinel-2 patch
+
two preceding yearly patches
```

The paired script therefore removes the temporal component while maintaining the target sample.

## Handling Class Imbalance

The paired experiment supports:

- Positive-class loss weighting
- Balanced sampling

By default, weighted binary loss is used.

## Positive-Class Weight

The script calculates:

```text
number of negative training samples
───────────────────────────────────
number of positive training samples
```

and uses the result as:

```text
pos_weight
```

in:

```text
BCEWithLogitsLoss
```

## Balanced Sampler

Balanced sampling can be enabled with:

```text
--balanced-sampler
```

The script assigns inverse-class-frequency sampling weights and uses:

```text
WeightedRandomSampler
```

with replacement.

When balanced sampling is active, positive-class weighting is disabled.

This avoids applying both balancing strategies simultaneously.

## Training Device

The device is selected as:

```text
CUDA
```

when available.

Otherwise:

```text
CPU
```

is used.

A specific PyTorch device can also be passed through:

```text
--device
```

## Loss Function

The model trains using:

```text
BCEWithLogitsLoss
```

for binary agave classification.

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

The paired model uses:

```text
ReduceLROnPlateau
```

with:

```text
mode = max
factor = 0.5
patience = 3
```

The scheduler responds to the same validation model-selection score used for checkpoint selection.

## Validation Model Selection

After every training epoch, the script evaluates the validation set at a temporary classification threshold of:

```text
0.5
```

It calculates:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- Confusion matrix
- ROC AUC
- Average precision

The primary model-selection score is:

```text
validation average precision
```

when both classes are present.

If average precision cannot be calculated, the fallback is:

```text
validation F1
```

## Training Console Output

Each epoch prints a summary such as:

```text
Epoch 028 train_loss=0.3012 val_loss=0.4125 val_f1=0.6037 val_AP=0.5874
```

The displayed values include:

- Epoch
- Training loss
- Validation loss
- Validation F1 at 0.5
- Validation average precision

## Training History

The complete history is saved to:

```text
training_history.csv
```

Each row records:

- Epoch
- Training loss
- Validation loss
- Validation F1 at 0.5
- Validation ROC AUC
- Validation average precision
- Learning rate

## Best Model Checkpoint

Whenever the validation model-selection score improves by more than:

```text
0.000001
```

the model is saved to:

```text
best_model.pt
```

The checkpoint contains:

- Model weights
- Band means
- Band standard deviations
- Dropout value
- Best epoch
- Best validation score
- Script version

## Early Stopping

Early stopping is supported but is **disabled by default**.

The default is:

```text
--patience 0
```

which means:

```text
run every requested epoch
```

A positive value enables early stopping.

For example:

```text
--patience 10
```

stops training after ten consecutive non-improving epochs.

The default maximum training length is:

```text
60 epochs
```

This behavior is explicitly implemented so that `patience=0` intentionally allows every requested epoch to run.

## Why Early Stopping Is Disabled by Default

For a paired ablation, using the requested training budget consistently can make comparison with the temporal experiment easier.

However, if the temporal model itself used early stopping, the experimenter should record the difference in optimization behavior when interpreting the comparison.

The script preserves the best-validation checkpoint regardless of whether all epochs are run.

## Validation Threshold Selection

After training, the best checkpoint is reloaded.

The script generates validation probabilities and tests thresholds from:

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

The threshold producing the highest validation F1 is selected.

Importantly:

```text
Test data never selects the threshold.
```



## Validation Predictions

The best model's validation results are written to:

```text
validation_predictions.csv
```

The file contains the paired validation records plus:

```text
probability_agave
prediction
```

This allows the selected threshold and individual validation behavior to be inspected directly.

## Test Predictions

The same validation-selected threshold is then applied to the held-out test probabilities.

The results are written to:

```text
test_predictions.csv
```

Each test observation receives:

```text
probability_agave
prediction
```

## Binary Metrics

The script reports:

- Sample count
- Positive sample count
- Classification threshold
- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- Confusion matrix
- ROC AUC
- Average precision

ROC AUC and average precision are set to `None` if only one target class is represented.

## Overall Metrics Report

The paired run writes:

```text
metrics.json
```

containing:

```text
script_version
experiment
temporal_run_dir
device
best_model_epoch
epochs_requested
train_n
validation_n
test_n
train_base_ids
validation_base_ids
test_base_ids
selected_threshold
test_metrics
per_target_year_metrics
```

The experiment is labeled:

```text
paired_single_year_vs_temporal_ablation
```

## Per-Target-Year Metrics

Test predictions are grouped by:

```text
year
```

which corresponds to the original temporal:

```text
target_year
```

The script calculates separate metrics for every represented target year.

This allows direct comparison of temporal and single-year performance by year.

## Temporal Comparison

If the source temporal run contains:

```text
metrics.json
```

the paired script automatically reads it after evaluating the single-year model.

It then creates:

```text
paired_comparison.json
```

## Overall Comparison Metrics

The automatic comparison includes:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC AUC
- Average precision

For every metric it records:

```text
paired_single_year
temporal_3yr
delta_single_minus_temporal
```

For example:

```text
paired_single_year = 0.61
temporal_3yr       = 0.66
delta              = -0.05
```

A negative delta means the temporal model performed better on that metric.

A positive delta means the paired single-year model performed better.

The comparison function explicitly computes the metric difference as single-year minus temporal.

## Sample Count Comparison

The comparison file also records:

```text
single_year_test_n
temporal_test_n
```

This provides a direct check that both models were evaluated against the same number of test observations.

Because the paired model is built from the temporal split files, these should normally match.

A mismatch should be investigated before interpreting metric differences.

## Per-Year Temporal Comparison

When the temporal model provides:

```text
per_target_year_metrics
```

the paired workflow performs an additional year-by-year comparison.

For each shared target year, it compares:

- F1
- Precision
- Recall
- ROC AUC
- Average precision

Each again receives:

```text
paired_single_year
temporal_3yr
delta_single_minus_temporal
```

This makes it possible to determine whether temporal context helps consistently or primarily during particular years.

## Why the Paired Comparison Is Important

Suppose an independently trained single-year CNN scores:

```text
F1 = 0.62
```

while a temporal CNN scores:

```text
F1 = 0.67
```

If those models used different samples or different geographic splits, the difference cannot be cleanly attributed to temporal information.

This paired workflow instead attempts to ensure:

```text
same target observations
same target years
same geographic partitions
```

so that the major input difference becomes:

```text
single image
vs.
multi-year sequence
```

## Running the Script

### 1. Complete the temporal CNN run first

The source directory must contain:

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

Preferably it should also contain:

```text
metrics.json
```

### 2. Activate the project environment

Use the Python or Conda environment containing:

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

### 3. Run the paired experiment

A basic run is:

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "temporal_cnn_run_01" ^
    --output-dir "paired_single_year_run_01"
```

### 4. Validate the source rasters

For the final experiment, raster validation can be enabled:

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "temporal_cnn_run_01" ^
    --output-dir "paired_single_year_run_01" ^
    --validate-rasters
```

### 5. Review the inherited paired splits

Inspect:

```text
train_paired.csv
validation_paired.csv
test_paired.csv
```

These should correspond one-for-one with the temporal sequence targets.

### 6. Review training performance

Inspect:

```text
training_history.csv
```

and:

```text
best_model.pt
```

### 7. Review paired model metrics

Inspect:

```text
metrics.json
```

Pay particular attention to:

- `best_model_epoch`
- `selected_threshold`
- `test_metrics`
- `per_target_year_metrics`

### 8. Review direct temporal comparison

When available, inspect:

```text
paired_comparison.json
```

This is the primary output for determining whether temporal context improved the model on the controlled paired test.

## Example Command with Explicit Settings

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "temporal_cnn_grouped" ^
    --output-dir "cnn_paired_single_year" ^
    --epochs 60 ^
    --batch-size 32 ^
    --learning-rate 0.001 ^
    --weight-decay 0.0001 ^
    --dropout 0.30 ^
    --validate-rasters
```

## Balanced-Sampling Example

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "temporal_cnn_grouped" ^
    --output-dir "cnn_paired_balanced" ^
    --balanced-sampler
```

Because balanced sampling changes training behavior, it should generally be treated as a separate experiment rather than overwriting the standard paired baseline.

## Early-Stopping Example

To enable early stopping:

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "temporal_cnn_grouped" ^
    --output-dir "cnn_paired_earlystop" ^
    --patience 10
```

## Important Configuration Options

| Setting                  |   Default | Purpose                                                 |
| ------------------------ | --------: | ------------------------------------------------------- |
| `--temporal-run-dir`     |  Required | Completed temporal CNN run supplying the exact splits   |
| `--output-dir`           |  Required | Destination for paired experiment outputs               |
| `--epochs`               |      `60` | Requested training epochs                               |
| `--batch-size`           |      `32` | Number of patches per batch                             |
| `--workers`              |       `0` | DataLoader worker processes                             |
| `--seed`                 |      `42` | Reproducibility seed                                    |
| `--learning-rate`        |   `0.001` | Initial AdamW learning rate                             |
| `--weight-decay`         |  `0.0001` | AdamW regularization                                    |
| `--dropout`              |    `0.30` | Classifier dropout                                      |
| `--patience`             |       `0` | Early stopping; `0` runs every requested epoch          |
| `--normalization-images` |    `2000` | Maximum paired training images used for normalization   |
| `--balanced-sampler`     |       Off | Enables inverse-frequency sampling                      |
| `--validate-rasters`     |       Off | Validates final-year GeoTIFF dimensions and band counts |
| `--device`               | Automatic | Explicit PyTorch device override                        |

These settings are defined in the script's `train` command.

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

The same seed is also used when selecting a subset of training images for normalization.

## PyTorch Checkpoint Loading

The script explicitly loads checkpoints using:

```python
weights_only=False
```

because the checkpoint contains metadata in addition to the model state dictionary.

This behavior is intended for local checkpoints created by the workflow.

The script notes the PyTorch 2.6 change in default `torch.load()` behavior.

## Recommendations

- Complete and finalize the temporal CNN experiment before running this paired ablation.
- Do not regenerate train, validation, or test splits for the paired model.
- Preserve the original temporal split CSVs exactly.
- Keep paired results in a separate output directory from the temporal run.
- Use `--validate-rasters` for the final reported paired experiment.
- Confirm that single-year and temporal test sample counts match before interpreting performance differences.
- Preserve `sequence_id` so individual paired predictions can be matched directly to temporal predictions.
- Preserve `source_temporal_image_column` so the selected target-year input is auditable.
- Use the final-year image only; adding earlier imagery would defeat the purpose of the ablation.
- Compute normalization from the paired training inputs rather than reusing statistics influenced by earlier temporal frames.
- Do not tune the classification threshold using test data.
- Compare average precision, ROC AUC, F1, precision, recall, and balanced accuracy rather than relying on accuracy alone.
- Examine `per_target_year_metrics` to determine whether temporal gains differ across years.
- Interpret `delta_single_minus_temporal` consistently: negative values favor the temporal model, while positive values favor the paired single-year model.
- Investigate any mismatch between paired and temporal test sample counts before drawing conclusions.
- Keep model architecture and optimization choices as comparable as practical when interpreting the temporal ablation.
- Record whether balanced sampling or early stopping was used, since those choices can affect comparability.
- Preserve `best_model.pt`, `normalization.json`, `metrics.json`, and `paired_comparison.json` with the final experiment.
- Treat this paired experiment as the preferred direct test of whether temporal context improves classification over target-year imagery alone.

## Current Workflow Summary

In sequence, the script:

1. Loads the completed temporal CNN run directory.
2. Locates `train_sequences.csv`, `validation_sequences.csv`, and `test_sequences.csv`.
3. Loads each temporal split independently.
4. Searches each split for all `image_path_N` columns.
5. Sorts those columns by their numerical sequence position.
6. Identifies the final temporal image column.
7. Extracts the final-year image from every temporal sequence.
8. Copies the original `base_id`.
9. Converts `target_year` into the paired `year`.
10. Copies the exact temporal target.
11. Preserves the original sequence ID when available.
12. Records which temporal image column supplied the paired input.
13. Preserves the original train, validation, or test assignment.
14. Checks that no `base_id` appears across multiple partitions.
15. Checks for duplicate sequence IDs.
16. Confirms every target is binary.
17. Reports the number of paired samples and unique locations in each split.
18. Reports the target-year distribution in the paired test set.
19. Optionally validates every unique target-year raster.
20. Confirms each validated raster contains nine bands.
21. Confirms each validated raster is 64 × 64 pixels.
22. Saves `train_paired.csv`.
23. Saves `validation_paired.csv`.
24. Saves `test_paired.csv`.
25. Samples paired training images for normalization.
26. Calculates per-band means and standard deviations.
27. Saves normalization statistics and records their source.
28. Builds paired training, validation, and test datasets.
29. Applies each raster's validity mask.
30. Replaces invalid values with paired training-band means.
31. Standardizes each spectral band.
32. Applies random flips and rotations to training images.
33. Optionally constructs an inverse-frequency weighted sampler.
34. Creates DataLoaders for all three partitions.
35. Selects CUDA or CPU.
36. Initializes the 9-band multispectral CNN.
37. Calculates the positive-class training weight.
38. Uses weighted binary cross-entropy unless balanced sampling is enabled.
39. Initializes AdamW.
40. Initializes a plateau-based learning-rate scheduler.
41. Trains for the requested maximum number of epochs.
42. Evaluates the validation split after every epoch.
43. Calculates validation F1, ROC AUC, and average precision.
44. Uses validation average precision as the preferred model-selection score.
45. Falls back to validation F1 when necessary.
46. Reduces the learning rate when validation performance plateaus.
47. Saves a new `best_model.pt` whenever validation performance improves.
48. Runs all requested epochs by default because patience is zero.
49. Optionally stops early when a positive patience value is configured.
50. Writes the complete `training_history.csv`.
51. Reloads the best-validation checkpoint.
52. Generates validation probabilities.
53. Tests thresholds from 0.05 through 0.95.
54. Selects the threshold maximizing validation F1.
55. Saves validation probabilities and predictions.
56. Generates held-out test probabilities.
57. Applies the validation-selected threshold.
58. Saves individual test predictions.
59. Calculates overall test metrics.
60. Calculates metrics separately for every target year.
61. Writes the paired experiment `metrics.json`.
62. Searches the source temporal run for its `metrics.json`.
63. Loads temporal test metrics when available.
64. Checks paired and temporal test sample counts.
65. Compares overall accuracy.
66. Compares balanced accuracy.
67. Compares precision.
68. Compares recall.
69. Compares F1.
70. Compares ROC AUC.
71. Compares average precision.
72. Calculates each metric as paired single-year minus temporal.
73. Retrieves temporal per-target-year metrics when available.
74. Compares paired and temporal F1 by year.
75. Compares paired and temporal precision by year.
76. Compares paired and temporal recall by year.
77. Compares paired and temporal ROC AUC by year.
78. Compares paired and temporal average precision by year.
79. Writes the complete comparison to `paired_comparison.json`.

This workflow provides a controlled single-year ablation for the temporal CNN experiment. By evaluating the same target samples under the same geographic splits while removing the earlier-year imagery, it creates a direct test of whether multi-year Sentinel-2 context contributes measurable predictive value beyond the target-year image alone.
