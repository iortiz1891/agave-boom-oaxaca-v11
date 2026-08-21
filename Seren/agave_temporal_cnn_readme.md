# Agave Three-Year Spatiotemporal CNN Workflow

## Short Summary

This script builds, trains, resumes, evaluates, and applies a spatiotemporal convolutional neural network using multi-year Sentinel-2 image sequences.

It:

- Reads an existing single-year CNN manifest
- Groups imagery by spatial `base_id`
- Constructs chronological rolling sequences of consecutive years
- Uses three-year sequences by default
- Predicts the agave/non-agave label for the final year in each sequence
- Requires complete consecutive-year windows
- Optionally validates every raster in each sequence
- Keeps all sequences from the same spatial location in the same data split
- Supports grouped spatial train/validation/test splitting
- Supports stricter future-year validation and test experiments
- Computes spectral normalization using training sequences only
- Applies identical spatial augmentation across every year in a sequence
- Uses a shared CNN to encode each yearly image
- Uses a 1D temporal CNN to learn changes across yearly embeddings
- Supports class balancing through weighted sampling or weighted loss
- Tracks validation F1 and average precision
- Saves the best-performing model
- Selects a final classification threshold from validation predictions
- Reports overall and per-target-year test metrics
- Can resume an existing training run
- Can recover final evaluation outputs from an existing checkpoint
- Predicts agave probabilities for new sequence manifests

The default input sequence contains:

```text
[t-2, t-1, t]
```

and the prediction target is the observed agave label at:

```text
t
```

All sequences belonging to the same `base_id` remain together during data splitting to prevent spatial leakage.

## Overview

This script extends the single-year multispectral CNN into a **spatiotemporal classification model**.

Instead of predicting agave presence from one Sentinel-2 image patch, the model receives a chronological sequence of patches from the same location.

For the default three-year sequence:

```text
2019 image
2020 image
2021 image
```

the target is:

```text
2021 label
```

The model therefore learns from both:

- Spatial and spectral information within each yearly Sentinel-2 patch
- Changes in the learned image representation across consecutive years

The script contains five primary commands:

```text
build-sequences
train
resume
finalize
predict
```

The script version is:

```text
1.2-resume-training
```

## Expected Source Manifest

The temporal workflow begins with an existing **single-year CNN manifest**.

The expected columns include:

```text
image_path
image_name
image_id
base_id
year
target
```

At minimum, the sequence builder requires:

```text
image_path
base_id
year
target
```

The target must be binary:

```text
0 = non-agave
1 = agave
```

Rows whose targets are not `0` or `1` are excluded before sequences are constructed.

## Expected Sentinel-2 Imagery

Each image in a temporal sequence must contain:

```text
9 bands
64 × 64 pixels
```

The expected band names are:

1. B2
2. B3
3. B4
4. B5
5. B6
6. B7
7. B8
8. B11
9. B12

The raster-validation function checks that every image contains exactly:

```text
9 × 64 × 64
```

in band-height-width form.

## Main Commands

| Command | Purpose |
|---|---|
| `build-sequences` | Convert single-year observations into chronological temporal sequences |
| `train` | Train a shared spatial encoder and temporal CNN |
| `resume` | Continue an existing training run through a requested total epoch count |
| `finalize` | Recover validation/test evaluation from an existing checkpoint |
| `predict` | Generate probabilities from an existing sequence manifest |

## `build-sequences`

The first stage converts the single-year manifest into rolling temporal sequences.

Example:

```bash
python agave_temporal_cnn.py build-sequences ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-csv "agave_temporal_sequences_3yr.csv"
```

The default sequence length is:

```text
3
```

This can be changed using:

```text
--sequence-length
```

## Sequence Construction

The source manifest is grouped by:

```text
base_id
```

Within each location, records are organized by year.

For each potential target year, the script constructs the required consecutive-year window.

With:

```text
sequence_length = 3
```

a target year of 2022 requires:

```text
2020
2021
2022
```

A target year of 2023 requires:

```text
2021
2022
2023
```

and so on.

## Consecutive Years Are Required

The workflow does not fill temporal gaps.

For example, this location:

```text
2019
2020
2022
```

cannot produce the three-year sequence:

```text
2020
2021
2022
```

because 2021 is missing.

That candidate window is skipped and counted under:

```text
candidate_windows_skipped_for_missing_years
```

This ensures every temporal input represents a consistent sequence of consecutive observations.

## Duplicate Location-Year Records

The source manifest must contain no duplicate:

```text
base_id + year
```

combinations.

If duplicate records exist, sequence construction stops.

The error includes examples containing:

- `base_id`
- `year`
- `image_path`

This prevents ambiguity over which image should represent a particular location during a particular year.

## Sequence Naming Convention

Each temporal observation receives:

```text
sequence_id
```

using:

```text
{base_id}_{target_year}_L{sequence_length}
```

For example:

```text
INT_0042_2023_L3
```

represents a three-year sequence ending in 2023.

## Sequence Manifest Structure

For a three-year sequence, each row contains information such as:

```text
sequence_id
base_id
target_year
target
sequence_length

year_0
image_path_0
image_id_0

year_1
image_path_1
image_id_1

year_2
image_path_2
image_id_2
```

The index corresponds to chronological sequence position.

For example:

```text
year_0 = 2020
year_1 = 2021
year_2 = 2022
target_year = 2022
```

## Sequence Target

The classification target is taken from the record corresponding to the **final year** in the sequence.

For:

```text
[2020, 2021, 2022]
```

the target is the agave/non-agave label associated with:

```text
2022
```

Earlier labels are not directly supplied as targets to the network.

Their imagery provides temporal context for predicting the final-year condition.

## Optional Raster Validation

Sequence creation can include:

```text
--validate-rasters
```

When enabled, every image in every generated sequence is checked.

Each raster must contain:

```text
9 bands
64 × 64 pixels
```

Sequences containing invalid imagery are removed from the final sequence manifest.

## Sequence-Building Report

The sequence builder creates a JSON report beside the output CSV.

For example:

```text
agave_temporal_sequences_3yr.csv
agave_temporal_sequences_3yr.report.json
```

The report contains:

- Script version
- Number of source rows
- Sequence length
- Number of sequences created
- Number of unique `base_id` groups
- Target class counts
- Target-year counts
- Number of candidate windows skipped for missing years
- Number of `base_id` groups with label transitions
- Number of invalid sequences
- Examples of raster-validation failures

## Label Transitions

The report also counts locations where the binary target changes across years.

For example:

```text
INT_0042
2019: 0
2020: 0
2021: 1
2022: 1
```

contains a label transition.

These changes are preserved because they may represent actual agave expansion or removal.

## Spectral Normalization

Before training, the script calculates per-band normalization statistics from the **training sequences only**.

For each of the nine Sentinel-2 bands, it calculates:

```text
mean
standard deviation
```

across the images contained in a sample of training sequences.

Each image is then standardized using:

```text
x_normalized = (x - mean) / standard deviation
```

The same normalization values are used for:

- Training
- Validation
- Testing
- Prediction

## Normalization Sampling

By default, normalization is calculated using up to:

```text
500 sequences
```

controlled by:

```text
--normalization-sequences
```

Each selected sequence contributes all of its yearly image patches to the calculation.

For a three-year model using 500 sampled sequences, normalization may therefore inspect up to approximately:

```text
1,500 image patches
```

assuming all sampled sequences contain three images.

## Invalid Pixel Handling

Each Sentinel-2 image is loaded together with its raster validity mask.

Pixels outside the valid raster area are set temporarily to non-finite values.

Any remaining invalid values are replaced with the corresponding training-band mean before normalization.

This prevents missing or invalid pixels from propagating through the neural network.

## Temporal Dataset Shape

Each sequence is stacked into an array with the structure:

```text
time × bands × height × width
```

For the default three-year workflow:

```text
3 × 9 × 64 × 64
```

A training batch therefore has the general shape:

```text
batch × time × bands × height × width
```

## Temporal Data Augmentation

Training sequences receive random spatial augmentation.

Possible operations include:

- Horizontal flip
- Vertical flip
- 90° rotation
- 180° rotation
- 270° rotation

Critically, the **same transformation is applied to every year in the sequence**.

For example, if the 2020 patch is horizontally flipped, then the 2021 and 2022 patches are also horizontally flipped.

This preserves spatial correspondence across time.

## Model Architecture

The temporal model consists of two major components:

1. A shared spatial encoder
2. A temporal 1D CNN

The spatial encoder processes every yearly image individually.

The temporal CNN then processes the sequence of learned yearly embeddings.

## Spatial Encoder

Every image passes through the same spatial CNN.

The channel progression is:

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

Each convolution block contains:

```text
3 × 3 convolution
Batch Normalization
ReLU
3 × 3 convolution
Batch Normalization
ReLU
```

Max pooling is used between the earlier convolutional stages.

The final 256-dimensional representation is projected to a configurable embedding size.

The default is:

```text
embed_dim = 128
```

## Shared Spatial Weights

The same `SpatialEncoder` processes every year.

For a sequence:

```text
2020
2021
2022
```

the network does **not** train separate spatial CNNs for each year.

Instead:

```text
2020 ─┐
2021 ─┼─→ shared SpatialEncoder
2022 ─┘
```

This means the yearly embeddings exist in the same learned feature space.

## Temporal Representation

After each image has been encoded, the embeddings are reorganized into a temporal feature sequence.

Conceptually:

```text
2020 image → embedding
2021 image → embedding
2022 image → embedding
```

becomes:

```text
[embedding_2020, embedding_2021, embedding_2022]
```

The default embedding dimension is:

```text
128
```

## Temporal CNN

The embedding sequence is passed through a one-dimensional convolutional network.

The default temporal structure is:

```text
Conv1D
BatchNorm1D
ReLU
Conv1D
BatchNorm1D
ReLU
Adaptive Average Pooling
```

Both temporal convolution layers use:

```text
kernel size = 3
```

and the default temporal channel count is:

```text
128
```

## Classification Head

After temporal pooling, the representation passes through:

```text
Linear → 64
ReLU
Dropout
Linear → 1
```

The default dropout is:

```text
0.30
```

The final output is one binary classification logit for the sequence target year.

## Model Flow

For a default three-year sequence, the architecture can be summarized as:

```text
2020 Sentinel-2 patch ─→ Shared Spatial Encoder ─→ Embedding 2020
2021 Sentinel-2 patch ─→ Shared Spatial Encoder ─→ Embedding 2021
2022 Sentinel-2 patch ─→ Shared Spatial Encoder ─→ Embedding 2022

                         ↓ chronological stack

                    1D Temporal CNN

                         ↓

                 Sequence representation

                         ↓

                  Binary classifier

                         ↓

             P(agave in target year 2022)
```

## Default Grouped Split

Without future-year holdout settings, the script uses grouped splitting by:

```text
base_id
```

The default fractions are:

```text
Validation: 15%
Test:       15%
Training:   approximately 70%
```

All sequences belonging to one `base_id` remain together.

The split mode is recorded as:

```text
grouped_location
```

## Spatial Leakage Prevention

After splitting, the script explicitly compares the `base_id` groups assigned to:

- Training
- Validation
- Testing

If any location occurs in more than one partition, training stops with:

```text
base_id leakage
```

This ensures sequences created from overlapping years at the same location do not appear on opposite sides of model evaluation.

## Future-Year Evaluation

The model also supports a stricter future-year split.

Specify:

```text
--test-year
```

and optionally:

```text
--val-year
```

For example:

```bash
--test-year 2025 --val-year 2024
```

produces:

```text
Training: target years before 2024
Validation: target year 2024
Test: target year 2025
```

If no validation year is specified, it defaults to:

```text
test_year - 1
```

## Spatial Independence in Future-Year Mode

Future-year splitting also preserves spatial independence.

All `base_id` values appearing in either the validation year or test year are removed from the training set.

For example, if:

```text
INT_0042
```

appears in the 2025 test sequences, earlier sequences from `INT_0042` are not allowed into training.

This produces a stricter test of whether the model generalizes simultaneously to:

- Newer temporal conditions
- Previously unseen locations

The future split logic is defined directly from `target_year` and excludes validation/test locations from earlier training sequences.

## Split Outputs

The exact partitions are saved separately as:

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

These files are also reused by the `resume` and `finalize` commands.

They should therefore remain in the training output directory.

## Handling Class Imbalance

The workflow supports two approaches to class imbalance:

- Positive-class loss weighting
- Balanced sampling

By default, loss weighting is used.

## Positive-Class Weight

Without balanced sampling, the training loss calculates:

```text
negative training sequences
───────────────────────────
positive training sequences
```

and uses that ratio as:

```text
pos_weight
```

for:

```text
BCEWithLogitsLoss
```

This increases the influence of errors on the less common positive class.

## Balanced Sampler

Balanced sampling can be enabled with:

```text
--balanced-sampler
```

The script assigns each sequence an inverse-frequency sampling weight and uses:

```text
WeightedRandomSampler
```

with replacement.

When this option is enabled, positive-class loss weighting is disabled.

This avoids simultaneously applying both imbalance corrections in the default implementation.

## Loss Function

Training uses:

```text
BCEWithLogitsLoss
```

for binary sequence classification.

## Optimizer

The model uses:

```text
AdamW
```

with default:

```text
learning rate = 0.001
weight decay  = 0.0001
```

## Learning-Rate Scheduler

Training uses:

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

## Validation Metrics

The script calculates:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- Confusion matrix
- ROC AUC
- Average precision

ROC AUC and average precision are only calculated when both classes are represented.

## Primary Model-Selection Metric

The preferred validation score is:

```text
average precision
```

When average precision cannot be calculated because only one class appears, the workflow falls back to:

```text
F1
```

The learning-rate scheduler and best-checkpoint selection both use this score.

## Training Console Output

Each epoch prints a compact summary resembling:

```text
Epoch 028 train_loss=0.2807 val_loss=0.3732 val_f1=0.6154 val_AP=0.5934671477
```

This makes it possible to monitor:

- Training loss
- Validation loss
- Validation F1
- Validation average precision

during training.

## Best Model

When the validation score improves, the script saves:

```text
best_model.pt
```

The checkpoint contains:

- Model weights
- Band means
- Band standard deviations
- Sequence length
- Embedding dimension
- Temporal-channel count
- Dropout
- Script version

## Early Stopping

Initial training uses early stopping.

The default patience is:

```text
10 epochs
```

If the validation score does not improve beyond the best score for 10 epochs, training stops.

The maximum default training duration is:

```text
60 epochs
```



## Training History

After training, the epoch history is saved to:

```text
training_history.csv
```

Each row records:

- Epoch
- Training loss
- Validation loss
- Validation F1
- Validation average precision
- Learning rate

## Validation Threshold Selection

After the best model has been selected, it is evaluated again on the validation set.

The script tests classification thresholds between:

```text
0.05
```

and:

```text
0.95
```

in increments of:

```text
0.01
```

The threshold producing the highest validation F1 is selected.

This threshold is then applied to the test predictions.

## Test Evaluation

The best checkpoint is loaded and evaluated against:

```text
test_sequences.csv
```

Each test sequence receives:

```text
probability_agave
prediction
```

The binary prediction is created using the validation-selected threshold.

The results are saved to:

```text
test_predictions.csv
```

## Test Metrics

The final evaluation contains:

- Script version
- Split mode
- Device
- Sequence length
- Training sequence count
- Validation sequence count
- Test sequence count
- Number of train `base_id` groups
- Number of validation `base_id` groups
- Number of test `base_id` groups
- Selected threshold
- Overall test metrics
- Per-target-year metrics

These results are saved to:

```text
metrics.json
```

## Per-Target-Year Metrics

When multiple target years occur in the test set, the predictions are grouped by:

```text
target_year
```

and metrics are calculated independently for each one.

For example:

```text
2019
2020
2021
2022
2023
2024
2025
```

can each receive separate:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC AUC
- Average precision
- Confusion matrix

This makes year-dependent changes in model performance easier to identify.

## PyTorch Checkpoint Compatibility

The script includes a dedicated checkpoint loader.

PyTorch 2.6 changed the default behavior of:

```python
torch.load()
```

to use:

```text
weights_only=True
```

The temporal CNN checkpoint contains more than model weights, including:

- Normalization arrays
- Sequence length
- Model configuration

The loader therefore explicitly uses:

```python
weights_only=False
```

when supported.

Older PyTorch versions that do not accept this argument fall back to the previous loading method.

Only trusted checkpoints should be loaded this way.

## `resume`

The `resume` command continues an existing temporal CNN run.

Example:

```bash
python agave_temporal_cnn.py resume ^
    --output-dir "temporal_cnn_run_01" ^
    --total-epochs 100
```

The command requires the existing run directory to contain:

```text
best_model.pt
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

It also reuses:

```text
training_history.csv
```

when available.

## Resume Behavior

The workflow resumes from the existing:

```text
best_model.pt
```

rather than starting the network from random weights.

It also preserves:

- Existing train split
- Existing validation split
- Existing test split
- Existing normalization values
- Existing model architecture
- Existing training history

This prevents a resumed run from accidentally changing the geographic partition used in the original experiment.

## Epoch Numbering During Resume

The command inspects:

```text
training_history.csv
```

to determine the highest completed epoch.

If the existing run contains 28 completed epochs and the command specifies:

```text
--total-epochs 60
```

the resumed run processes:

```text
Epoch 29 through Epoch 60
```

rather than running 60 additional epochs.

If the requested total is not greater than the completed epoch count, the script stops with an error.

## Resume Optimizer Limitation

The older checkpoints used by this workflow do not store optimizer or learning-rate-scheduler state.

Therefore, when training resumes:

- Model weights are restored
- Normalization values are restored
- Geographic splits are restored
- Optimizer state is **not** restored
- Scheduler state is **not** restored

A new AdamW optimizer and scheduler are created.

The resumed model therefore continues from its learned network weights but not from the exact optimizer momentum/state of the earlier run.

## Resume Learning Rate

The default learning rate for resumed training is:

```text
0.0001
```

rather than the initial training default of:

```text
0.001
```

This provides a smaller learning rate when continuing an already-trained model.

It can be changed using:

```text
--learning-rate
```

## Resume Early Stopping

A key difference in resumed training is that early stopping is **disabled by default**.

The default is:

```text
--patience 0
```

which means all remaining requested epochs are allowed to run.

Early stopping can be re-enabled by specifying a positive patience value.

For example:

```bash
--patience 10
```

This behavior is useful when the goal is specifically to continue a previous run through a fixed total number of epochs.

## `last_model.pt`

During resumed training, the most recent epoch is written to:

```text
last_model.pt
```

This checkpoint records:

- Current model weights
- Normalization values
- Sequence length
- Embedding dimension
- Temporal channels
- Dropout
- Completed epoch
- Script version

Unlike `best_model.pt`, this represents the latest resumed epoch rather than necessarily the best validation epoch.

## Updating `best_model.pt`

During resumed training, validation performance is compared with the best score already present in the historical training record.

When the resumed model produces a better score:

```text
best_model.pt
```

is replaced with the new checkpoint.

Otherwise, the previous best checkpoint remains unchanged.

## Resumed Training History

Resumed epochs are appended to the existing:

```text
training_history.csv
```

and include:

```text
resumed = True
```

This distinguishes newly continued epochs from those produced during the original training session.

The file is updated after every resumed epoch.

## Evaluation After Resume

When resumed training ends, the script loads the best checkpoint again.

It then:

1. Generates validation probabilities.
2. Selects the F1-maximizing threshold.
3. Evaluates the test split.
4. Rewrites `test_predictions.csv`.
5. Recalculates per-target-year metrics.
6. Rewrites `metrics.json`.

The resulting metrics report includes:

```text
resumed_training = True
```

as well as:

- Requested total epochs
- Highest recorded epoch
- Sequence length
- Dataset sizes
- Selected threshold
- Overall test metrics
- Per-year metrics

## `finalize`

The `finalize` command completes evaluation for a training run that already contains a usable best model and split files.

Example:

```bash
python agave_temporal_cnn.py finalize ^
    --output-dir "temporal_cnn_run_01"
```

This command was designed for situations where training itself completed, but an older checkpoint-loading issue caused the script to fail before final evaluation outputs were written.

## Finalize Requirements

The output directory must contain:

```text
best_model.pt
validation_sequences.csv
test_sequences.csv
```

No retraining occurs.

The command simply restores the saved model and performs the validation/test evaluation stage.

## Finalize Workflow

The command:

1. Loads `best_model.pt`.
2. Restores the saved normalization statistics.
3. Loads the validation sequence split.
4. Loads the test sequence split.
5. Reconstructs the temporal CNN.
6. Loads the saved weights.
7. Calculates validation probabilities.
8. Selects the best validation F1 threshold.
9. Calculates test probabilities.
10. Produces binary test predictions.
11. Writes `test_predictions.csv`.
12. Calculates overall metrics.
13. Calculates per-target-year metrics.
14. Writes `metrics.json`.

The resulting metrics file contains:

```text
recovered_from_existing_checkpoint = True
```

to indicate that evaluation was recovered from an already completed run.

## When to Use `resume` vs. `finalize`

Use:

```text
resume
```

when additional training epochs are desired.

Use:

```text
finalize
```

when training is already complete and only the missing evaluation outputs need to be generated.

For example:

```text
Training stopped at epoch 28 and more training is wanted
→ resume
```

```text
Training stopped normally, but checkpoint loading failed before test metrics
→ finalize
```

## `predict`

The prediction command applies a trained temporal CNN to an existing sequence manifest.

Example:

```bash
python agave_temporal_cnn.py predict ^
    --model "temporal_cnn_run_01\best_model.pt" ^
    --sequence-manifest "agave_temporal_sequences_3yr.csv" ^
    --output-csv "temporal_predictions.csv"
```

The script:

1. Loads the model checkpoint.
2. Loads the sequence manifest.
3. Restores band means and standard deviations.
4. Builds the temporal dataset.
5. Reconstructs the model architecture.
6. Runs each sequence through the temporal CNN.
7. Converts the output logits to agave probabilities.
8. Writes the probabilities to the output CSV.

## Prediction Output

The original sequence-manifest columns are preserved.

A new column is added:

```text
probability_agave
```

For example:

```text
sequence_id       INT_0042_2023_L3
target_year       2023
probability_agave 0.8174
```

The `predict` command outputs probabilities only. It does not add a thresholded binary prediction column.

## Important Training Configuration Options

| Setting | Default | Purpose |
|---|---:|---|
| `--sequence-length` | `3` | Number of consecutive yearly patches per sequence |
| `--epochs` | `60` | Maximum initial training epochs |
| `--batch-size` | `8` | Number of temporal sequences per batch |
| `--workers` | `0` | DataLoader worker processes |
| `--learning-rate` | `0.001` | Initial AdamW learning rate |
| `--weight-decay` | `0.0001` | AdamW regularization |
| `--dropout` | `0.30` | Dropout in temporal classifier |
| `--embed-dim` | `128` | Spatial embedding size for each year |
| `--temporal-channels` | `128` | Feature width of temporal Conv1D layers |
| `--patience` | `10` | Initial training early-stopping patience |
| `--seed` | `42` | Random seed |
| `--val-fraction` | `0.15` | Validation fraction in grouped mode |
| `--test-fraction` | `0.15` | Test fraction in grouped mode |
| `--test-year` | None | Optional future target year for testing |
| `--val-year` | `test_year - 1` | Optional future validation year |
| `--normalization-sequences` | `500` | Maximum training sequences used for normalization |
| `--balanced-sampler` | Off | Enables inverse-frequency sequence sampling |
| `--device` | Automatic | Manually specifies PyTorch device |

## Important Resume Configuration Options

| Setting | Default | Purpose |
|---|---:|---|
| `--total-epochs` | `60` | Total desired epoch number, not additional epochs |
| `--batch-size` | `8` | Resumed training batch size |
| `--learning-rate` | `0.0001` | Learning rate used after resume |
| `--weight-decay` | `0.0001` | AdamW regularization |
| `--scheduler-patience` | `3` | Plateau scheduler patience |
| `--patience` | `0` | Resume early stopping; `0` disables it |
| `--balanced-sampler` | Off | Enables balanced sampling during resumed training |
| `--seed` | `42` | Random seed |
| `--device` | Automatic | Selected PyTorch device |

These command-line defaults are defined in the parser for the sequence, training, resume, finalize, and prediction stages.

## Running the Workflow

### 1. Activate the project environment

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

### 2. Confirm the single-year CNN manifest

Verify that the existing manifest contains:

```text
image_path
base_id
year
target
```

and preferably:

```text
image_name
image_id
```

### 3. Build the temporal sequences

```bash
python agave_temporal_cnn.py build-sequences ^
    --manifest "agave_cnn_manifest.csv" ^
    --output-csv "agave_temporal_sequences_3yr.csv" ^
    --validate-rasters
```

### 4. Review the sequence report

Inspect:

```text
agave_temporal_sequences_3yr.report.json
```

Pay particular attention to:

- `source_rows`
- `sequences_created`
- `unique_base_ids`
- `target_counts`
- `target_year_counts`
- `candidate_windows_skipped_for_missing_years`
- `base_ids_with_label_transitions`
- `invalid_sequences`

### 5. Train the grouped temporal baseline

```bash
python agave_temporal_cnn.py train ^
    --sequence-manifest "agave_temporal_sequences_3yr.csv" ^
    --output-dir "temporal_cnn_grouped"
```

### 6. Run a future-year experiment

For example:

```bash
python agave_temporal_cnn.py train ^
    --sequence-manifest "agave_temporal_sequences_3yr.csv" ^
    --output-dir "temporal_cnn_future_2025" ^
    --test-year 2025 ^
    --val-year 2024
```

### 7. Review training outputs

Inspect:

```text
best_model.pt
training_history.csv
train_sequences.csv
validation_sequences.csv
test_sequences.csv
normalization.json
test_predictions.csv
metrics.json
```

### 8. Continue training when needed

For example, if the run has completed 28 epochs and should continue through epoch 60:

```bash
python agave_temporal_cnn.py resume ^
    --output-dir "temporal_cnn_grouped" ^
    --total-epochs 60
```

Because resume patience defaults to `0`, the remaining requested epochs will run unless another error interrupts training.

### 9. Recover evaluation when necessary

```bash
python agave_temporal_cnn.py finalize ^
    --output-dir "temporal_cnn_grouped"
```

Use this only when the saved model and split files already exist and additional training is not needed.

### 10. Generate probabilities for sequence data

```bash
python agave_temporal_cnn.py predict ^
    --model "temporal_cnn_grouped\best_model.pt" ^
    --sequence-manifest "agave_temporal_sequences_3yr.csv" ^
    --output-csv "temporal_cnn_predictions.csv"
```

## Main Output Files

A standard training run can produce:

```text
best_model.pt
train_sequences.csv
validation_sequences.csv
test_sequences.csv
normalization.json
training_history.csv
test_predictions.csv
metrics.json
```

A resumed run can additionally produce:

```text
last_model.pt
```

The sequence-building stage produces:

```text
<sequence_manifest>.csv
<sequence_manifest>.report.json
```

## Recommendations

- Build the single-year CNN manifest before creating temporal sequences.
- Preserve the same `base_id` definition throughout both the single-year and temporal workflows.
- Check for duplicate `base_id`/year rows before sequence construction.
- Do not interpolate missing years simply to increase the number of sequences.
- Review the number of candidate windows lost to missing years.
- Use `--validate-rasters` when creating the final temporal manifest.
- Keep all years from the same location in one data partition.
- Preserve `train_sequences.csv`, `validation_sequences.csv`, and `test_sequences.csv` with every model run.
- Use future-year splitting when evaluating generalization to later observation periods.
- Remember that future-year mode is also spatially independent and may substantially reduce the available training set.
- Compute normalization only from training sequences.
- Apply identical augmentation to all years in the same sequence.
- Use average precision and F1 alongside accuracy when interpreting model performance.
- Review metrics by target year rather than relying only on aggregate test performance.
- Preserve `best_model.pt`, `normalization.json`, and `training_history.csv` together.
- Use `resume` rather than restarting the experiment when additional epochs are needed and the original geographic splits should be preserved.
- Remember that resumed training restores model weights but not the previous AdamW optimizer or scheduler state.
- Leave resumed `--patience` at `0` when the explicit goal is to run through all remaining requested epochs.
- Use `finalize` instead of retraining when the only problem was failure during post-training checkpoint loading or evaluation.
- Only load checkpoints created by this workflow or another trusted source.
- Treat this model as a temporal classification model, not a pixel-level segmentation model.
- Compare temporal CNN performance directly with the single-year CNN to determine whether multi-year imagery provides measurable improvement.

## Current Workflow Summary

In sequence, the script:

1. Loads the existing single-year CNN manifest.
2. Confirms the required image path, spatial ID, year, and target columns.
3. Removes rows without binary targets.
4. Converts years and targets to integer values.
5. Checks for duplicate `base_id`/year observations.
6. Groups source observations by spatial `base_id`.
7. Sorts the available observations chronologically.
8. Creates rolling consecutive-year windows for every possible target year.
9. Skips candidate windows containing missing years.
10. Assigns each sequence a stable ID containing its location, target year, and sequence length.
11. Stores each yearly image path in chronological sequence order.
12. Optionally validates every raster in each sequence.
13. Removes sequences containing invalid imagery when validation is enabled.
14. Saves the completed temporal sequence manifest.
15. Counts target classes and target years.
16. Counts locations whose labels change through time.
17. Writes the sequence-building report.
18. Loads the temporal manifest for training.
19. Creates a grouped spatial split by default.
20. Alternatively assigns specific future target years to validation and testing.
21. Removes overlapping validation/test locations from future-mode training.
22. Confirms `base_id` groups are disjoint across train, validation, and test.
23. Saves all three sequence partitions separately.
24. Samples training sequences for normalization.
25. Calculates nine-band training means and standard deviations across sequence imagery.
26. Saves the normalization statistics.
27. Loads each sequence as a time × bands × height × width tensor.
28. Replaces invalid imagery values with training-band means.
29. Normalizes every yearly image.
30. Applies identical random augmentation across all years in each training sequence.
31. Optionally builds an inverse-frequency balanced sampler.
32. Creates the shared spatial CNN encoder.
33. Encodes every yearly image into a learned feature vector.
34. Stacks the yearly feature vectors in chronological order.
35. Passes the sequence through the 1D temporal CNN.
36. Pools the temporal representation.
37. Produces one binary target-year classification logit.
38. Calculates class-weighted binary cross-entropy by default.
39. Optimizes the model using AdamW.
40. Monitors validation average precision when available.
41. Falls back to validation F1 when average precision is unavailable.
42. Reduces the learning rate when validation performance plateaus.
43. Saves the best temporal CNN checkpoint.
44. Stops initial training when early-stopping patience is reached.
45. Writes the training history.
46. Reloads the best checkpoint.
47. Uses validation probabilities to select an F1-maximizing decision threshold.
48. Generates held-out test probabilities.
49. Creates thresholded test predictions.
50. Saves individual test predictions.
51. Calculates overall test metrics.
52. Calculates metrics separately for every target year.
53. Writes the final evaluation report.
54. Loads PyTorch checkpoints with explicit compatibility handling for the `weights_only` behavior.
55. Allows an existing training run to resume from its best saved weights.
56. Reuses the original geographic splits during resumed training.
57. Reuses the original normalization values.
58. Reads prior training history to determine the completed epoch count.
59. Continues epoch numbering through the requested total epoch.
60. Reinitializes AdamW and the learning-rate scheduler during resume.
61. Saves the latest resumed model separately as `last_model.pt`.
62. Updates `best_model.pt` only when resumed validation performance improves.
63. Appends resumed epochs to the existing training history.
64. Disables resumed early stopping by default.
65. Re-evaluates the best checkpoint after resumed training.
66. Rewrites test predictions and metrics after resume.
67. Provides a `finalize` mode for recovering evaluation from an already completed checkpoint.
68. Reconstructs validation and test datasets without retraining during finalization.
69. Recalculates the validation-selected classification threshold.
70. Rewrites recovered test predictions and metrics.
71. Loads existing sequence manifests for prediction.
72. Restores the saved spatial and temporal model architecture.
73. Restores the saved spectral normalization values.
74. Generates agave probabilities for each temporal sequence.
75. Writes those probabilities back to a CSV containing the original sequence information.

This workflow provides a three-year spatiotemporal CNN pipeline that combines a shared multispectral spatial encoder with learned temporal features, allowing agave classification to incorporate consecutive Sentinel-2 observations while maintaining strict spatial separation between training and evaluation data.