# Agave CNN and U-Net Model Workflow

## Short Summary

This document explains how to run and interpret the Agave deep-learning workflow beginning with:

```text
agave_cnn_64.py
```

and continuing through:

```text
agave_temporal_cnn.py
agave_cnn_paired_test.py
agave_unet_weak.py
agave_unet_partial.py
```

The scripts are connected, but they do not all answer the same research question.

The main experiments are:

1. **Single-Year CNN** — classifies one Sentinel-2 patch as agave or non-agave.
2. **Temporal CNN** — uses several consecutive years to classify the target year.
3. **Paired Single-Year CNN** — repeats the temporal experiment using only the target-year image while preserving the exact same samples and geographic splits.
4. **Weak U-Net** — uses patch-level labels to create exploratory pixel-level agave probability maps.
5. **Partial Pseudo-Label U-Net** — converts only high-confidence weak-U-Net pixels into segmentation supervision and trains a conventional U-Net on those pixels.

The classification models should primarily be used to compare agave/non-agave prediction at the **patch level**.

The U-Net workflows should be treated separately because they produce **pixel-level maps**, and neither the weak nor partial U-Net uses independently verified human field masks.

---

# 1. Scripts Covered

The scripts covered by this guide are:

```text
agave_cnn_64.py
agave_temporal_cnn.py
agave_cnn_paired_test.py
agave_unet_weak.py
agave_unet_partial.py
```

---

# 2. Recommended Workflow Order

Run the scripts in this order:

1. **`agave_cnn_64.py` — Build the single-year manifest.**  
   Match the 64 × 64 Sentinel-2 patches with their year-specific agave labels and validate the imagery.

2. **`agave_cnn_64.py` — Train the single-year CNN baseline.**  
   This establishes the main one-year patch-classification baseline.

3. **`agave_temporal_cnn.py` — Build temporal sequences.**  
   Convert the single-year manifest into consecutive multi-year sequences, using three years by default.

4. **`agave_temporal_cnn.py` — Train the temporal CNN.**  
   Test whether prior-year imagery improves target-year agave classification.

5. **`agave_cnn_paired_test.py` — Run the paired single-year ablation.**  
   Train on only the target-year image while using the exact samples and geographic splits from the temporal model.

6. **`agave_unet_weak.py` — Train the weakly supervised U-Net.**  
   Use patch-level labels to train a model that produces exploratory pixel-level agave probability maps.

7. **`agave_unet_weak.py` — Generate weak probability rasters.**  
   These probability maps become the input to the partial pseudo-label workflow.

8. **`agave_unet_partial.py` — Build partial pseudo-labels.**  
   Convert only high-confidence weak-U-Net predictions into pixel labels while leaving uncertain pixels ignored.

9. **`agave_unet_partial.py` — Train the partial U-Net.**  
   Train a conventional segmentation model using only the confident pseudo-labeled pixels.

10. **`agave_unet_partial.py` — Generate final probability maps and masks.**

11. **Compare the experiments.**  
   Keep patch-classification results separate from pseudo-segmentation results.

---

# 3. Relationship Between the Scripts

| Script | Main Input | Main Purpose | Feeds Into |
|---|---|---|---|
| `agave_cnn_64.py` | Sentinel-2 patches + year-specific labels | Single-year patch classification baseline | `agave_temporal_cnn.py`, `agave_unet_weak.py` |
| `agave_temporal_cnn.py` | Single-year CNN manifest | Multi-year target-year classification | `agave_cnn_paired_test.py` |
| `agave_cnn_paired_test.py` | Completed temporal CNN split files | Controlled test of temporal-context value | Final temporal comparison |
| `agave_unet_weak.py` | Single-year patch manifest | Weak pixel localization from patch labels | `agave_unet_partial.py` |
| `agave_unet_partial.py` | Weak U-Net probability rasters + preserved splits | High-confidence pseudo-label refinement and segmentation | Final spatial probability maps and masks |

The single-year manifest produced for the CNN becomes a common input for the later workflows.

The temporal CNN depends on the single-year manifest.

The paired CNN depends on a completed temporal CNN run.

The partial U-Net depends on probability rasters produced by the weak U-Net.

---

# 4. Shared Sentinel-2 Image Requirements

All five scripts use standardized Sentinel-2 imagery containing:

```text
9 spectral bands
64 × 64 pixels
```

The expected bands are:

```text
B2
B3
B4
B5
B6
B7
B8
B11
B12
```

The single-year CNN is designed around 9-band, 64 × 64 Sentinel-2 patches and year-specific labels.

The temporal CNN uses the same image format but combines several years from one location into a chronological sequence.

---

# 5. Recommended Environment

Activate the Agave project environment before running the scripts.

For example:

```bash
conda activate agave-oaxaca
```

The main dependencies include:

```text
torch
numpy
pandas
rasterio
scikit-learn
tqdm
```

Manifest-building operations may also require:

```text
geopandas
pyogrio
shapely
```

A typical installation command is:

```bash
pip install torch numpy pandas rasterio scikit-learn tqdm geopandas pyogrio shapely
```

---

# 6. Recommended Output Organization

Keep every experiment in its own output directory.

For example:

```text
outputs/
    cnn_single_year/
    temporal_cnn_3yr/
    temporal_cnn_future2025/
    paired_single_year/
    unet_weak/
    unet_partial/
```

Do not overwrite one model run with another.

Each training run should preserve:

```text
best_model.pt
normalization.json
training_history.csv
metrics.json
```

along with the exact split files used by that experiment.

---

# PART I — `agave_cnn_64.py`

# 7. Purpose

`agave_cnn_64.py` provides the main single-year deep-learning baseline.

Its input is one:

```text
9-band
64 × 64
Sentinel-2 patch
```

and its output is one:

```text
P(agave)
```

for the entire patch.

It does not perform segmentation.

Its primary research question is:

> How accurately can agave be classified using spectral-spatial information from one Sentinel-2 observation?

---

# 8. Build the Single-Year Manifest

Run:

```bash
python agave_cnn_64.py build-manifest ^
    --image-dir "data\sentinel2_64" ^
    --label-file "data\processed\unet_labels\training_5000_polygons.gpkg" ^
    --output-csv "data\agave_cnn_manifest.csv" ^
    --validate-rasters
```

The resulting manifest contains fields such as:

```text
image_path
image_name
image_id
base_id
year
target
```

The script matches imagery to year-specific labels, standardizes labels, and keeps `base_id` as the spatial grouping variable.

---

# 9. Review the Manifest Report

The build command creates:

```text
agave_cnn_manifest.report.json
```

Review:

```text
images_found
usable_labeled_images
class_counts
year_counts
missing_label_count
excluded_unknown_labels
duplicate_id_conflicts
invalid_image_count
unique_base_ids
base_ids_with_multiple_years
base_ids_with_label_transitions
```

Unexpected missing labels, duplicate conflicts, or invalid rasters should be resolved before final training.

---

# 10. Train the Single-Year CNN

Run:

```bash
python agave_cnn_64.py train ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "outputs\cnn_single_year" ^
    --epochs 60 ^
    --batch-size 32
```

The default split keeps all observations from the same:

```text
base_id
```

together.

This prevents imagery from the same physical location from appearing in both training and evaluation.

---

# 11. Important Outputs

Preserve:

```text
best_model.pt
manifest_with_splits.csv
normalization.json
training_history.csv
test_predictions.csv
metrics.json
```

Important metrics include:

```text
balanced_accuracy
precision
recall
f1
roc_auc
average_precision
```

Accuracy alone should not be used to judge performance when the target classes are imbalanced.

---

# 12. Role in the Overall Workflow

`agave_cnn_64.py` serves two important purposes:

1. It establishes the primary one-year CNN baseline.
2. It creates the standardized manifest used by the temporal and weak-U-Net workflows.

---

# PART II — `agave_temporal_cnn.py`

# 13. Purpose

The temporal CNN tests whether previous years improve target-year classification.

The default sequence is:

```text
t-2
t-1
t
```

and the target is:

```text
label at t
```

For example:

```text
2020 image
2021 image
2022 image

Target:
2022 agave label
```

The script constructs rolling consecutive-year sequences from the single-year manifest.

---

# 14. Build Temporal Sequences

Run:

```bash
python agave_temporal_cnn.py build-sequences ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-csv "data\agave_temporal_sequences_3yr.csv" ^
    --sequence-length 3 ^
    --validate-rasters
```

The default sequence length is:

```text
3 years
```

The years must be consecutive.

A location with:

```text
2020
2021
2022
```

can create a sequence.

A location with:

```text
2020
2022
```

cannot.

Skipped windows are counted as:

```text
candidate_windows_skipped_for_missing_years
```

---

# 15. Review the Sequence Report

Inspect:

```text
agave_temporal_sequences_3yr.report.json
```

Important fields include:

```text
source_rows
sequence_length
sequences_created
unique_base_ids
target_counts
target_year_counts
candidate_windows_skipped_for_missing_years
base_ids_with_label_transitions
invalid_sequences
```

The number of temporal sequences will normally be lower than the number of single-year observations.

---

# 16. Train the Temporal CNN

Run:

```bash
python agave_temporal_cnn.py train ^
    --sequence-manifest "data\agave_temporal_sequences_3yr.csv" ^
    --output-dir "outputs\temporal_cnn_3yr" ^
    --epochs 60 ^
    --batch-size 8
```

The model contains:

- A shared spatial encoder
- One learned image embedding per year
- A 1D temporal CNN
- A final binary target-year classifier

The same spatial encoder processes each yearly image before the embeddings are passed through the temporal network.

---

# 17. Default Grouped Split

The normal temporal experiment uses:

```text
grouped_location
```

with approximately:

```text
70% training
15% validation
15% test
```

All sequences from one `base_id` remain together.

This should be the primary temporal model used for comparison with the normal and paired single-year CNNs.

---

# 18. Future-Year Experiment

A stricter experiment can use:

```bash
python agave_temporal_cnn.py train ^
    --sequence-manifest "data\agave_temporal_sequences_3yr.csv" ^
    --output-dir "outputs\temporal_cnn_future2025" ^
    --test-year 2025 ^
    --val-year 2024
```

This creates:

```text
Training:
target years before 2024

Validation:
target year 2024

Test:
target year 2025
```

Locations represented in validation or test are removed from training, providing both spatial and temporal independence.

Treat this as a separate experiment from the grouped temporal baseline.

---

# 19. Important Outputs

Preserve:

```text
best_model.pt
normalization.json
training_history.csv

train_sequences.csv
validation_sequences.csv
test_sequences.csv

test_predictions.csv
metrics.json
```

The three sequence split files are required by `agave_cnn_paired_test.py`.

---

# 20. Resume Training

To continue a temporal run:

```bash
python agave_temporal_cnn.py resume ^
    --output-dir "outputs\temporal_cnn_3yr" ^
    --total-epochs 100
```

`--total-epochs` means the desired final epoch number.

If the run already contains 28 epochs and:

```text
--total-epochs 60
```

is supplied, the script runs:

```text
epochs 29–60
```

The saved model weights and original geographic splits are reused.

The optimizer and scheduler are reinitialized because older checkpoints did not save their state.

---

# 21. Resume Early Stopping

The default resume behavior is:

```text
--patience 0
```

which disables renewed early stopping.

To enable early stopping during resume:

```bash
--patience 10
```

---

# 22. Finalize a Completed Temporal Run

If training completed but the final evaluation failed:

```bash
python agave_temporal_cnn.py finalize ^
    --output-dir "outputs\temporal_cnn_3yr"
```

This does not retrain the model.

It uses the saved checkpoint and split files to recreate the test predictions and final metrics.

---

# PART III — `agave_cnn_paired_test.py`

# 23. Purpose

This script provides the strongest controlled test of whether temporal context improves classification.

The temporal model receives:

```text
t-2
t-1
t
```

while the paired model receives:

```text
t
```

but both use:

```text
the same target samples
the same target years
the same labels
the same base_id split assignments
```



---

# 24. Run the Paired Single-Year CNN

The temporal directory must already contain:

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

and preferably:

```text
metrics.json
```

Run:

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "outputs\temporal_cnn_3yr" ^
    --output-dir "outputs\paired_single_year" ^
    --epochs 60 ^
    --validate-rasters
```

For the standard three-year sequence, the paired model uses only the final:

```text
image_path_2
```

image.

---

# 25. Why the Paired Experiment Matters

An independently trained single-year CNN may not contain exactly the same observations as the temporal CNN.

Therefore:

```text
Single-Year CNN
vs.
Temporal CNN
```

is useful but not perfectly controlled.

The paired experiment fixes this by using exactly the same temporal targets and geographic partitions.

The cleanest temporal comparison is therefore:

```text
Paired Single-Year CNN
vs.
Temporal CNN
```

---

# 26. Paired Comparison Output

When the temporal directory contains `metrics.json`, the paired script creates:

```text
paired_comparison.json
```

It compares:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
average_precision
```

and reports:

```text
paired_single_year
temporal_3yr
delta_single_minus_temporal
```

Interpret:

```text
negative delta
= temporal CNN performed better

positive delta
= paired single-year CNN performed better
```



---

# 27. Main Temporal Comparison to Report

When answering:

> Does multi-year context improve agave classification?

place the greatest emphasis on:

```text
Temporal CNN
vs.
Paired Single-Year CNN
```

The original `agave_cnn_64.py` result remains an important general baseline.

---

# PART IV — `agave_unet_weak.py`

# 28. Purpose

`agave_unet_weak.py` begins the pixel-localization branch.

Unlike the CNN classifiers, it produces a:

```text
64 × 64
```

pixel probability map.

However, it is trained without true pixel-level segmentation masks.

Its available supervision is only:

```text
agave patch
or
non-agave patch
```

The script therefore uses weak multiple-instance-learning-style supervision to connect pixel predictions to the patch label.

---

# 29. Train the Weak U-Net

Run:

```bash
python agave_unet_weak.py train ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "outputs\unet_weak" ^
    --epochs 60 ^
    --batch-size 16 ^
    --validate-rasters
```

Default weak-supervision parameters include:

```text
topk_fraction = 0.10
tv_weight = 0.02
sparsity_weight = 0.02
```

---

# 30. Weak U-Net Training Logic

The weak U-Net combines:

```text
patch-level top-k classification loss
negative-patch pixel penalty
spatial smoothness penalty
positive-patch sparsity penalty
```

The top-k pooling operation selects only the highest-scoring portion of the pixel logits when constructing the patch-level prediction.

This allows a positive image to contain only a localized agave region rather than requiring the entire patch to be positive.

---

# 31. Weak U-Net Metrics

The main quantitative evaluation during training remains **patch-level classification**.

Metrics include:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
average_precision
```

These metrics should not be described as segmentation accuracy.

No true pixel masks are used.

---

# 32. Generate Weak Probability Maps

Run:

```bash
python agave_unet_weak.py predict ^
    --model "outputs\unet_weak\best_model.pt" ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "data\weak_unet_predictions"
```

This produces files such as:

```text
*_weak_unet_probability.tif
*_weak_unet_pseudomask.tif
pseudo_mask_summary.csv
```

The most important input for the next stage is:

```text
*_weak_unet_probability.tif
```

---

# 33. Interpreting Weak U-Net Outputs

The pixel outputs are:

```text
pseudo-segmentation maps
```

not verified field boundaries.

The script explicitly records that true pixel masks were not used.

Therefore:

```text
high patch-level F1
```

does not establish:

```text
high field-boundary accuracy
```

The weak maps are best treated as:

- Exploratory localization
- Candidate agave regions
- Inputs for pseudo-label generation
- Intermediate supervision for `agave_unet_partial.py`

---

# PART V — `agave_unet_partial.py`

# 34. Purpose

`agave_unet_partial.py` converts the weak U-Net's probability maps into conservative pixel-level pseudo-labels.

The mask values are:

```text
0   = confident non-agave
1   = confident agave
255 = unknown / ignore
```

The workflow then trains a conventional U-Net using only the pixels labeled `0` or `1`.

---

# 35. Build Partial Pseudo-Labels

Run:

```bash
python agave_unet_partial.py build-pseudolabels ^
    --manifest "data\weak_unet_manifest_with_splits.csv" ^
    --probability-dir "data\weak_unet_predictions" ^
    --output-dir "data\partial_pseudolabels"
```

The default thresholds are:

```text
positive_threshold = 0.80
negative_threshold = 0.10
```

---

# 36. Positive Patch Rules

For:

```text
target = 1
```

the weak U-Net probability is converted as:

```text
probability >= 0.80 → 1
probability <= 0.10 → 0
everything else      → 255
```

This keeps only high-confidence positive and negative pixels.

---

# 37. Negative Patch Rules

For:

```text
target = 0
```

every valid pixel becomes:

```text
0
```

This is treated as strong negative supervision.

---

# 38. Partial Pseudo-Label Outputs

The command creates:

```text
partial_unet_manifest.csv
partial_unet_manifest.report.json
*_partial_mask.tif
```

The manifest includes useful quality-control fields:

```text
confident_pixels
confident_positive_pixels
confident_negative_pixels
ignored_pixels
fraction_confident
fraction_positive
```

These should be inspected before training.

---

# 39. Pseudo-Label Threshold Tradeoff

More conservative thresholds such as:

```text
positive = 0.90
negative = 0.05
```

create:

```text
fewer labeled pixels
higher required confidence
```

More relaxed thresholds such as:

```text
positive = 0.70
negative = 0.20
```

create:

```text
more labeled pixels
greater risk of pseudo-label errors
```

Treat each threshold configuration as a separate experiment.

---

# 40. Train the Partial U-Net

Run:

```bash
python agave_unet_partial.py train ^
    --manifest "data\partial_pseudolabels\partial_unet_manifest.csv" ^
    --output-dir "outputs\unet_partial" ^
    --epochs 60 ^
    --batch-size 16
```

The partial manifest must preserve the original:

```text
train
validation
test
```

split.

The script does not create a new split and checks `base_id` independence before training.

---

# 41. Partial U-Net Loss

The model trains with:

```text
masked binary cross-entropy
+
masked Dice loss
```

Only pixels where:

```text
mask != 255
```

contribute to the loss.

Ignored pixels contribute nothing.

---

# 42. Partial U-Net Evaluation

Evaluation also includes only confident pseudo-labeled pixels.

Metrics may include:

```text
accuracy
balanced_accuracy
precision
recall
f1
roc_auc
average_precision
```

but they are calculated against machine-generated pseudo-labels.

The script explicitly records:

```text
Evaluation is against pseudo-label pixels, not human field masks.
```



Do not describe these metrics as human-ground-truth segmentation accuracy.

---

# 43. Generate Partial U-Net Predictions

Run:

```bash
python agave_unet_partial.py predict ^
    --model "outputs\unet_partial\best_model.pt" ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "outputs\unet_partial_predictions"
```

The outputs include:

```text
*_partial_unet_probability.tif
*_partial_unet_mask.tif
prediction_summary.csv
```

The default mask threshold is:

```text
0.5
```

The prediction summary records per-image mean probability and fraction of pixels predicted positive.

---

# 44. Required vs. Optional Experiments

## Core Classification Workflow

For classification and temporal analysis:

```text
agave_cnn_64.py
agave_temporal_cnn.py
agave_cnn_paired_test.py
```

These provide:

- Single-year baseline
- Multi-year model
- Controlled temporal ablation

## Spatial Localization Workflow

For exploratory pixel localization:

```text
agave_unet_weak.py
agave_unet_partial.py
```

The weak U-Net must run before the partial U-Net.

---

# 45. Which Models Can Be Compared Directly?

The following models produce patch-level classification results:

```text
agave_cnn_64.py
agave_temporal_cnn.py
agave_cnn_paired_test.py
agave_unet_weak.py
```

Their patch-level metrics can be compared cautiously when the samples and splits are compatible.

The strongest direct comparison is:

```text
agave_temporal_cnn.py
vs.
agave_cnn_paired_test.py
```

because they use the same target samples and geographic split.

---

# 46. Which Metrics Should Not Be Compared Directly?

Do not directly compare:

```text
CNN patch F1
```

with:

```text
Partial U-Net confident-pixel F1
```

as if they measure the same task.

They do not.

## CNN Evaluation Unit

```text
one image patch
```

Question:

> Is this patch agave?

## Partial U-Net Evaluation Unit

```text
one confident pseudo-labeled pixel
```

Question:

> Does this pixel match the pseudo-label generated from the weak model?

These metrics should appear in separate result sections.

---

# 47. Recommended Classification Results Table

| Model | Input | Evaluation Split | F1 | Balanced Accuracy | ROC AUC | Average Precision |
|---|---|---|---:|---:|---:|---:|
| `agave_cnn_64.py` | `t` | Grouped location |  |  |  |  |
| `agave_temporal_cnn.py` | `t-2, t-1, t` | Grouped location |  |  |  |  |
| `agave_cnn_paired_test.py` | `t` | Exact temporal split |  |  |  |  |

The temporal and paired models should be emphasized when evaluating the benefit of temporal context.

---

# 48. Recommended Spatial Results Table

| Model | Supervision | Evaluation Target | F1 | Average Precision | Interpretation |
|---|---|---|---:|---:|---|
| `agave_unet_weak.py` | Patch-level labels | Patch classification |  |  | Pixel maps are exploratory |
| `agave_unet_partial.py` | High-confidence pseudo-labels | Confident pseudo-label pixels |  |  | Not human ground truth |
| Future fully supervised U-Net | Human masks | Human-labeled pixels |  |  | True segmentation evaluation |

---

# 49. Recommended Experiment Naming

Use descriptive output directories.

Examples:

```text
cnn_single_year_grouped/
temporal_cnn_3yr_grouped/
temporal_cnn_3yr_future2025/
paired_single_year_temporal3yr/
unet_weak_topk010/
unet_partial_p080_n010/
unet_partial_p090_n005/
```

The directory name should make the major experimental configuration obvious.

---

# 50. Files That Should Always Be Preserved

For every model run, preserve:

```text
best_model.pt
normalization.json
training_history.csv
metrics.json
```

Also preserve the exact split files.

## `agave_cnn_64.py`

```text
manifest_with_splits.csv
```

## `agave_temporal_cnn.py`

```text
train_sequences.csv
validation_sequences.csv
test_sequences.csv
```

## `agave_cnn_paired_test.py`

```text
train_paired.csv
validation_paired.csv
test_paired.csv
```

## `agave_unet_weak.py`

```text
train_manifest.csv
validation_manifest.csv
test_manifest.csv
```

## `agave_unet_partial.py`

```text
partial_unet_manifest.csv
partial_unet_manifest.report.json

train_manifest.csv
validation_manifest.csv
test_manifest.csv
```

These files are necessary for experiment reproduction and auditing.

---

# 51. Spatial Leakage Rule

Across all experiments, the same physical:

```text
base_id
```

should not appear in both training and evaluation data.

For example:

```text
INT_0042_2020
INT_0042_2021
INT_0042_2022
```

all belong to the same underlying location:

```text
INT_0042
```

These records should remain together when spatial independence is required.

---

# 52. Temporal Label Changes Are Not Automatically Errors

A location may legitimately contain:

```text
2019 → not_agave
2020 → not_agave
2021 → agave
2022 → agave
```

This should not automatically be treated as contradictory labeling.

The temporal workflow intentionally preserves these changes because they may represent the land-use transition the project is attempting to detect.

---

# 53. Recommended Minimum Experiment Set

A strong minimum set is:

## Experiment A — Single-Year CNN

Script:

```text
agave_cnn_64.py
```

Input:

```text
t
```

Purpose:

```text
single-year classification baseline
```

---

## Experiment B — Temporal CNN

Script:

```text
agave_temporal_cnn.py
```

Input:

```text
t-2, t-1, t
```

Purpose:

```text
multi-year target-year classification
```

---

## Experiment C — Paired Single-Year Ablation

Script:

```text
agave_cnn_paired_test.py
```

Input:

```text
t
```

Samples and split:

```text
exactly matched to temporal CNN
```

Purpose:

```text
isolate the value of temporal context
```

---

## Experiment D — Weak U-Net

Script:

```text
agave_unet_weak.py
```

Supervision:

```text
patch labels
```

Purpose:

```text
exploratory weak pixel localization
```

---

## Experiment E — Partial U-Net

Script:

```text
agave_unet_partial.py
```

Supervision:

```text
high-confidence weak pseudo-label pixels
```

Purpose:

```text
pseudo-label segmentation refinement
```

---

# 54. Optional Additional Experiments

After establishing the primary models, additional experiments may test:

```text
balanced sampling
future-year holdout
different temporal sequence lengths
different temporal embedding sizes
different CNN dropout
different weak-U-Net top-k fractions
different pseudo-label confidence thresholds
different U-Net base channel sizes
different prediction thresholds
```

Whenever possible, change one major factor at a time.

---

# 55. Recommended Temporal Experiment Progression

A useful order is:

1. `agave_cnn_64.py` grouped baseline
2. `agave_temporal_cnn.py` three-year grouped model
3. `agave_cnn_paired_test.py` using the temporal model's exact split
4. Optional future-year temporal experiment

This separates two questions:

> Does temporal context help?

from:

> Does the temporal model generalize to later years and unseen locations?

---

# 56. Recommended Partial Pseudo-Label Experiments

Begin with:

```text
positive threshold = 0.80
negative threshold = 0.10
```

Potential follow-up experiments include:

```text
0.90 / 0.05
```

and:

```text
0.70 / 0.20
```

For each configuration, record:

```text
masks created
missing probability rasters
mean fraction confident
mean fraction positive
confident positive pixel count
ignored pixel count
validation average precision
test confident-pixel F1
```

Do not select pseudo-label thresholds solely because they maximize pseudo-label test performance.

---

# 57. Quality-Control Checklist Before Training

Before each final training run, confirm:

- Sentinel-2 files exist.
- Images contain 9 bands.
- Images are 64 × 64.
- Labels are binary.
- Years are correct.
- `base_id` values are correct.
- No unintended duplicate location-year records exist.
- Train/validation/test groups do not leak across `base_id`.
- Both classes are represented where the selected metrics require them.
- The output directory is dedicated to the current experiment.
- The correct normalization and split files are preserved.
- For U-Net workflows, the interpretation of the supervision source is clearly recorded.

---

# 58. What to Watch During Training

Review:

```text
training loss
validation loss
validation F1
validation average precision
validation ROC AUC
learning rate
```

Do not assume the final epoch is the best epoch.

Use:

```text
best_model.pt
```

for final evaluation unless there is a specific reason to inspect another checkpoint.

---

# 59. Threshold Selection

When a script optimizes a classification threshold, the correct sequence is:

```text
validation probabilities
→ choose threshold
→ evaluate test predictions
```

Do not use the held-out test data to select the threshold.

The paired single-year script explicitly optimizes its classification threshold on validation F1 before applying it to the test data.

---

# 60. Patch Thresholds and Pixel Thresholds Are Different

The U-Net branch may involve two different concepts.

## Patch Threshold

Used to decide:

```text
agave patch
vs.
non-agave patch
```

## Pixel Threshold

Used to decide:

```text
agave pixel
vs.
non-agave pixel
```

A patch-classification threshold should not automatically be reused as the best segmentation threshold.

Without independent human masks, there is no true pixel ground truth for optimizing a final segmentation threshold.

---

# 61. How to Interpret Each Script

## `agave_cnn_64.py`

Question:

> Can current-year Sentinel-2 imagery identify agave patches?

---

## `agave_temporal_cnn.py`

Question:

> Does adding prior-year imagery improve target-year agave classification?

---

## `agave_cnn_paired_test.py`

Question:

> Does temporal context improve prediction when target samples and geographic splits are held constant?

---

## `agave_unet_weak.py`

Question:

> Can patch-level labels produce useful spatial localization without pixel masks?

---

## `agave_unet_partial.py`

Question:

> Can high-confidence weak-model predictions provide useful partial supervision for a conventional segmentation model?

---

# 62. Important U-Net Limitation

The weak and partial U-Net workflows do not independently establish real-world segmentation accuracy.

The partial U-Net explicitly evaluates against:

```text
pseudo-label pixels
```

rather than human field masks.

Appropriate wording is:

```text
The partial U-Net achieved an F1 of X against held-out high-confidence pseudo-label pixels.
```

Avoid wording such as:

```text
The U-Net delineated real agave fields with an F1 of X.
```

unless independent human segmentation masks are eventually used for evaluation.

---

# 63. Recommended Reporting Order

For the final analysis, report models in this order:

1. **`agave_cnn_64.py`** — single-year baseline
2. **`agave_temporal_cnn.py`** — multi-year model
3. **`agave_cnn_paired_test.py`** — controlled temporal ablation
4. **Future-year temporal experiment**, if performed
5. **`agave_unet_weak.py`** — weak spatial localization
6. **`agave_unet_partial.py`** — pseudo-label refinement
7. **Future fully supervised U-Net**, if human masks become available

This keeps the classification results separate from the exploratory segmentation branch.

---

# 64. Full Example Workflow

## Step 1 — Build the Single-Year Manifest

```bash
python agave_cnn_64.py build-manifest ^
    --image-dir "data\sentinel2_64" ^
    --label-file "data\processed\unet_labels\training_5000_polygons.gpkg" ^
    --output-csv "data\agave_cnn_manifest.csv" ^
    --validate-rasters
```

## Step 2 — Train the Single-Year CNN

```bash
python agave_cnn_64.py train ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "outputs\cnn_single_year"
```

## Step 3 — Build Temporal Sequences

```bash
python agave_temporal_cnn.py build-sequences ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-csv "data\agave_temporal_sequences_3yr.csv" ^
    --sequence-length 3 ^
    --validate-rasters
```

## Step 4 — Train the Temporal CNN

```bash
python agave_temporal_cnn.py train ^
    --sequence-manifest "data\agave_temporal_sequences_3yr.csv" ^
    --output-dir "outputs\temporal_cnn_3yr"
```

## Step 5 — Run the Paired Single-Year Ablation

```bash
python agave_cnn_paired_test.py train ^
    --temporal-run-dir "outputs\temporal_cnn_3yr" ^
    --output-dir "outputs\paired_single_year" ^
    --validate-rasters
```

## Step 6 — Train the Weak U-Net

```bash
python agave_unet_weak.py train ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "outputs\unet_weak" ^
    --validate-rasters
```

## Step 7 — Generate Weak Probability Maps

```bash
python agave_unet_weak.py predict ^
    --model "outputs\unet_weak\best_model.pt" ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "data\weak_unet_predictions"
```

## Step 8 — Build Partial Pseudo-Labels

```bash
python agave_unet_partial.py build-pseudolabels ^
    --manifest "data\weak_unet_manifest_with_splits.csv" ^
    --probability-dir "data\weak_unet_predictions" ^
    --output-dir "data\partial_pseudolabels"
```

## Step 9 — Train the Partial U-Net

```bash
python agave_unet_partial.py train ^
    --manifest "data\partial_pseudolabels\partial_unet_manifest.csv" ^
    --output-dir "outputs\unet_partial"
```

## Step 10 — Generate Partial U-Net Predictions

```bash
python agave_unet_partial.py predict ^
    --model "outputs\unet_partial\best_model.pt" ^
    --manifest "data\agave_cnn_manifest.csv" ^
    --output-dir "outputs\unet_partial_predictions"
```

Paths should be adjusted to match the actual project structure.

---

# 65. Final Workflow Summary

The five scripts form a progression from classification to temporal analysis and then to exploratory spatial localization.

### `agave_cnn_64.py`

Creates the standardized single-year dataset and establishes the primary patch-classification baseline.

### `agave_temporal_cnn.py`

Uses consecutive yearly observations to determine whether temporal context improves target-year classification.

### `agave_cnn_paired_test.py`

Removes the earlier temporal imagery while preserving the exact temporal samples and geographic split, providing the cleanest direct test of whether temporal context contributes predictive value.

### `agave_unet_weak.py`

Uses patch-level labels to train a weakly supervised U-Net that produces exploratory pixel probability maps without claiming access to true segmentation masks.

### `agave_unet_partial.py`

Uses only high-confidence regions of those weak probability maps as partial segmentation labels and trains a conventional U-Net while ignoring uncertain pixels.

Across all five scripts, the most important workflow principles are:

- Preserve `base_id` geographic independence.
- Preserve exact train/validation/test split files.
- Keep temporal label transitions rather than automatically treating them as errors.
- Normalize from training data only.
- Select classification thresholds from validation data rather than test data.
- Keep every experiment in a separate output directory.
- Preserve checkpoints, metrics, normalization statistics, and training histories.
- Compare only models that evaluate the same task.
- Use `agave_cnn_paired_test.py` as the strongest controlled test of temporal value.
- Treat `agave_unet_weak.py` outputs as weak pseudo-localization.
- Treat `agave_unet_partial.py` metrics as performance against pseudo-label pixels rather than human segmentation ground truth.
- Use independent human field masks for final segmentation validation if they become available.

Together, these scripts provide a structured workflow for moving from **single-year agave classification**, to **multi-year temporal classification**, to a **controlled temporal ablation**, and finally to **weakly and partially supervised spatial localization** using the same standardized Sentinel-2 patch dataset.
