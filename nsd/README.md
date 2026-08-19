# NSD Encoding Models

This directory contains the recovered code and SLURM launchers used for the Natural Scenes Dataset (NSD) feature-extraction, voxelwise encoding-model, and variance-partitioning analyses reported in:

**Diaz, D. M., & Henderson, M. M. (2026). _Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field_. Proceedings of the 9th Conference on Cognitive Computational Neuroscience.**

## Structure

```text
nsd/
├── sbatch/
│   ├── extract_eccbias.sh
│   ├── fit_eccbias.sh
│   └── fit_eccbias_varpart.sh
├── extract_simclr_eccbias.py
├── fit_eccbias_model_nsd.py
├── fit_eccbias_varpart_model_nsd.py
└── model_fitting_utils.py
```

The scripts retain their recovered filenames, internal model names, and cluster-oriented paths so that they remain associated with the original SLURM launchers and saved artifacts.

## Pipeline overview

The recovered NSD pipeline consists of three main stages:

```text
NSD 224 × 224 stimuli
        ↓
ResNet-18 feature extraction
        ↓
spatial reduction + PCA by layer
        ↓
concatenate layer features
        ↓
voxelwise ridge encoding models
        ↓
R² / correlation on held-out NSD images
        ↓
ROI summaries and variance partitioning
```

### 1. Feature extraction

`extract_simclr_eccbias.py` loads the preprocessed **224 × 224 NSD stimuli** and extracts activations from pretrained ResNet models.

For the ResNet-18 analyses used in the paper, activations are taken from:

- `conv1`
- `layer1.1`
- `layer2.1`
- `layer3.1`
- `layer4.1`
- `avgpool`

Convolutional feature maps are spatially reduced before PCA to keep the intermediate feature dimensionality manageable. The recovered implementation uses a target pre-PCA dimensionality of approximately **5,000 features per layer** where spatial reduction is required.

Images are normalized using the standard ImageNet mean and standard deviation:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

All compared encoders are evaluated on the same intact NSD stimuli. The VEDB Fovea-Gaze, Periph, and Periph-NF image transformations are **not** applied to NSD images during encoding-model evaluation; those labels refer to how the corresponding encoders were pretrained.

### 2. PCA

PCA is performed separately for each extracted layer.

The Python function in `extract_simclr_eccbias.py` has a historical default of 100 components, but the recovered production configuration used **200 PCA components per layer** for the paper analyses.

For the six ResNet-18 feature stages above, this yields up to:

```text
6 layers × 200 components = 1,200 model features
```

before addition of the regression intercept.

The PCA-transformed arrays are saved as subject-, model-, and layer-specific `.npy` files and are then consumed by the encoding-model scripts.

#### PCA fitting scope

PCA was fit separately for each **NSD subject, visual model, and feature layer**. In the recovered analysis pipeline, PCA was applied to the full subject-specific feature matrix **before** the encoding-model training, nested-holdout, and final held-out partitions were applied.

This is distinct from the later feature-normalization step. Feature-wise means and standard deviations were estimated only from the combined training and nested-holdout partitions and then applied to the final held-out evaluation set.

### 3. Voxelwise encoding models

`fit_eccbias_model_nsd.py` fits voxelwise ridge-regression encoding models.

For each subject:

- PCA features from the selected ResNet-18 layers are concatenated;
- features are z-scored before fitting;
- a constant intercept column is appended;
- ridge regression is fit separately for each voxel;
- the ridge penalty is selected using a nested held-out partition;
- final performance is evaluated on held-out shared NSD images.

With six 200-component layers, the fitted design matrix contains **1,201 columns**: 1,200 model features plus the intercept.

The recovered code evaluates **20 ridge penalties** spanning a broad logarithmic range.

### NSD splits

The recovered fitting code uses:

- **1,000 shared NSD images** as the final held-out validation/test set;
- a random **10% nested held-out partition** from the remaining subject-specific data for ridge-parameter selection;
- the remaining valid images for model fitting.

Feature normalization parameters are estimated using the training and nested partitions together and then applied to the held-out 1,000-image set.

The final encoding-model files store quantities including:

- model weights
- selected ridge-penalty indices
- voxelwise R²
- voxelwise correlation
- voxel masks/indices
- voxel noise-ceiling information
- feature-file provenance

## Variance partitioning

`fit_eccbias_varpart_model_nsd.py` fits three encoding models for a pair of feature spaces:

1. model 1 alone
2. model 2 alone
3. the concatenated feature spaces from both models

These fits are used to quantify variance uniquely explained by each representation and variance shared between the pair.

For two models A and B, unique variance is computed from held-out voxelwise `R²` as:

- `Unique R² for A = Combined R² - B-only R²`
- `Unique R² for B = Combined R² - A-only R²`

Thus, the variance uniquely attributed to one representation is the variance explained by the combined feature space minus the variance explained by the other representation alone.

The paper-facing variance-partitioning comparisons are:

- **Fovea-Gaze vs. Periph**
- **Periph vs. Periph-NF**

The script supports other pairings, but those two comparisons correspond to the variance-partitioning results reported in the paper.

## `model_fitting_utils.py`

`model_fitting_utils.py` contains shared utilities used by the encoding-model scripts, including:

- feature splitting and normalization
- ridge-regression fitting
- nested ridge-penalty selection
- R² calculation
- correlation calculation
- GPU-oriented helper functions

The recovered ridge implementation solves the regularized linear system separately across the candidate penalty values and selects the best penalty for each voxel using the nested held-out data.

## Model and artifact naming

The released encoding-model files retain the identifiers used by the original NSD feature-extraction and model-fitting pipeline. These are analysis-time identifiers rather than names stored inside the underlying SimCLR checkpoints.

| Public model name | Encoding-analysis identifier |
|---|---|
| **Baseline** | `resnet18-Baseline` |
| **Fovea-Gaze** | `resnet18-FoveaGaze` |
| **Periph** | `resnet18-PeriphNonTTM` |
| **Periph-NF** | `resnet18-PeriphTTM` |
| **STL-10** | `resnet18-pretrained-simclr` |
| **ImageNet-100** | `resnet18-simclr-imgnet100` |
| **ImageNet-1K** | `resnet18-simclr-imgnet1k` |

Related recovered scripts and paths may contain slight variations in capitalization, hyphenation, or shortened labels such as `Base`, `FoveaGaze`, `PeriphNonTTM`, and `PeriphTTM`. These historical names are retained where needed for compatibility with the original feature directories, SLURM launchers, and saved model-fit artifacts.

The internal identifier `resnet18-pretrained-simclr` corresponds to the STL-10 SimCLR ResNet-18 checkpoint obtained from the external [Spijkervet/SimCLR](https://github.com/Spijkervet/SimCLR) release. The ImageNet-100 and ImageNet-1K reference encoders were trained for this project using SimCLR implemented with the Lightly self-supervised learning framework.

Fields such as `model`, `model1`, `model2`, `features_file_list`, `features_file_list1`, and `features_file_list2` may preserve these historical identifiers and original cluster paths for provenance. Those paths are not expected to resolve outside the original computing environment.

## Recovered model artifacts

Large NSD encoding-model arrays are not stored in this GitHub repository.

The released encoding-model artifacts are available at:

https://huggingface.co/DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models

The VEDB SimCLR checkpoints used to extract the NSD features are available in the associated Hugging Face collection:

https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb

Detailed checkpoint specifications, training hyperparameters, architecture details, and model-loading notes are provided in the Hugging Face model cards and collection.

## Important artifact limitation

The released encoding-model `.npy` files contain the fitted encoding weights and associated evaluation information, but they are **not standalone end-to-end prediction packages**.

In particular, the saved model-fit arrays do not contain all preprocessing objects required to reconstruct the exact feature pipeline from arbitrary new images, including the fitted PCA transforms and feature-normalization parameters. Reproducing predictions from new stimuli therefore requires rerunning the feature-extraction/PCA pipeline or separately preserving the corresponding transforms.

The final element of each fitted weight vector corresponds to the added intercept term.

## Reproducibility notes

This directory preserves the recovered research code rather than presenting a fully refactored NSD software package.

Users should expect to adapt:

- NSD dataset locations
- checkpoint locations
- feature-output paths
- model-fit output paths
- SLURM resource settings
- Python environments and package versions

Several scripts contain absolute paths from the original Henderson Lab computing environment. These are intentionally retained as provenance and must be changed for another system.

Also note that the feature-extraction script also contains code paths for models and architectures that were explored during development but were not all part of the final paper comparisons. The paper-facing analyses should be identified from the associated SLURM launchers, released model artifacts, and analysis notebooks rather than from every branch implemented in the general-purpose extraction script.

## Analysis notebooks

The publication-facing NSD plotting and statistical analyses are stored separately under:

```text
analysis/nsd/
```

Those notebooks summarize voxelwise encoding performance across subjects and ROIs, compare the VEDB-trained encoders with reference models, and generate the variance-partitioning figures reported in the paper.

## Data access

The full [Natural Scenes Dataset](https://www.naturalscenesdataset.org/) is not redistributed through this repository.

Users should obtain NSD from the original dataset source and comply with its access and licensing requirements.

## Citation

```bibtex
@inproceedings{diaz2026eccentricity,
  author    = {Diaz, Dylan M. and Henderson, Margaret M.},
  title     = {Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field},
  booktitle = {Proceedings of the 9th Conference on Cognitive Computational Neuroscience},
  address   = {New York, NY, USA},
  year      = {2026},
  doi       = {10.32470/0416gfsq}
}
```
