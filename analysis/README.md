# Analysis

This directory contains the publication-facing analysis notebooks used to reproduce the figures and tables associated with:

**Diaz, D. M., & Henderson, M. M. (2026). _Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field_. Proceedings of the 9th Conference on Cognitive Computational Neuroscience.**

The notebooks here were cleaned from the original working analyses to retain the code relevant to the final reported figures and tables while removing exploratory plots, duplicate analyses, deprecated code, and intermediate development cells.

## Structure

```text
analysis/
├── vedb/
└── nsd/
```

### `vedb/`

Contains analysis and plotting notebooks for the VEDB SimCLR and downstream classification results, including:

- SimCLR pretraining curves reported in Fig. 2A
- in-domain linear-probe results reported in Fig. 2B
- VGGFace2 linear-probe results reported in Fig. 2C
- Places365 linear-probe results reported in Fig. 2D
- classwise in-domain confusion-structure analyses used for Fig. 3
- classwise ΔF1/bootstrap analyses used for Table 2 and supplementary analyses
- row-normalized in-domain confusion-matrix visualizations

The downstream scripts used in the project support exploratory fine-tuning, but **the results reported in the paper use the frozen-backbone linear-probe stage only**. The notebooks in this directory therefore select and plot ONLY the probe-stage results used for the publication.

### `nsd/`

Contains analysis and plotting notebooks for the Natural Scenes Dataset encoding-model analyses, including:

- encoding-model performance comparisons across VEDB-trained models
- comparisons against reference models
- noise-ceiling-normalized summaries
- variance-partitioning analyses reported in the paper

Large NSD model arrays and learned encoding-model artifacts are not stored in this GitHub repository. See the associated Hugging Face release for model artifacts and checkpoint details.

## Inputs

The notebooks generally operate on small derived analysis artifacts such as:

- `history.json`
- metrics/configuration JSON files
- confusion matrices
- per-class summary CSV files
- NSD subject/ROI summary arrays or tables

Large pretrained checkpoints and encoding-model files are hosted separately from GitHub and within a [HuggingFace collection](https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb).

Detailed checkpoint specifications, training hyperparameters, architecture details, and model-loading notes are provided in the associated Hugging Face collection and individual model cards.

## Reproducibility notes

These notebooks are intended to preserve the analyses that generated the reported figures rather than to serve as a fully refactored analysis package. Some notebooks may therefore retain:

- historical internal condition names,
- paths that should be adapted to the local directory structure,
- plotting conventions inherited from the original analysis,
- code organized around the exact intermediate files produced during the project.

Where possible, the cleaned notebooks preserve the plotting logic, ordering, labels, and source metrics used for the final paper figures.

## Condition names

Public-facing condition names used throughout the release are:

- **Baseline**
- **Fovea-Gaze**
- **Periph**
- **Periph-NF**

Historical scripts, filenames, and intermediate artifacts may use earlier internal names, sometimes with slight variations. The approximate mapping is:

| Public-facing name | Historical/internal name |
|---|---|
| **Baseline** | `Base`, `Baseline` |
| **Fovea-Gaze** | `Fovea-Gaze`, `FoveaGaze` |
| **Periph** | `periphNonTTM`, `PeriphNonTTM` |
| **Periph-NF** | `periphTTM`, `PeriphTTM` |

The historical names are retained where necessary for compatibility with the original scripts, SLURM launchers, file paths, and saved artifacts.

## Model artifacts

VEDB SimCLR checkpoints:

- https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb

NSD encoding-model artifacts:

- https://huggingface.co/DM-Diaz/VEDB-NSD-ResNet18-Encoding-Models

Non-Egocentric SimCLR checkpoints:

- [SimCLR ResNet-18 — ImageNet-1K](https://huggingface.co/DM-Diaz/SimCLR-ResNet18-ImageNet1K)
- [SimCLR ResNet-18 — ImageNet-100](https://huggingface.co/DM-Diaz/SimCLR-ResNet18-ImageNet100)
- [SimCLR ResNet-18 — STL-10](https://github.com/Spijkervet/SimCLR) *(external pretrained reference model; checkpoint provided by Spijkervet/SimCLR and not redistributed by this project)*

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
