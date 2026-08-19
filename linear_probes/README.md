# Linear Probes

This directory contains the downstream classification code, recovered SLURM launchers, reference-model utilities, and small output artifacts used for the transfer and in-domain evaluations reported in:

**Diaz, D. M., & Henderson, M. M. (2026). _Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field_. Proceedings of the 9th Conference on Cognitive Computational Neuroscience.**

## Structure

```text
linear_probes/
├── outputs/
├── reference_models/
├── sbatch/
├── in_domain_classifier_simclr_staged_robust_v3.py
├── places365_scene_classifier_v2.py
└── vggface2_identity_classifier_v2.py
```

## Downstream tasks

The three main classifier scripts correspond to the downstream evaluations used in the paper:

| Script | Evaluation |
|---|---|
| `in_domain_classifier_simclr_staged_robust_v3.py` | In-domain VEDB classification |
| `vggface2_identity_classifier_v2.py` | VGGFace2 identity classification |
| `places365_scene_classifier_v2.py` | Places365 scene classification |

The scripts retain their recovered filenames so that they remain directly associated with the original launch commands in `sbatch/`.

## Reported evaluation protocol

The downstream scripts support multiple stages, including exploratory fine-tuning. **The results reported in the paper use the frozen-backbone linear-probe stage only.**

For the reported analyses, the pretrained ResNet-18 encoder was frozen and a linear classifier was trained on top of the learned representation. Exploratory fine-tuning functionality present in the scripts and some recovered job configurations was not used for the reported paper results.

Accordingly, files or checkpoints produced after a fine-tuning stage should not be interpreted as the source of the published linear-probe results unless their stage has been independently verified.

## `outputs/`

`outputs/` contains small recovered run artifacts used to reconstruct and verify downstream analyses. Depending on the run, these include files such as:

- `history.json` — epoch-level training/evaluation history
- `config.json` — run configuration
- `run_fingerprint.json` — run/provenance information
- `label2id.json` — class-label mapping
- `best_top_confusions.csv` — confusion-summary output used in downstream analysis

Large model checkpoints are intentionally not stored in this GitHub repository.

For publication-facing analyses and plots, see `../analysis/vedb/`.

## `sbatch/`

Contains recovered SLURM launchers for the downstream evaluations.

These launchers preserve the original script names, command-line arguments, and cluster-oriented configuration used during the project. They may contain filesystem paths, resource requests, or historical condition names specific to the original HPC environment and should be adapted before reuse.

Some launchers may include arguments enabling both linear probing and exploratory fine-tuning. This does **not** indicate that the fine-tuning stage was used in the paper; reported results were taken from the linear-probe stage.

## `reference_models/`

Contains code and supporting materials for the non-VEDB reference encoders used in downstream comparisons.

The ImageNet-100 and ImageNet-1K SimCLR reference models were trained using **LightlySSL**, a PyTorch-based framework for self-supervised learning. LightlySSL provides components for contrastive learning, including SimCLR-style transformations, projection heads, and self-supervised losses.

LightlySSL documentation:

https://docs.lightly.ai/self-supervised-learning/getting_started/lightly_at_a_glance.html

The STL-10 SimCLR reference model originated from the external [Spijkervet SimCLR implementation/release](https://github.com/Spijkervet/SimCLR) rather than the LightlySSL training pipeline. Third-party licenses and attribution should be retained with the corresponding materials.

Detailed reference-model hyperparameters, checkpoint specifications, architecture information, and model-loading notes are documented in the associated Hugging Face model cards and collection.

## Model checkpoints

The four VEDB-trained SimCLR encoders are hosted on Hugging Face:

- Baseline: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Baseline
- Fovea-Gaze: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Fovea-Gaze
- Periph: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Periph
- Periph-NF: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Periph-NF

Collection:

https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb

Reference-model checkpoints are also hosted separately from this GitHub repository. See the corresponding Hugging Face repositories/model cards for exact checkpoint details.

## Historical condition names

Public-facing condition names and common historical/internal names are approximately:

| Public-facing name | Historical/internal name |
|---|---|
| **Baseline** | `Base`, `Baseline` |
| **Fovea-Gaze** | `Fovea-Gaze`, `FoveaGaze` |
| **Periph** | `periphNonTTM`, `PeriphNonTTM` |
| **Periph-NF** | `periphTTM`, `PeriphTTM` |

Capitalization, hyphenation, and underscore conventions vary slightly across recovered scripts and artifacts. Historical names are retained where necessary for compatibility with the original paths, launchers, and saved outputs.

## Reproducibility notes

This directory preserves recovered production code rather than presenting a fully refactored software package. Users should therefore expect to adapt:

- local or cluster filesystem paths,
- dataset locations,
- SLURM resource settings,
- checkpoint paths,
- environment-specific dependencies.

The original `.py` filenames were intentionally preserved because the recovered `.sbatch` launchers refer to those names directly.

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
