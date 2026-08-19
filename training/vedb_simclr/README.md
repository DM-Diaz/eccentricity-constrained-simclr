# VEDB SimCLR Training

This directory contains the recovered training code and SLURM launchers used to pretrain the four VEDB SimCLR ResNet-18 models reported in:

**Diaz, D. M., & Henderson, M. M. (2026). _Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field_. Proceedings of the 9th Conference on Cognitive Computational Neuroscience.**

The four public-facing training conditions are:

- **Baseline**
- **Fovea-Gaze**
- **Periph**
- **Periph-NF**

## Structure

```text
training/
└── vedb_simclr/
    ├── models/
    │   └── resnet_simclr.py
    ├── sbatch/
    ├── run_updated_fixed_v5.py
    └── simclr.py
```

This directory preserves the recovered project training files and their relationship to the original SimCLR implementation. It is **not intended to be a standalone reimplementation of SimCLR**.

## Upstream dependency: `sthalles/PyTorch-SimCLR`

The VEDB training pipeline was built from the open-source PyTorch SimCLR implementation:

https://github.com/sthalles/PyTorch-SimCLR

The relationship between the released files and the upstream repository is:

| File | Provenance |
|---|---|
| `run_updated_fixed_v5.py` | Project-modified version of the upstream `run.py` training entry point |
| `simclr.py` | From the upstream `sthalles/PyTorch-SimCLR` repository |
| `models/resnet_simclr.py` | From the upstream `sthalles/PyTorch-SimCLR` repository |

`run_updated_fixed_v5.py` was adapted for this project to support the recovered VEDB training workflow, including manifest-driven datasets and project-specific configuration.

The complete upstream repository is **not vendored here**. Users wishing to rerun training should first obtain `sthalles/PyTorch-SimCLR`, which provides the remaining package structure, utilities, augmentations, configuration support, and dependencies expected by the training code.

A typical setup is:

```bash
git clone https://github.com/sthalles/PyTorch-SimCLR.git
cd PyTorch-SimCLR
```

Then place the released project-specific training entry point into the corresponding repository layout, or otherwise adapt its imports and paths to your local installation.

The recovered copies of `simclr.py` and `models/resnet_simclr.py` are retained for provenance and to make clear which upstream implementation the project used.

Please preserve the original upstream license and attribution when reusing or redistributing code derived from `sthalles/PyTorch-SimCLR`.

## Training pipeline

At a high level:

```text
processed VEDB frames
        ↓
session-level train / validation split
        ↓
SimCLR data augmentation
        ↓
ResNet-18 encoder
        ↓
projection head
        ↓
NT-Xent contrastive objective
        ↓
120-epoch pretrained checkpoint
```

Each of the four visual-field conditions used the same model architecture and contrastive-learning framework. The conditions differed in the processed imagery supplied to the training pipeline.

Condition construction is documented under:

```text
vedb_processing/vedb/
```

Frame identities and split assignments are documented under:

```text
metadata/vedb/
```

## Architecture

The recovered model uses a **ResNet-18** backbone with the SimCLR projection head implemented in `models/resnet_simclr.py`.

For the released VEDB checkpoints, the projection head is:

```text
Linear(512, 512)
ReLU
Linear(512, 128)
```

The projection head is installed as `backbone.fc` in the recovered implementation.

For downstream use, the 512-dimensional encoder representation can be recovered by replacing the projection head with an identity mapping after loading the checkpoint.

Conceptually:

```python
encoder = model.backbone
encoder.fc = torch.nn.Identity()
```

## Recovered training configuration

The recovered production artifacts support the following VEDB SimCLR configuration:

| Setting | Recovered value |
|---|---|
| Backbone | ResNet-18 |
| Input size | `224 × 224` |
| Training epochs | 120 |
| Projection dimension | 128 |
| Contrastive loss | NT-Xent |
| Temperature (`tau`) | `0.07` |
| Optimizer | Adam |
| Learning rate | `6e-4` |
| Weight decay | `1e-4` |
| Learning-rate scheduler | Cosine annealing |
| Recovered production batch size | 512 |

The recovered training code uses `CosineAnnealingLR`, and the SimCLR training loop advances the scheduler during training according to the recovered implementation.

### Batch-size documentation discrepancy

The published methods described a batch size of **64**, whereas the recovered production training artifacts indicate a batch size of **512** for the VEDB SimCLR runs.

The public release preserves the recovered code and launch configuration rather than changing them to match the paper description.

## SimCLR augmentations

The project used the augmentation pipeline provided through the `sthalles/PyTorch-SimCLR` training framework, applied after condition-specific VEDB preprocessing.

The four conditions therefore enter SimCLR training as already constructed images:

```text
Baseline
Fovea-Gaze
Periph
Periph-NF
        ↓
shared SimCLR augmentation pipeline
        ↓
contrastive training
```

Condition-specific manipulations are not performed inside the SimCLR model itself.

## Checkpoint format

The recovered VEDB checkpoints are full training checkpoints rather than backbone-only state dictionaries.

Recovered checkpoint keys include:

```text
epoch
arch
state_dict
optimizer
```

The model parameters are stored under the recovered `backbone.*` naming convention, including the projection head under `backbone.fc.*`.

The released checkpoints correspond to epoch 120.

## SLURM launchers

The `sbatch/` directory contains recovered HPC launchers associated with the VEDB SimCLR training workflow.

These files preserve historical:

- cluster filesystem paths,
- resource requests,
- environment activation commands,
- condition names,
- command-line arguments.

They should be treated as provenance for the original computing environment and adapted before use on another system.

Some recovered launchers may represent smoke tests, debugging runs, or intermediate configurations rather than the final 120-epoch production jobs. The released epoch-120 checkpoints and associated metadata should be used together with the recovered training code when reconstructing the final training configuration.

## Historical condition names

Historical training scripts and paths may use earlier internal names:

| Public-facing name | Historical/internal name |
|---|---|
| **Baseline** | `Base`, `Baseline` |
| **Fovea-Gaze** | `Fovea-Gaze`, `FoveaGaze` |
| **Periph** | `periphNonTTM`, `PeriphNonTTM` |
| **Periph-NF** | `periphTTM`, `PeriphTTM` |

Capitalization, hyphenation, and underscore conventions may vary slightly across recovered files.

## Released pretrained models

The four VEDB SimCLR ResNet-18 checkpoints are hosted on Hugging Face:

- Baseline: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Baseline
- Fovea-Gaze: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Fovea-Gaze
- Periph: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Periph
- Periph-NF: https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Periph-NF

Collection:

https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb

The individual model cards contain detailed checkpoint specifications, architecture information, model-loading notes, and condition-specific documentation.

## Reference models

The ImageNet-100 and ImageNet-1K SimCLR reference models used in the study were trained separately using the **Lightly** self-supervised learning framework and are not produced by the VEDB training code in this directory.

The STL-10 reference model was obtained from the external Spijkervet/SimCLR release.

Reference-model code and documentation are provided separately under:

```text
linear_probes/reference_models/
```

## Reproducibility notes

This directory is a recovered research-code release rather than a packaged training library.

Users should expect to adapt:

- VEDB image and manifest paths,
- checkpoint/output paths,
- cluster-specific configuration,
- SLURM settings,
- Python/package versions,
- imports relative to the upstream `sthalles/PyTorch-SimCLR` repository.

The project-specific files are intentionally kept close to their recovered form so that the public release reflects the actual training workflow rather than a rewritten implementation.

## Third-party code and licensing

Portions of this directory originate from or are derived from:

**sthalles/PyTorch-SimCLR**  
https://github.com/sthalles/PyTorch-SimCLR

Those files remain subject to the upstream project's license and attribution requirements.

Project-authored modifications and other original release materials are governed by the repository's own license, while third-party code retains its original terms.

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
