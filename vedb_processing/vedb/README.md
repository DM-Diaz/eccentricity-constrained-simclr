# VEDB Processing

This directory contains the recovered preprocessing code used to construct the four Visual Experience Dataset (VEDB) training conditions reported in:

**Diaz, D. M., & Henderson, M. M. (2026). _Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field_. Proceedings of the 9th Conference on Cognitive Computational Neuroscience.**

The full VEDB dataset is **not redistributed through this repository**. These scripts operate on VEDB video, gaze, and derived frame data obtained from the original dataset source.

## Processing overview

```text
VEDB raw video
    ↓
frame_sampling/
eb_frameprocessing_cluster_2026.py
    ↓
Baseline 224×224 frames
    │
    ├───────────────┐
    ↓               ↓
gaze alignment    Baseline
    ↓
gaze/
eb_2026_HumanGazeProcessing_1.py
    ↓
frame-aligned gaze
    ↓
conditions/
eb_foveal_cropping.py
    ├── fixation metadata
    └── Fovea-Gaze
    ↓
eb_periph_gaze_cropping_v2.py
    └── Periph
          ↓
neurofovea/
Metamer_Transform_fixed_v2_fastbatch.py
          ↓
       Periph-NF
```

The four public-facing training conditions are:

- **Baseline**
- **Fovea-Gaze**
- **Periph**
- **Periph-NF**

<p align="center">
  <img
    src="https://huggingface.co/DM-Diaz/VEDB-SimCLR-ResNet18-Baseline/resolve/main/VEDB_Frame_Manipulation.png"
    alt="Example VEDB frames under the Baseline, Fovea-Gaze, Periph, and Periph-NF training conditions"
    width="850"
  >
</p>

## Directory structure

```text
vedb_processing/
└── vedb/
    ├── frame_sampling/
    ├── gaze/
    ├── conditions/
    └── neurofovea/
```

Each subdirectory corresponds to a distinct stage of the recovered preprocessing pipeline.

## `frame_sampling/`

The main recovered production script is:

```text
eb_frameprocessing_cluster_2026.py
```

This script:

- reads VEDB video segments;
- samples frames from the source videos;
- resizes frames to 256 pixels using bicubic interpolation;
- applies a 224 × 224 center crop;
- writes the processed Baseline frames used as the common source imagery for the visual-field conditions;
- records metadata used to construct the recovered manifests.

The recovered production configuration uses default values:

```text
STRIDE_SEC = 1
MAX_FRAMES = 1000
```

Adapt STRIDE_SEC to your data needs, but keep in mind, lowering STRIDE_SEC results in more temporally autocorrelated frames (and higher storage costs/overhead).

The released master manifest is available under:

```text
metadata/vedb/all_manifests_combined_synmapped_with_split.csv.gz
```

## `gaze/`

The primary recovered gaze-processing script is:

```text
eb_2026_HumanGazeProcessing_1.py
```

This stage aligns VEDB gaze measurements with sampled video frames.

The recovered implementation uses the VEDB synchronization/calibration information to map gaze timestamps to corresponding video-frame indices, producing frame-aligned gaze coordinates that can then be used for gaze-contingent image transformations.

These frame-aligned gaze estimates are consumed by the Fovea-Gaze and peripheral-condition preprocessing scripts.

## `conditions/`

### Fovea-Gaze

```text
eb_foveal_cropping.py
```

This script constructs the **Fovea-Gaze** condition from the Baseline frames and aligned gaze coordinates.

The recovered implementation uses a gaze-centered **112 × 112** crop, clamps the crop to image boundaries, upsamples it to **224 × 224**, and applies a feathered circular aperture with a gray surround.

The Fovea-Gaze condition should be interpreted as a **gaze-centered central-only input condition**, rather than as a biological simulation of the human fovea.

The script also produces fixation metadata used by later processing stages.

### Periph

```text
eb_periph_gaze_cropping_v2.py
```

This script constructs the **Periph** condition by masking the gaze-centered central region of each Baseline frame.

The recovered implementation applies a gray circular scotoma centered on gaze. The transition at the scotoma boundary is Gaussian-feathered; image content outside the central mask is not otherwise blurred by this script.

## `neurofovea/`

The recovered project-specific NeuroFovea renderer is:

```text
Metamer_Transform_fixed_v2_fastbatch.py
```

This stage constructs the **Periph-NF** condition.

### Recovered processing order

The recovered production pipeline applies the transformations in the following order:

```text
Baseline frame
    ↓
gaze-centered central scotoma
    ↓
Periph
    ↓
NeuroFovea transform
    ↓
Periph-NF
```

In other words, the recovered implementation applies the NeuroFovea transformation **after** the central scotoma has already been added. The scotoma is therefore part of the image passed through the NeuroFovea transformation.

NOTE that this differs from the ordering described in the published methods, which described NeuroFovea as preceding the central mask. The public release preserves the recovered production behavior and documents the discrepancy.

The recovered NeuroFovea configuration used a scale parameter of:

```text
scale = 0.4
```

The NeuroFovea implementation is derived from:

https://github.com/ArturoDeza/NeuroFovea_PyTorch

Third-party attribution and licensing should be preserved for the NeuroFovea-derived code.

## Historical condition names

Historical scripts, paths, and intermediate artifacts may use earlier internal names. Common mappings are:

| Public-facing name | Historical/internal name |
|---|---|
| **Baseline** | `Base`, `Baseline` |
| **Fovea-Gaze** | `Fovea-Gaze`, `FoveaGaze` |
| **Periph** | `periphNonTTM`, `PeriphNonTTM` |
| **Periph-NF** | `periphTTM`, `PeriphTTM` |

Capitalization, hyphenation, and underscore conventions vary slightly across recovered artifacts. Historical names are retained where needed for compatibility with original paths, scripts, and saved outputs.

## Related metadata

Recovered frame manifests are stored under:

```text
metadata/vedb/
```

These include the master sampled-frame manifest, the Fovea-Gaze derivative manifest, and the filtered Core-17 manifest used for in-domain classification.

The full VEDB imagery is not redistributed through this repository.

## Related model checkpoints

The four SimCLR ResNet-18 checkpoints trained from these processed VEDB conditions are available in the associated Hugging Face collection:

https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb

Detailed checkpoint specifications, training hyperparameters, architecture information, and model-loading notes are provided in the individual model cards.

## Reproducibility notes

This directory preserves recovered research code rather than presenting a fully refactored preprocessing package.

Users should expect to adapt:

- VEDB dataset locations;
- local or cluster filesystem paths;
- SLURM resource settings;
- output directories;
- environment-specific dependencies.

Some scripts contain historical paths or naming conventions from the original computing environment. These are retained for provenance and compatibility with the recovered workflow.

Legacy or exploratory preprocessing scripts may also exist alongside the production-facing files. The scripts documented above correspond to the recovered pipeline used to construct the released conditions.

## Data access

The underlying [Visual Experience Dataset](https://jov.arvojournals.org/article.aspx?articleid=2802101) should be obtained from the original source. This repository does not redistribute the full VEDB dataset.

Researchers using VEDB should comply with the dataset's original access requirements, licensing terms, and citation guidance.

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
