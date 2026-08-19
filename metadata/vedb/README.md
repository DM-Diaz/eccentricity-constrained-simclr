# Metadata

This directory contains recovered metadata and manifests used to define the VEDB frame corpus, session-level data splits, and the in-domain classification subset for:

**Diaz, D. M., & Henderson, M. M. (2026). _Eccentricity-Constrained CNN Training Reveals Adaptive Information Coding Around the Visual Field_. Proceedings of the 9th Conference on Cognitive Computational Neuroscience.**

## Structure

```text
metadata/
└── vedb/
    ├── all_manifests_combined_synmapped_with_split.csv.gz
    ├── all_manifests_combined_synmapped_with_split_fovea.csv.gz
    └── in_domain_core17_manifest_final.csv.gz
```

The full Visual Experience Database (VEDB) is **not redistributed through this repository**. These files contain metadata identifying the sampled frames used by the project rather than the underlying video/image data.

## Master VEDB manifest

`all_manifests_combined_synmapped_with_split.csv.gz` is the recovered master manifest for the matched VEDB frame corpus used in the project.

Recovered totals:

- **514 sessions**
- **433,564 frames**
- **455 training sessions / 377,462 frames**
- **28 validation sessions / 26,026 frames**
- **31 test sessions / 30,076 frames**

The split is defined at the **session level**, so frames from a given session remain within one split.

Common manifest fields include:

| Column | Description |
|---|---|
| `session` / `session.1` | VEDB session identifier |
| `sample_i` | within-session sampled-frame index |
| `segment_id` | source video segment identifier |
| `segment_start` | segment start metadata |
| `segment_end` | segment end metadata |
| `task` | processed task/activity label |
| `location` | recovered location/context field |
| `frame_idx` | sampled frame index within the source video |
| `video_frame_idx` | corresponding source-video frame index |
| `filename` | processed-frame filename/path recorded by the pipeline |
| `ok` | processing-success flag |
| `error` | processing error field |
| `task_raw` | original/raw task label |
| `split` | session-level `train`, `val`, or `test` assignment |

The manifest should be treated as the authoritative record of the recovered frame identities and split assignments used in the project.

## Fovea-Gaze manifest

`all_manifests_combined_synmapped_with_split_fovea.csv.gz` is the corresponding recovered manifest for the Fovea-Gaze processed frames.

It preserves the same underlying sampled frame identities and split assignments as the master manifest, while the processed filenames refer to the Fovea-Gaze condition.

## In-domain Core-17 manifest

`in_domain_core17_manifest_final.csv.gz` defines the filtered VEDB subset used for the in-domain classification analysis.

Recovered totals:

- **167,437 frames**
- **429 sessions**
- **17 task/activity classes**
- **116,540 training frames**
- **23,626 validation frames**
- **27,271 test frames**

This file is a filtered derivative of the master VEDB manifest. The retained `(session, frame_idx)` pairs map back to the master manifest, preserving the relevant source-frame and split metadata.

The recovered master manifest also contains the session-level split:

- 455 train sessions
- 28 validation sessions
- 31 test sessions

## Relationship to preprocessing code

The metadata in this directory should be interpreted together with the recovered VEDB preprocessing code under:

```text
vedb_processing/vedb/
├── gaze/
├── frame_sampling/
├── conditions/
└── neurofovea/
```

In particular, the frame-sampling scripts define how source-video frames were selected and processed, while these manifests record the resulting frame identities, task labels, and split assignments.

## Condition correspondence

The four public-facing VEDB conditions are:

| Public-facing name | Common historical/internal name |
|---|---|
| **Baseline** | `Base`, `Baseline` |
| **Fovea-Gaze** | `Fovea-Gaze`, `FoveaGaze` |
| **Periph** | `periphNonTTM`, `PeriphNonTTM` |
| **Periph-NF** | `periphTTM`, `PeriphTTM` |

Historical capitalization, hyphenation, and underscore conventions vary slightly across recovered files.

## Data access

The underlying VEDB data should be obtained from the original dataset source. See [VEDB](https://jov.arvojournals.org/article.aspx?articleid=2802101). This repository redistributes only project-specific metadata required to identify and reconstruct the sampled corpus.

Users are responsible for complying with the original VEDB license and access terms.

## Related model artifacts

The VEDB SimCLR checkpoints corresponding to these manifests are available in the associated Hugging Face collection:

https://huggingface.co/collections/DM-Diaz/eccentricity-constrained-simclr-models-vedb

Detailed checkpoint specifications, training hyperparameters, architecture information, and model-loading notes are provided in the individual model cards.

## Citation

If these metadata are used to reproduce the analyses, please cite the associated paper:

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
