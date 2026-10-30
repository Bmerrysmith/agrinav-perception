# AgriNav Perception

Rice and weed perception for an autonomous paddy tractor: data tooling, segmentation pretraining, detector training, and evaluation, packaged as an installable Python library with a unified CLI.

Senior engineering project, Florida Gulf Coast University. I led object detection and plant discrimination; this repository is the perception codebase I wrote and maintain.

> Research code, not a deployed system. Nothing here drives a sprayer or any actuator; the inference entry point is intentionally a disabled safety stub.

## Start here

- **Data integrity:** [deterministic dataset builder](src/agrinav/data/build_rice_phase2.py) and [validation tests](tests/test_build_rice_phase2.py).
- **Training and evaluation:** [baseline controls](src/agrinav/training/baseline_det_control.py), [COCO evaluator](src/agrinav/evaluation/runner.py), and [baseline findings](docs/baselines.md).
- **Engineering decisions:** [architecture records](docs/adr/) and [CPU CI workflow](.github/workflows/ci.yml).

## What's in here

| Area | What it does |
|---|---|
| `src/agrinav/data/` | Converts three annotation sources into one COCO schema, builds hash-verified train/valid/test splits, and runs fail-closed preflight checks (per-image hashes, decoded dimensions, in-bounds boxes, cross-split duplicates) |
| `src/agrinav/training/` | RiceSEG segmentation pretraining, detector training with atomic checkpoints and `metrics.jsonl`, stock DeepLabV3 / SegFormer / Faster R-CNN baseline controls, and a pilot report that recommends a gradient-clip threshold from measured gradient-norm quantiles |
| `src/agrinav/evaluation/` | Model-to-COCO adapter and COCO AP/AR evaluation used to select checkpoints |
| `src/agrinav/models/` | Custom single-stage detector (ATSS assigner, stride-4 FPN level for small, dense objects) |
| `tests/` | 300+ pytest tests, CPU only, no dataset required |
| `docs/adr/` | Architecture decision records for layout, packaging/CI, and postprocessing |

## Results

| Finding | Number |
|---|---|
| Dataset of record | 2,579 images, 81,201 boxes, 33 objects per image, ~29 px median box |
| Split leakage caught by manifest diffing | 940 mis-assigned files that hash-duplicate checks missed |
| RiceSEG segmentation pretraining | 0.583 mIoU, reproduced within 0.001 |
| Gradient-clipping default (0.5) | Truncated 100% of optimizer steps; retuning from measured norms raised overfit-gate AP50 from 0.006 to 0.875 |
| Segmentation ablation (weed IoU, only the model varies) | Custom 0.202, DeepLabV3 0.483, SegFormer-B2 0.522 |
| Same-protocol detection baseline | Custom detector 0.092 val AP, stock Faster R-CNN ~0.179 |

The last two rows are the most important result: under an identical split, resolution, and evaluator, stock architectures beat the custom ones. The project now ports the custom ideas onto a trusted baseline instead of defending a bespoke stack. Details: [baseline comparison](docs/baselines.md) and [segmentation results and limitations](docs/research/RICESEG_PRETRAIN_RESULTS.md). These are development findings; the small-set overfit gate is a pipeline check, not held-out field accuracy.

## Quick start

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell (use instead of the line above):
# .\.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
agrinav --help
python -m pytest
ruff check .
black --check .
```

The base install (`pip install -e .`) is data tooling only; `.[train]` adds torch, torchvision, and transformers.

## Engineering notes

- `src/` layout, single `agrinav` CLI dispatching to every data and training tool
- GitHub Actions CI on every pull request: ruff, black, mypy, pytest on CPU, wheel build
- Training runs write manifests recording git commit, seed, environment, and artifact hashes
- `.claude/agents/` holds the multi-agent Claude Code setup used during development (lead, data, ML, safety, review, docs, token-audit agents); review agents approve work they did not generate

## Not included

Datasets, trained weights, and run outputs are not published: the source datasets carry their own licenses and are not redistributed here. Some code comments refer to internal audit and handoff documents that are also not part of this repository.

## License

No license is granted. The code is shared for viewing and evaluation only; all rights reserved.

Author: Benjamin Merryman-Smith
