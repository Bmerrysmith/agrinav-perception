# Training the detector

Runs need a GPU (development used a Colab A100) and a local copy of the dataset, which is not distributed with this repository. `<DATA_ROOT>` stands for wherever your data and pretrained backbone live.

## 2-epoch pilot

The pilot records the gradient-norm distribution, the positive/negative loss split, and train-vs-eval BatchNorm parity. Run it before any full training run.

RiceSEG-initialized backbone:

```bash
python -m agrinav.training.weeddet_train --epochs 2 --bn-policy freeze_pretrained --bn-freeze-scope backbone --riceseg-backbone <DATA_ROOT>/riceseg_backbone.pth --dump-grad-norms --checkpoint-dir checkpoints/pilot_riceseg
```

ImageNet-initialized backbone:

```bash
python -m agrinav.training.weeddet_train --epochs 2 --bn-policy freeze_pretrained --bn-freeze-scope imagenet --dump-grad-norms --checkpoint-dir checkpoints/pilot_imagenet
```

Then summarize both runs, including the recommended gradient-clip threshold:

```bash
python -m agrinav.training.pilot_report --help
```

Every command in a fenced bash block that calls the trainer is checked by `tests/test_documented_commands.py`, so a renamed or invented flag fails CI.
