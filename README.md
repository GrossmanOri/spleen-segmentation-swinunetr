# Spleen Segmentation with Swin UNETR

Medical-image segmentation of the spleen from abdominal CT using a **Swin UNETR** vision transformer (PyTorch + MONAI), with **attention-based explainability** to verify the model looks at the right anatomy.

Final project for the **Neural Networks** course, Shenkar College of Engineering, Design and Art.
**Authors:** Ori Grossman & Amit Eliya.

---

## Overview

- Trained and evaluated a Swin UNETR model to segment the spleen from CT volumes (Medical Segmentation Decathlon, **Task09_Spleen**).
- Best validation **Dice 0.9465**.
- Ran a controlled, epoch-matched **ablation** isolating data augmentation.
- Built **attention-based explainability** and tested it for faithfulness (occlusion), not just correlation.

## Results

![Validation predictions on all 9 scans — red = prediction, green = ground truth](figures/results_gallery.png)

| Run | Schedule | Augmentation | Epochs | Val Dice |
|---|---|---|---|---|
| baseline | fixed LR 1e-4 | none | 60 | 0.9437 |
| exp1 | cosine + 15 ep warm-up | **on** | 80 | 0.9088 |
| exp2 | cosine + 10 ep warm-up + LR floor | **on** | 100 | 0.9108 |
| **exp3 (chosen)** | cosine + 10 ep warm-up + LR floor | none | 70 | **0.9465** |

Note the baseline is *also* un-augmented, so baseline → exp3 isolates the **LR schedule**, not augmentation. The clean augmentation contrast is **exp2 vs exp3** (identical schedule, augmentation the only difference).

- **Epoch-matched control:** the augmented recipe re-run at exactly 70 epochs reached 0.8927 vs exp3's 0.9465 — a ~5.4-point single-variable gap. That run peaked at epoch 40 and then plateaued, so it was converged, not undertrained.
- **Boundary quality (best model):** HD95 3.95 mm · Surface-Dice 0.889 @2 mm · precision 0.944 · recall 0.950.
- **Convergence:** exp3 dips once, to 0.7573 at epoch 40, recovers by epoch 45, then holds a 0.013-wide band (0.9331–0.9465) over its last 25 epochs — against a 0.231-wide swing for the baseline over its last 30. Roughly 17× tighter, which is why exp3 was chosen.
- **Stability:** 3-seed mean 0.9477 ± 0.0028. A paired Wilcoxon of **exp3 vs the baseline** gives **p = 0.20 — not significant**: exp3 is chosen for how tightly it converges, *not* for a demonstrated accuracy advantage over the baseline.

> **Limitation.** Results are on **9 validation volumes with no held-out test set**, and every hyper-parameter (schedule, warm-up, LR floor, augmentation on/off) was selected on those same 9 volumes. All numbers here are therefore optimistic to an unknown degree.

## Explainability

![Attention-on-spleen ratio and occlusion faithfulness test](figures/explainability_evidence.png)

Hooked the deepest Swin self-attention to see where the model attends:
- Attention concentrates on the spleen (attention-on-spleen ratio **1.47–2.33** across validation cases).
- **Null control:** a trained model's attention ratio (~1.90) is well above an untrained model (~1.11) and a random box (~1.28) → the concentration is *learned*.
- **Occlusion faithfulness:** masking the top-attention region drops Dice by **0.44**, vs **0.01** for random regions (Wilcoxon p = 0.004) → the attention is *faithful*, not merely correlational.

## Stack

PyTorch · MONAI 1.4.0 · Swin UNETR · trained on an NVIDIA L4 GPU.

## Repository

| Path | What it is |
|---|---|
| `Spleen_Segmentation_SwinUNETR.ipynb` | The full, self-contained notebook — code, outputs and figures. |
| `demo_inference.ipynb` | Short inference-only demo (reference code): segments a validation volume and reproduces the attention analysis. Requires the trained exp3 checkpoint from the Shenkar lab GPU — checkpoints are gitignored and not distributed, so it is not runnable as-is. |
| `scripts/train.py` | The training script used for all four runs — CLI-configurable schedule, augmentation and epoch budget. |
| `scripts/submit.sh` | Slurm batch job that runs the trainer inside the lab's Apptainer PyTorch container. |
| `docs/Report.pdf` | Written technical report. |
| `docs/Presentation.pdf` | Project presentation slides. |
| `figures/` | Result figures (learning curves, ablation, attention maps, prediction gallery). |

## Data

The dataset is **not** included in this repo. See [`docs/DATA.md`](docs/DATA.md) to download the Medical Segmentation Decathlon **Task09_Spleen** volumes from the official source.

## Reproducing

The notebook was run on a university GPU lab and is committed **with all outputs saved**, so it reads end-to-end without re-running. To re-run, download the dataset (see `docs/DATA.md`) and use a CUDA GPU with **MONAI 1.4.0** + PyTorch (`pip install -r requirements.txt`).

`scripts/train.py` is CLI-configurable and reproduces all four runs:

```bash
# baseline — fixed LR, no augmentation, 60 epochs
python scripts/train.py --exp_name baseline --scheduler none --epochs 60 --lr 1e-4

# exp1 — cosine + 15-epoch warm-up + augmentation, 80 epochs
python scripts/train.py --exp_name exp1_cosine_aug --scheduler cosine --warmup 15 --augment --epochs 80

# exp2 — cosine + 10-epoch warm-up + LR floor + augmentation, 100 epochs
python scripts/train.py --exp_name exp2_cosine_aug_floor --scheduler cosine --warmup 10 --augment --epochs 100

# exp3 (chosen) — same schedule, augmentation OFF, 70 epochs
python scripts/train.py --exp_name exp3_cosine_noaug --scheduler cosine --warmup 10 --epochs 70
```

Other flags: `--data_dir`, `--out_root`, `--weight_decay`, `--feature_size`, `--num_samples`, `--batch_size`, `--val_interval`, `--seed`. `scripts/submit.sh` wraps the same script as a Slurm job.

## Selected figures

![Experiment comparison](figures/all_experiments_comparison.png)
![Attention explanation](figures/attention_explanation.png)
![Prediction gallery](figures/results_gallery.png)
