# MORSE: Morphological Observation with Reweighted Spectral Encoding

> **Paper:** *MORSE: Leveraging Morphological and Spectral Priors for Early Disease Detection*
> **Journal:** Journal of Computers, Mechanical and Management (JCMM), 2026
> **Repository:** https://github.com/asmaalhasami/MORSE

A lightweight **~0.40 M parameter** medical image classifier trained **from random initialisation** — no ImageNet pretraining required. MORSE combines three domain-grounded components:

- **Multi-Scale Morphological Extraction** — fixed disk erosion/dilation at radii {1, 2, 4}
- **Gated Haar Wavelet Spectral Branch** — adaptive LL / high-frequency weighting
- **Memory-Efficient Cross-Attention** — fixed 7×7 token grid, constant attention cost

---

## Files

| File | Description |
|------|-------------|
| `proposed.py` | Full MORSE model (`MorphSpectralClassifier`) |
| `train.py` | Benchmark training script — MORSE + 8 baselines |
| `confusion_matrices.json` | Empirical confusion matrices for primary benchmarks (Tables 4 and 5) and reference distributions |
| `predictions_summary.csv` | Summary metrics per model and dataset |
| `run_ablation.py` | Reproduces Table 6 ablation and replacement evaluations from evaluated counts |
| `run_pretrained.py` | Reproduces Table 7 pretrained baseline comparisons |
| `run_seeds.py` | Reproduces Table 8 multi-seed evaluation (seeds 0 to 4) from evaluated per-seed records |
| `run_sensitivity.py` | Reproduces Table 9 hyperparameter sensitivity analysis across 3 seeds (seeds 0 to 2) |
| `make_tables.py` | Comprehensive verification script reconstructing all paper tables |
| `final_tables/` | Per-seed evaluation CSV archives (`table8_per_seed.csv`, `table9_per_seed.csv`, `table6_ablation_skin.csv`) |
| `download.sh` | Auto-downloads all 4 Kaggle datasets |

---

## Datasets

Use the provided `download.sh` to fetch all datasets automatically (requires Kaggle API credentials):

```bash
bash download.sh
```

Or download manually from Kaggle and place inside `datasets/`:

| Dataset | Classes | Images | Kaggle |
|---------|---------|--------|--------|
| Skin Disease | 9 | 878 | [Link](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-classification-image-dataset) |
| Nail Disease | 3 | 1,466 | [Link](https://www.kaggle.com/datasets/josephrasanjana/nail-disease-image-classification-dataset) |
| Eye Disease | 4 | 4,217 | [Link](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) |
| Alzheimer's MRI | 4 | 44,000 | [Link](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented) |

Expected structure after download:

```
datasets/
  skin_disease/   acne/  atopic_dermatitis/  cellulitis/  eczema/  impetigo/  melanoma/  psoriasis/  ringworm/  vitiligo/
  nail_disease/   acral_lentiginous_melanoma/  onychomycosis/  nail_psoriasis/
  eye_disease/    normal/  cataract/  diabetic_retinopathy/  glaucoma/
  alzheimers/     NonDemented/  VeryMildDemented/  MildDemented/  ModerateDemented/
```

---

## RadImageNet Pretrained Weights

The `resnet50_radimagenet` baseline (Table 7) requires the official RadImageNet-pretrained ResNet-50 checkpoint, released by [BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet) (Mei et al., 2022).

1. Download the PyTorch checkpoint from the official RadImageNet repository above.
2. Place it at `checkpoints/ResNet50_RadImageNet_pytorch.pt`.
3. Run `train.py` with the `resnet50_radimagenet` model key — the checkpoint is loaded automatically at that path (see `create_model()` in `train.py`).

If the checkpoint is not found, `train.py` prints a warning and falls back to a randomly initialized ResNet-50 rather than failing silently.

---

## Setup

```bash
pip install torch torchvision timm scikit-learn numpy pandas tqdm Pillow
```

---

## Reproducing Results

Measured empirical confusion matrices are archived for the primary benchmark models (Tables 4 and 5) in `confusion_matrices.json`. Per-sample predictions for the secondary experiments (Tables 6, 8, 9, and Section 4.5.4) were not preserved; for those experiments, the evaluated correct prediction counts and accuracies are reported in `final_tables/` (`table8_per_seed.csv`, `table9_per_seed.csv`, `table6_ablation_skin.csv`), and any per-class reference matrices in `confusion_matrices.json` are reconstructed distributions calibrated from those counts rather than measured per-sample outputs.

### Note on Environmental and Hardware Reproducibility
All primary models are trained under an identical stratified 70/15/15 split on an NVIDIA Tesla L4 GPU. On small-cohort evaluation benchmarks (such as the 132-image Skin Disease test set, where each individual test sample accounts for approximately 0.76% in Overall Accuracy), minor numerical variations ($\pm 1$ to $2$ samples) can naturally emerge from CUDA/cuDNN floating-point non-determinism. Empirical confusion matrices for primary benchmarks are preserved in `confusion_matrices.json`.

---

## Training from Scratch

```bash
# Train MORSE + all 8 baselines on all 4 datasets
python train.py
```

Key hyperparameters (set in `CONFIG` inside `train.py`):

| Setting | Value |
|---------|-------|
| Epochs | 30 |
| Batch size | 6 |
| Optimizer | AdamW (lr=1e-3, wd=0.01) |
| Early stopping patience | 10 |
| Seed | 42 |
| Split | 70 / 15 / 15 (train / val / test) |
| Augmentation | Disabled (`use_augmentation: False` in `CONFIG`) to evaluate raw inductive capability |

Results and logs are saved to `Results/`.
