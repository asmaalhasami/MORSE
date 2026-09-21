# MORSE: Morphological Observation with Reweighted Spectral Encoding



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
| `download.sh` | Auto-downloads all 4 Kaggle datasets |
| `confusion_matrices.json` | Per-model confusion matrices (all 36) |
| `predictions_summary.csv` | Test prediction records |

---

## Datasets

Use the provided `download.sh` to fetch all datasets automatically (requires Kaggle API credentials):

```bash
bash download.sh
```

Or download manually from Kaggle and place inside `datasets/`:

| Dataset | Classes | Images |
|---------|---------|--------|
| Skin Disease | 9 | 878 |
| Nail Disease | 3 | 1,466 |
| Eye Disease | 4 | 4,217 |
| Alzheimer's MRI | 4 | 44,000 |

Expected structure after download:

```
datasets/
  skin_disease/   acne/  atopic_dermatitis/  cellulitis/  ...
  nail_disease/   acral_lentiginous_melanoma/  onychomycosis/  nail_psoriasis/
  eye_disease/    normal/  cataract/  diabetic_retinopathy/  glaucoma/
  alzheimers/     NonDemented/  VeryMildDemented/  MildDemented/  ModerateDemented/
```

---

## Setup

```bash
pip install torch torchvision timm scikit-learn numpy pandas tqdm Pillow
```

---

## Training

```bash
# Train MORSE + all 8 baselines on all 4 datasets
python train.py
```

Key hyperparameters (set in `CONFIG` inside `train.py`):

| Setting | Value |
|---------|-------|
| Epochs | 30 |
| Batch size | 6 (× 4 grad accum = 24 effective) |
| Optimizer | AdamW (lr=1e-3, wd=0.01) |
| Early stopping patience | 10 |
| Seed | 42 |

Results are saved to `Results/`.
