# HABIFN

English | [中文](README.md)

A reference implementation of a **source–tract asymmetric fusion network** for Parkinson's disease (PD) speech detection.

This repository accompanies the method and experiments in Chapters 4–5 of the master's thesis: F0 represents the glottal source excitation and MFCC the vocal-tract response. PD/HC classification is performed with Attribute-Prototype-guided refinement (APAG), Balanced Cross-Attention (BCA), and a global–local fusion module (MAM).

> This repository contains code and a small set of result files only. Speech corpora, preprocessed features, and trained checkpoints are **not** included.

## 1. Method Overview

| Module | Description |
| --- | --- |
| Dual-branch backbone | The MFCC and F0 branches each produce a 256-dim, 40-step sequence representation |
| APAG | Attribute-prototype-guided refinement that strengthens local responses consistent with stable pathological patterns |
| BCA | Two stacked bidirectional cross-attention layers; the balance factor α (α1+α2=1) scales the query vectors and controls the interaction strength of each direction |
| MAM | Global-guided fusion of local responses; the fusion weight λ is derived from α (λ_M=α1, λ_F=α2) |
| Output | Concatenation → fully connected → Sigmoid, producing the PD probability |

The balance factor α is generated from the frame-wise mean and standard deviation of each branch representation:

```
σ = [μ(H_M); s(H_M); μ(H_F); s(H_F)]        # 4 × 256 = 1024
r = softplus(W2 · LeakyReLU(W1 · LayerNorm(σ) + b1) + b2)
α_i = r_i / (r1 + r2 + ε)                    # α1 + α2 = 1
```

## 2. Repository Structure

```
.
├── train.py                       # Main entry (raw audio → preprocessing → training)
├── train_framed.py                # Frame-level input variant
├── modules/
│   ├── model.py                   # Main model (APAG + BCA + MAM)
│   ├── preprocess.py              # 320-dim feature extraction (MFCC + F0)
│   ├── preprocess_framed.py       # Frame-level feature extraction
│   ├── train_utils.py             # Training, evaluation, metrics
│   ├── features.py / path_utils.py
├── alpha_ablation/
│   ├── model_ablation.py          # Ablation model (α mode and fusion variant selectable)
│   ├── train_ablation.py          # Ablation training script
│   └── README.md                  # Ablation configurations and commands
├── tools/
│   ├── analysis/                  # Feature extraction, α analysis, noise evaluation
│   ├── attention_analysis/        # α controls, convergence, mechanism figures
│   └── experiments/               # Subject-level experiments
├── ui/                            # Desktop demo interface (PySide6)
└── results/
    ├── figures/                   # Paper figures (attention/balance, full)
    └── metrics/                   # Summary metric CSV files
```

## 3. Environment

- Python 3.9
- TensorFlow 2.10.1 (GPU, CUDA 11.8 / cuDNN 8.6), Keras 2.10.0
- Other dependencies are listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

On Windows, if loading the CUDA DLLs fails, set:

```powershell
$env:TF_CUDA_LIB_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
```

## 4. Data Preparation

Due to the license restrictions of PC-GITA, IPVS, and similar corpora, no data are distributed with this repository. Please apply for the corpora and generate the following files with `modules/preprocess.py`:

```
data/processed_features/train_data.csv
data/processed_features/val_data.csv
data/processed_features/test_data.csv
```

CSV format (322 columns):

| Column | Description |
| --- | --- |
| `filename` | Audio file name |
| `label` | 0 = HC, 1 = PD |
| `feat_0 … feat_159` | MFCC statistics (40 dimensions × 4 statistic groups) |
| `feat_160 … feat_319` | F0 contour, first difference, second difference, and voicing probability (40 frames × 4 groups) |

To re-extract features from raw audio, set the corpus root directory before running the noise/α analysis scripts, for example:

```bash
export PCGITA_AUDIO_DIR=/path/to/PC-GITA/All   # On Windows: $env:PCGITA_AUDIO_DIR
```

## 5. Training

```bash
# Main model (raw audio path; preprocessing is performed inside the script)
python train.py --data_path /path/to/audio --full_training --seed 42

# Ablation: both (learned α) and none (fixed α=0.5)
python alpha_ablation/train_ablation.py --alpha_mode both --input_type full \
    --batch_size 16 --seed 42 --tag demo_both
python alpha_ablation/train_ablation.py --alpha_mode none --input_type full \
    --batch_size 16 --seed 42 --tag demo_none
```

Outputs are written to `results/alpha_ablation/<tag>/` (report, predictions, history, config) and
`checkpoints/alpha_ablation/<tag>/final`.

## 6. Analysis and Reproduction

The following scripts read checkpoints from `checkpoints/alpha_ablation/` (**not released with this repository**; train the models first):

```bash
python tools/attention_analysis/alpha_convergence.py          # Quantify α convergence
python tools/attention_analysis/alpha_control_eval.py         # α permutation / fixed-mean controls
python tools/attention_analysis/balance_mechanism.py          # Figure: α distribution + intervention
python tools/attention_analysis/plot_alpha_mechanism_3seed.py # Figure: 3-seed availability response
```

## 7. Result Files

| File | Content |
| --- | --- |
| `results/figures/attention/balance/fig3_dose_response.png` | α distribution, α vs. attention entropy, forced-α intervention, and decision-level effect |
| `results/figures/attention/balance/fig_alpha_mechanism.png` | Effect of branch attenuation on α and performance across 3 seeds |
| `results/figures/full/` | Main-model training curves, confusion matrix, FDR-IIDR |
| `results/metrics/main_seed/main_both_none.csv` | Paired both-vs-none results over 3 seeds |
| `results/metrics/alpha_control/alpha_control.csv` | α permutation / fixed-mean controls |
| `results/metrics/alpha_convergence/convergence.csv` | α convergence metrics |
