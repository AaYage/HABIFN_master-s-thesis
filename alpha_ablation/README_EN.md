# α Ablation Experiments (Standalone Directory)

English | [中文](README.md)

Used to verify the role of the balance factor (α) in Chapter 4. This directory is fully isolated from `modules/` and `train_framed.py`; no original files are modified.

## α Definition (consistent with Section 4.2.2)

- `alpha_net` input: the frame-wise **mean + standard deviation** of MFCC and F0 (4 × 256 dims); this reflects branch stability better than the mean alone;
- The output passes through softplus and is then normalized so that `α1 + α2 = 1` and both are non-negative; `α1 = α2 = 0.5` is the neutral point;
- It is fixed at 0.5 during the first epoch and learned afterwards;
- Balance-loss weight **0.001**, to avoid suppressing directional differences.

## MAM Dynamic Weight (Section 5.2.3)

The MAM weight λ is computed directly from α; no separate quality scorer is used:

```
λ_M = α1 / (α1 + α2)
λ_F = 1 - λ_M
```

Here `α1` is the MFCC-query-F0 direction (the update is applied to MFCC), corresponding to `α*_fm` in the thesis.
In `none` mode, `α1 = α2 = 0.5`, so λ = 0.5, which is equivalent to fixed equal-weight fusion.

## Four Configurations

| Mode | α1 (MFCC→F0) | α2 (F0→MFCC) | Description |
| --- | --- | --- | --- |
| `none` | 0.5 | 0.5 | Balance factor disabled; λ = 0.5, fixed equal-weight fusion |
| `a1` | Learned | 0.5 | Only the MFCC→F0 direction uses a learned α; bidirectional attention is retained |
| `a2` | 0.5 | Learned | Only the F0→MFCC direction uses a learned α; bidirectional attention is retained |
| `both` | Learned | Learned | Same as the original model |

The balance loss is applied only in `both`; it is not used in `none` or the single-path modes.

## Input Types

The scripts support two input formats; the model structure is identical except for data loading:

| `--input_type` | Input | Time steps | Usage |
| --- | --- | --- | --- |
| `full` | MFCC 160 + F0 160 (+ other 0), CSV | 40 steps after the backbone | Legacy full320 main model |
| `framed` | MFCC (200,120) + F0 (200,4), npz | 200 steps | Frame-level model |

The `full` split is exactly the same as `modules/train_utils.py::train_full` (`StratifiedShuffleSplit(test_size=0.2, seed)` + `StandardScaler`). When the main model uses `batch_size=16`, the ablation should also pass `--batch_size 16`; otherwise the metrics are not comparable.

## Running

```powershell
# Run from the repository root

# GPU (recommended): four full320 ablations
$py = 'python'
foreach ($m in @('both','none','a1','a2')) {
  & $py alpha_ablation\train_ablation.py --alpha_mode $m --input_type full `
        --batch_size 16 --seed 42 --tag "alpha_full_$m"
}

# Frame-level: omit --input_type (default framed, seq_len=200)
& $py alpha_ablation\train_ablation.py --alpha_mode both --tag alpha_both
```

## Outputs

```text
results/alpha_ablation/alpha_full_<mode>/
  report.txt        # input_type / alpha_mode / Acc / AUC / sensitivity / specificity / confusion matrix
  predictions.csv   # per-sample probabilities
  history.csv       # per-epoch loss / AUC / alpha1 / alpha2
  config.json
checkpoints/alpha_ablation/alpha_full_<mode>/final
```

When `--tag` is omitted in frame-level mode, outputs go to `results/alpha_ablation/alpha_<mode>/`.

## Interpretation

- The Acc/AUC gap between `both` and `none` determines whether the balance factor is worth keeping;
- Comparing `a1` / `a2` with `both` verifies the "bidirectional synergy";
- `alpha1` / `alpha2` in `history.csv` can be used to plot convergence curves (Figure 4-3 in the thesis).
