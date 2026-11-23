"""量化 α 的收敛性（主模型 3 个 seed）。

指标：
    final           : 最后一个 epoch 的 α1
    last10_sd       : 最后 10 个 epoch 的 α1 标准差
    last10_range    : 最后 10 个 epoch 的 α1 极差
    last10_step     : 最后 10 个 epoch 相邻差的平均绝对值（振荡幅度）
    conv_epoch_1pp  : 首次进入并保持在 final±0.01 的 epoch
    conv_ratio      : conv_epoch_1pp / 总 epoch 数

输出：
    results/metrics/alpha_convergence/convergence.csv
"""
import os

import numpy as np
import pandas as pd

TAGS = ("alpha_full_main_both_s42", "alpha_full_main_both_s43", "alpha_full_main_both_s44")
ROOT = "results/alpha_ablation"
OUT = "results/metrics/alpha_convergence"
TOL = 0.01
LAST = 10


def parse_val(s):
    return float(str(s).strip("[]").split()[0])


def metrics(a, final_tol=TOL, last=LAST, win=10, win_tol=0.02):
    a = np.asarray(a, dtype=float)
    n = len(a)
    tail = a[-last:]
    final = a[-1]
    dev = np.abs(a - final)
    conv = None
    for e in range(n):
        if np.all(dev[e:] <= final_tol):
            conv = e + 1
            break
    conv_win = None
    for e in range(n - win + 1):
        if all(a[k:k + win].ptp() <= win_tol for k in range(e, n - win + 1)):
            conv_win = e + 1
            break
    k = np.arange(min(20, n))
    slope = float(np.polyfit(k, a[-len(k):], 1)[0])
    return {
        "epochs": n,
        "alpha1_final": final,
        "last10_mean": float(tail.mean()),
        "last10_sd": float(tail.std(ddof=1)),
        "last10_range": float(tail.max() - tail.min()),
        "last10_step": float(np.abs(np.diff(tail)).mean()),
        "conv_epoch_1pp": conv if conv is not None else np.nan,
        "conv_ratio": (conv / n) if conv is not None else np.nan,
        "conv_epoch_win2pp": conv_win if conv_win is not None else np.nan,
        "slope_last20": slope,
    }


def main():
    rows = []
    for tag in TAGS:
        df = pd.read_csv(os.path.join(ROOT, tag, "history.csv"))
        m = metrics(df["alpha1"].values)
        m["seed"] = int(tag.rsplit("s", 1)[-1])
        m["split"] = "train"
        rows.append(m)
        mv = metrics([parse_val(v) for v in df["val_alpha_tracker"].values])
        mv["seed"] = m["seed"]
        mv["split"] = "val"
        rows.append(mv)

    out = pd.DataFrame(rows)[["seed", "split", "epochs", "alpha1_final", "last10_mean",
                              "last10_sd", "last10_range", "last10_step",
                              "conv_epoch_1pp", "conv_ratio", "conv_epoch_win2pp",
                              "slope_last20"]]
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/convergence.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\n== summary (mean over seeds)")
    print(out.groupby("split")[["alpha1_final", "last10_sd", "last10_range",
                                "last10_step", "conv_ratio"]].agg(["mean", "std"])
          .round(4).to_string())
    print("ALPHA_CONVERGENCE_DONE")


if __name__ == "__main__":
    main()
