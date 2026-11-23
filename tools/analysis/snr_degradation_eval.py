"""单分支 SNR 退化实验：检验 α 是否随分支可靠度变化，以及 both/none 的鲁棒性差异。

做法（推理级扰动，不改模型、不重训）：
    1. 读取与探针一致的测试集与标准化流程；
    2. 每次只对一条分支（MFCC 或 F0）加高斯噪声，噪声功率按目标 SNR 相对该分支自身方差设定；
    3. 同一 (分支, SNR) 下使用同一份噪声，保证 both / none 的对比是配对的；
    4. 记录 Acc/AUC/敏感度/特异度，以及 α1/α2/λ 与第一层注意力熵。

用法：
    python tools/analysis/snr_degradation_eval.py

产物：
    results/metrics/snr_degradation/metrics.csv
    results/metrics/snr_degradation/summary.json
    results/figures/attention/balance/snr_alpha_response.png
    results/figures/attention/balance/snr_robustness.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402

from attention_balance_probe import FIG_DIR, build_test_inputs, load_model  # noqa: E402

METRIC_DIR = "results/metrics/snr_degradation"
SNR_LEVELS = [20, 10, 5, 0]  # dB；None 表示 clean
BRANCHES = ["mfcc", "f0"]
MODELS = {"both": "checkpoints/alpha_ablation/alpha_full_both/final",
          "none": "checkpoints/alpha_ablation/alpha_full_none/final"}


def add_noise(x, snr_db, rng):
    """按每条样本自身的特征方差设定噪声功率，使分支特征的信噪比为 snr_db。"""
    if snr_db is None:
        return x
    std = x.std(axis=1, keepdims=True) + 1e-8
    noise_pow = std ** 2 / (10.0 ** (snr_db / 10.0))
    return x + rng.normal(0.0, np.sqrt(noise_pow), size=x.shape).astype(np.float32)


def evaluate(model, bca, inputs, y):
    prob = model(inputs, training=False).numpy().ravel()
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    last = bca.bca_layers[-1]
    first = bca.bca_layers[0]
    return {
        "acc": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "alpha1_mean": float(last.last_alpha1.numpy().mean()),
        "alpha2_mean": float(last.last_alpha2.numpy().mean()),
        "alpha1_std": float(last.last_alpha1.numpy().std()),
        "entropy_m_l1": float(first.last_entropy_m.numpy().mean()),
        "entropy_f_l1": float(first.last_entropy_f.numpy().mean()),
    }


def run():
    X, y = build_test_inputs()
    models = {name: load_model(path, name) for name, path in MODELS.items()}

    rows = []
    for name, (model, bca) in models.items():
        # clean 基线
        base = evaluate(model, bca, [X[0], X[1], X[2]], y)
        rows.append({"model": name, "branch": "clean", "snr_db": None, **base})
        for branch in BRANCHES:
            for snr in SNR_LEVELS:
                seed = {"mfcc": 1, "f0": 2}[branch] * 1000 + int(snr) + 17
                rng = np.random.default_rng(seed)
                corrupted = [X[0].copy(), X[1].copy(), X[2]]
                idx = 0 if branch == "mfcc" else 1
                corrupted[idx] = add_noise(corrupted[idx], snr, rng)
                res = evaluate(model, bca, corrupted, y)
                rows.append({"model": name, "branch": branch, "snr_db": snr, **res})

    df = pd.DataFrame(rows)
    os.makedirs(METRIC_DIR, exist_ok=True)
    df.to_csv(f"{METRIC_DIR}/metrics.csv", index=False)

    summary = {}
    for name in MODELS:
        d = df[df["model"] == name]
        summary[name] = {b: d[d["branch"] == b].to_dict("records") for b in ["clean"] + BRANCHES}
    with open(f"{METRIC_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_alpha_response(df)
    plot_robustness(df)
    return df


def plot_alpha_response(df):
    """α 随分支 SNR 的响应：噪声加在哪条分支，该分支的 α 是否下降。"""
    c_b, c_n = "#2E5EAA", "#B0413E"
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, branch, label in zip(axes, BRANCHES, ["MFCC branch degraded", "F0 branch degraded"]):
        d = df[(df["model"] == "both") & (df["branch"] == branch)]
        xs = [25] + list(d["snr_db"].values)  # clean 放在 25 dB 左侧占位
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        a1 = np.r_[df[df["branch"] == "clean"]["alpha1_mean"].values[0], d["alpha1_mean"].values][order]
        a2 = np.r_[df[df["branch"] == "clean"]["alpha2_mean"].values[0], d["alpha2_mean"].values][order]
        ax.plot(xs, a1, "o-", color=c_b, ms=3.5, label=r"$\alpha_1$ (MFCC)")
        ax.plot(xs, a2, "s-", color="#7FA6D8", ms=3.5, label=r"$\alpha_2$ (F0)")
        ax.axhline(0.5, color=c_n, lw=0.9, ls="--", label="Neutral 0.5")
        ax.set_xlim(27, -1)
        ax.set_xlabel("SNR (dB), clean at 25")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Mean balance factor")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/snr_alpha_response.png", dpi=600)
    plt.close(fig)


def plot_robustness(df):
    """both vs none 在单分支退化下的性能曲线。"""
    c_b, c_n = "#2E5EAA", "#B0413E"
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2), sharex=True)
    for j, branch in enumerate(BRANCHES):
        for i, metric in enumerate(["auc", "acc"]):
            ax = axes[i, j]
            for name, color, marker in [("both", c_b, "o"), ("none", c_n, "s")]:
                d = df[(df["model"] == name) & (df["branch"] == branch)]
                clean = df[(df["model"] == name) & (df["branch"] == "clean")]
                xs = [25] + list(d["snr_db"].values)
                ys = [clean[metric].values[0]] + list(d[metric].values)
                order = np.argsort(xs)
                ax.plot(np.array(xs)[order], np.array(ys)[order], marker + "-",
                        color=color, ms=3.0, label=f"{name}")
            ax.set_xlim(27, -1)
            ax.set_title(f"{branch.upper()} degraded")
            ax.set_ylabel(metric.upper())
            if i == 1:
                ax.set_xlabel("SNR (dB), clean at 25")
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/snr_robustness.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    out = run()
    print(out.to_string(index=False))
    print("SNR_EVAL_DONE")
