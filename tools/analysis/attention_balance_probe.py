"""平衡因子机制探针：导出逐样本 α/λ、注意力熵，并绘制机制图。

用法：
    python tools/analysis/attention_balance_probe.py

产物：
    results/metrics/attention_balance/*.csv, summary.json
    results/figures/attention/balance/fig3_dose_response.png

fig3 为 2x3 机制总览：
    (a) 学到的 α1 分布（动态性）
    (b)(c) 第一层注意力熵 vs 学到的 α（观测证据）
    (d)(e) 强制 α 的剂量-反应，阴影为均值的 95% CI（因果证据）
    (f) 相对 α1=0.5 的配对概率变化，误差带为 95% CI（下游影响）
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import keras
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr, ttest_rel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
})

from alpha_ablation.model_ablation import (  # noqa: E402
    BalancedCrossAttentionLayerAblation,
    ModalCrossAttentionAblation,
)
from modules.model import AlphaTracker  # noqa: E402
from modules.train_utils import load_data  # noqa: E402

CUSTOM_OBJECTS = {
    "AlphaTracker": AlphaTracker,
    "BalancedCrossAttentionLayerAblation": BalancedCrossAttentionLayerAblation,
    "ModalCrossAttentionAblation": ModalCrossAttentionAblation,
}

TRAIN_CSV = "data/processed_features/train_data.csv"
TEST_CSV = "data/processed_features/test_data.csv"
FIG_DIR = "results/figures/attention/balance"
METRIC_DIR = "results/metrics/attention_balance"
ALPHA_GRID = np.round(np.arange(0.15, 0.86, 0.10), 2)


def build_test_inputs(seed=42):
    """复现 train_full 的标准化流程：在训练集内部划分上拟合 scaler，再变换测试集。"""
    X_tr, y_tr = load_data(TRAIN_CSV)
    X_te, y_te = load_data(TEST_CSV)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for tr_idx, _ in sss.split(X_tr, y_tr):
        scaler = StandardScaler().fit(X_tr[tr_idx])
    X_te = scaler.transform(X_te)
    return [X_te[:, :160], X_te[:, 160:320], X_te[:, 320:]], y_te


def read_train_config(path):
    """从 checkpoints/alpha_ablation/<tag>/final 反查训练配置，保证重建结构一致。"""
    parts = os.path.normpath(path).split(os.sep)
    if "alpha_ablation" not in parts:
        return {}
    tag = parts[parts.index("alpha_ablation") + 1]
    cfg = os.path.join("results", "alpha_ablation", tag, "config.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def load_model(path, alpha_mode=None, gate_mode=None):
    """SavedModel 的 __call__ 会复用已追踪的图，不会执行 Python call，
    因此缓存属性拿不到。这里重建同结构模型并迁移权重，保证逐样本探针可用。"""
    from alpha_ablation.model_ablation import build_full_model_ablation

    cfg = read_train_config(path)
    alpha_mode = alpha_mode or cfg.get("alpha_mode", "both")
    gate_mode = gate_mode or cfg.get("gate_mode", "residual")

    saved = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
    model, bca = build_full_model_ablation(alpha_mode=alpha_mode, gate_mode=gate_mode)
    model.set_weights(saved.get_weights())
    return model, bca


def collect(model, bca, inputs, y):
    prob = model(inputs, training=False).numpy().ravel()
    last = bca.bca_layers[-1]
    first = bca.bca_layers[0]
    return pd.DataFrame({
        "label": y,
        "prob": prob,
        "alpha1": last.last_alpha1.numpy(),
        "alpha2": last.last_alpha2.numpy(),
        "entropy_m": last.last_entropy_m.numpy(),
        "entropy_f": last.last_entropy_f.numpy(),
        "entropy_m_l1": first.last_entropy_m.numpy(),
        "entropy_f_l1": first.last_entropy_f.numpy(),
        "alpha1_l1": first.last_alpha1.numpy(),
        "alpha2_l1": first.last_alpha2.numpy(),
    })


def dose_response(model, bca, inputs, grid=ALPHA_GRID):
    """固定权重，强制 α1 扫描（α2=1-α1）。逐层记录熵，用于分离直接效应与层间传播。"""
    originals = [layer.alpha_net for layer in bca.bca_layers]
    records = []
    probs_list = []
    try:
        for a1 in grid:
            # alpha_net 的输出已经过 softplus，因此这里直接给 (alpha1, alpha2) 即可
            const = tf.constant([[a1, 1.0 - a1]], dtype=tf.float32)
            for layer in bca.bca_layers:
                layer.alpha_net = lambda x, c=const: tf.tile(c, [tf.shape(x)[0], 1])
            prob = model(inputs, training=False).numpy().ravel()
            probs_list.append(prob)
            for li, layer in enumerate(bca.bca_layers):
                em = layer.last_entropy_m.numpy()
                ef = layer.last_entropy_f.numpy()
                records.append({
                    "layer": li,
                    "target_alpha1": float(a1),
                    "target_alpha2": float(1.0 - a1),
                    "real_alpha1": float(layer.last_alpha1.numpy().mean()),
                    "entropy_m": float(em.mean()),
                    "entropy_f": float(ef.mean()),
                    "entropy_m_std": float(em.std()),
                    "entropy_f_std": float(ef.std()),
                    "prob_mean": float(prob.mean()),
                })
    finally:
        for layer, net in zip(bca.bca_layers, originals):
            layer.alpha_net = net
    return pd.DataFrame(records), np.vstack(probs_list)


def _add_dual_axis(ax, top_label=r"$\alpha_2 = 1 - \alpha_1$"):
    sec = ax.secondary_xaxis("top", functions=(lambda v: 1.0 - v, lambda v: 1.0 - v))
    sec.set_xlabel(top_label, fontsize=8)


def _fit_line(ax, x, y, color):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, k * xs + b, "-", color=color, lw=1.2, alpha=0.9)


def fig_balance_overview(df_both, df_none, dose_both, dose_none, stats, path, probs_both):
    """机制总览 2x3：观测(α 分布/相关) -> 干预(强制 α) -> 结果(输出)。"""
    c_m, c_f, c_n = "#2E5EAA", "#7FA6D8", "#B0413E"
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2))
    s1 = stats["both"]["layer1_real"]

    # (a) 学到的 α 分布
    ax = axes[0, 0]
    a1 = df_both["alpha1"].values
    ax.hist(a1, bins=18, color=c_m, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(0.5, color="k", ls="--", lw=0.9, label="Neutral value 0.5")
    ax.set_xlabel(r"Balance factor $\alpha_1$")
    ax.set_ylabel("Number of test samples")
    ax.set_title(r"(a) Distribution of learned $\alpha_1$")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.03, 0.95, "mean = %.3f\nSD = %.3f" % (a1.mean(), a1.std()),
            transform=ax.transAxes, va="top", fontsize=8)

    # (b) 观测证据：MFCC 分支熵 vs 学到的 α1
    ax = axes[0, 1]
    xb, yb = df_both["alpha1_l1"], df_both["entropy_m_l1"]
    ax.scatter(xb, yb, s=9, c=c_m, alpha=0.55, label="Test samples")
    _fit_line(ax, xb, yb, c_m)
    ax.set_xlabel(r"Learned balance factor $\alpha_1$")
    ax.set_ylabel("Attention entropy (layer 1)")
    ax.set_title(r"(b) MFCC branch vs learned $\alpha_1$")
    ax.text(0.04, 0.06, f"Spearman \u03c1: {s1['rho_hm_alpha1']:.2f}" + "\n" + r"$p < 0.001$",
            transform=ax.transAxes, va="bottom", fontsize=10)

    # (c) 观测证据：F0 分支熵 vs 学到的 α2
    ax = axes[0, 2]
    xc, yc = df_both["alpha2_l1"], df_both["entropy_f_l1"]
    ax.scatter(xc, yc, s=9, c=c_f, alpha=0.7, marker="s", label="Test samples")
    _fit_line(ax, xc, yc, c_f)
    ax.set_xlabel(r"Learned balance factor $\alpha_2$")
    ax.set_ylabel("Attention entropy (layer 1)")
    ax.set_title(r"(c) F0 branch vs learned $\alpha_2$")
    ax.text(0.04, 0.06, f"Spearman \u03c1: {s1['rho_hf_alpha2']:.2f}" + "\n" + r"$p < 0.001$",
            transform=ax.transAxes, va="bottom", fontsize=10)

    b = dose_both[dose_both["layer"] == 0]
    n = dose_none[dose_none["layer"] == 0]

    # (d) 干预：MFCC 分支随强制 α1（阴影为均值的 95% CI）
    ax = axes[1, 0]
    n_s = probs_both.shape[1]
    ci_m = 1.96 * b["entropy_m_std"] / np.sqrt(n_s)
    ax.fill_between(b["target_alpha1"], b["entropy_m"] - ci_m,
                    b["entropy_m"] + ci_m, color=c_m, alpha=0.22, lw=0)
    ax.plot(b["target_alpha1"], b["entropy_m"], "o-", color=c_m, ms=3.5, label="Adaptive \u03b1")
    ax.plot(n["target_alpha1"], n["entropy_m"], "-", color=c_n, lw=1.0, label="Fixed \u03b1 = 0.5")
    ax.set_xlabel(r"Forced balance factor $\alpha_1$")
    ax.set_ylabel("Attention entropy (layer 1)")
    ax.set_title(r"(d) MFCC branch under forced $\alpha_1$")
    ax.legend(frameon=False)

    # (e) 干预：F0 分支随强制 α2（横轴为 α2，单独一条轴）
    ax = axes[1, 1]
    bs = b.sort_values("target_alpha2")
    ns = n.sort_values("target_alpha2")
    ci_f = 1.96 * bs["entropy_f_std"] / np.sqrt(n_s)
    ax.fill_between(bs["target_alpha2"], bs["entropy_f"] - ci_f,
                    bs["entropy_f"] + ci_f, color=c_f, alpha=0.25, lw=0)
    ax.plot(bs["target_alpha2"], bs["entropy_f"], "s-", color=c_f, ms=3.5, label="Adaptive \u03b1")
    ax.plot(ns["target_alpha2"], ns["entropy_f"], "-", color=c_n, lw=1.0, label="Fixed \u03b1 = 0.5")
    ax.set_xlabel(r"Forced balance factor $\alpha_2$")
    ax.set_ylabel("Attention entropy (layer 1)")
    ax.set_title(r"(e) F0 branch under forced $\alpha_2$")
    ax.legend(frameon=False)

    # (f) 结果：相对 α1=0.5 的配对概率变化（95% CI）
    grid = b["target_alpha1"].values
    base = np.array([np.interp(0.5, grid, probs_both[:, i]) for i in range(n_s)])
    delta = probs_both - base[None, :]
    d_mean = delta.mean(axis=1)
    d_ci = 1.96 * delta.std(axis=1, ddof=1) / np.sqrt(n_s)

    ax = axes[1, 2]
    ax.fill_between(grid, d_mean - d_ci, d_mean + d_ci, color=c_m, alpha=0.25, lw=0)
    ax.plot(grid, d_mean, "o-", color=c_m, ms=3.5, label="Adaptive \u03b1")
    ax.axhline(0.0, color=c_n, lw=1.0, label="Fixed \u03b1 = 0.5")
    ax.set_xlabel(r"Forced balance factor $\alpha_1$")
    ax.set_ylabel(r"$\Delta$ predicted probability vs $\alpha_1 = 0.5$")
    ax.set_title("(f) Paired effect on decision output")
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRIC_DIR, exist_ok=True)
    inputs, y = build_test_inputs()

    model_both, bca_both = load_model("checkpoints/alpha_ablation/alpha_full_both/final", "both")
    model_none, bca_none = load_model("checkpoints/alpha_ablation/alpha_full_none/final", "none")

    df_both = collect(model_both, bca_both, inputs, y)
    df_none = collect(model_none, bca_none, inputs, y)
    df_both.to_csv(f"{METRIC_DIR}/samples_both.csv", index=False)
    df_none.to_csv(f"{METRIC_DIR}/samples_none.csv", index=False)

    dose_both, probs_both = dose_response(model_both, bca_both, inputs)
    dose_none, _ = dose_response(model_none, bca_none, inputs)
    dose_both.to_csv(f"{METRIC_DIR}/dose_response_both.csv", index=False)
    dose_none.to_csv(f"{METRIC_DIR}/dose_response_none.csv", index=False)

    stats = {}
    for name, df in [("both", df_both), ("none", df_none)]:
        stats[name] = {}
        for key in ["entropy_m", "entropy_f"]:
            rho, p = spearmanr(df["alpha1"], df[key])
            stats[name][key] = {"rho": float(rho), "p": float(p)}
        stats[name]["alpha_std"] = float(df["alpha1"].std())
        stats[name]["alpha_min"] = float(df["alpha1"].min())
        stats[name]["alpha_max"] = float(df["alpha1"].max())
        stats[name]["hm_hf_pearson"] = float(pearsonr(df["entropy_m"], df["entropy_f"])[0])
        # 第一层独立验证：用模型自己学到的 α 与第一层熵做相关（非强制扫描）
        stats[name]["layer1_real"] = {}
        rho, p = spearmanr(df["alpha1_l1"], df["entropy_m_l1"])
        stats[name]["layer1_real"]["rho_hm_alpha1"] = float(rho)
        stats[name]["layer1_real"]["p_hm_alpha1"] = float(p)
        rho, p = spearmanr(df["alpha2_l1"], df["entropy_f_l1"])
        stats[name]["layer1_real"]["rho_hf_alpha2"] = float(rho)
        stats[name]["layer1_real"]["p_hf_alpha2"] = float(p)

    for name, dose in [("both", dose_both), ("none", dose_none)]:
        stats[name]["dose"] = {}
        for li in sorted(dose["layer"].unique()):
            d = dose[dose["layer"] == li]
            stats[name]["dose"][f"layer{li}"] = {
                "entropy_m_range": float(d["entropy_m"].max() - d["entropy_m"].min()),
                "entropy_f_range": float(d["entropy_f"].max() - d["entropy_f"].min()),
                "prob_range": float(d["prob_mean"].max() - d["prob_mean"].min()),
                "rho_hm_alpha1": float(spearmanr(d["target_alpha1"], d["entropy_m"])[0]),
                "rho_hf_alpha1": float(spearmanr(d["target_alpha1"], d["entropy_f"])[0]),
            }

    # 配对效应：α1=0.15 相对中性点 α1=0.5 的输出概率变化（同一批样本）
    grid = ALPHA_GRID
    n_s = probs_both.shape[1]
    base = np.array([np.interp(0.5, grid, probs_both[:, i]) for i in range(n_s)])
    i0 = 0
    d = probs_both[i0] - base
    _, p_pair = ttest_rel(probs_both[i0], base)
    stats["both"]["paired_effect"] = {
        "alpha1": float(grid[i0]),
        "delta_mean": float(d.mean()),
        "delta_ci95_low": float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(n_s)),
        "delta_ci95_high": float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(n_s)),
        "p_value": float(p_pair),
    }

    with open(f"{METRIC_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    fig_balance_overview(df_both, df_none, dose_both, dose_none, stats,
                         f"{FIG_DIR}/fig3_dose_response.png", probs_both)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("BALANCE_PROBE_DONE")


if __name__ == "__main__":
    main()
