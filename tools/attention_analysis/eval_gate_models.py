"""评估 α 门控融合（gate）版本：掩蔽扰动 + 音频级 SNR。

对比：
    both_drop / none_drop —— 分支丢弃训练 + 残差连接（α 只调温度）
    both_gate / none_gate —— 分支丢弃训练 + α 直接控制自身/交互融合比例

输出：
    results/metrics/gate_eval/metrics_masking.csv
    results/metrics/gate_eval/metrics_snr.csv
    results/figures/attention/balance/gate_masking_preview.png
    results/figures/attention/balance/gate_snr_preview.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from attention_balance_probe import build_test_inputs, load_model  # noqa: E402
from modules.train_utils import load_data  # noqa: E402

OUT_DIR = "results/metrics/gate_eval"
FIG_DIR = "results/figures/attention/balance"
SNR_DIR = "results/metrics/snr_audio"
CONDITIONS = ["clean", "20dB", "10dB", "5dB", "0dB"]
MODELS = {
    "both_drop": "checkpoints/alpha_ablation/alpha_full_both_drop/final",
    "none_drop": "checkpoints/alpha_ablation/alpha_full_none_drop/final",
    "both_gate": "checkpoints/alpha_ablation/alpha_full_both_gate/final",
    "none_gate": "checkpoints/alpha_ablation/alpha_full_none_gate/final",
    "both_resmask": "checkpoints/alpha_ablation/alpha_full_both_resmask/final",
    "none_resmask": "checkpoints/alpha_ablation/alpha_full_none_resmask/final",
    "both_gatemask": "checkpoints/alpha_ablation/alpha_full_both_gatemask/final",
    "none_gatemask": "checkpoints/alpha_ablation/alpha_full_none_gatemask/final",
    "both_resmask2": "checkpoints/alpha_ablation/alpha_full_both_resmask2/final",
    "none_resmask2": "checkpoints/alpha_ablation/alpha_full_none_resmask2/final",
}
COLORS = {
    "both_drop": "#8FA8C8", "none_drop": "#C9A0A0",
    "both_gate": "#1B3B6F", "none_gate": "#B0413E",
    "both_resmask": "#2E6F5E", "none_resmask": "#E08A5A",
    "both_gatemask": "#203864", "none_gatemask": "#7A9CC6",
    "both_resmask2": "#1F6F5C", "none_resmask2": "#8FBFB0",
}


def masked_inputs(X, branch, ratio):
    idx = 0 if branch == "mfcc" else 1
    x = X[idx].copy()
    k = int(round(ratio * x.shape[1]))
    if k > 0:
        cols = np.random.default_rng((11 if branch == "mfcc" else 23) + int(ratio * 100)) \
            .choice(x.shape[1], size=k, replace=False)
        x[:, cols] = 0.0
    out = [X[0], X[1], X[2]]
    out[idx] = x
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sel = sys.argv[1:] or list(MODELS)
    loaded = {n: load_model(MODELS[n], "none" if n.startswith("none") else "both")
              for n in sel}
    X, y = build_test_inputs()

    rows = []
    for name, (model, bca) in loaded.items():
        last = bca.bca_layers[-1]
        for branch in ("mfcc", "f0"):
            for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
                prob = model(masked_inputs(X, branch, ratio), training=False).numpy().ravel()
                rows.append({"model": name, "branch": branch, "ratio": ratio,
                             "acc": accuracy_score(y, (prob >= 0.5).astype(int)),
                             "auc": roc_auc_score(y, prob),
                             "alpha1": float(last.last_alpha1.numpy().mean()),
                             "alpha2": float(last.last_alpha2.numpy().mean())})
    mask_df = pd.DataFrame(rows)
    mask_df.to_csv(f"{OUT_DIR}/metrics_masking.csv", index=False)

    X_tr, y_tr = load_data("data/processed_features/train_data.csv")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for idx, _ in sss.split(X_tr, y_tr):
        scaler = StandardScaler().fit(X_tr[idx])
    srows = []
    for cond in CONDITIONS:
        d = np.load(f"{SNR_DIR}/features_{cond}.npz")
        xs = scaler.transform(d["X"])
        inputs = [xs[:, :160].astype(np.float32), xs[:, 160:320].astype(np.float32), xs[:, 320:]]
        for name, (model, bca) in loaded.items():
            prob = model(inputs, training=False).numpy().ravel()
            last = bca.bca_layers[-1]
            srows.append({"condition": cond, "model": name,
                          "acc": accuracy_score(d["y"], (prob >= 0.5).astype(int)),
                          "auc": roc_auc_score(d["y"], prob),
                          "alpha2": float(last.last_alpha2.numpy().mean())})
    snr_df = pd.DataFrame(srows)
    snr_df.to_csv(f"{OUT_DIR}/metrics_snr.csv", index=False)

    plot(mask_df, snr_df)
    print(mask_df.round(4).to_string(index=False))
    print(snr_df.round(4).to_string(index=False))
    print("GATE_EVAL_DONE")


def plot(mask_df, snr_df):
    names = list(mask_df["model"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.8))
    for ax, branch, col, title in zip(axes[0], ("mfcc", "f0", "mfcc"),
                                      ("alpha1", "alpha2", "auc"),
                                      ("(a) MFCC masking", "(b) F0 masking", "(c) AUC")):
        for name in names:
            d = mask_df[(mask_df["model"] == name) & (mask_df["branch"] == branch)]
            ls = "-" if name.startswith("both") else "--"
            ax.plot(d["ratio"], d[col], "o" + ls, ms=3.0, lw=1.0,
                    color=COLORS.get(name), label=name)
        if col != "auc":
            ax.axhline(0.5, color="#B0413E", ls=":", lw=0.8)
        ax.set_xlabel(f"{branch.upper()} mask ratio" if col != "auc" else "Mask ratio")
        ax.set_ylabel(r"Mean $\alpha$ of masked branch" if col != "auc" else "AUC")
        ax.set_title(title)
    axes[0, 0].legend(frameon=False, fontsize=6.0)

    for ax, col, title in zip(axes[1, :2], ("alpha2", "auc"),
                              ("(d) SNR vs alpha2", "(e) SNR vs AUC")):
        for name in names:
            d = snr_df[snr_df["model"] == name]
            ls = "-" if name.startswith("both") else "--"
            ax.plot(range(len(CONDITIONS)), d[col], "o" + ls, ms=3.0, lw=1.0,
                    color=COLORS.get(name), label=name)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(["clean", "20", "10", "5", "0"])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(r"Mean $\alpha_2$" if col == "alpha2" else "AUC")
        ax.set_title(title)
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/gate_masking_preview.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
