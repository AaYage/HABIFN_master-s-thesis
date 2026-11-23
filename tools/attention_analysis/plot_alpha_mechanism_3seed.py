"""正式图：平衡因子的可用性感知门控（3 seed 聚合）。

四个面板：
    (a) MFCC 衰减 vs Δα1（相对基线，跨 3 seed 的 mean ± SD，细线为单 seed）
    (b) F0 衰减 vs Δα2（同上）
    (c) 衰减 vs AUC（adaptive / fixed，两条分支，mean ± SD）
    (d) 各条件下 AUC 差值（adaptive − fixed，含干净集与退化集）

输出：
    results/figures/attention/balance/fig_alpha_mechanism.png
    results/figures/attention/balance/fig_alpha_mechanism_caption.txt
    results/metrics/gate_eval/mechanism_3seed.csv
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

from attention_balance_probe import build_test_inputs, load_model  # noqa: E402

OUT_DIR = "results/metrics/gate_eval"
FIG_DIR = "results/figures/attention/balance"
CKPT = "checkpoints/alpha_ablation"
SCALES = (1.0, 0.7, 0.4, 0.1)
TAGS = {
    42: {"adaptive": "alpha_full_both_gatemask", "fixed": "alpha_full_none_gatemask"},
    43: {"adaptive": "alpha_full_both_gatemask_s43", "fixed": "alpha_full_none_gatemask_s43"},
    44: {"adaptive": "alpha_full_both_gatemask_s44", "fixed": "alpha_full_none_gatemask_s44"},
}


def main():
    X, y = build_test_inputs()
    rows = []
    for seed, tags in TAGS.items():
        for name, tag in tags.items():
            model, bca = load_model(os.path.join(CKPT, tag, "final"),
                                    "both" if name == "adaptive" else "none")
            for branch, idx in (("mfcc", 0), ("f0", 1)):
                for s in SCALES:
                    inp = [X[0], X[1], X[2]]
                    inp[idx] = X[idx] * s
                    prob = model(inp, training=False).numpy().ravel()
                    row = {"seed": seed, "model": name, "branch": branch, "scale": s,
                           "acc": accuracy_score(y, prob >= 0.5),
                           "auc": roc_auc_score(y, prob)}
                    if name == "adaptive":
                        a = bca.bca_layers[-1]
                        row["alpha1"] = float(a.last_alpha1.numpy().mean())
                        row["alpha2"] = float(a.last_alpha2.numpy().mean())
                    rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/mechanism_3seed.csv", index=False)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "mathtext.fontset": "stix", "mathtext.default": "regular",
    })
    c_m, c_f, c_ctrl = "#1B3B6F", "#7FA6D8", "#B0413E"
    c_none = "#9A9A9A"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    axes = axes.ravel()

    # (a)(b) α 剂量-反应（相对未衰减基线，消除 seed 间基线差异）
    for ax, branch, col, color, tag in zip(
            axes[:2], ("mfcc", "f0"), ("alpha1", "alpha2"), (c_m, c_f), ("(a)", "(b)")):
        d = df[(df["model"] == "adaptive") & (df["branch"] == branch)].copy()
        base = d[d["scale"] == 1.0].set_index("seed")[col]
        d["delta"] = d[col] - d["seed"].map(base)
        styles = (0, (1, 1)), (0, (3, 1.2)), (0, (5, 1.5))
        for (seed, g), ls in zip(d.groupby("seed"), styles):
            ax.plot(g["scale"], g["delta"], ls=ls, color=color, alpha=0.6, lw=0.9)
        mean = d.groupby("scale")["delta"].mean()
        sd = d.groupby("scale")["delta"].std(ddof=1)
        xs = mean.index.values
        ax.fill_between(xs, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)
        ax.plot(xs, mean.values, "o-", color=color, ms=3.6, lw=1.1)
        ax.axhline(0.0, color=c_ctrl, ls="--", lw=0.9)
        ax.set_xlim(1.08, 0.02)
        ax.set_xlabel(f"{branch.upper()} attenuation factor $s$")
        ax.set_ylabel(r"$\Delta\alpha_1$" if col == "alpha1" else r"$\Delta\alpha_2$")
        ax.set_title(f"{tag} {'MFCC' if branch == 'mfcc' else 'F0'} availability")

    # (c) AUC vs 衰减
    ax = axes[2]
    for branch, color, mk in (("mfcc", c_m, "o"), ("f0", c_f, "s")):
        for name, ls in (("adaptive", "-"), ("fixed", "--")):
            d = df[(df["model"] == name) & (df["branch"] == branch)]
            mean = d.groupby("scale")["auc"].mean()
            sd = d.groupby("scale")["auc"].std(ddof=1)
            ax.errorbar(mean.index.values, mean.values, yerr=sd.values,
                        marker=mk, ls=ls, color=color, ms=3.0, lw=1.0, capsize=1.8,
                        mfc=color if name == "adaptive" else "white",
                        label=f"{name}, {branch.upper()}")
    ax.set_xlim(1.08, 0.02)
    ax.set_xlabel("Attenuation factor $s$")
    ax.set_ylabel("AUC")
    ax.set_title("(c) Classification performance")
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")

    # (d) 各条件下 AUC 差值（adaptive − fixed）
    ax = axes[3]
    piv = df.pivot_table(index=["seed", "branch", "scale"], columns="model",
                         values="auc").reset_index()
    piv["delta"] = piv["adaptive"] - piv["fixed"]
    conds = (("clean", "mfcc", 1.0), ("MFCC 0.4", "mfcc", 0.4), ("MFCC 0.1", "mfcc", 0.1),
             ("F0 0.4", "f0", 0.4), ("F0 0.1", "f0", 0.1))
    for i, (_, branch, s) in enumerate(conds):
        vals = piv[(piv["branch"] == branch) & (piv["scale"] == s)]["delta"].values
        ax.plot(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals,
                "o", color=c_m, ms=3.2, alpha=0.75)
        ax.errorbar([i], [vals.mean()], yerr=[vals.std(ddof=1)], fmt="s", color=c_ctrl,
                    ms=4.2, capsize=2.2, lw=1.0)
    ax.axhline(0.0, color="#666666", ls="--", lw=0.9)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c[0] for c in conds], fontsize=7, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylabel(r"$\Delta$AUC (adaptive $-$ fixed)")
    ax.set_title("(d) Performance difference")
    ax.set_xlabel("Test condition")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_alpha_mechanism.png", dpi=600)
    plt.close(fig)

    caption = (
        "Availability-aware gating of the balance factor. (a, b) Change in alpha1 and alpha2 "
        "relative to the unattenuated branch as the MFCC and F0 branches are attenuated by a "
        "factor s, averaged over three random seeds; shaded regions denote SD across seeds and "
        "thin dashed lines show individual seeds. (c) Test AUC under the same attenuation for "
        "adaptive (solid) and fixed-weight (dashed) fusion; error bars denote SD across seeds. "
        "(d) Difference in test AUC between the two schemes for the clean test set and four "
        "attenuation conditions; dots are individual seeds, squares with whiskers denote mean "
        "and SD, and the dashed line marks zero. The red dashed line in (a, b) marks no change."
    )
    with open(f"{FIG_DIR}/fig_alpha_mechanism_caption.txt", "w", encoding="ascii") as fh:
        fh.write(caption + "\n")

    print(df.groupby(["model", "branch", "scale"])[["auc", "acc"]].mean().round(4).to_string())
    print("ALPHA_MECHANISM_DONE")


if __name__ == "__main__":
    main()
