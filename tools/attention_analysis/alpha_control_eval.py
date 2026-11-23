"""α 对照实验：学习到的 α vs 固定 0.5 vs 固定均值 vs 跨样本置换。

目的：判断 α 的“样本特异性”是否携带决策相关信息。
    - learned      : 模型原始输出
    - fixed_0.5    : 所有样本 α=(0.5,0.5)
    - fixed_mean   : 所有样本 α=训练集平均 α
    - shuffled     : 把测试集样本间的 α 随机置换（20 次平均）

测试条件：clean 与 MFCC/F0 分支衰减 s=0.4、0.1。
输出：results/metrics/alpha_control/alpha_control.csv
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis")))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from attention_balance_probe import build_test_inputs, load_model
from modules.train_utils import load_data

CKPT = "checkpoints/alpha_ablation"
OUT = "results/metrics/alpha_control"
SEEDS = (42, 43, 44)
N_PERM = 20
TEST_CONDS = (("clean", None, 1.0), ("mfcc_s0.4", 0, 0.4), ("mfcc_s0.1", 0, 0.1),
              ("f0_s0.4", 1, 0.4), ("f0_s0.1", 1, 0.1))


def set_override(bca, alpha):
    for layer in bca.bca_layers:
        layer.alpha_override = alpha


def forward(model, bca, inputs, alpha=None):
    set_override(bca, alpha)
    prob = model(inputs, training=False).numpy().ravel()
    learned = None
    if alpha is None:
        last = bca.bca_layers[-1]
        learned = np.stack([last.last_alpha1.numpy(), last.last_alpha2.numpy()], axis=1)
    return prob, learned


def train_mean_alpha(model, bca):
    x_tr, y_tr = load_data("data/processed_features/train_data.csv")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, _ = next(sss.split(x_tr, y_tr))
    scaler = StandardScaler().fit(x_tr[tr_idx])
    x_tr = scaler.transform(x_tr)
    inputs = [x_tr[:, :160], x_tr[:, 160:320], x_tr[:, 320:]]
    _, learned = forward(model, bca, inputs)
    return learned.mean(axis=0)


def evaluate(prob, y):
    return accuracy_score(y, prob >= 0.5), roc_auc_score(y, prob)


def main():
    X, y = build_test_inputs()
    rows = []
    for seed in SEEDS:
        model, bca = load_model(os.path.join(CKPT, f"alpha_full_main_both_s{seed}", "final"))
        _, learned_alpha = forward(model, bca, X)
        mean_alpha = np.tile(train_mean_alpha(model, bca), (len(y), 1))
        rng = np.random.default_rng(seed)

        for cond_name, branch, s in TEST_CONDS:
            inputs = [X[0], X[1], X[2]]
            if branch is not None:
                inputs[branch] = X[branch] * s

            prob, _ = forward(model, bca, inputs)
            acc, auc = evaluate(prob, y)
            rows.append({"seed": seed, "alpha_mode": "learned", "test_cond": cond_name,
                         "acc": acc, "auc": auc, "n_perm": 1})

            prob, _ = forward(model, bca, inputs, np.full((len(y), 2), 0.5, np.float32))
            acc, auc = evaluate(prob, y)
            rows.append({"seed": seed, "alpha_mode": "fixed_0.5", "test_cond": cond_name,
                         "acc": acc, "auc": auc, "n_perm": 1})

            prob, _ = forward(model, bca, inputs, mean_alpha.astype(np.float32))
            acc, auc = evaluate(prob, y)
            rows.append({"seed": seed, "alpha_mode": "fixed_mean", "test_cond": cond_name,
                         "acc": acc, "auc": auc, "n_perm": 1})

            accs, aucs = [], []
            for _ in range(N_PERM):
                perm = rng.permutation(len(y))
                ov = learned_alpha[perm].astype(np.float32)
                prob, _ = forward(model, bca, inputs, ov)
                a, u = evaluate(prob, y)
                accs.append(a)
                aucs.append(u)
            rows.append({"seed": seed, "alpha_mode": "shuffled", "test_cond": cond_name,
                         "acc": float(np.mean(accs)), "auc": float(np.mean(aucs)),
                         "n_perm": N_PERM})

        set_override(bca, None)
        print(f"seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/alpha_control.csv", index=False)
    for cond in [c[0] for c in TEST_CONDS]:
        sub = df[df.test_cond == cond]
        print(f"\n== {cond}")
        print(sub.groupby("alpha_mode")[["acc", "auc"]].mean().round(4).to_string())
    print("ALPHA_CONTROL_DONE")


if __name__ == "__main__":
    main()
