"""音频级 SNR 退化实验：对原始测试音频加噪后重新提取 MFCC/F0，再评估 both / none。

与特征级版本的差别：
    - 噪声加在音频波形上，特征由加噪后的音频重新提取，α 的输入统计量因此是
      "真实退化" 的结果，而不是被人为扰动的特征；
    - 由于测试集文件名都是 _augN（增强样本随机生成、无法复现），这里统一改用
      PC-GITA 原始录音，clean 基线也在同一批原始音频上重算，因此数值与主表不可直接比较。

用法：
    python tools/analysis/snr_audio_eval.py

产物：
    results/metrics/snr_audio/metrics.csv, summary.json, features_*.npz
    results/figures/attention/balance/snr_audio_alpha.png

说明：性能曲线（AUC/Acc 随 SNR）经检验无鲁棒性优势，已不再出图；
      本脚本只保留 α 对分支信息量的响应证据。
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

TEST_CSV = "data/processed_features/test_data.csv"
TRAIN_CSV = "data/processed_features/train_data.csv"
AUDIO_ROOT = os.environ.get("PCGITA_AUDIO_DIR", os.path.join("data", "raw", "PC-GITA", "All"))
METRIC_DIR = "results/metrics/snr_audio"
FIG_DIR = "results/figures/attention/balance"
SNR_LEVELS = [20, 10, 5, 0]
SR = 16000


def _strip_aug(name):
    base = name.rsplit(".", 1)[0]
    if "_aug" in base:
        base = base.rsplit("_aug", 1)[0]
    return base


def build_audio_index():
    idx = {}
    for root, _, files in os.walk(AUDIO_ROOT):
        for f in files:
            if f.lower().endswith(".wav"):
                idx[os.path.splitext(f)[0]] = os.path.join(root, f)
    return idx


def _extract_one(args):
    """工作进程：读取音频 → 加噪 → 提取 320 维特征。"""
    import librosa

    from modules.preprocess import extract_features

    wav_path, snr_db, seed = args
    y, sr = librosa.load(wav_path, sr=SR)
    y = librosa.util.normalize(y)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p_sig = float(np.mean(y ** 2)) + 1e-12
        noise = rng.normal(0.0, np.sqrt(p_sig / (10.0 ** (snr_db / 10.0))), size=y.shape)
        y = (y + noise).astype(np.float32)
    feat = extract_features(y, sr)
    return feat


def extract_all(paths, snr_db, seed_base):
    tasks = [(p, snr_db, seed_base + i) for i, p in enumerate(paths)]
    with ProcessPoolExecutor(max_workers=6) as ex:
        return list(ex.map(_extract_one, tasks, chunksize=2))


def main():
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    from attention_balance_probe import build_test_inputs  # noqa: F401  (保持同目录依赖)
    from attention_balance_probe import load_model

    df = pd.read_csv(TEST_CSV, usecols=["filename", "label"])
    idx = build_audio_index()
    stems = [_strip_aug(f) for f in df["filename"]]
    paths = [idx.get(s) for s in stems]
    ok = [i for i, p in enumerate(paths) if p is not None]
    print(f"audio found: {len(ok)}/{len(paths)}")
    df = df.iloc[ok].reset_index(drop=True)
    paths = [paths[i] for i in ok]
    y = df["label"].values.astype(int)

    # 与训练流程一致的标准化：在训练集内部划分上拟合 scaler
    from modules.train_utils import load_data

    X_tr, y_tr = load_data(TRAIN_CSV)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_idx, _ in sss.split(X_tr, y_tr):
        scaler = StandardScaler().fit(X_tr[tr_idx])

    feats = {}
    for snr in [None] + SNR_LEVELS:
        key = "clean" if snr is None else f"{snr}dB"
        arr = np.vstack(extract_all(paths, snr, seed_base=(0 if snr is None else int(snr) * 1000)))
        np.savez_compressed(f"{METRIC_DIR}/features_{key}.npz", X=arr, y=y, filename=df["filename"].values)
        feats[key] = scaler.transform(arr)
        print("extracted", key, arr.shape)

    models = {
        "both": load_model("checkpoints/alpha_ablation/alpha_full_both/final", "both"),
        "none": load_model("checkpoints/alpha_ablation/alpha_full_none/final", "none"),
    }

    rows = []
    for name, (model, bca) in models.items():
        for key, X in feats.items():
            inputs = [X[:, :160].astype(np.float32), X[:, 160:320].astype(np.float32), X[:, 320:]]
            prob = model(inputs, training=False).numpy().ravel()
            pred = (prob >= 0.5).astype(int)
            from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            last = bca.bca_layers[-1]
            first = bca.bca_layers[0]
            rows.append({
                "model": name, "condition": key,
                "acc": float(accuracy_score(y, pred)),
                "auc": float(roc_auc_score(y, prob)),
                "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
                "alpha1_mean": float(last.last_alpha1.numpy().mean()),
                "alpha2_mean": float(last.last_alpha2.numpy().mean()),
                "alpha1_std": float(last.last_alpha1.numpy().std()),
                "entropy_m_l1": float(first.last_entropy_m.numpy().mean()),
                "entropy_f_l1": float(first.last_entropy_f.numpy().mean()),
            })

    out = pd.DataFrame(rows)
    out.to_csv(f"{METRIC_DIR}/metrics.csv", index=False)
    with open(f"{METRIC_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    plot(out)
    print(out.to_string(index=False))
    print("SNR_AUDIO_DONE")


def plot(df):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "mathtext.fontset": "stix", "mathtext.default": "regular",
    })
    order = ["clean", "20dB", "10dB", "5dB", "0dB"]
    x = np.arange(len(order))

    c_b, c_n = "#2E5EAA", "#B0413E"
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    d = df[df["model"] == "both"].set_index("condition").loc[order]
    ax.plot(x, d["alpha1_mean"], "o-", color=c_b, ms=3.5, label=r"$\alpha_1$ (MFCC)")
    ax.plot(x, d["alpha2_mean"], "s-", color="#7FA6D8", ms=3.5, label=r"$\alpha_2$ (F0)")
    ax.axhline(0.5, color=c_n, ls="--", lw=0.9, label="Neutral 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "20", "10", "5", "0"])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean balance factor")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/snr_audio_alpha.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(METRIC_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    main()
