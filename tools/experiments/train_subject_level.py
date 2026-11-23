# -*- coding: utf-8 -*-
"""
按受试者划分的完整流程：
原始音频 -> 受试者级 5 折 -> 仅训练集增强 -> MFCC+F0 特征 -> HABIFN 训练/评估。

与 train.py 的区别：
1. 先按 subject 分组划分，保证同一个人的所有录音只出现在一个折；
2. 增强只发生在训练折，验证/测试折只用原始录音；
3. 特征提取不再使用 librosa.load / pyin（本机 libsndfile 不可用且 pyin 很慢），
   改为 scipy 读取 + numpy MFCC + Praat Pitch F0，速度快且可复现。

用法（/a/ 持续元音，推荐）：
    python train_subject_level.py --data_path "data\\raw\\PC-GITA\\test" --task a
全部任务：
    python train_subject_level.py --data_path "data\\raw\\PC-GITA\\All"
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.io import wavfile
from scipy.signal import resample_poly, savgol_filter
from scipy.fftpack import dct
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from tqdm import tqdm

import sys

import keras
import tensorflow as tf

# 脚本位于 tools/experiments/，把工程根目录加入导入路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.model import build_model
from modules.preprocess import extract_features as librosa_extract_features


# ----------------------------- 基础工具 -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def scan_audio(root, task_filter=None):
    root = Path(root)
    rows = []
    for dirpath, _, files in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        parts = [p.lower() for p in rel.parts]
        if "hc" in parts:
            label = 0
        elif "pd" in parts:
            label = 1
        else:
            continue
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            base = Path(f).stem
            m = re.match(r"([A-Za-z]+\d+)", base)
            if not m:
                continue
            t = re.search(r"([a-z])\d+$", base)
            task = t.group(1) if t else ""
            if task_filter and task_filter != "all" and task != task_filter:
                continue
            rows.append({
                "filename": f,
                "filepath": str(Path(dirpath) / f),
                "subject": m.group(1),
                "label": label,
                "task": task,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["filepath"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"没有扫描到音频：{root}")
    print(f"扫描到 {len(df)} 条录音，{df['subject'].nunique()} 个受试者，"
          f"HC={int((df.label == 0).sum())} / PD={int((df.label == 1).sum())}")
    return df


def load16k(path):
    sr, y = wavfile.read(path)
    y = y.astype(np.float32)
    mx = np.abs(y).max()
    if mx > 1:
        y = y / mx
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = resample_poly(y, 16000, sr).astype(np.float32)
    return y, 16000


def augment_audio(y, sr, n_aug=2):
    """训练集增强：噪声 + 音高偏移；不再使用 time_stretch=1.0 这种无效增强。"""
    import librosa

    out = [y.astype(np.float32)]
    for i in range(n_aug):
        z = y.copy().astype(np.float32)
        try:
            if i % 2 == 0:
                z = z + np.random.normal(0, 0.006, len(z)).astype(np.float32)
            else:
                z = librosa.effects.pitch_shift(z, sr=sr, n_steps=1).astype(np.float32)
            z = np.clip(z, -1.0, 1.0)
            if len(z) < len(y):
                z = np.pad(z, (0, len(y) - len(z)))
            out.append(z[:len(y)])
        except Exception as e:
            print(f"增强失败，跳过：{e}")
    return out


# ----------------------------- 快速 MFCC+F0 -----------------------------
def mel_filterbank(sr, n_fft, n_mels=40, fmin=20, fmax=8000):
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def mfcc_feat(y, sr, n_mfcc=40, frame_ms=25, hop_ms=10, n_fft=512, n_mels=40):
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    n_frames = 1 + (len(y) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hamming(frame_len)[None, :]
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    fb = mel_filterbank(sr, n_fft, n_mels=n_mels)
    logmel = np.log(power @ fb.T + 1e-10)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :n_mfcc].T
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1)
    if width >= 3:
        d1 = savgol_filter(mfcc, width, 1, axis=1)
        d2 = savgol_filter(mfcc, width, 2, axis=1)
    else:
        d1 = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
        d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), d1.mean(axis=1), d2.mean(axis=1)])


def f0_feat(y, sr):
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float32)
    if np.any(f0 > 0):
        good = f0 > 0
        idx = np.arange(len(f0))
        contour = np.interp(np.linspace(0, len(f0) - 1, 40), idx[good], f0[good])
    else:
        contour = np.zeros(40, dtype=np.float32)
    d1 = np.diff(contour, prepend=contour[0])
    d2 = np.diff(d1, prepend=d1[0])
    voiced = (contour > 0).astype(np.float32)
    return np.concatenate([contour, d1, d2, voiced])


def extract_one(y, sr):
    return np.concatenate([mfcc_feat(y, sr), f0_feat(y, sr)]).astype(np.float32)


# ----------------------------- 特征缓存 -----------------------------
def build_features(df, out_dir, n_aug=2, reuse=False, backend="librosa"):
    cache = os.path.join(out_dir, "subject_features.csv")
    if reuse and os.path.exists(cache):
        print("复用特征缓存:", cache)
        return pd.read_csv(cache)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        try:
            y, sr = load16k(r["filepath"])
            for aug, z in enumerate(augment_audio(y, sr, n_aug=n_aug)):
                if backend == "librosa":
                    f = librosa_extract_features(z, sr)
                else:
                    f = extract_one(z, sr)
                if f is None:
                    continue
                row = {
                    "filename": r["filename"], "subject": r["subject"],
                    "label": int(r["label"]), "task": r["task"], "aug": aug,
                }
                row.update({f"feat_{i}": float(v) for i, v in enumerate(f)})
                rows.append(row)
        except Exception as e:
            print("跳过", r["filepath"], e)
    feat = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    feat.to_csv(cache, index=False, encoding="utf-8-sig")
    print("特征已保存:", cache, feat.shape)
    return feat


# ----------------------------- 受试者级划分 -----------------------------
def make_folds(df, seed=42):
    sub = df[["subject", "label"]].drop_duplicates().reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(sub, sub["label"], groups=sub["subject"])):
        tr_sub = sub.iloc[tr_idx]
        te_sub = sub.iloc[te_idx]
        rng = np.random.RandomState(seed + fold)
        val_subjects = []
        for lab in sorted(tr_sub["label"].unique()):
            s = tr_sub.loc[tr_sub["label"] == lab, "subject"].tolist()
            rng.shuffle(s)
            n_val = max(1, int(round(len(s) * 0.15)))
            val_subjects.extend(s[:n_val])
        val_set = set(val_subjects)
        train_subjects = [s for s in tr_sub["subject"].tolist() if s not in val_set]
        test_subjects = te_sub["subject"].tolist()
        assert not (set(train_subjects) & set(test_subjects))
        assert not (set(val_subjects) & set(test_subjects))
        folds.append({
            "fold": fold,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })
    return folds


# ----------------------------- 训练 -----------------------------
def set_epoch_recursive(layer, epoch):
    if hasattr(layer, "set_current_epoch"):
        try:
            layer.set_current_epoch(epoch)
        except Exception:
            pass
    for sub in getattr(layer, "layers", []):
        set_epoch_recursive(sub, epoch)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "acc": accuracy_score(y, pred),
        "auc": roc_auc_score(y, p),
        "sen": tp / (tp + fn) if (tp + fn) else 0.0,
        "spe": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y, pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--task", default="a", help="a/e/i/o/u 或 all，默认 a")
    ap.add_argument("--out_dir", default=os.path.join("results", "subject_level"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_aug", type=int, default=2, help="训练集每条录音的增强数量")
    ap.add_argument("--backend", choices=["librosa", "praat"], default="librosa",
                    help="librosa: 调用主流程 extract_features（pyin，论文一致但慢）；praat: 快速实现")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse_features", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    df = scan_audio(args.data_path, args.task)
    feat = build_features(df, args.out_dir, n_aug=args.n_aug, reuse=args.reuse_features, backend=args.backend)
    cols = [c for c in feat.columns if c.startswith("feat_")]

    folds = make_folds(df, seed=args.seed)
    with open(os.path.join(args.out_dir, "folds.json"), "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)

    all_rows = []
    for fold_info in folds:
        k = fold_info["fold"]
        tr = feat[feat["subject"].isin(fold_info["train_subjects"])].copy()
        va = feat[(feat["subject"].isin(fold_info["val_subjects"])) & (feat["aug"] == 0)].copy()
        te = feat[(feat["subject"].isin(fold_info["test_subjects"])) & (feat["aug"] == 0)].copy()
        print(f"\n===== fold {k}: train {len(tr)} / val {len(va)} / test {len(te)} =====")

        scaler = StandardScaler().fit(tr[cols].values)
        Xtr, Xva, Xte = (scaler.transform(x[cols].values) for x in (tr, va, te))
        ytr, yva, yte = tr["label"].values, va["label"].values, te["label"].values

        model, _ = build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(args.lr),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        classes = np.unique(ytr)
        cw = compute_class_weight("balanced", classes=classes, y=ytr)
        class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

        class EpochSetter(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                set_epoch_recursive(self.model, epoch)

        callbacks = [
            EpochSetter(),
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=20, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
        ]
        model.fit(
            [Xtr[:, :160], Xtr[:, 160:320], Xtr[:, 320:]],
            ytr,
            validation_data=([Xva[:, :160], Xva[:, 160:320], Xva[:, 320:]], yva),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )
        p = model.predict([Xte[:, :160], Xte[:, 160:320], Xte[:, 320:]], verbose=0).ravel()
        m = metrics(yte, p)
        m.update({"fold": k, "n_train": len(tr), "n_val": len(va), "n_test": len(te)})
        all_rows.append(m)
        pd.DataFrame({
            "filename": te["filename"].values,
            "subject": te["subject"].values,
            "task": te["task"].values,
            "label": yte,
            "prob": p,
        }).to_csv(os.path.join(args.out_dir, f"predictions_fold{k}.csv"), index=False)
        print(f"fold {k}: {m}")

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(args.out_dir, "subject_level_summary.csv"), index=False)
    print("\n===== 每折 =====")
    print(res.round(4).to_string(index=False))
    print("\n===== 均值 ± 标准差 =====")
    for c in ["acc", "auc", "sen", "spe", "f1"]:
        print(f"{c}: {res[c].mean():.4f} ± {res[c].std(ddof=1):.4f}")
    print("\n结果目录:", args.out_dir)


if __name__ == "__main__":
    main()
