# -*- coding: utf-8 -*-
"""
受试者级评估辅助脚本（不改动主训练流程，不覆盖原有结果）。

目的：
  1. 先按受试者分组，再做特征提取与增强，避免同一受试者的录音跨 train/test；
  2. 给出 kNN 基线，快速观察“样本级 -> 受试者级”准确率会掉多少；
  3. 可选按折训练当前 HABIFN 模型，输出每折与均值±标准差。

用法：
  只生成受试者级特征和折文件：
    python subject_level_eval.py --data_path "D:\\path\\to\\audio" --mode prepare

  生成 + kNN 基线：
    python subject_level_eval.py --data_path "D:\\path\\to\\audio" --mode knn

  生成 + kNN + 深度模型 5 折：
    python subject_level_eval.py --data_path "D:\\path\\to\\audio" --mode all --epochs 100

  已有 features_aug.csv 时复用，不重新提取：
    python subject_level_eval.py --data_path "D:\\path\\to\\audio" --mode all --reuse

注意：
  - 需要原始音频目录，且目录名/路径中包含 PD/Parkinson 或 HC/Control/Healthy 关键字；
  - 特征提取使用 modules.preprocess.extract_features（当前 MFCC 160 + F0 160）；
  - 训练折使用原始 + 增强，验证/测试折只用原始样本；
  - 该脚本用于协议对比，训练循环做了简化，不作为论文最终结果。
"""
import argparse
import json
import os
import re
import sys

import keras

import numpy as np
import pandas as pd

# 脚本位于 tools/experiments/，工程根目录在上两级
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="原始音频根目录")
    p.add_argument("--out_dir", default=os.path.join(PROJECT_DIR, "data", "subject_level"))
    p.add_argument("--mode", choices=["prepare", "knn", "train", "all"], default="all")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.15, help="从训练受试者中划验证集的比例")
    p.add_argument("--max_augments", type=int, default=3, help="每个文件保留的版本数（含原始）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件，用于快速调试")
    p.add_argument("--reuse", action="store_true", help="复用 out_dir/features_aug.csv")
    p.add_argument("--subject_regex", default=r"([A-Za-z]+\d+)")
    p.add_argument("--pd_keywords", default="pd,parkinson")
    p.add_argument("--hc_keywords", default="hc,control,healthy")
    return p.parse_args()


def label_from_path(path, pd_kw, hc_kw):
    low = path.lower()
    if any(k in low for k in pd_kw):
        return 1
    if any(k in low for k in hc_kw):
        return 0
    return None


def scan_audio(data_path, args):
    pd_kw = [k.strip().lower() for k in args.pd_keywords.split(",") if k.strip()]
    hc_kw = [k.strip().lower() for k in args.hc_keywords.split(",") if k.strip()]
    rows = []
    for root, _, files in os.walk(data_path):
        # 只用相对目录判断标签，避免 data_path 本身含 "parkinson" 导致全部标成 PD
        rel_root = os.path.relpath(root, data_path)
        for f in files:
            if not f.lower().endswith((".wav", ".mp3")):
                continue
            path = os.path.join(root, f)
            label = label_from_path(rel_root, pd_kw, hc_kw)
            if label is None:
                continue
            m = re.match(args.subject_regex, os.path.splitext(f)[0])
            if not m:
                continue
            rows.append({"filepath": path, "filename": f, "subject": m.group(1), "label": int(label)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("没有扫描到音频：请检查 --data_path 和目录名中的 PD/HC 关键字")
    mixed = df.groupby("subject")["label"].nunique()
    if (mixed > 1).any():
        bad = mixed[mixed > 1].index.tolist()
        raise RuntimeError(f"同一受试者出现多个标签，请先核查：{bad}")
    return df


def extract_all(df, args):
    import librosa
    from tqdm import tqdm
    from modules.preprocess import apply_augmentation, extract_features

    np.random.seed(args.seed)
    rows = []
    items = df.to_dict("records")
    if args.limit:
        items = items[: args.limit]
    for r in tqdm(items, desc="提取特征"):
        try:
            y, sr = librosa.load(r["filepath"], sr=None)
            # apply_augmentation 返回“原始 + max_augments 个增强”，这里转换成总版本数
            audios = apply_augmentation(y, sr, max_augments=max(0, args.max_augments - 1))
            base = os.path.splitext(r["filename"])[0]
            for aug_idx, audio in enumerate(audios):
                feat = extract_features(audio, sr)
                if feat is None:
                    continue
                row = {
                    "filename": f"{base}_aug{aug_idx}.npz",
                    "subject": r["subject"],
                    "label": int(r["label"]),
                    "aug": int(aug_idx),
                    "source_file": r["filepath"],
                }
                row.update({f"feat_{i}": float(v) for i, v in enumerate(feat)})
                rows.append(row)
        except Exception as e:
            print(f"[跳过] {r['filepath']}: {e}")
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("特征提取失败，未生成任何样本")
    return out


def make_folds(feats, args):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    subjects = feats[["subject", "label"]].drop_duplicates().reset_index(drop=True)
    try:
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(subjects, subjects["label"], groups=subjects["subject"])
    except Exception:
        splitter = GroupKFold(n_splits=args.n_splits)
        split_iter = splitter.split(subjects, subjects["label"], groups=subjects["subject"])

    folds = []
    for k, (tr_idx, te_idx) in enumerate(split_iter):
        tr_sub = subjects.iloc[tr_idx]
        te_sub = subjects.iloc[te_idx]
        rng = np.random.RandomState(args.seed + k)
        val_subjects = []
        for lab in sorted(tr_sub["label"].unique()):
            s = tr_sub.loc[tr_sub["label"] == lab, "subject"].tolist()
            rng.shuffle(s)
            n_val = max(1, int(round(len(s) * args.val_ratio)))
            val_subjects.extend(s[:n_val])
        val_set = set(val_subjects)
        train_subjects = [s for s in tr_sub["subject"].tolist() if s not in val_set]
        test_subjects = te_sub["subject"].tolist()
        assert not (set(train_subjects) & set(test_subjects))
        assert not (set(val_subjects) & set(test_subjects))
        folds.append({
            "fold": k,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })
    return folds


def fold_frames(feats, fold):
    tr = feats[feats["subject"].isin(fold["train_subjects"])].copy()
    va = feats[(feats["subject"].isin(fold["val_subjects"])) & (feats["aug"] == 0)].copy()
    te = feats[(feats["subject"].isin(fold["test_subjects"])) & (feats["aug"] == 0)].copy()
    return tr, va, te


def metric_dict(y, p):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "acc": float(accuracy_score(y, pred)),
        "sen": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "spe": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n": int(len(y)),
    }
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    return out


def eval_knn(tr, va, te):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in tr.columns if c.startswith("feat_")]
    scaler = StandardScaler().fit(tr[cols].values)
    clf = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(tr[cols].values), tr["label"].values)
    p = clf.predict_proba(scaler.transform(te[cols].values))[:, 1]
    return metric_dict(te["label"].values, p)


def set_epoch_recursive(layer, epoch):
    if hasattr(layer, "set_current_epoch"):
        try:
            layer.set_current_epoch(epoch)
        except Exception:
            pass
    for sub in getattr(layer, "layers", []):
        set_epoch_recursive(sub, epoch)


def eval_deep(tr, va, te, args):
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from modules.model import build_model

    cols = [c for c in tr.columns if c.startswith("feat_")]
    scaler = StandardScaler().fit(tr[cols].values)
    Xtr, Xva, Xte = (scaler.transform(x[cols].values) for x in (tr, va, te))
    ytr, yva, yte = tr["label"].values, va["label"].values, te["label"].values

    model = build_model((160,), (160,), (0,))
    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    classes = np.unique(ytr)
    weights = compute_class_weight("balanced", classes=classes, y=ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

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
    return metric_dict(yte, p), p


def summarize(rows):
    df = pd.DataFrame(rows)
    print("\n===== 每折结果 =====")
    print(df.round(4).to_string(index=False))
    print("\n===== 均值 ± 标准差 =====")
    for col in ["knn_acc", "knn_auc", "deep_acc", "deep_auc", "deep_sen", "deep_spe"]:
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals):
                print(f"{col}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")
    return df


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    feat_path = os.path.join(args.out_dir, "features_aug.csv")
    folds_path = os.path.join(args.out_dir, "folds.json")

    if args.reuse and os.path.exists(feat_path):
        print(f"复用特征：{feat_path}")
        feats = pd.read_csv(feat_path)
    else:
        raw = scan_audio(args.data_path, args)
        print(f"扫描到 {len(raw)} 个文件，{raw['subject'].nunique()} 个受试者")
        print(raw.groupby("label")["subject"].nunique().rename("subjects").to_string())
        feats = extract_all(raw, args)
        feats.to_csv(feat_path, index=False)
        print(f"特征已保存：{feat_path}，形状 {feats.shape}")

    folds = make_folds(feats, args)
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
    print(f"折文件已保存：{folds_path}")

    if args.mode == "prepare":
        return

    rows = []
    for fold in folds:
        tr, va, te = fold_frames(feats, fold)
        row = {
            "fold": fold["fold"],
            "n_train_subjects": len(fold["train_subjects"]),
            "n_val_subjects": len(fold["val_subjects"]),
            "n_test_subjects": len(fold["test_subjects"]),
            "n_train_samples": len(tr),
            "n_test_samples": len(te),
        }
        if args.mode in ("knn", "all"):
            m = eval_knn(tr, va, te)
            row.update({f"knn_{k}": v for k, v in m.items()})
        if args.mode in ("train", "all"):
            m, p = eval_deep(tr, va, te, args)
            row.update({f"deep_{k}": v for k, v in m.items()})
            pred_path = os.path.join(args.out_dir, f"predictions_fold{fold['fold']}.csv")
            pd.DataFrame({"filename": te["filename"].values, "subject": te["subject"].values,
                          "label": te["label"].values, "prob": p}).to_csv(pred_path, index=False)
        rows.append(row)
        print(f"fold {fold['fold']} 完成")

    summary = summarize(rows)
    summary.to_csv(os.path.join(args.out_dir, "subject_level_summary.csv"), index=False)
    print(f"\n汇总已保存：{os.path.join(args.out_dir, 'subject_level_summary.csv')}")


if __name__ == "__main__":
    main()
