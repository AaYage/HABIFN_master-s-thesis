# -*- coding: utf-8 -*-
"""帧级 MFCC+F0 训练脚本（APAG / BCA / MAM 在真实时间维上运行）。

特征由 ``modules.preprocess_framed`` 生成：
    data/processed_features/framed_train.npz
    data/processed_features/framed_val.npz
    data/processed_features/framed_test.npz

用法：
    python train_framed.py --full_training --tune --seed 42
    python train_framed.py                      # 5 折（按 base 分组）
"""
import os
import sys
import json
import time
import locale
import logging
import argparse

# Windows 下 Python 3.8+ 不再从 PATH 搜索 DLL，需显式注册 CUDA 库目录
_CUDA_DLL_DIR = os.environ.get("TF_CUDA_LIB_DIR", "")
if (
    sys.platform == "win32"
    and _CUDA_DLL_DIR
    and hasattr(os, "add_dll_directory")
    and os.path.isdir(_CUDA_DLL_DIR)
):
    _CUDA_DLL_HANDLE = os.add_dll_directory(_CUDA_DLL_DIR)

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Keras 2.12 起才支持 .keras 格式；TF 2.10 自带的 Keras 2.10 用 SavedModel 目录保存
_LEGACY_KERAS = tuple(int(p) for p in keras.__version__.split(".")[:2]) < (2, 12)

from modules.path_utils import safe_path
from modules.model import build_framed_model, AlphaEpochVisualizationCallback
from modules import train_utils


class ASCIIFilter(logging.Filter):
    def filter(self, record):
        for uni, asc in {"✅": "[OK]", "❌": "[ERROR]", "🔄": "[PROCESS]",
                         "🏋️": "[TRAIN]", "🎉": "[SUCCESS]"}.items():
            record.msg = record.msg.replace(uni, asc)
        return True


def configure_logging(log_file):
    try:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    except locale.Error:
        pass
    if sys.platform.startswith("win") and sys.version_info >= (3, 7):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.addFilter(ASCIIFilter())


def load_split(path):
    """读取帧级 npz，缺失文件时给出明确报错。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 {path}，请先运行 modules.preprocess_framed 生成帧级特征"
        )
    with np.load(path, allow_pickle=False) as data:
        return {
            "mfcc": data["mfcc"].astype(np.float32),
            "f0": data["f0"].astype(np.float32),
            "y": data["y"].astype(np.int32),
            "filename": data["filename"],
            "subject": data["subject"],
            "base": data["base"],
            "aug": data["aug"],
            "length": data["length"],
        }


def take(data, index):
    """按索引取子集，保持 npz 内各字段对齐。"""
    return {key: value[index] for key, value in data.items()}


def check_shapes(data, seq_len, split_name):
    if data["mfcc"].shape[1] != seq_len:
        raise ValueError(
            f"{split_name} 的帧数 {data['mfcc'].shape[1]} 与 --seq_len {seq_len} 不一致，"
            f"请确认特征提取时使用的 --seq_len"
        )
    logging.info(
        "%s: mfcc=%s f0=%s 标签=%s",
        split_name, data["mfcc"].shape, data["f0"].shape,
        np.bincount(data["y"]).tolist(),
    )


def evaluate(model, data, threshold=0.5):
    """返回概率、预测与常用指标。"""
    prob = model.predict([data["mfcc"], data["f0"]], batch_size=64, verbose=0).reshape(-1)
    y = data["y"].astype(int)
    pred = (prob >= threshold).astype(int)

    acc = float(accuracy_score(y, pred))
    try:
        auc = float(roc_auc_score(y, prob))
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "prob": prob,
        "pred": pred,
        "acc": acc,
        "auc": auc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": cm,
    }


def build_metrics_report(metrics, y):
    cm = metrics["confusion_matrix"]
    report = classification_report(
        y, metrics["pred"], labels=[0, 1],
        target_names=["Negative (HC)", "Positive (PD)"],
        zero_division=0,
    )
    lines = [
        "              precision    recall  f1-score   support",
        "",
        report,
        f"Confusion Matrix:\n{cm}",
        "",
        f"accuracy: {metrics['acc'] * 100:.2f}%",
        f"auc: {metrics['auc']:.4f}",
        f"sensitivity (PD recall): {metrics['sensitivity'] * 100:.2f}%",
        f"specificity (HC recall): {metrics['specificity'] * 100:.2f}%",
    ]
    return "\n".join(lines)


def save_predictions(data, metrics, path):
    df = pd.DataFrame({
        "filename": data["filename"],
        "subject": data["subject"],
        "base": data["base"],
        "aug": data["aug"],
        "length": data["length"],
        "label": data["y"],
        "prob_pd": metrics["prob"],
        "pred": metrics["pred"],
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_callbacks(model, modal_cross_attn, train_inputs, val_inputs, args,
                   artifact_dir, identifier, save_artifacts):
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=args.patience,
                      min_delta=0.001, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
        train_utils.PrototypeKMeansCallback(
            model, train_inputs, num_prototypes=4, warmup_epoch=args.warmup
        ),
        AlphaEpochVisualizationCallback(modal_cross_attn, subdir="framed"),
        train_utils.TrainingProgressLogger(identifier),
    ]
    if save_artifacts:
        checkpoint_path = (
            os.path.join(artifact_dir, "best") if _LEGACY_KERAS
            else os.path.join(artifact_dir, "best.keras")
        )
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_auc", mode="max", save_best_only=True, verbose=0,
            )
        )
    return callbacks


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def run_experiment(train_data, val_data, test_data, args, lr, batch_size,
                   identifier, artifact_dir, save_artifacts):
    """构建、训练并评估一次模型。"""
    keras.backend.clear_session()
    train_utils.set_global_seed(args.seed)
    model, modal_cross_attn = build_framed_model(
        seq_len=args.seq_len,
        mfcc_dim=train_data["mfcc"].shape[-1],
        f0_dim=train_data["f0"].shape[-1],
    )
    compile_model(model, lr)

    train_inputs = [train_data["mfcc"], train_data["f0"]]
    val_inputs = [val_data["mfcc"], val_data["f0"]]

    start = time.time()
    history = model.fit(
        train_inputs, train_data["y"],
        validation_data=(val_inputs, val_data["y"]),
        epochs=args.epochs,
        batch_size=batch_size,
        callbacks=make_callbacks(
            model, modal_cross_attn, train_inputs, val_inputs, args,
            artifact_dir, identifier, save_artifacts,
        ),
        verbose=0,
    )
    logging.info("[%s] 训练耗时 %.1f 秒", identifier, time.time() - start)

    metrics = evaluate(model, test_data)
    report = build_metrics_report(metrics, test_data["y"])

    if save_artifacts:
        os.makedirs(artifact_dir, exist_ok=True)
        with open(os.path.join(artifact_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        save_predictions(test_data, metrics, os.path.join(artifact_dir, "predictions.csv"))
        pd.DataFrame(history.history).to_csv(
            os.path.join(artifact_dir, "history.csv"), index=False, encoding="utf-8-sig"
        )
        train_utils.save_training_plots(history, identifier)
        train_utils.save_confusion_matrix(
            model, [test_data["mfcc"], test_data["f0"]], test_data["y"], identifier
        )
        alpha1, alpha2 = modal_cross_attn.get_alpha_epoch_values()
        train_utils.save_alpha_visualization(alpha1, alpha2, identifier)

        model_dir = safe_path(f"checkpoints/framed/{args.tag}")
        os.makedirs(model_dir, exist_ok=True)
        if _LEGACY_KERAS:
            model.save(os.path.join(model_dir, "final"))
        else:
            model.save(os.path.join(model_dir, "final.keras"))
        logging.info("[SAVE] 模型与产物已保存到 %s", artifact_dir)

    logging.info("[%s] 测试集结果:\n%s", identifier, report)
    return model, history, metrics


def tune_hyperparams(train_data, val_data, args, artifact_dir):
    """在训练/验证集上搜索 lr 与 batch_size，按验证集 AUC 选择。"""
    grid = [
        {"lr": 1e-5, "batch_size": 32},
        {"lr": 1e-4, "batch_size": 32},
        {"lr": 1e-5, "batch_size": 16},
        {"lr": 1e-4, "batch_size": 16},
    ]
    epochs = min(args.epochs, 60)
    patience = min(args.patience, 12)
    warmup = min(args.warmup, max(1, epochs // 4))

    rows = []
    best = None
    for i, cfg in enumerate(grid):
        keras.backend.clear_session()
        train_utils.set_global_seed(args.seed + i)
        model, _ = build_framed_model(
            seq_len=args.seq_len,
            mfcc_dim=train_data["mfcc"].shape[-1],
            f0_dim=train_data["f0"].shape[-1],
        )
        compile_model(model, cfg["lr"])
        history = model.fit(
            [train_data["mfcc"], train_data["f0"]], train_data["y"],
            validation_data=([val_data["mfcc"], val_data["f0"]], val_data["y"]),
            epochs=epochs,
            batch_size=cfg["batch_size"],
            callbacks=[
                EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                              min_delta=0.001, restore_best_weights=True),
                train_utils.PrototypeKMeansCallback(
                    model, [train_data["mfcc"], train_data["f0"]],
                    num_prototypes=4, warmup_epoch=warmup,
                ),
            ],
            verbose=0,
        )
        val_aucs = history.history.get("val_auc", [0.0])
        val_accs = history.history.get("val_accuracy", [0.0])
        best_idx = int(np.argmax(val_aucs))
        row = {
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "val_auc": float(val_aucs[best_idx]),
            "val_acc": float(val_accs[best_idx]),
            "best_epoch": best_idx + 1,
        }
        rows.append(row)
        logging.info(
            "[TUNE] lr=%g, batch=%d, val_auc=%.4f, val_acc=%.4f, best_epoch=%d",
            row["lr"], row["batch_size"], row["val_auc"], row["val_acc"], row["best_epoch"],
        )
        if best is None or (row["val_auc"], row["val_acc"]) > (best["val_auc"], best["val_acc"]):
            best = row
        del model

    os.makedirs(artifact_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(artifact_dir, "hyperparam_search.csv"), index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(artifact_dir, "hyperparam_best.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "best": best, "trials": rows}, f, ensure_ascii=False, indent=2)
    logging.info("[TUNE] 最优配置: %s", best)
    return best


def run_folds(train_data, args, artifact_dir):
    """按 base（或 subject）分组的 5 折交叉验证，报告折外指标。"""
    groups = train_data["subject"] if args.group == "subject" else train_data["base"]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_data["mfcc"], train_data["y"], groups=groups)):
        logging.info("=== 第 %d 折 ===", fold)
        fold_dir = os.path.join(artifact_dir, f"fold_{fold}")
        _, _, metrics = run_experiment(
            take(train_data, tr_idx), take(train_data, va_idx), take(train_data, va_idx),
            args, args.lr, args.batch_size, f"{args.tag}_fold{fold}", fold_dir, False,
        )
        rows.append({
            "fold": fold,
            "acc": metrics["acc"],
            "auc": metrics["auc"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(artifact_dir, "folds_summary.csv"), index=False, encoding="utf-8-sig")
    logging.info(
        "[FOLDS] acc=%.4f±%.4f auc=%.4f±%.4f",
        summary["acc"].mean(), summary["acc"].std(),
        summary["auc"].mean(), summary["auc"].std(),
    )


def main():
    parser = argparse.ArgumentParser(description="帧级 MFCC+F0 训练（APAG/BCA/MAM）")
    parser.add_argument("--data_dir", default="data/processed_features", help="帧级 npz 所在目录")
    parser.add_argument("--full_training", action="store_true", help="训练集训练、验证集早停、测试集评估")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="仅全量模式下搜索 lr/batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10, help="原型 K-means 初始化前的预热轮数")
    parser.add_argument("--group", choices=["base", "subject"], default="base", help="5 折分组的单位")
    parser.add_argument("--tag", default="framed_full", help="结果目录与文件标识")
    args = parser.parse_args()

    configure_logging(safe_path("logs/framed_training.log"))
    train_utils.set_global_seed(args.seed)
    logging.info("=== 帧级训练启动 | seed=%d ===", args.seed)

    data_dir = safe_path(args.data_dir)
    train_data = load_split(os.path.join(data_dir, "framed_train.npz"))
    val_data = load_split(os.path.join(data_dir, "framed_val.npz"))
    test_data = load_split(os.path.join(data_dir, "framed_test.npz"))
    for name, data in (("train", train_data), ("val", val_data), ("test", test_data)):
        check_shapes(data, args.seq_len, name)

    artifact_dir = safe_path(os.path.join("results/framed", args.tag))
    os.makedirs(artifact_dir, exist_ok=True)

    if args.full_training:
        lr, batch_size = args.lr, args.batch_size
        if args.tune:
            best = tune_hyperparams(train_data, val_data, args, artifact_dir)
            lr, batch_size = best["lr"], best["batch_size"]
            logging.info("[TUNE] 采用最优配置: lr=%g, batch_size=%d", lr, batch_size)

        run_experiment(
            train_data, val_data, test_data, args, lr, batch_size,
            args.tag, artifact_dir, True,
        )
        with open(os.path.join(artifact_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "seed": args.seed, "lr": lr, "batch_size": batch_size,
                "epochs": args.epochs, "patience": args.patience,
                "seq_len": args.seq_len, "warmup": args.warmup,
                "group": args.group, "tune": args.tune,
                "shapes": {
                    "train": list(train_data["mfcc"].shape),
                    "val": list(val_data["mfcc"].shape),
                    "test": list(test_data["mfcc"].shape),
                },
            }, f, ensure_ascii=False, indent=2)
        logging.info("[SUCCESS] 全量训练完成")
    else:
        if args.tune:
            logging.warning("[TUNE] 超参搜索仅支持 --full_training，5 折模式已跳过")
        run_folds(train_data, args, artifact_dir)
        logging.info("[SUCCESS] 5 折训练完成")


if __name__ == "__main__":
    main()
