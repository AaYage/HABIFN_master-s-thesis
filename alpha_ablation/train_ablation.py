# -*- coding: utf-8 -*-
"""α 消融训练脚本（独立于 train_framed.py）。

用法：
    python alpha_ablation/train_ablation.py --alpha_mode none --input_type full
    python alpha_ablation/train_ablation.py --alpha_mode a1 --input_type full
    python alpha_ablation/train_ablation.py --alpha_mode a2 --input_type full
    python alpha_ablation/train_ablation.py --alpha_mode both --input_type full
    python alpha_ablation/train_ablation.py --alpha_mode none   # 默认帧级输入

产物：
    results/alpha_ablation/<tag>/{report.txt,predictions.csv,history.csv,config.json}
    checkpoints/alpha_ablation/<tag>/final
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from alpha_ablation.model_ablation import (
    ALPHA_MODES, AlphaTracker, BalancedCrossAttentionLayerAblation,
    ModalCrossAttentionAblation, build_full_model_ablation, build_framed_model_ablation,
)
from modules.path_utils import safe_path
from modules.train_utils import PrototypeKMeansCallback, load_data, set_global_seed

CUSTOM_OBJECTS = {
    "AlphaTracker": AlphaTracker,
    "BalancedCrossAttentionLayerAblation": BalancedCrossAttentionLayerAblation,
    "ModalCrossAttentionAblation": ModalCrossAttentionAblation,
}


def load_npz(path):
    d = np.load(safe_path(path))
    return d["mfcc"].astype(np.float32), d["f0"].astype(np.float32), d["y"].astype(np.int32)


class AlphaEpochCallback(keras.callbacks.Callback):
    def __init__(self, bca):
        super().__init__()
        self.bca = bca

    def on_epoch_begin(self, epoch, logs=None):
        self.bca.set_current_epoch(epoch)
        self.bca.reset_tracker()

    def on_epoch_end(self, epoch, logs=None):
        a1, a2 = self.bca.get_epoch_alpha()
        logs = logs or {}
        logs["alpha1"] = a1
        logs["alpha2"] = a2
        print(f"  epoch {epoch + 1}: alpha1={a1:.4f}, alpha2={a2:.4f}")


def _noise_on_selected(x, sel, sigma):
    """对 sel 为真的样本，按 sigma 加高斯噪声；其余样本保持不变。"""
    shape = [1] * len(x.shape)
    shape[0] = -1
    scale = tf.reshape(tf.where(sel, sigma, tf.zeros_like(sigma)), shape)
    return x + tf.random.normal(tf.shape(x)) * scale


def _graded_mask(x, sel, ratio):
    """对 sel 为真的样本，按各自 ratio 随机掩蔽部分维度（置 0）。"""
    shape = [-1] + [1] * (len(x.shape) - 1)
    r = tf.reshape(ratio, shape)
    s = tf.reshape(tf.cast(sel, x.dtype), shape)
    rand = tf.random.uniform(tf.shape(x))
    keep = 1.0 - s * tf.cast(rand < r, x.dtype)
    return x * keep


def make_train_dataset(inputs, y, batch_size, perturb_prob, snr_min, snr_max, seed,
                       drop_prob=0.0, mask_prob=0.0):
    """训练集管线：分支丢弃（modality dropout）与单分支加噪。

    drop_prob: 每条样本以该概率整条丢弃一个分支（两条分支各半），
               迫使模型在单分支可用时也能预测，从而给 α 提供梯度。
    """
    n_in = len(inputs)
    ds = tf.data.Dataset.from_tensor_slices(((tuple(tf.constant(x) for x in inputs)), tf.constant(y)))
    ds = ds.shuffle(4096, seed=seed, reshuffle_each_iteration=True).batch(batch_size)

    if perturb_prob <= 0 and drop_prob <= 0 and mask_prob <= 0:
        return ds.prefetch(tf.data.AUTOTUNE)

    def augment(xs, yy):
        xs = list(xs)
        n = tf.shape(yy)[0]

        if mask_prob > 0:
            masked = tf.random.uniform((n,)) < mask_prob
            ratio = tf.random.uniform((n,), 0.0, 1.0)
            pick_f0 = tf.random.uniform((n,)) < 0.5
            xs[0] = _graded_mask(xs[0], tf.logical_and(masked, ~pick_f0), ratio)
            xs[1] = _graded_mask(xs[1], tf.logical_and(masked, pick_f0), ratio)

        if drop_prob > 0:
            dropped = tf.random.uniform((n,)) < drop_prob
            drop_f0 = tf.random.uniform((n,)) < 0.5
            keep_m = tf.cast(~tf.logical_and(dropped, ~drop_f0), xs[0].dtype)
            keep_f = tf.cast(~tf.logical_and(dropped, drop_f0), xs[1].dtype)
            xs[0] = xs[0] * tf.reshape(keep_m, (-1,) + (1,) * (len(xs[0].shape) - 1))
            xs[1] = xs[1] * tf.reshape(keep_f, (-1,) + (1,) * (len(xs[1].shape) - 1))

        if perturb_prob > 0:
            masked = tf.random.uniform((n,)) < perturb_prob
            pick_f0 = tf.random.uniform((n,)) < 0.5
            snr_db = tf.random.uniform((n,), snr_min, snr_max)
            sigma = tf.sqrt(1.0 / tf.pow(10.0, snr_db / 10.0))
            xs[0] = _noise_on_selected(xs[0], tf.logical_and(masked, ~pick_f0), sigma)
            xs[1] = _noise_on_selected(xs[1], tf.logical_and(masked, pick_f0), sigma)

        return tuple(xs), yy

    return ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def evaluate(model, test_inputs, y_test):
    prob = np.asarray(model.predict(test_inputs, verbose=0)).reshape(-1)
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    return prob, pred, {"accuracy": acc, "auc": auc, "sensitivity": sen, "specificity": spe}, cm


def main():
    ap = argparse.ArgumentParser(description="α 消融训练")
    ap.add_argument("--alpha_mode", choices=ALPHA_MODES, required=True)
    ap.add_argument("--input_type", choices=("full", "framed"), default="framed",
                    help="full=旧 320 维输入；framed=帧级输入")
    ap.add_argument("--data_dir", default="data/processed_features")
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--balance_weight", type=float, default=0.001)
    ap.add_argument("--perturb_prob", type=float, default=0.0,
                    help="训练期单分支加噪的样本比例（0 关闭，0.5 推荐）")
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--drop_prob", type=float, default=0.0,
                    help="训练期整分支丢弃概率（modality dropout）")
    ap.add_argument("--gate_mode", choices=("residual", "gate", "res_gate"), default="residual",
                    help="gate=α 控制自身/交互凸组合；res_gate=自身+增益后的交互分支")
    ap.add_argument("--mask_prob", type=float, default=0.0,
                    help="训练期分级掩蔽的样本比例（每条样本随机掩蔽 0–100% 维度）")
    ap.add_argument("--init_ckpt", default="",
                    help="可选：从已有 SavedModel 初始化权重，用于课程式微调")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"alpha_{args.alpha_mode}"
    set_global_seed(args.seed)

    if args.input_type == "full":
        # 与 train_utils.train_full 完全一致的划分与标准化
        X_train, y_train = load_data(os.path.join(args.data_dir, "train_data.csv"))
        X_test, y_test = load_data(os.path.join(args.data_dir, "test_data.csv"))
        y_te = y_test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        for tr_idx, va_idx in sss.split(X_train, y_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_test)
        inputs_tr = [X_tr[:, :160], X_tr[:, 160:320], X_tr[:, 320:]]
        inputs_va = [X_va[:, :160], X_va[:, 160:320], X_va[:, 320:]]
        inputs_te = [X_te[:, :160], X_te[:, 160:320], X_te[:, 320:]]
        model, bca = build_full_model_ablation(alpha_mode=args.alpha_mode,
                                               balance_weight=args.balance_weight,
                                               gate_mode=args.gate_mode)
    else:
        tr = os.path.join(args.data_dir, "framed_train.npz")
        va = os.path.join(args.data_dir, "framed_val.npz")
        te = os.path.join(args.data_dir, "framed_test.npz")
        x_tr, f_tr, y_tr = load_npz(tr)
        x_va, f_va, y_va = load_npz(va)
        x_te, f_te, y_te = load_npz(te)
        if x_tr.shape[1] != args.seq_len:
            raise ValueError(f"seq_len 不匹配: 数据 {x_tr.shape[1]} vs 参数 {args.seq_len}")
        inputs_tr = [x_tr, f_tr]
        inputs_va = [x_va, f_va]
        inputs_te = [x_te, f_te]
        model, bca = build_framed_model_ablation(
            alpha_mode=args.alpha_mode, seq_len=args.seq_len,
            mfcc_dim=x_tr.shape[-1], f0_dim=f_tr.shape[-1],
            balance_weight=args.balance_weight,
            gate_mode=args.gate_mode,
        )
    if args.init_ckpt:
        saved = keras.models.load_model(safe_path(args.init_ckpt), custom_objects=CUSTOM_OBJECTS,
                                        compile=False)
        model.set_weights(saved.get_weights())
        print(f"initialized from: {args.init_ckpt}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    out_dir = safe_path(os.path.join("results/alpha_ablation", tag))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = safe_path(os.path.join("checkpoints/alpha_ablation", tag, "final"))
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        AlphaEpochCallback(bca),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                      patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(ckpt_dir, monitor="val_auc", mode="max", save_best_only=True),
        PrototypeKMeansCallback(model, inputs_tr, num_prototypes=4, warmup_epoch=args.warmup),
    ]

    train_ds = make_train_dataset(inputs_tr, y_tr, args.batch_size,
                                  args.perturb_prob, args.snr_min, args.snr_max, args.seed,
                                  drop_prob=args.drop_prob, mask_prob=args.mask_prob)
    history = model.fit(
        train_ds,
        validation_data=(inputs_va, y_va),
        epochs=args.epochs, batch_size=None,
        callbacks=callbacks, verbose=2,
    )

    prob, pred, metrics, cm = evaluate(model, inputs_te, y_te)

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"input_type: {args.input_type}\n")
        fh.write(f"alpha_mode: {args.alpha_mode}\n")
        fh.write(f"accuracy: {metrics['accuracy']:.4f}\nauc: {metrics['auc']:.4f}\n")
        fh.write(f"sensitivity: {metrics['sensitivity']:.4f}\nspecificity: {metrics['specificity']:.4f}\n\n")
        fh.write(classification_report(y_te, pred, target_names=["HC", "PD"]))
        fh.write(f"\nconfusion matrix:\n{cm}\n")

    pd.DataFrame({"prob_pd": prob, "pred": pred, "label": y_te}) \
        .to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    pd.DataFrame(history.history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("saved:", out_dir)


if __name__ == "__main__":
    main()
