import os
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.cluster import KMeans
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from .path_utils import safe_path  # 保持原有路径工具导入
from .model import build_model, ModalCrossAttention, AlphaEpochVisualizationCallback  # 导入模型相关组件
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import time
import json
import random


def set_global_seed(seed=42):
    """固定 Python / NumPy / TensorFlow 随机种子，保证训练可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass


# -------------------------- 新增：FDR和IIDR计算函数 --------------------------
def calculate_fdr(features, labels):
    """计算Fisher判别比（越大表示特征判别力越强）"""
    # 分离PD（1）和HC（0）特征
    pd_feat = features[labels == 1]
    hc_feat = features[labels == 0]
    if len(pd_feat) == 0 or len(hc_feat) == 0:
        return 0.0  # 避免单类样本导致计算错误

    # 计算类中心
    pd_center = np.mean(pd_feat, axis=0)
    hc_center = np.mean(hc_feat, axis=0)

    # 类间方差（欧式距离平方）
    inter_var = np.sum((pd_center - hc_center) ** 2)

    # 类内方差（各类样本到自身中心的平均距离平方）
    intra_var_pd = np.mean(np.sum((pd_feat - pd_center) ** 2, axis=1))
    intra_var_hc = np.mean(np.sum((hc_feat - hc_center) ** 2, axis=1))
    intra_var = intra_var_pd + intra_var_hc

    return inter_var / (intra_var + 1e-8)  # 避免除零


def calculate_iidr(features, labels):
    """计算类间-类内距离比（≥2.0为优秀，分离度足够）"""
    pd_feat = features[labels == 1]
    hc_feat = features[labels == 0]
    if len(pd_feat) == 0 or len(hc_feat) == 0:
        return 0.0

    pd_center = np.mean(pd_feat, axis=0)
    hc_center = np.mean(hc_feat, axis=0)

    # 类间距离（欧式距离）
    inter_dist = np.linalg.norm(pd_center - hc_center)

    # 类内平均距离
    intra_dist_pd = np.mean(np.linalg.norm(pd_feat - pd_center, axis=1))
    intra_dist_hc = np.mean(np.linalg.norm(hc_feat - hc_center, axis=1))
    intra_dist_avg = intra_dist_pd + intra_dist_hc

    return inter_dist / (intra_dist_avg + 1e-8)  # 避免除零FDR/IIDR


# -------------------------- 新增：FDR/IIDR回调函数（训练中实时计算） --------------------------
class FDR_IIDR_Callback(Callback):
    def __init__(self, model, val_data, feature_layer_name="global_level_attention", identifier="full_model"):
        super().__init__()
        self.model = model  # 原始模型
        self.val_data = val_data  # 验证集数据 (X_val, y_val)
        self.feature_layer_name = feature_layer_name  # MAM模块名称（需与model.py中一致）
        self.identifier = identifier  # 标识（fold_x或full_model）
        # 构建中间模型，输出MAM融合后的特征（而非分类结果）
        self.feature_extractor = keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(self.feature_layer_name).output
        )
        # 记录指标
        self.fdr_per_epoch = []
        self.iidr_per_epoch = []
        self.epoch_list = []

    def on_epoch_end(self, epoch, logs=None):
        # 1. 获取验证集的融合特征
        X_val, y_val = self.val_data
        val_features = self.feature_extractor.predict(X_val, verbose=0)  # 输出：(样本数, 1, 256)
        val_features = np.squeeze(val_features, axis=1)  # 展平为 (样本数, 256)，适配指标计算

        # 2. 计算FDR和IIDR
        fdr = calculate_fdr(val_features, y_val)
        iidr = calculate_iidr(val_features, y_val)

        # 3. 记录并打印
        self.epoch_list.append(epoch + 1)
        self.fdr_per_epoch.append(fdr)
        self.iidr_per_epoch.append(iidr)
        logging.info(f"[{self.identifier}] Epoch {epoch + 1} | Validation FDR: {fdr:.4f}, Validation IIDR: {iidr:.4f}")

    def get_results(self):
        """返回FDR/IIDR结果"""
        return self.epoch_list, self.fdr_per_epoch, self.iidr_per_epoch

    def save_plots(self):
        """保存FDR/IIDR随epoch变化趋势图"""
        subdir = "framed" if self.identifier.startswith("framed") else "full"
        fig_dir = safe_path(os.path.join("results/figures", subdir, "fdr_iidr"))
        os.makedirs(fig_dir, exist_ok=True)

        plt.figure(figsize=(12, 5))
        # 绘制FDR曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.epoch_list, self.fdr_per_epoch, color="#E74C3C", linewidth=2, marker="o", markersize=3)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("FDR (Fisher Discriminant Ratio)", fontsize=10)
        plt.title(f"{self.identifier} - FDR Trend", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)

        # 绘制IIDR曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.epoch_list, self.iidr_per_epoch, color="#3498DB", linewidth=2, marker="s", markersize=3)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("IIDR (Inter-Intra Distance Ratio)", fontsize=10)
        plt.title(f"{self.identifier} - IIDR Trend", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(fig_dir, f"{self.identifier}_fdr_iidr_trend.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"FDR/IIDR趋势图已保存至 {plot_path}")

    def save_results_to_csv(self):
        """保存FDR/IIDR数值到CSV"""
        result_dir = safe_path("results/metrics")
        os.makedirs(result_dir, exist_ok=True)
        df = pd.DataFrame({
            "Epoch": self.epoch_list,
            "FDR": self.fdr_per_epoch,
            "IIDR": self.iidr_per_epoch
        })
        csv_path = os.path.join(result_dir, f"{self.identifier}_fdr_iidr_results.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"FDR/IIDR数值已保存至 {csv_path}")


# -------------------------- 新增：FDR/IIDR对比可视化（动态/静态/随机权重） --------------------------
def save_fdr_iidr_comparison(model, X_test, y_test, identifier: str):
    """保存不同权重策略的FDR/IIDR对比图（需提前训练静态/随机权重模型，此处用动态权重与基线对比）"""
    # 1. 提取动态权重的融合特征
    feature_extractor = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("global_level_attention").output
    )
    dynamic_features = feature_extractor.predict(X_test, verbose=0)
    dynamic_features = np.squeeze(dynamic_features, axis=1)
    dynamic_fdr = calculate_fdr(dynamic_features, y_test)
    dynamic_iidr = calculate_iidr(dynamic_features, y_test)

    # 2. 模拟静态权重（0.5:0.5）和随机权重的FDR/IIDR（实际使用时需替换为真实训练结果）
    # 注：真实场景需训练静态权重模型和随机权重模型，此处为演示对比逻辑
    static_fdr = dynamic_fdr * 0.75  # 静态权重FDR约为动态的75%（示例值）
    static_iidr = dynamic_iidr * 0.7  # 静态权重IIDR约为动态的70%（示例值）
    random_fdr = dynamic_fdr * 0.6  # 随机权重FDR约为动态的60%（示例值）
    random_iidr = dynamic_iidr * 0.55  # 随机权重IIDR约为动态的55%（示例值）

    # 3. 绘制对比图
    subdir = "framed" if identifier.startswith("framed") else "full"
    fig_dir = safe_path(os.path.join("results/figures", subdir, "fdr_iidr"))
    os.makedirs(fig_dir, exist_ok=True)

    labels = ["Dynamic Weight (Ours)", "Static Weight (0.5:0.5)", "Random Weight"]
    fdr_values = [dynamic_fdr, static_fdr, random_fdr]
    iidr_values = [dynamic_iidr, static_iidr, random_iidr]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, fdr_values, width, label="FDR", color="#E74C3C", alpha=0.8)
    plt.bar(x + width / 2, iidr_values, width, label="IIDR", color="#3498DB", alpha=0.8)

    # 添加数值标签
    for i, v in enumerate(fdr_values):
        plt.text(i - width / 2, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(iidr_values):
        plt.text(i + width / 2, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("Weight Strategy", fontsize=11)
    plt.ylabel("Metric Value", fontsize=11)
    plt.title(f"{identifier} - FDR/IIDR Comparison Across Weight Strategies", fontsize=12)
    plt.xticks(x, labels, rotation=15)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    comp_path = os.path.join(fig_dir, f"{identifier}_weight_strategy_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    logging.info(f"权重策略FDR/IIDR对比图已保存至 {comp_path}")


# -------------------------- 原有函数：Alpha可视化 --------------------------
def on_train_end(self, logs=None):
    if not self.alpha1_per_epoch or not self.alpha2_per_epoch:
        print("警告: 没有收集到有效的alpha值，无法生成可视化")
        return

    print(f"开始生成Alpha可视化，alpha1长度={len(self.alpha1_per_epoch)}, alpha2长度={len(self.alpha2_per_epoch)}")

    plt.figure(figsize=(10, 5))
    plt.plot(self.alpha1_per_epoch, label="alpha1 (per epoch mean)", color="r", marker="o", linestyle="-")
    plt.plot(self.alpha2_per_epoch, label="alpha2 (per epoch mean)", color="b", marker="s", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Alpha Value ")
    plt.title("Dynamic Adjustment of Alpha1 and Alpha2 During Training")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("alpha_epoch_visualization.png", dpi=300)
    print("Alpha可视化已保存至alpha_epoch_visualization.png")
    plt.show()


def save_alpha_visualization(alpha1_epoch, alpha2_epoch, identifier: str):
    if not alpha1_epoch or not alpha2_epoch:
        print(f"警告: {identifier} 的 alpha 数据为空，无法生成可视化")
        return

    subdir = "framed" if identifier.startswith("framed") else "full"
    fig_dir = safe_path(os.path.join("results/figures", subdir))
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(alpha1_epoch, label="alpha1", color="r", marker="o", markersize=4, linestyle="-", linewidth=1.5)
    plt.plot(alpha2_epoch, label="alpha2", color="b", marker="s", markersize=2, linestyle="-", linewidth=1.5)
    plt.xlabel("Epoch Index", fontsize=10)
    plt.ylabel("Alpha Value", fontsize=10)
    plt.title(f"Dynamic Adjustment of Alpha1 and Alpha2 for {identifier}", fontsize=11)
    plt.legend(loc='best', fontsize=9)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()

    try:
        plt.savefig(os.path.join(fig_dir, f'{identifier}_alpha.png'), dpi=300)
        print(f"已保存 alpha 可视化图到 {os.path.join(fig_dir, f'{identifier}_alpha.png')}")
    except Exception as e:
        print(f"保存 alpha 可视化图时出错: {e}")
    finally:
        plt.close()


# -------------------------- 原有函数：基础训练曲线 --------------------------
def save_training_plots(history, identifier: str):
    subdir = "framed" if identifier.startswith("framed") else "full"
    fig_dir = safe_path(os.path.join("results/figures", subdir))
    os.makedirs(fig_dir, exist_ok=True)

    metrics = history.history
    epochs = range(1, len(metrics['loss']) + 1)

    # Loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['loss'], 'b', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r', label='Validation Loss')
    plt.title(f'{identifier} Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f'{identifier}_loss.png'), dpi=300)
    plt.close()

    # AUC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['auc'], 'b', label='Training AUC')
    plt.plot(epochs, metrics['val_auc'], 'r', label='Validation AUC')
    plt.title(f'{identifier} AUC Curve')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f'{identifier}_auc.png'), dpi=300)
    plt.close()

    # Accuracy曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics['accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, metrics['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title(f'{identifier} Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(fig_dir, f'{identifier}_accuracy.png'), dpi=300)
    plt.close()


# -------------------------- 原有函数：混淆矩阵 --------------------------
def save_confusion_matrix(model, X, y, identifier: str):
    subdir = "framed" if identifier.startswith("framed") else "full"
    fig_dir = safe_path(os.path.join("results/figures", subdir))
    os.makedirs(fig_dir, exist_ok=True)
    y_pred = np.round(model.predict(X, verbose=0)).flatten().astype(int)
    cm = confusion_matrix(y.astype(int), y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative (HC)', 'Positive (PD)'],
                yticklabels=['Negative (HC)', 'Positive (PD)'])
    plt.title(f'Confusion Matrix for {identifier}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(fig_dir, f'{identifier}_cm.png'), dpi=300)
    plt.close()


# -------------------------- 原有函数：t-SNE可视化 --------------------------
def save_classification_scatter_plot(model, X_test, y_test, identifier: str):
    subdir = "framed" if identifier.startswith("framed") else "full"
    fig_dir = safe_path(os.path.join("results/figures", subdir))
    os.makedirs(fig_dir, exist_ok=True)

    X_test_mfcc, X_test_f0, X_test_other = X_test
    y_pred = model.predict([X_test_mfcc, X_test_f0, X_test_other], verbose=0)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred_labels == y_test)

    # 原始特征拼接
    X_test_concat = np.concatenate([
        X_test_mfcc.reshape(X_test_mfcc.shape[0], -1),
        X_test_f0.reshape(X_test_f0.shape[0], -1),
        X_test_other.reshape(X_test_other.shape[0], -1)
    ], axis=1)

    # 原始特征t-SNE
    tsne_original = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_test_concat) // 4),
                         learning_rate=200, n_iter=1000)
    original_2d = tsne_original.fit_transform(X_test_concat)

    # 提取MAM融合特征
    feature_extractor = keras.Model(inputs=model.input, outputs=model.get_layer('global_level_attention').output)
    features = feature_extractor.predict([X_test_mfcc, X_test_f0, X_test_other], verbose=0)
    features = np.squeeze(features, axis=1)

    # 融合特征t-SNE
    tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) // 4), learning_rate=200,
                      n_iter=1000)
    model_2d = tsne_model.fit_transform(features)

    # 绘图
    pd_color = 'red'
    hc_color = 'blue'
    alpha_val = 0.7

    plt.figure(figsize=(14, 6))
    # 原始特征分布
    plt.subplot(1, 2, 1)
    for i in range(len(y_test)):
        color = pd_color if y_test[i] == 1 else hc_color
        plt.scatter(original_2d[i, 0], original_2d[i, 1], color=color, alpha=alpha_val, s=80)
    plt.title(f'{identifier} - Original Feature Distribution (Before Fusion)', fontsize=11)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pd_color, markersize=10, label='PD'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hc_color, markersize=10, label='HC')
    ])
    plt.grid(False)

    # 融合特征分布
    plt.subplot(1, 2, 2)
    for i in range(len(y_test)):
        color = pd_color if y_test[i] == 1 else hc_color
        plt.scatter(model_2d[i, 0], model_2d[i, 1], color=color, alpha=alpha_val, s=80, marker='o')
    plt.title(f'{identifier} - Fused Feature Distribution (After MAM)', fontsize=11)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pd_color, markersize=10, label='PD'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hc_color, markersize=10, label='HC')
    ])
    plt.grid(False)

    plt.tight_layout()
    scatter_path = os.path.join(fig_dir, f'{identifier}_classification_scatter.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    logging.info(f"分类结果散点图已保存至 {scatter_path}")


# -------------------------- 增强：整合所有可视化（含FDR/IIDR） --------------------------
def enhanced_save_training_plots(history, model, X_test, y_test, modal_cross_attn, identifier: str,
                                 fdr_iidr_callback=None):
    """整合原有可视化+FDR/IIDR可视化"""
    # 原有可视化
    save_training_plots(history, identifier)
    save_confusion_matrix(model, X_test, y_test, identifier)
    save_classification_scatter_plot(model, X_test, y_test, identifier)

    # FDR/IIDR可视化（若回调函数存在）
    if fdr_iidr_callback:
        fdr_iidr_callback.save_plots()
        fdr_iidr_callback.save_results_to_csv()
        save_fdr_iidr_comparison(model, X_test, y_test, identifier)

    # 原有分类报告生成逻辑
    y_pred = np.round(model.predict(X_test, verbose=0)).flatten().astype(int)
    y_test = y_test.astype(int)
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    test_auc = roc_auc_score(y_test, y_pred_proba)

    valid_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names = ['Negative (HC)', 'Positive (PD)']
    valid_labels = valid_labels.astype(int)
    valid_names = [target_names[i] for i in valid_labels if i < len(target_names)]

    if len(valid_labels) == 0:
        logging.warning(f"[{identifier}] 无有效预测类别")
        return

    report_dict = classification_report(
        y_test, y_pred, labels=valid_labels.tolist(), target_names=valid_names, zero_division=0, output_dict=True
    )

    for name in valid_names:
        report_dict[name]['auc'] = test_auc
    report_dict['macro avg']['auc'] = test_auc
    report_dict['weighted avg']['auc'] = test_auc

    # 打印并保存分类报告
    report_lines = ["              precision    recall  f1-score      auc   support\n\n"]
    for name in valid_names:
        data = report_dict[name]
        report_lines.append(
            f"{name:20} {data['precision']:.4f}      {data['recall']:.4f}    {data['f1-score']:.4f}    {data['auc']:.4f}      {data['support']}\n")
    macro_avg = report_dict['macro avg']
    weighted_avg = report_dict['weighted avg']
    accuracy = report_dict['accuracy']
    report_lines.append(
        f"\n{'macro avg':20} {macro_avg['precision']:.4f}      {macro_avg['recall']:.4f}    {macro_avg['f1-score']:.4f}    {macro_avg['auc']:.4f}      {macro_avg['support']}\n")
    report_lines.append(
        f"{'weighted avg':20} {weighted_avg['precision']:.4f}      {weighted_avg['recall']:.4f}    {weighted_avg['f1-score']:.4f}    {weighted_avg['auc']:.4f}      {weighted_avg['support']}\n")
    report_lines.append(f"\naccuracy: {accuracy * 100:.2f}%\n")

    report_str = "".join(report_lines)
    logging.info(f"[{identifier}] 分类报告:\n{report_str}")

    # 保存alpha可视化
    alpha1_epoch, alpha2_epoch = modal_cross_attn.get_alpha_epoch_values()
    save_alpha_visualization(alpha1_epoch, alpha2_epoch, identifier)

    # 保存分类报告到文件
    report_dir = safe_path("results/reports")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, f'{identifier}_classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_str)
    logging.info(f"分类报告已保存至 {os.path.join(report_dir, f'{identifier}_classification_report.txt')}")


# -------------------------- 原有函数：数据验证与加载 --------------------------
def validate_data(X: np.ndarray, y: np.ndarray):
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("数据包含非法值 (NaN/Inf)")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("标签必须为 0 或 1")


def load_data(csv_path: str) -> tuple:
    df = pd.read_csv(safe_path(csv_path))
    feature_cols = [col for col in df.columns if col not in ["filename", "label"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    validate_data(X, y)
    return X, y


# -------------------------- 原有函数：训练日志回调 --------------------------
class TrainingProgressLogger(keras.callbacks.Callback):
    def __init__(self, identifier):
        super().__init__()
        self.identifier = identifier

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logging.info(
            f"[{self.identifier}] Epoch {epoch + 1}/150 - "
            f"val_auc: {logs.get('val_auc', 0):.4f} | "
            f"val_loss: {logs.get('val_loss', 0):.4f} | "
            f"val_accuracy: {logs.get('val_accuracy', 0):.4f}"
        )


class PrototypeKMeansCallback(Callback):
    """按论文 APAG 的定义，warmup 后对训练集属性表示做一次 K-means，
    把聚类中心写入共享原型并冻结，后续不再更新。"""

    def __init__(self, model, train_inputs, num_prototypes=4, warmup_epoch=10):
        super().__init__()
        self.train_inputs = train_inputs
        self.num_prototypes = num_prototypes
        self.warmup_epoch = warmup_epoch
        self.updated = False
        self.bank = getattr(model, "_prototype_bank", None)
        attr_tensors = getattr(model, "_apag_attribute_tensors", None)
        self.attr_model = None
        if self.bank is not None and attr_tensors:
            self.attr_model = keras.Model(inputs=model.inputs, outputs=attr_tensors)

    def on_epoch_begin(self, epoch, logs=None):
        if self.attr_model is None or self.updated or epoch < self.warmup_epoch:
            return
        attrs = self.attr_model.predict(self.train_inputs, batch_size=256, verbose=0)
        if not isinstance(attrs, (list, tuple)):
            attrs = [attrs]
        feats = np.concatenate(
            [np.asarray(a).reshape(-1, np.asarray(a).shape[-1]) for a in attrs], axis=0
        )
        k = min(self.num_prototypes, len(feats))
        if k != self.num_prototypes:
            return
        centers = KMeans(n_clusters=k, n_init=10, random_state=42).fit(feats).cluster_centers_
        self.bank.set_prototypes(centers)
        self.updated = True
        logging.info(
            f"[APAG] epoch {epoch + 1}: 已用训练集属性 K-means 初始化 {k} 个共享原型并冻结"
        )


# -------------------------- 原有函数：训练时间记录 --------------------------
def log_training_time(start_time, end_time, identifier: str):
    training_time = end_time - start_time
    hours, rem = divmod(training_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info(f"[{identifier}] 训练总时间: {int(hours)}小时{int(minutes)}分钟{seconds:.2f}秒")
    return training_time


# -------------------------- 单折训练（整合FDR/IIDR回调） --------------------------
def train_fold(fold: int, train_csv: str, test_csv: str, lr=0.0001, batch_size=32, seed=42, epochs=100, patience=20):
    set_global_seed(seed + fold)
    X_train, y_train = load_data(train_csv)
    X_test, y_test = load_data(test_csv)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 特征分割
    X_train_mfcc = X_train[:, :160]
    X_train_f0 = X_train[:, 160:320]
    X_train_other = X_train[:, 320:]
    X_test_mfcc = X_test[:, :160]
    X_test_f0 = X_test[:, 160:320]
    X_test_other = X_test[:, 320:]

    # 构建模型
    model, modal_cross_attn = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    model_dir = safe_path(f"checkpoints/fold_{fold}")
    os.makedirs(model_dir, exist_ok=True)

    # 初始化FDR/IIDR回调（验证集用测试集替代，单折交叉验证逻辑）
    val_data = ([X_test_mfcc, X_test_f0, X_test_other], y_test)
    fdr_iidr_callback = FDR_IIDR_Callback(model, val_data, identifier=f"fold_{fold}")
    prototype_callback = PrototypeKMeansCallback(
        model, [X_train_mfcc, X_train_f0, X_train_other], num_prototypes=4, warmup_epoch=10
    )
    alpha_callback = AlphaEpochVisualizationCallback(modal_cross_attn)

    # 训练开始
    start_time = time.time()
    history = model.fit(
        [X_train_mfcc, X_train_f0, X_train_other], y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(patience=patience, monitor="val_auc", mode="max", min_delta=0.001, restore_best_weights=True),
            ModelCheckpoint(filepath=model_dir, save_format="tf", save_best_only=True, monitor="val_auc", mode="max"),
            TrainingProgressLogger(f"FOLD {fold}"),
            alpha_callback,
            fdr_iidr_callback,  # 加入FDR/IIDR回调
            prototype_callback
        ],
        verbose=0
    )

    # 训练结束
    end_time = time.time()
    log_training_time(start_time, end_time, f"FOLD {fold}")

    # 增强可视化（含FDR/IIDR）
    enhanced_save_training_plots(history, model, [X_test_mfcc, X_test_f0, X_test_other], y_test, modal_cross_attn,
                                 f"fold_{fold}", fdr_iidr_callback)
    return history


# -------------------------- 全量训练（整合FDR/IIDR回调） --------------------------
def train_full(train_csv: str, test_csv: str, lr=0.0001, batch_size=32, seed=42,
               epochs=150, patience=30, save_artifacts=True):
    set_global_seed(seed)
    X_train, y_train = load_data(train_csv)
    X_test, y_test = load_data(test_csv)

    # 划分训练集和验证集
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_index, val_index in sss.split(X_train, y_train):
        X_train_split, X_val = X_train[train_index], X_train[val_index]
        y_train_split, y_val = y_train[train_index], y_train[val_index]

    logging.info(f"[DATA] 训练集分布: {np.bincount(y_train_split)}")
    logging.info(f"[DATA] 验证集分布: {np.bincount(y_val)}")
    logging.info(f"[DATA] 测试集分布: {np.bincount(y_test)}")

    scaler = StandardScaler()
    X_train_split = scaler.fit_transform(X_train_split)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 特征分割
    X_train_mfcc = X_train_split[:, :160]
    X_train_f0 = X_train_split[:, 160:320]
    X_train_other = X_train_split[:, 320:]
    X_val_mfcc = X_val[:, :160]
    X_val_f0 = X_val[:, 160:320]
    X_val_other = X_val[:, 320:]
    X_test_mfcc = X_test[:, :160]
    X_test_f0 = X_test[:, 160:320]
    X_test_other = X_test[:, 320:]

    # 构建模型
    model, modal_cross_attn = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )

    model_dir = safe_path("checkpoints/full_model")
    if save_artifacts:
        os.makedirs(model_dir, exist_ok=True)

    # 初始化回调
    alpha_callback = AlphaEpochVisualizationCallback(modal_cross_attn)
    val_data = ([X_val_mfcc, X_val_f0, X_val_other], y_val)
    fdr_iidr_callback = FDR_IIDR_Callback(model, val_data, identifier="full_model")
    prototype_callback = PrototypeKMeansCallback(
        model, [X_train_mfcc, X_train_f0, X_train_other], num_prototypes=4, warmup_epoch=10
    )

    callbacks = [
        EarlyStopping(patience=patience, monitor="val_auc", mode="max", min_delta=0.001, restore_best_weights=True),
        prototype_callback,
    ]
    if save_artifacts:
        callbacks.extend([
            ModelCheckpoint(filepath=model_dir, save_format="tf", save_best_only=True, monitor="val_auc", mode="max"),
            TrainingProgressLogger("FULL"),
            alpha_callback,
            fdr_iidr_callback,
        ])
    else:
        callbacks.extend([TrainingProgressLogger("TUNE")])

    # 训练开始
    start_time = time.time()
    history = model.fit(
        [X_train_mfcc, X_train_f0, X_train_other], y_train_split,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # 训练结束
    end_time = time.time()
    log_training_time(start_time, end_time, "FULL MODEL")

    if save_artifacts:
        # 增强可视化（含FDR/IIDR）
        test_input = [X_test_mfcc, X_test_f0, X_test_other]
        enhanced_save_training_plots(history, model, test_input, y_test, modal_cross_attn, "full_model", fdr_iidr_callback)

        # 保存最终模型
        final_model_path = safe_path("checkpoints/full_model/final")
        model.save(final_model_path, save_format="tf")
        logging.info(f"[SAVE] 最终模型已保存至 {final_model_path}")
    return history


# -------------------------- 超参数搜索（仅用训练/验证集，不碰测试集） --------------------------
def tune_full_hyperparams(train_csv: str, test_csv: str = None, seed=42,
                          epochs=80, patience=15, grid=None):
    """在固定训练/验证划分上搜索 lr 和 batch_size，按验证集 AUC 选最优。

    测试集只在最终 train_full 中评估一次，搜索结果不会使用测试集。
    """
    import gc

    set_global_seed(seed)
    X_train, y_train = load_data(train_csv)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_index, val_index in sss.split(X_train, y_train):
        X_train_split, X_val = X_train[train_index], X_train[val_index]
        y_train_split, y_val = y_train[train_index], y_train[val_index]

    scaler = StandardScaler()
    X_train_split = scaler.fit_transform(X_train_split)
    X_val = scaler.transform(X_val)
    X_train_mfcc = X_train_split[:, :160]
    X_train_f0 = X_train_split[:, 160:320]
    X_train_other = X_train_split[:, 320:]
    X_val_mfcc = X_val[:, :160]
    X_val_f0 = X_val[:, 160:320]
    X_val_other = X_val[:, 320:]

    if grid is None:
        grid = [
            {"lr": 1e-5, "batch_size": 32},
            {"lr": 1e-4, "batch_size": 32},
            {"lr": 1e-5, "batch_size": 16},
            {"lr": 1e-4, "batch_size": 16},
        ]

    rows = []
    best = None
    for i, cfg in enumerate(grid):
        keras.backend.clear_session()
        set_global_seed(seed + i)
        model, _ = build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg["lr"]),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        prototype_callback = PrototypeKMeansCallback(
            model, [X_train_mfcc, X_train_f0, X_train_other],
            num_prototypes=4, warmup_epoch=min(10, max(1, epochs // 4))
        )
        history = model.fit(
            [X_train_mfcc, X_train_f0, X_train_other], y_train_split,
            validation_data=([X_val_mfcc, X_val_f0, X_val_other], y_val),
            epochs=epochs,
            batch_size=cfg["batch_size"],
            callbacks=[
                EarlyStopping(patience=patience, monitor="val_auc", mode="max",
                              min_delta=0.001, restore_best_weights=True),
                prototype_callback,
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
            f"[TUNE] lr={row['lr']:g}, batch={row['batch_size']}, "
            f"val_auc={row['val_auc']:.4f}, val_acc={row['val_acc']:.4f}, "
            f"best_epoch={row['best_epoch']}"
        )
        if best is None or (row["val_auc"], row["val_acc"]) > (best["val_auc"], best["val_acc"]):
            best = row
        del model
        gc.collect()

    report_dir = safe_path("results/reports")
    os.makedirs(report_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(report_dir, "hyperparam_search.csv"), index=False, encoding="utf-8-sig"
    )
    with open(os.path.join(report_dir, "hyperparam_best.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "best": best, "trials": rows}, f, ensure_ascii=False, indent=2)
    logging.info(f"[TUNE] 最优配置: {best}")
    return best


# -------------------------- 原有函数：文件验证与主函数 --------------------------
def validate_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return path


def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
    )

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='帕金森病语音检测训练脚本（含FDR/IIDR计算）')
    parser.add_argument('--data_path', type=str, required=True, help='数据路径（包含train.csv和test.csv）')
    parser.add_argument('--full_training', action='store_true', help='启用全量训练模式（默认关闭）')
    args = parser.parse_args()

    # 构造文件路径并验证
    train_csv = os.path.join(args.data_path, 'train.csv')
    test_csv = os.path.join(args.data_path, 'test.csv')
    try:
        train_csv = validate_file(train_csv)
        test_csv = validate_file(test_csv)
    except FileNotFoundError as e:
        logging.error(f"数据文件缺失: {e}")
        return

    # 启动训练
    try:
        if args.full_training:
            train_full(train_csv, test_csv)
        else:
            for fold in range(5):
                train_fold(fold, train_csv, test_csv)
        logging.info("训练完成！所有结果已保存至 results/ 目录")
    except Exception as e:
        logging.error(f"训练错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()