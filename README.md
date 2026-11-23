# HABIFN

[English](README_EN.md) | 中文

面向帕金森病（PD）语音检测的**声源—声道非对称融合网络**参考实现。

本仓库对应硕士学位论文第 4–5 章的方法与实验：以 F0 表征声源激励、以 MFCC 表征声道响应，通过属性原型引导的特征精炼（APAG）、双向平衡交叉注意力（BCA）与全局—局部融合（MAM）完成 PD/HC 判别。

> 本仓库仅包含代码与少量结果文件，**不包含语音语料、预处理特征、训练权重**。

## 1. 方法概览

| 模块    | 作用                                            |
| ----- | --------------------------------------------- |
| 双分支骨干 | MFCC 分支与 F0 分支各自提取 256 维、40 步的序列表示            |
| APAG  | 属性原型引导的特征精炼，增强与稳定病理模式一致的局部响应                  |
| BCA   | 2 层双向交叉注意力；平衡因子 α（α1+α2=1）乘在查询向量上，调节两个方向的交互强度 |
| MAM   | 全局引导与局部响应融合；融合权重 λ 由 α 导出（λ_M=α1，λ_F=α2）      |
| 输出    | 拼接 → 全连接 → Sigmoid，输出 PD 概率                   |

平衡因子 α 由各分支表示的帧均值与帧标准差生成：

```
σ = [μ(H_M); s(H_M); μ(H_F); s(H_F)]        # 4 × 256 = 1024
r = softplus(W2 · LeakyReLU(W1 · LayerNorm(σ) + b1) + b2)
α_i = r_i / (r1 + r2 + ε)                    # α1 + α2 = 1
```

## 2. 仓库结构

```
.
├── train.py                       # 主训练入口（原始音频 → 预处理 → 训练）
├── train_framed.py                # 帧级输入版本
├── modules/
│   ├── model.py                   # 主模型（APAG + BCA + MAM）
│   ├── preprocess.py              # 320 维特征提取（MFCC + F0）
│   ├── preprocess_framed.py       # 帧级特征提取
│   ├── train_utils.py             # 训练、评估、指标
│   ├── features.py / path_utils.py
├── alpha_ablation/
│   ├── model_ablation.py          # 消融模型（α 模式、融合方式可选）
│   ├── train_ablation.py          # 消融训练脚本
│   └── README.md                  # 消融配置与运行说明
├── tools/
│   ├── analysis/                  # 特征提取、α 分析、噪声评估
│   ├── attention_analysis/        # α 对照、收敛性、机制图
│   └── experiments/               # 受试者级实验
├── ui/                            # 桌面演示界面（PySide6）
└── results/
    ├── figures/                   # 论文图件（attention/balance、full）
    └── metrics/                   # 汇总指标 CSV
```

## 3. 环境

- Python 3.9
- TensorFlow 2.10.1（GPU，CUDA 11.8 / cuDNN 8.6），Keras 2.10.0
- 其他依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

Windows 下若出现 CUDA DLL 加载失败，可设置：

```powershell
$env:TF_CUDA_LIB_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
```

## 4. 数据准备

由于 PC-GITA、IPVS 等语料的使用许可限制，本仓库不提供数据。请自行申请后按 `modules/preprocess.py` 生成：

```
data/processed_features/train_data.csv
data/processed_features/val_data.csv
data/processed_features/test_data.csv
```

CSV 格式（322 列）：

| 列                     | 含义                               |
| --------------------- | -------------------------------- |
| `filename`            | 音频文件名                            |
| `label`               | 0 = HC，1 = PD                    |
| `feat_0 … feat_159`   | MFCC 统计特征（40 维 × 4 组统计量）         |
| `feat_160 … feat_319` | F0 轮廓、一阶差分、二阶差分与发声概率（40 帧 × 4 组） |

若需从原始音频重新提取，设置语料根目录后运行噪声/α 分析脚本，例如：

```bash
export PCGITA_AUDIO_DIR=/path/to/PC-GITA/All   # Windows 用 $env:PCGITA_AUDIO_DIR
```

## 5. 训练

```bash
# 主模型（原始音频路径，脚本内部完成预处理）
python train.py --data_path /path/to/audio --full_training --seed 42

# 消融：both（学习 α）与 none（固定 α=0.5）
python alpha_ablation/train_ablation.py --alpha_mode both --input_type full \
    --batch_size 16 --seed 42 --tag demo_both
python alpha_ablation/train_ablation.py --alpha_mode none --input_type full \
    --batch_size 16 --seed 42 --tag demo_none
```

输出写入 `results/alpha_ablation/<tag>/`（report、predictions、history、config）与
`checkpoints/alpha_ablation/<tag>/final`。

## 6. 分析与复现

以下脚本默认读取 `checkpoints/alpha_ablation/` 下的权重（**未随仓库发布**，需先训练）：

```bash
python tools/attention_analysis/alpha_convergence.py          # α 收敛性量化
python tools/attention_analysis/alpha_control_eval.py         # α 置换 / 固定均值对照
python tools/attention_analysis/balance_mechanism.py          # 图：α 分布 + 干预实验
python tools/attention_analysis/plot_alpha_mechanism_3seed.py # 图：3 seed 可用性响应
```

## 7. 结果文件

| 文件                                                          | 内容                         |
| ----------------------------------------------------------- | -------------------------- |
| `results/figures/attention/balance/fig3_dose_response.png`  | α 分布、α 与注意力熵、强制 α 干预、决策层影响 |
| `results/figures/attention/balance/fig_alpha_mechanism.png` | 3 seed 下分支衰减对 α 与性能的影响     |
| `results/figures/full/`                                     | 主模型训练曲线、混淆矩阵、FDR-IIDR      |
| `results/metrics/main_seed/main_both_none.csv`              | 3 seed 的 both vs none 配对结果 |
| `results/metrics/alpha_control/alpha_control.csv`           | α 置换 / 固定均值对照              |
| `results/metrics/alpha_convergence/convergence.csv`         | α 收敛性指标                    |
