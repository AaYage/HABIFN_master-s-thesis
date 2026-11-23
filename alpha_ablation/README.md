# α 消融实验（独立目录）

[English](README_EN.md) | 中文

用于验证第 4 章平衡因子（α）的作用，与现有 `modules/`、`train_framed.py` 完全隔离，不修改任何原文件。

## α 定义（与论文 4.2.2 一致）

- alpha_net 输入：mfcc 与 f0 的**帧均值 + 帧标准差**（4 × 256 维），比只用均值更能反映分支稳定性；
- 输出经 softplus 后归一化 `α1 + α2 = 1` 且非负；两支平衡时 `α1 = α2 = 0.5` 为中性点；
- 训练第 1 轮固定为 0.5，之后由网络学习并分离；
- 平衡损失权重 **0.001**，避免压制方向差异。

## MAM 的动态权重（论文 5.2.3）

MAM 的 λ 直接由 α 计算，不再使用独立质量打分器：

```
λ_M = α1 / (α1 + α2)
λ_F = 1 - λ_M
```

其中 `α1` 是 mfcc 查询 f0 的方向（输出更新 mfcc），对应论文的 `α*_fm`。
`none` 模式下 `α1 = α2 = 0.5`，因此 λ = 0.5，等价固定等权融合。

## 四种配置

| 模式 | α1 (mfcc→f0) | α2 (f0→mfcc) | 说明 |
|---|---|---|---|
| `none` | 0.5 | 0.5 | 关闭平衡因子，λ=0.5 固定等权融合 |
| `a1` | 学习 | 0.5 | 仅 mfcc→f0 方向使用学习 α；双向注意力仍保留 |
| `a2` | 0.5 | 学习 | 仅 f0→mfcc 方向使用学习 α；双向注意力仍保留 |
| `both` | 学习 | 学习 | 与原模型一致 |

`both` 才施加平衡损失；`none` / 单路径不做该约束。

## 输入类型

脚本支持两种输入，除数据加载外模型结构一致：

| `--input_type` | 输入 | 时间步 | 用途 |
|---|---|---|---|
| `full` | mfcc 160 + f0 160（+ other 0），CSV | backbone 后 40 步 | 旧 full320 主模型 |
| `framed` | mfcc (200,120) + f0 (200,4)，npz | 200 步 | 帧级模型 |

`full` 的划分与 `modules/train_utils.py::train_full` 完全一致（`StratifiedShuffleSplit(test_size=0.2, seed)` + `StandardScaler`），主模型用 `batch_size=16` 时消融也应传 `--batch_size 16`，否则指标不可比。

## 运行

```powershell
# 在仓库根目录执行

# GPU（推荐）：full320 四组消融
$py = 'python'
foreach ($m in @('both','none','a1','a2')) {
  & $py alpha_ablation\train_ablation.py --alpha_mode $m --input_type full `
        --batch_size 16 --seed 42 --tag "alpha_full_$m"
}

# 帧级：去掉 --input_type 即可（默认 framed，seq_len=200）
& $py alpha_ablation\train_ablation.py --alpha_mode both --tag alpha_both
```

## 产物

```text
results/alpha_ablation/alpha_full_<mode>/
  report.txt        # input_type / alpha_mode / Acc / AUC / 敏感度 / 特异度 / 混淆矩阵
  predictions.csv   # 逐样本概率
  history.csv       # 每轮 loss/auc/alpha1/alpha2
  config.json
checkpoints/alpha_ablation/alpha_full_<mode>/final
```

帧级不传 `--tag` 时默认写到 `results/alpha_ablation/alpha_<mode>/`。

## 判读

- `both` 与 `none` 的 Acc/AUC 差距决定平衡因子是否值得保留；
- `a1` / `a2` 与 `both` 对比用于验证「双向协同」；
- `history.csv` 中的 alpha1/alpha2 可画收敛曲线（对应论文图 4-3）。

