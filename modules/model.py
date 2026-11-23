# -*- coding: utf-8 -*-
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import layers
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.regularizers import l2

try:
    from keras.utils import register_keras_serializable
except ImportError:  # Keras 2.10（TensorFlow 2.10）没有该符号
    from tensorflow.keras.utils import register_keras_serializable


# 学习率调度函数
def cosine_warmup_schedule(initial_lr=0.00001, warmup_epochs=20, total_epochs=150):
    def schedule(epoch, lr):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            cosine_decay = 0.5 * (1 + tf.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            return initial_lr * cosine_decay

    return LearningRateScheduler(schedule)


@register_keras_serializable(package="HABIFN")
class AttributeEncoder(layers.Layer):
    """APAG 第一步：局部上下文编码 + 属性空间映射（论文 3.2.2 / 式 3-7）"""

    def __init__(self, embed_dim=256, attr_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.attr_dim = attr_dim
        self.local_conv = layers.Conv1D(
            embed_dim, 5, padding="same", use_bias=False, kernel_regularizer=l2(0.003)
        )
        self.local_norm = layers.BatchNormalization()
        self.local_act = layers.LeakyReLU(0.1)
        self.attr_proj = layers.Dense(attr_dim, use_bias=False, kernel_regularizer=l2(0.001))

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "attr_dim": self.attr_dim})
        return config

    def call(self, inputs, training=None):
        x = self.local_conv(inputs)
        x = self.local_norm(x, training=training)
        x = self.local_act(x)
        return self.attr_proj(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.attr_dim)


@register_keras_serializable(package="HABIFN")
class SharedPrototypeBank(layers.Layer):
    """APAG 第二步：共享属性原型库（K 个原型，两个分支共用，论文式 3-8~3-11）

    原型以非训练参数保存；训练时由 PrototypeKMeansCallback 在 warmup 后
    对训练集属性表示执行一次 K-means，并把聚类中心写回原型后冻结，
    从而与论文“训练集聚类得到共享属性原型”一致。
    簇内偏差项用于约束属性表示靠近对应原型。
    """

    def __init__(self, num_prototypes=4, attr_dim=128, compact_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.num_prototypes = num_prototypes
        self.attr_dim = attr_dim
        self.compact_weight = compact_weight
        self.prototypes = self.add_weight(
            name="prototypes",
            shape=(num_prototypes, attr_dim),
            initializer="he_normal",
            trainable=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_prototypes": self.num_prototypes,
            "attr_dim": self.attr_dim,
            "compact_weight": self.compact_weight,
        })
        return config

    def call(self, attr):
        proto = tf.math.l2_normalize(self.prototypes, axis=-1)
        attr_n = tf.math.l2_normalize(attr, axis=-1)

        logits = tf.matmul(attr_n, proto, transpose_b=True) / tf.sqrt(tf.cast(self.attr_dim, tf.float32))
        assign = tf.nn.softmax(logits, axis=-1)  # (B, T, K) 原型匹配权重
        guide = tf.matmul(assign, proto)  # (B, T, A) 原型引导表示

        # 聚类目标：最小化簇内属性表示与原型中心的偏差
        diff = tf.expand_dims(attr_n, axis=2) - tf.expand_dims(proto, axis=0)
        dist = tf.reduce_sum(tf.square(diff), axis=-1)
        compact = tf.reduce_mean(tf.reduce_sum(assign * dist, axis=-1))
        self.add_loss(self.compact_weight * compact)

        return guide

    def set_prototypes(self, centers):
        """用训练集属性表示的 K-means 聚类中心更新共享原型。"""
        centers = np.asarray(centers, dtype=np.float32)
        expected = tuple(self.prototypes.shape)
        if centers.shape != expected:
            raise ValueError(f"原型中心形状应为 {expected}，实际为 {centers.shape}")
        self.prototypes.assign(centers)

    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package="HABIFN")
class PrototypeGuidedRefinement(layers.Layer):
    """APAG 第三步：原型引导注意力 + 逐维门控 + 残差补偿（论文式 3-12~3-19）"""

    def __init__(self, embed_dim=256, num_heads=4, guide_scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.guide_scale = guide_scale

        self.q_proj = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=l2(0.001))
        self.k_proj = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=l2(0.001))
        self.v_proj = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=l2(0.001))
        self.guide_proj = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=l2(0.001))

        self.gate_proj = layers.Dense(embed_dim, activation="sigmoid", kernel_regularizer=l2(0.001))
        self.beta_raw = self.add_weight(
            name="residual_beta", shape=(), initializer=keras.initializers.Constant(0.0), trainable=True
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "guide_scale": self.guide_scale,
        })
        return config

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        x, guide_attr = inputs
        batch_size = tf.shape(x)[0]

        q = self._split_heads(self.q_proj(x), batch_size)
        k = self._split_heads(self.k_proj(x), batch_size)
        v = self._split_heads(self.v_proj(x), batch_size)
        guide = self._split_heads(self.guide_proj(guide_attr), batch_size)

        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale
        # 原型引导项：在基础注意力得分上叠加由共享原型重构出的引导得分
        scores = scores + self.guide_scale * tf.matmul(q, guide, transpose_b=True) / scale
        attn = tf.nn.softmax(scores, axis=-1)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.embed_dim))

        # 逐维门控 + 残差补偿：beta 控制原始增强特征的保留比例
        gate = self.gate_proj(out)
        beta = tf.sigmoid(self.beta_raw)
        refined = (1.0 - beta) * gate * out + beta * out

        return self.norm(x + refined)


class AlphaTracker(keras.metrics.Metric):
    def __init__(self, name='alpha_tracker', **kwargs):
        super(AlphaTracker, self).__init__(name=name, **kwargs)
        self.alpha1_sum = self.add_weight(name='alpha1_sum', initializer='zeros')
        self.alpha2_sum = self.add_weight(name='alpha2_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, alpha1, alpha2, sample_weight=None):
        self.alpha1_sum.assign_add(tf.reduce_sum(alpha1))
        self.alpha2_sum.assign_add(tf.reduce_sum(alpha2))
        self.count.assign_add(tf.cast(tf.shape(alpha1)[0], tf.float32))

    def result(self):
        return self.alpha1_sum / (self.count + 1e-8), self.alpha2_sum / (self.count + 1e-8)

    def reset_state(self):
        self.alpha1_sum.assign(0.0)
        self.alpha2_sum.assign(0.0)
        self.count.assign(0.0)


@register_keras_serializable(package="HABIFN")
class BalancedCrossAttentionLayer(layers.Layer):
    """BCA 单层：双向交叉注意力 + 平衡因子 + 残差/归一化/FFN（论文第 4 章）"""

    def __init__(self, embed_dim=256, num_heads=4, ff_dim=None, dropout_rate=0.2, balance_weight=0.001, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.balance_weight = balance_weight
        self.dropout_rate = dropout_rate

        # 平衡因子：分支统计（均值+标准差）-> 非线性映射 -> softplus
        # 输入不再只用帧均值，加入帧间标准差，使 α 能感知分支的稳定性/质量差异。
        # 注意：不能对 alpha_net 加较强 L2 或零初始化，否则输出会被压回中性值，
        # α 退化为常数（实测逐样本方差仅 1e-4），平衡因子失效。
        self.alpha_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.alpha_net = keras.Sequential([
            layers.Dense(128, kernel_initializer="he_normal"),
            layers.LeakyReLU(0.1),
            layers.Dense(
                2,
                activation="softplus",
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer="zeros",
            ),
        ])
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)

        # 两个方向各自独立的投影
        self.mfcc_query = layers.Dense(embed_dim)
        self.f0_key = layers.Dense(embed_dim)
        self.f0_value = layers.Dense(embed_dim)
        self.f0_query = layers.Dense(embed_dim)
        self.mfcc_key = layers.Dense(embed_dim)
        self.mfcc_value = layers.Dense(embed_dim)

        self.norm_mfcc = layers.LayerNormalization(epsilon=1e-6)
        self.norm_f0 = layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = keras.Sequential([
            layers.Dense(self.ff_dim),
            layers.LeakyReLU(0.1),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "balance_weight": self.balance_weight,
        })
        return config

    def _split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def set_current_epoch(self, epoch):
        self.current_epoch.assign(epoch)

    def call(self, mfcc, f0, training=None):
        batch_size = tf.shape(mfcc)[0]

        # 分支级统计：均值反映整体能量，标准差反映帧间波动（与信噪比/发声稳定性相关）
        m_mean = tf.reduce_mean(mfcc, axis=1)
        m_std = tf.math.reduce_std(mfcc, axis=1)
        f_mean = tf.reduce_mean(f0, axis=1)
        f_std = tf.math.reduce_std(f0, axis=1)
        # 四组统计量量纲差异大，先按分支分别做 LayerNorm，再拼接送入 alpha_net
        joint_state = self.alpha_norm(tf.stack([m_mean, m_std, f_mean, f_std], axis=1))
        m_mean, m_std, f_mean, f_std = tf.unstack(joint_state, axis=1)
        joint_state = tf.concat([m_mean, m_std, f_mean, f_std], axis=-1)
        raw_alpha = self.alpha_net(joint_state) + 1e-6

        # 归一化：α1+α2=1（论文 4.2.2），中性点为 0.5
        norm_alpha = raw_alpha / tf.reduce_sum(raw_alpha, axis=-1, keepdims=True)
        alpha1 = norm_alpha[:, 0]
        alpha2 = norm_alpha[:, 1]

        # 训练第 1 轮：两个平衡因子固定为中性值 0.5；之后才由网络学习并分离
        training_flag = tf.cast(False if training is None else training, tf.bool)
        first_epoch = tf.logical_and(training_flag, tf.equal(self.current_epoch, 0))
        neutral = tf.fill(tf.shape(alpha1), 0.5)
        alpha1 = tf.where(first_epoch, neutral, alpha1)
        alpha2 = tf.where(first_epoch, neutral, alpha2)

        # 平衡因子作用于查询，调节两个方向的查询强度
        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))

        q_m = self._split_heads(self.mfcc_query(mfcc), batch_size)
        k_f = self._split_heads(self.f0_key(f0), batch_size)
        v_f = self._split_heads(self.f0_value(f0), batch_size)
        q_m = q_m * tf.reshape(alpha1, (-1, 1, 1, 1))
        scores_m = tf.matmul(q_m, k_f, transpose_b=True) / scale
        attn_m = tf.nn.softmax(scores_m, axis=-1)
        out_m = tf.matmul(attn_m, v_f)
        out_m = tf.transpose(out_m, perm=[0, 2, 1, 3])
        out_m = tf.reshape(out_m, (batch_size, -1, self.embed_dim))

        q_f = self._split_heads(self.f0_query(f0), batch_size)
        k_m = self._split_heads(self.mfcc_key(mfcc), batch_size)
        v_m = self._split_heads(self.mfcc_value(mfcc), batch_size)
        q_f = q_f * tf.reshape(alpha2, (-1, 1, 1, 1))
        scores_f = tf.matmul(q_f, k_m, transpose_b=True) / scale
        attn_f = tf.nn.softmax(scores_f, axis=-1)
        out_f = tf.matmul(attn_f, v_m)
        out_f = tf.transpose(out_f, perm=[0, 2, 1, 3])
        out_f = tf.reshape(out_f, (batch_size, -1, self.embed_dim))

        mfcc_updated = self.norm_mfcc(mfcc + self.feed_forward(out_m, training=training))
        f0_updated = self.norm_f0(f0 + self.feed_forward(out_f, training=training))

        # 缓存平衡因子与注意力统计，供机制分析（熵/权重）使用。
        # 不加 training 条件判断：部分加载路径会把 training=False 转成张量，
        # 导致缓存被跳过；无条件缓存不影响训练逻辑。
        if True:
            def _attn_stats(attn):
                p = tf.reduce_mean(attn, axis=1)  # 头平均 (B, T, T)
                ent = -tf.reduce_sum(p * tf.math.log(p + 1e-9), axis=-1)  # (B, T)
                seq = tf.cast(tf.shape(attn)[-1], tf.float32)
                ent = tf.reduce_mean(ent, axis=-1) / tf.maximum(tf.math.log(seq), 1e-6)
                return p, ent

            self.last_attn_m, self.last_entropy_m = _attn_stats(attn_m)
            self.last_attn_f, self.last_entropy_f = _attn_stats(attn_f)
            self.last_alpha1, self.last_alpha2 = alpha1, alpha2

        # 平衡损失：避免交互系数长期偏向单一分支
        self.add_loss(self.balance_weight * tf.reduce_mean(tf.square(alpha1 - alpha2)))

        return mfcc_updated, f0_updated, alpha1, alpha2

    def compute_output_shape(self, input_shape):
        mfcc_shape, f0_shape = input_shape
        return [mfcc_shape, f0_shape, (mfcc_shape[0],), (f0_shape[0],)]


@register_keras_serializable(package="HABIFN")
class ModalCrossAttention(layers.Layer):
    """BCA Module：n 层递进式双向平衡交叉注意力"""

    def __init__(self, embed_dim=256, num_heads=4, ff_dim=None, dropout_rate=0.2, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bca_layers = [
            BalancedCrossAttentionLayer(embed_dim, num_heads, self.ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]

        self.alpha_tracker = AlphaTracker()
        self.alpha1_per_epoch = []
        self.alpha2_per_epoch = []
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "num_layers": self.num_layers,
        })
        return config

    def call(self, mfcc, f0, training=None):
        mfcc_cur, f0_cur = mfcc, f0
        alpha1, alpha2 = None, None
        for layer in self.bca_layers:
            mfcc_cur, f0_cur, alpha1, alpha2 = layer(mfcc_cur, f0_cur, training=training)

        self.alpha_tracker.update_state(alpha1, alpha2)
        return mfcc_cur, f0_cur, alpha1, alpha2

    def compute_output_shape(self, input_shape):
        mfcc_shape, f0_shape = input_shape
        return [mfcc_shape, f0_shape, (mfcc_shape[0],), (f0_shape[0],)]

    def set_current_epoch(self, epoch):
        self.current_epoch.assign(epoch)
        for layer in self.bca_layers:
            layer.set_current_epoch(epoch)

    def reset_tracker(self):
        self.alpha_tracker.reset_state()

    def get_epoch_alpha(self):
        alpha1, alpha2 = self.alpha_tracker.result()
        return alpha1.numpy(), alpha2.numpy()

    def get_alpha_epoch_values(self):
        return self.alpha1_per_epoch, self.alpha2_per_epoch


class AlphaEpochVisualizationCallback(Callback):
    def __init__(self, modal_cross_attention_layer, subdir: str = "full"):
        super().__init__()
        self.modal_cross_attention_layer = modal_cross_attention_layer
        self.subdir = subdir
        self.alpha1_per_epoch = []
        self.alpha2_per_epoch = []

    def on_epoch_begin(self, epoch, logs=None):
        self.modal_cross_attention_layer.reset_tracker()
        self.modal_cross_attention_layer.set_current_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        try:
            alpha1_mean, alpha2_mean = self.modal_cross_attention_layer.get_epoch_alpha()
            self.alpha1_per_epoch.append(alpha1_mean)
            self.alpha2_per_epoch.append(alpha2_mean)
            self.modal_cross_attention_layer.alpha1_per_epoch.append(alpha1_mean)
            self.modal_cross_attention_layer.alpha2_per_epoch.append(alpha2_mean)

            print(f"Epoch {epoch + 1} | alpha1_mean: {alpha1_mean:.4f}, alpha2_mean: {alpha2_mean:.4f}")
        except Exception as e:
            print(f"获取 alpha 值时出错: {e}")

    def on_train_end(self, logs=None):
        """训练结束时生成可视化"""
        if not self.alpha1_per_epoch or not self.alpha2_per_epoch:
            print("警告: 没有收集到有效的alpha值，无法生成可视化")
            return

        print(f"开始生成Alpha可视化，alpha1长度={len(self.alpha1_per_epoch)}, alpha2长度={len(self.alpha2_per_epoch)}")

        plt.figure(figsize=(10, 5))
        plt.plot(self.alpha1_per_epoch, label="alpha1 (per epoch mean)", color="r",
                 marker="o", markersize=4, linestyle="-", linewidth=1.5)
        plt.plot(self.alpha2_per_epoch, label="alpha2 (per epoch mean)", color="b",
                 marker="s", markersize=4, linestyle="-", linewidth=1.5)

        plt.xlabel("Epoch Index", fontsize=10)
        plt.ylabel("Alpha Value (Mean)", fontsize=10)
        plt.title("Dynamic Adjustment of Alpha1 and Alpha2 During Training", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
        plt.legend(loc='best', fontsize=9)
        plt.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        fig_dir = os.path.join("results", "figures", self.subdir)
        os.makedirs(fig_dir, exist_ok=True)
        save_path = os.path.join(fig_dir, "alpha_curve.png")
        plt.savefig(save_path, dpi=300)
        print(f"Alpha可视化已保存至{save_path}")
        plt.close()


@register_keras_serializable(package="HABIFN")
class GlobalLevelAttention(layers.Layer):
    """MAM Module：全局相关性图 + 局部注意力 + 样本级动态相关性 + 动态权重（论文第 5 章）"""

    def __init__(self, embed_dim=256, num_heads=4, seq_len=40, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_len = seq_len

        # 映射到统一相关性空间
        self.proj_mfcc = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=l2(0.001))
        self.proj_f0 = layers.Dense(embed_dim, use_bias=False, kernel_regularizer=l2(0.001))

        # 可学习的全局相关性图 W_g（论文 5.2）
        self.global_corr = self.add_weight(
            name="global_correlation_Wg",
            shape=(seq_len, seq_len),
            initializer="zeros",
            trainable=True,
        )
        # 局部注意力图
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)

        self.fc_integration = layers.Dense(embed_dim, kernel_regularizer=l2(0.001))
        self.integration_act = layers.LeakyReLU(0.1)
        self.fc_out = layers.Dense(embed_dim, kernel_regularizer=l2(0.001))
        self.out_act = layers.LeakyReLU(0.1)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "seq_len": self.seq_len,
        })
        return config

    def _sample_corr(self, mfcc, f0):
        """样本级跨特征动态相关性（论文式 5-6）"""
        mfcc_flat = tf.reshape(mfcc, [tf.shape(mfcc)[0], -1])
        f0_flat = tf.reshape(f0, [tf.shape(f0)[0], -1])

        mfcc_center = mfcc_flat - tf.reduce_mean(mfcc_flat, axis=1, keepdims=True)
        f0_center = f0_flat - tf.reduce_mean(f0_flat, axis=1, keepdims=True)

        cov = tf.reduce_mean(mfcc_center * f0_center, axis=1)
        var_m = tf.reduce_mean(tf.square(mfcc_center), axis=1)
        var_f = tf.reduce_mean(tf.square(f0_center), axis=1)

        corr = cov / (tf.sqrt(var_m * var_f) + 1e-8)
        corr = (corr + 1.0) * 0.5
        return tf.reshape(corr, [-1, 1, 1])

    def call(self, mfcc_updated, f0_updated, alpha1_star, alpha2_star, training=None):
        mfcc_proj = self.proj_mfcc(mfcc_updated)
        f0_proj = self.proj_f0(f0_updated)

        # 全局相关性图：投影相似度 + 可学习全局相关性图 W_g
        scale = tf.sqrt(tf.cast(self.embed_dim, tf.float32))
        global_map = tf.nn.softmax(
            tf.matmul(mfcc_proj, f0_proj, transpose_b=True) / scale + self.global_corr,
            axis=-1,
        )
        mfcc_global = tf.matmul(global_map, f0_proj)
        f0_global = tf.matmul(global_map, mfcc_proj, transpose_a=True)

        # 局部注意力响应，并由样本级相关性校准
        mfcc_local = self.mha(mfcc_updated, mfcc_updated)
        f0_local = self.mha(f0_updated, f0_updated)
        corr = self._sample_corr(mfcc_updated, f0_updated)
        mfcc_local = mfcc_local * corr
        f0_local = f0_local * corr

        # 动态权重直接由 BCA 的平衡因子计算（论文 5.2.3）：
        # λ_M = α_fm*/(α_fm*+α_mf*)，λ_F = 1 - λ_M。
        # 代码中 alpha1 是 mfcc 查询 f0 的方向（输出更新 mfcc），对应 α_fm*。
        alpha_sum = alpha1_star + alpha2_star + 1e-8
        lambda_m = tf.reshape(alpha1_star / alpha_sum, [-1, 1, 1])
        lambda_f = 1.0 - lambda_m

        # 全局引导局部加权
        mfcc_fused = lambda_m * mfcc_global + (1.0 - lambda_m) * (mfcc_updated + mfcc_local)
        f0_fused = lambda_f * f0_global + (1.0 - lambda_f) * (f0_updated + f0_local)

        integrated = self.integration_act(self.fc_integration(tf.concat([mfcc_updated, f0_updated], axis=-1)))
        combined = self.out_act(self.fc_out(tf.concat([mfcc_fused, f0_fused], axis=-1)))
        out = self.norm(combined + integrated)

        # 输出保持 (B, 1, D)，与训练/可视化代码中的 squeeze(axis=1) 兼容
        return tf.reduce_mean(out, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1, self.embed_dim)


def _build_backbone(input_tensor, first_filters, name_prefix):
    """声学分支的时序特征提取（论文 3.2.2 的局部特征提取部分）"""
    x = layers.Reshape((160, 1), name=f"{name_prefix}_reshape")(input_tensor)
    x = layers.Conv1D(first_filters, 5 if first_filters == 256 else 3, padding="same", use_bias=False,
                      kernel_regularizer=l2(0.003), name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.LeakyReLU(0.1, name=f"{name_prefix}_act1")(x)
    x = layers.Dropout(0.2, name=f"{name_prefix}_drop1")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_pool1")(x)  # 160 -> 80

    x = layers.Conv1D(256, 3, padding="same", use_bias=False,
                      kernel_regularizer=l2(0.003), name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.LeakyReLU(0.1, name=f"{name_prefix}_act2")(x)
    x = layers.SpatialDropout1D(0.15, name=f"{name_prefix}_drop2")(x)
    x = layers.MaxPooling1D(2, name=f"{name_prefix}_pool2")(x)  # 80 -> 40

    return x


def build_model():
    # 输入定义
    mfcc_input = layers.Input(shape=(160,), name="mfcc_input")
    f0_input = layers.Input(shape=(160,), name="f0_input")
    other_input = layers.Input(shape=(0,), name="other_input")

    # ----------------------- 时序特征提取（保持序列，不做全局池化） -----------------------
    mfcc = _build_backbone(mfcc_input, 256, "mfcc")
    f0 = _build_backbone(f0_input, 64, "f0")

    # ----------------------- APAG Module -----------------------
    mfcc_attr = AttributeEncoder(name="apag_encoder_mfcc")(mfcc)
    f0_attr = AttributeEncoder(name="apag_encoder_f0")(f0)

    # 两个分支共享同一个原型库
    prototype_bank = SharedPrototypeBank(name="shared_attribute_prototypes")
    mfcc_guide = prototype_bank(mfcc_attr)
    f0_guide = prototype_bank(f0_attr)

    mfcc = PrototypeGuidedRefinement(name="apag_refine_mfcc")([mfcc, mfcc_guide])
    f0 = PrototypeGuidedRefinement(name="apag_refine_f0")([f0, f0_guide])

    # ----------------------- BCA Module -----------------------
    modal_cross_attn = ModalCrossAttention(name="bca_module")
    mfcc, f0, alpha1, alpha2 = modal_cross_attn(mfcc, f0)

    # ----------------------- MAM Module -----------------------
    merged = GlobalLevelAttention(name="global_level_attention")(mfcc, f0, alpha1, alpha2)
    merged = layers.Flatten()(merged)

    # ----------------------- 分类层 -----------------------
    merged = layers.Dense(128, activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.005))(merged)
    outputs = layers.Dense(1, activation="sigmoid")(merged)

    # 构建模型
    model = Model(inputs=[mfcc_input, f0_input, other_input], outputs=outputs)

    # 暴露 APAG 属性表示与共享原型库，供训练时的 K-means 回调使用
    model._apag_attribute_tensors = [mfcc_attr, f0_attr]
    model._prototype_bank = prototype_bank

    return model, modal_cross_attn


def _build_framed_backbone(input_tensor, first_filters, name_prefix):
    """帧级输入的时序特征提取：保留 (T, D)，不做时间维池化。"""
    x = layers.Conv1D(
        first_filters,
        5 if first_filters >= 128 else 3,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(0.003),
        name=f"{name_prefix}_conv1",
    )(input_tensor)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.LeakyReLU(0.1, name=f"{name_prefix}_act1")(x)
    x = layers.Dropout(0.2, name=f"{name_prefix}_drop1")(x)

    x = layers.Conv1D(
        256,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(0.003),
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.LeakyReLU(0.1, name=f"{name_prefix}_act2")(x)
    x = layers.SpatialDropout1D(0.15, name=f"{name_prefix}_drop2")(x)
    return x


def build_framed_model(seq_len=200, mfcc_dim=120, f0_dim=4):
    mfcc_input = layers.Input(shape=(seq_len, mfcc_dim), name="mfcc_input")
    f0_input = layers.Input(shape=(seq_len, f0_dim), name="f0_input")

    mfcc = _build_framed_backbone(mfcc_input, 128, "mfcc")
    f0 = _build_framed_backbone(f0_input, 64, "f0")

    # APAG
    mfcc_attr = AttributeEncoder(name="apag_encoder_mfcc")(mfcc)
    f0_attr = AttributeEncoder(name="apag_encoder_f0")(f0)
    prototype_bank = SharedPrototypeBank(name="shared_attribute_prototypes")
    mfcc_guide = prototype_bank(mfcc_attr)
    f0_guide = prototype_bank(f0_attr)
    mfcc = PrototypeGuidedRefinement(name="apag_refine_mfcc")([mfcc, mfcc_guide])
    f0 = PrototypeGuidedRefinement(name="apag_refine_f0")([f0, f0_guide])

    # BCA
    modal_cross_attn = ModalCrossAttention(name="bca_module")
    mfcc, f0, alpha1, alpha2 = modal_cross_attn(mfcc, f0)

    # MAM：W_g 使用真实帧数 seq_len
    merged = GlobalLevelAttention(name="global_level_attention", seq_len=seq_len)(mfcc, f0, alpha1, alpha2)
    merged = layers.Flatten()(merged)

    merged = layers.Dense(128, activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.005))(merged)
    outputs = layers.Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[mfcc_input, f0_input], outputs=outputs)
    model._apag_attribute_tensors = [mfcc_attr, f0_attr]
    model._prototype_bank = prototype_bank
    return model, modal_cross_attn


# 主函数
def main():
    # 构建模型和获取交叉注意力层
    model, modal_cross_attn = build_model()
    model.summary()

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(), keras.metrics.Recall(), keras.metrics.Precision()],
    )

    # 定义回调函数
    alpha_callback = AlphaEpochVisualizationCallback(modal_cross_attn)
    # 示例数据
    n_samples = 100
    train_mfcc = np.random.rand(n_samples, 160).astype(np.float32)
    train_f0 = np.random.rand(n_samples, 160).astype(np.float32)
    train_other = np.zeros((n_samples, 0)).astype(np.float32)
    train_labels = np.random.randint(0, 2, n_samples).astype(np.float32)

    # 划分训练集和验证集
    val_split = 0.2
    val_idx = int(n_samples * (1 - val_split))
    x_train = {
        "mfcc_input": train_mfcc[:val_idx],
        "f0_input": train_f0[:val_idx],
        "other_input": train_other[:val_idx],
    }
    y_train = train_labels[:val_idx]
    x_val = {
        "mfcc_input": train_mfcc[val_idx:],
        "f0_input": train_f0[val_idx:],
        "other_input": train_other[val_idx:],
    }
    y_val = train_labels[val_idx:]

    # 训练模型
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=30,
        batch_size=16,
        callbacks=[alpha_callback],
        verbose=1,
    )

    print("Training completed. Alpha visualization saved.")


if __name__ == "__main__":
    main()
