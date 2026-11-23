# -*- coding: utf-8 -*-
"""α 消融用的模型变体。

与 modules/model.py 的区别仅在「平衡因子是否启用」：
    none  : α1 = α2 = 0.5，等价关闭平衡因子
    a1    : 仅 mfcc→f0 方向使用学习到的 α1，α2 固定为 0.5
    a2    : 仅 f0→mfcc 方向使用学习到的 α2，α1 固定为 0.5
    both  : 两个方向都使用学习到的 α（与原模型一致）

其余结构（backbone / APAG / MAM / 分类头）与 build_framed_model 完全一致。
"""
import tensorflow as tf
import keras
from keras import layers

from modules.model import (
    AlphaTracker,
    AttributeEncoder,
    BalancedCrossAttentionLayer,
    GlobalLevelAttention,
    PrototypeGuidedRefinement,
    SharedPrototypeBank,
    _build_backbone,
    _build_framed_backbone,
)

ALPHA_MODES = ("none", "a1", "a2", "both")


class BalancedCrossAttentionLayerAblation(BalancedCrossAttentionLayer):
    """支持 α 配置的单层 BCA。"""

    def __init__(self, *args, alpha_mode="both", gate_mode="residual", **kwargs):
        if alpha_mode not in ALPHA_MODES:
            raise ValueError(f"alpha_mode 必须是 {ALPHA_MODES}，收到 {alpha_mode}")
        self.alpha_mode = alpha_mode
        self.gate_mode = gate_mode
        # 机制分析用：外部覆盖 α（置换/固定均值等），默认 None 时行为不变
        self.alpha_override = None
        super().__init__(*args, **kwargs)

    def call(self, mfcc, f0, training=None):
        batch_size = tf.shape(mfcc)[0]

        m_mean = tf.reduce_mean(mfcc, axis=1)
        m_std = tf.math.reduce_std(mfcc, axis=1)
        f_mean = tf.reduce_mean(f0, axis=1)
        f_std = tf.math.reduce_std(f0, axis=1)
        joint_state = self.alpha_norm(tf.stack([m_mean, m_std, f_mean, f_std], axis=1))
        m_mean, m_std, f_mean, f_std = tf.unstack(joint_state, axis=1)
        joint_state = tf.concat([m_mean, m_std, f_mean, f_std], axis=-1)
        raw_alpha = self.alpha_net(joint_state) + 1e-6
        norm_alpha = raw_alpha / tf.reduce_sum(raw_alpha, axis=-1, keepdims=True)
        alpha1 = norm_alpha[:, 0]
        alpha2 = norm_alpha[:, 1]

        # 与原实现一致：训练第 1 轮固定为中性值 0.5
        training_flag = tf.cast(False if training is None else training, tf.bool)
        first_epoch = tf.logical_and(training_flag, tf.equal(self.current_epoch, 0))
        neutral = tf.fill(tf.shape(alpha1), 0.5)
        alpha1 = tf.where(first_epoch, neutral, alpha1)
        alpha2 = tf.where(first_epoch, neutral, alpha2)

        # 消融：按模式固定对应方向的系数（中性值 0.5）
        if self.alpha_mode == "none":
            alpha1 = tf.fill(tf.shape(alpha1), 0.5)
            alpha2 = tf.fill(tf.shape(alpha2), 0.5)
        elif self.alpha_mode == "a1":
            alpha2 = tf.fill(tf.shape(alpha2), 0.5)
        elif self.alpha_mode == "a2":
            alpha1 = tf.fill(tf.shape(alpha1), 0.5)

        if self.alpha_override is not None:
            ov = tf.cast(self.alpha_override, tf.float32)
            alpha1 = ov[..., 0]
            alpha2 = ov[..., 1]

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

        if self.gate_mode == "gate":
            # α 直接控制“自身分支 vs 交互分支”的融合比例
            a1 = tf.reshape(alpha1, (-1, 1, 1))
            a2 = tf.reshape(alpha2, (-1, 1, 1))
            mfcc_updated = self.norm_mfcc(
                a1 * mfcc + (1.0 - a1) * self.feed_forward(out_m, training=training))
            f0_updated = self.norm_f0(
                a2 * f0 + (1.0 - a2) * self.feed_forward(out_f, training=training))
        elif self.gate_mode == "res_gate":
            # 保留自身分支全部信息，只调节交互分支的增益；中性点 α=0.5 时与原残差一致
            a1 = tf.reshape(alpha1, (-1, 1, 1))
            a2 = tf.reshape(alpha2, (-1, 1, 1))
            mfcc_updated = self.norm_mfcc(
                mfcc + 2.0 * (1.0 - a1) * self.feed_forward(out_m, training=training))
            f0_updated = self.norm_f0(
                f0 + 2.0 * (1.0 - a2) * self.feed_forward(out_f, training=training))
        else:
            mfcc_updated = self.norm_mfcc(mfcc + self.feed_forward(out_m, training=training))
            f0_updated = self.norm_f0(f0 + self.feed_forward(out_f, training=training))

        # 无条件缓存，避免 training=False 被转成张量时跳过缓存
        if True:
            def _attn_stats(attn):
                p = tf.reduce_mean(attn, axis=1)
                ent = -tf.reduce_sum(p * tf.math.log(p + 1e-9), axis=-1)
                seq = tf.cast(tf.shape(attn)[-1], tf.float32)
                return p, tf.reduce_mean(ent, axis=-1) / tf.maximum(tf.math.log(seq), 1e-6)

            self.last_attn_m, self.last_entropy_m = _attn_stats(attn_m)
            self.last_attn_f, self.last_entropy_f = _attn_stats(attn_f)
            self.last_alpha1, self.last_alpha2 = alpha1, alpha2

        # 只在双路径都启用时施加平衡损失，单路径/关闭时不做无意义约束
        if self.alpha_mode == "both":
            self.add_loss(self.balance_weight * tf.reduce_mean(tf.square(alpha1 - alpha2)))

        return mfcc_updated, f0_updated, alpha1, alpha2


class ModalCrossAttentionAblation(layers.Layer):
    """BCA 模块，支持 α 配置。"""

    def __init__(self, embed_dim=256, num_heads=4, ff_dim=None, dropout_rate=0.2,
                 num_layers=2, alpha_mode="both", balance_weight=0.001,
                 gate_mode="residual", **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.alpha_mode = alpha_mode
        self.bca_layers = [
            BalancedCrossAttentionLayerAblation(
                embed_dim, num_heads, self.ff_dim, dropout_rate,
                alpha_mode=alpha_mode, balance_weight=balance_weight, gate_mode=gate_mode,
            )
            for _ in range(num_layers)
        ]
        self.alpha_tracker = AlphaTracker()
        self.alpha1_per_epoch = []
        self.alpha2_per_epoch = []
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)

    def call(self, mfcc, f0, training=None):
        mfcc_cur, f0_cur = mfcc, f0
        alpha1 = alpha2 = None
        for layer in self.bca_layers:
            mfcc_cur, f0_cur, alpha1, alpha2 = layer(mfcc_cur, f0_cur, training=training)
        self.alpha_tracker.update_state(alpha1, alpha2)
        return mfcc_cur, f0_cur, alpha1, alpha2

    def set_current_epoch(self, epoch):
        self.current_epoch.assign(epoch)
        for layer in self.bca_layers:
            layer.set_current_epoch(epoch)

    def reset_tracker(self):
        self.alpha_tracker.reset_state()

    def get_epoch_alpha(self):
        alpha1, alpha2 = self.alpha_tracker.result()
        return float(alpha1.numpy()), float(alpha2.numpy())


def build_full_model_ablation(alpha_mode="both", balance_weight=0.001, gate_mode="residual"):

    l2 = keras.regularizers.l2

    mfcc_input = layers.Input(shape=(160,), name="mfcc_input")
    f0_input = layers.Input(shape=(160,), name="f0_input")
    other_input = layers.Input(shape=(0,), name="other_input")

    mfcc = _build_backbone(mfcc_input, 256, "mfcc")
    f0 = _build_backbone(f0_input, 64, "f0")

    mfcc_attr = AttributeEncoder(name="apag_encoder_mfcc")(mfcc)
    f0_attr = AttributeEncoder(name="apag_encoder_f0")(f0)
    prototype_bank = SharedPrototypeBank(name="shared_attribute_prototypes")
    mfcc_guide = prototype_bank(mfcc_attr)
    f0_guide = prototype_bank(f0_attr)
    mfcc = PrototypeGuidedRefinement(name="apag_refine_mfcc")([mfcc, mfcc_guide])
    f0 = PrototypeGuidedRefinement(name="apag_refine_f0")([f0, f0_guide])

    bca = ModalCrossAttentionAblation(name="bca_module", alpha_mode=alpha_mode,
                                      balance_weight=balance_weight, gate_mode=gate_mode)
    mfcc, f0, alpha1, alpha2 = bca(mfcc, f0)

    merged = GlobalLevelAttention(name="global_level_attention", seq_len=40)(mfcc, f0, alpha1, alpha2)
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.005))(merged)
    outputs = layers.Dense(1, activation="sigmoid")(merged)

    model = keras.Model(inputs=[mfcc_input, f0_input, other_input], outputs=outputs)
    model._apag_attribute_tensors = [mfcc_attr, f0_attr]
    model._prototype_bank = prototype_bank
    return model, bca


def build_framed_model_ablation(alpha_mode="both", seq_len=200, mfcc_dim=120, f0_dim=4,
                                balance_weight=0.001, gate_mode="residual"):
    l2 = keras.regularizers.l2

    mfcc_input = layers.Input(shape=(seq_len, mfcc_dim), name="mfcc_input")
    f0_input = layers.Input(shape=(seq_len, f0_dim), name="f0_input")

    mfcc = _build_framed_backbone(mfcc_input, 128, "mfcc")
    f0 = _build_framed_backbone(f0_input, 64, "f0")

    mfcc_attr = AttributeEncoder(name="apag_encoder_mfcc")(mfcc)
    f0_attr = AttributeEncoder(name="apag_encoder_f0")(f0)
    prototype_bank = SharedPrototypeBank(name="shared_attribute_prototypes")
    mfcc_guide = prototype_bank(mfcc_attr)
    f0_guide = prototype_bank(f0_attr)
    mfcc = PrototypeGuidedRefinement(name="apag_refine_mfcc")([mfcc, mfcc_guide])
    f0 = PrototypeGuidedRefinement(name="apag_refine_f0")([f0, f0_guide])

    bca = ModalCrossAttentionAblation(name="bca_module", alpha_mode=alpha_mode,
                                      balance_weight=balance_weight, gate_mode=gate_mode)
    mfcc, f0, alpha1, alpha2 = bca(mfcc, f0)

    merged = GlobalLevelAttention(name="global_level_attention", seq_len=seq_len)(mfcc, f0, alpha1, alpha2)
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.005))(merged)
    outputs = layers.Dense(1, activation="sigmoid")(merged)

    model = keras.Model(inputs=[mfcc_input, f0_input], outputs=outputs)
    model._apag_attribute_tensors = [mfcc_attr, f0_attr]
    model._prototype_bank = prototype_bank
    return model, bca
