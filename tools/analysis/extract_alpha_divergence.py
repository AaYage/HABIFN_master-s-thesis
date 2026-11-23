import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import re
import json
import argparse
import sys
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# 脚本位于 tools/analysis/，工程根目录在上两级
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
DEFAULT_AUDIO_DIR = os.environ.get(
    'PCGITA_AUDIO_DIR',
    os.path.join(BASE_DIR, 'data', 'raw', 'PC-GITA', 'All', 'test'),
)
DEFAULT_OUT_ROOT = os.path.join(BASE_DIR, 'results', 'alpha_analysis')
ALPHA_PATH_VERSION = 'post_attention_path_gate_v1'

PD_COLOR = '#D55E00'
HC_COLOR = '#0072B2'


def configure_utf8_console(stream):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8')


def require_corrected_model_protocol(manifest_path):
    if not os.path.exists(manifest_path):
        raise RuntimeError('Checkpoint protocol is missing; retrain with the corrected alpha path')
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('alpha_path_version') != ALPHA_PATH_VERSION:
        raise RuntimeError('Checkpoint uses a different alpha path; retrain with the corrected alpha path')


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
        return self.alpha1_sum / self.count, self.alpha2_sum / self.count

    def reset_state(self):
        self.alpha1_sum.assign(0.0)
        self.alpha2_sum.assign(0.0)
        self.count.assign(0.0)


class ModalCrossAttention(layers.Layer):
    def __init__(self, embed_dim=256, num_heads=4, ff_dim=None, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim or embed_dim * 4
        self.alpha_net = keras.Sequential([
            layers.Dense(64, activation="LeakyReLU", kernel_initializer="he_normal", kernel_regularizer=l2(0.001)),
            layers.Dense(2, activation="softplus", kernel_initializer="he_normal",
                         bias_initializer=keras.initializers.Constant(0.5413), kernel_regularizer=l2(0.001)),
        ])
        self.mfcc_query = layers.Dense(embed_dim)
        self.f0_key = layers.Dense(embed_dim)
        self.f0_value = layers.Dense(embed_dim)
        self.f0_query = layers.Dense(embed_dim)
        self.mfcc_key = layers.Dense(embed_dim)
        self.mfcc_value = layers.Dense(embed_dim)
        self.norm_mfcc = layers.LayerNormalization(epsilon=1e-6)
        self.norm_f0 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = keras.Sequential([
            layers.Dense(self.ff_dim, activation="LeakyReLU"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        self.alpha_tracker = AlphaTracker()
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.alpha_override = self.add_weight(
            name='alpha_override', shape=(2,),
            initializer=keras.initializers.Constant(-1.0), trainable=False)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, mfcc, f0, training=None):
        batch_size = tf.shape(mfcc)[0]
        if training is None:
            training = False
        combined_features = tf.concat([mfcc, f0], axis=-1)
        alphas = self.alpha_net(combined_features)
        alpha1 = tf.cond(
            tf.logical_and(tf.cast(training, tf.bool), tf.equal(self.current_epoch, 0)),
            lambda: tf.ones([batch_size]),
            lambda: alphas[:, 0]
        )
        alpha2 = tf.cond(
            tf.logical_and(tf.cast(training, tf.bool), tf.equal(self.current_epoch, 0)),
            lambda: tf.ones([batch_size]),
            lambda: alphas[:, 1]
        )
        self.alpha_tracker.update_state(alpha1, alpha2)
        mfcc_3d = tf.expand_dims(mfcc, 1)
        f0_3d = tf.expand_dims(f0, 1)
        Q = self.mfcc_query(mfcc_3d)
        K = self.f0_key(f0_3d)
        V = self.f0_value(f0_3d)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        scores = tf.matmul(Q, K, transpose_b=True)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        scaled_attention = tf.matmul(attn_weights, V)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, 1, self.embed_dim))
        att_mfcc = tf.squeeze(concat_attention, axis=1)

        Q_f0 = self.f0_query(f0_3d)
        K_mfcc = self.mfcc_key(mfcc_3d)
        V_mfcc = self.mfcc_value(mfcc_3d)
        Q_f0 = self.split_heads(Q_f0, batch_size)
        K_mfcc = self.split_heads(K_mfcc, batch_size)
        V_mfcc = self.split_heads(V_mfcc, batch_size)
        scores_f0 = tf.matmul(Q_f0, K_mfcc, transpose_b=True)
        attn_weights_f0 = tf.nn.softmax(scores_f0, axis=-1)
        scaled_attention_f0 = tf.matmul(attn_weights_f0, V_mfcc)
        scaled_attention_f0 = tf.transpose(scaled_attention_f0, perm=[0, 2, 1, 3])
        concat_attention_f0 = tf.reshape(scaled_attention_f0, (batch_size, 1, self.embed_dim))
        att_f0 = tf.squeeze(concat_attention_f0, axis=1)

        effective_alpha1 = tf.where(self.alpha_override[0] >= 0, self.alpha_override[0], alpha1)
        effective_alpha2 = tf.where(self.alpha_override[1] >= 0, self.alpha_override[1], alpha2)
        att_mfcc_ffn = self.feed_forward(att_mfcc) * tf.expand_dims(effective_alpha1, -1)
        att_f0_ffn = self.feed_forward(att_f0) * tf.expand_dims(effective_alpha2, -1)
        mfcc_updated = self.norm_mfcc(mfcc + att_mfcc_ffn)
        f0_updated = self.norm_f0(f0 + att_f0_ffn)
        return mfcc_updated, f0_updated

    def set_alpha_override(self, alpha1=None, alpha2=None):
        self.alpha_override.assign([
            -1.0 if alpha1 is None else float(alpha1),
            -1.0 if alpha2 is None else float(alpha2),
        ])

    def set_current_epoch(self, epoch):
        self.current_epoch.assign(epoch)

    def reset_tracker(self):
        self.alpha_tracker.reset_state()

    def get_epoch_alpha(self):
        alpha1, alpha2 = self.alpha_tracker.result()
        return alpha1.numpy(), alpha2.numpy()


class ECAModule(layers.Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.avg_pool = layers.GlobalAveragePooling1D()
        self.conv = layers.Conv1D(1, kernel_size=kernel_size, padding="same", use_bias=False)
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, inputs):
        x = self.avg_pool(inputs)
        x = tf.expand_dims(x, -1)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = tf.squeeze(x, -1)
        x = tf.expand_dims(x, 1)
        return layers.Multiply()([inputs, x])


class FeatureLevelAttention(layers.Layer):
    def __init__(self, embed_dim=256, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.conv_block = keras.Sequential([
            layers.DepthwiseConv1D(5, padding="same", activation=None, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
            layers.Conv1D(embed_dim, 1, activation=None, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.1),
        ])
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.gate = layers.Dense(embed_dim, activation="sigmoid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        conv_out = self.conv_block(inputs)
        att_out = self.mha(conv_out, conv_out)
        gate_value = self.gate(att_out)
        residual = att_out + gate_value * att_out
        return self.norm(residual)


class GlobalLevelAttention(layers.Layer):
    def __init__(self, embed_dim=256, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.fc_integration = layers.Dense(embed_dim, activation="LeakyReLU")
        self.fc_out = layers.Dense(embed_dim, activation="LeakyReLU")

    def call(self, mfcc_updated, f0_updated):
        combined = tf.concat([mfcc_updated, f0_updated], axis=-1)
        integrated_feat = self.fc_integration(combined)
        att_out = self.mha(integrated_feat, integrated_feat)
        mfcc_attended = mfcc_updated * att_out
        f0_attended = f0_updated * att_out
        combined_attended = tf.concat([mfcc_attended, f0_attended], axis=-1)
        combined_attended = self.fc_out(combined_attended)
        residual_output = combined_attended + integrated_feat
        return self.norm(residual_output)


class AlphaEpochCurveCallback(Callback):
    def __init__(self, mca_layer, save_path):
        super().__init__()
        self.mca = mca_layer
        self.save_path = save_path
        self.alpha1_curve = []
        self.alpha2_curve = []

    def on_epoch_begin(self, epoch, logs=None):
        self.mca.reset_tracker()
        self.mca.set_current_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            a1, a2 = 1.0, 1.0
        else:
            a1, a2 = self.mca.get_epoch_alpha()
        self.alpha1_curve.append(float(a1))
        self.alpha2_curve.append(float(a2))
        if (epoch + 1) % 10 == 0:
            print('  epoch %d | alpha1=%.4f alpha2=%.4f | val_auc=%s' % (
                epoch + 1, a1, a2, '%.4f' % logs.get('val_auc', float('nan')) if logs else 'na'))

    def on_train_end(self, logs=None):
        if not self.alpha1_curve:
            return
        ep = np.arange(1, len(self.alpha1_curve) + 1)
        plt.figure(figsize=(8, 4.5))
        plt.plot(ep, self.alpha1_curve, color=PD_COLOR, marker='o', ms=3, lw=1.5, label=r'$\alpha_1$ (per-epoch mean)')
        plt.plot(ep, self.alpha2_curve, color=HC_COLOR, marker='s', ms=3, lw=1.5, label=r'$\alpha_2$ (per-epoch mean)')
        plt.scatter([1], [1.0], s=40, c='red', zorder=5, label='fixed at 1 (epoch 1)')
        plt.xlabel('Epoch')
        plt.ylabel('Balanced factor value')
        plt.title('Training dynamics of the balanced factors')
        plt.grid(True, ls='--', alpha=0.5, lw=0.5)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=300)
        plt.close()


def build_model():
    mfcc_input = layers.Input(shape=(160,), name="mfcc_input")
    f0_input = layers.Input(shape=(160,), name="f0_input")
    mfcc = layers.Reshape((160, 1))(mfcc_input)
    conv1 = layers.Conv1D(256, 5, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(mfcc)
    conv1 = layers.Dropout(0.2)(conv1)
    conv1 = ECAModule()(conv1)
    conv2 = layers.Conv1D(256, 3, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(mfcc)
    conv2 = layers.Dropout(0.2)(conv2)
    conv2 = ECAModule()(conv2)
    conv3 = layers.Conv1D(256, 1, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(mfcc)
    conv3 = layers.Dropout(0.2)(conv3)
    conv3 = ECAModule()(conv3)
    mfcc = layers.Concatenate()([conv1, conv2, conv3])
    mfcc = layers.BatchNormalization()(mfcc)
    mfcc = layers.MaxPooling1D(2)(mfcc)
    mfcc = layers.Conv1D(256, 3, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(mfcc)
    mfcc = layers.Dropout(0.2)(mfcc)
    mfcc = layers.BatchNormalization()(mfcc)
    mfcc = layers.MaxPooling1D(2)(mfcc)
    mfcc = ECAModule()(mfcc)
    mfcc = layers.Conv1D(256, 3, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(mfcc)
    mfcc = layers.SpatialDropout1D(0.15)(mfcc)
    mfcc = layers.BatchNormalization()(mfcc)
    mfcc = layers.GlobalAveragePooling1D()(mfcc)

    f0 = layers.Reshape((160, 1))(f0_input)
    f0 = layers.Conv1D(64, 3, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(f0)
    f0 = layers.Dropout(0.2)(f0)
    f0 = layers.BatchNormalization()(f0)
    f0 = layers.MaxPooling1D(2)(f0)
    f0 = ECAModule()(f0)
    f0 = layers.Conv1D(256, 3, padding="same", activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.003))(f0)
    f0 = layers.SpatialDropout1D(0.15)(f0)
    f0 = layers.BatchNormalization()(f0)
    f0 = layers.GlobalAveragePooling1D()(f0)

    mfcc = FeatureLevelAttention()(tf.expand_dims(mfcc, 1))
    f0 = FeatureLevelAttention()(tf.expand_dims(f0, 1))
    mfcc = tf.squeeze(mfcc, 1)
    f0 = tf.squeeze(f0, 1)

    modal_cross_attn = ModalCrossAttention()
    mfcc, f0 = modal_cross_attn(mfcc, f0)

    mfcc_expanded = layers.Reshape((1, -1))(mfcc)
    f0_expanded = layers.Reshape((1, -1))(f0)
    merged = GlobalLevelAttention()(mfcc_expanded, f0_expanded)
    merged = layers.Flatten()(merged)
    merged = layers.Dense(128, activation=layers.LeakyReLU(0.1), kernel_regularizer=l2(0.005))(merged)
    outputs = layers.Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[mfcc_input, f0_input], outputs=outputs)
    return model, modal_cross_attn


def build_alpha_model(model):
    mca = None
    for layer in model.layers:
        if isinstance(layer, ModalCrossAttention):
            mca = layer
            break
    if mca is None:
        raise RuntimeError('ModalCrossAttention layer not found in model')
    node = mca.inbound_nodes[0]
    node_inputs = node.keras_inputs
    merged_features = tf.concat([node_inputs[0], node_inputs[1]], axis=-1)
    alpha_out = mca.alpha_net(merged_features)
    alpha_model = Model(model.inputs, alpha_out)
    return alpha_model, mca


def predict_path_ablations(model, mca, inputs, batch_size=256):
    predictions = []
    try:
        for alpha1, alpha2 in [(0.0, None), (None, 0.0)]:
            mca.set_alpha_override(alpha1=alpha1, alpha2=alpha2)
            predictions.append(model.predict(inputs, batch_size=batch_size, verbose=0).ravel())
    finally:
        mca.set_alpha_override()
    return predictions


def parse_stem(stem, regex_pattern, hc_prefix):
    m = re.match(regex_pattern, stem)
    if not m:
        raise ValueError(f'unexpected feature file name: {stem}, pattern={regex_pattern}')
    subject = m.group(1)
    task = m.group(2)
    aug = int(m.group(3))
    is_pd = not subject.startswith(hc_prefix)
    return subject, task, aug, is_pd


def load_dataset(data_dir, regex_pattern, hc_prefix):
    records = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for f in sorted(os.listdir(split_dir)):
            if not f.endswith('.npz'):
                continue
            z = np.load(os.path.join(split_dir, f), allow_pickle=True)
            stem = f[:-4]
            subject, task, aug, is_pd = parse_stem(stem, regex_pattern, hc_prefix)
            records.append({
                'orig_split': split,
                'filename': stem + '.wav',
                'subject': subject,
                'task': task,
                'aug': aug,
                'label': int(z['label']),
                'is_pd': is_pd,
                'feature': z['feature_1d'].astype(np.float64),
            })
    df = pd.DataFrame(records)
    before = len(df)
    df = df.drop_duplicates(subset=['subject', 'task', 'aug'], keep='first').reset_index(drop=True)
    print('[DATA] raw npz files: %d -> deduplicated (subject,task,aug): %d' % (before, len(df)))
    label_check = (df['is_pd'].astype(int) == df['label']).all()
    if not label_check:
        print('[WARN] subject prefix and npz label disagree for some samples')
    return df


def assign_subject_split(df, seed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """按比例划分受试者，不再硬编码35/8/7"""
    subjects = sorted(df['subject'].unique())
    pd_subjects = [s for s in subjects if df.loc[df['subject'] == s, 'label'].iloc[0] == 1]
    hc_subjects = [s for s in subjects if df.loc[df['subject'] == s, 'label'].iloc[0] == 0]
    rng = np.random.RandomState(seed)
    rng.shuffle(pd_subjects)
    rng.shuffle(hc_subjects)

    def split_list(lst, r, vr):
        n = len(lst)
        tr = int(n * r)
        vr_ = int(n * vr)
        return set(lst[:tr]), set(lst[tr:tr+vr_]), set(lst[tr+vr_:])

    pd_train, pd_val, pd_test = split_list(pd_subjects, train_ratio, val_ratio)
    hc_train, hc_val, hc_test = split_list(hc_subjects, train_ratio, val_ratio)

    train_subj = pd_train | hc_train
    val_subj = pd_val | hc_val
    test_subj = pd_test | hc_test

    print('[SPLIT] subjects: train=%d val=%d test=%d (PD %d/%d/%d, HC %d/%d/%d)' % (
        len(train_subj), len(val_subj), len(test_subj),
        len(pd_train), len(pd_val), len(pd_test),
        len(hc_train), len(hc_val), len(hc_test)))
    return train_subj, val_subj, test_subj


def split_features(X, mean, scale):
    Xs = (X - mean) / scale
    return Xs[:, :160].astype(np.float32), Xs[:, 160:320].astype(np.float32)


def compute_recording_acoustics(wav_path):
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < sr * 0.2:
        return np.nan, np.nan, np.nan
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr, frame_length=1024, hop_length=32)
    f0v = f0[~np.isnan(f0)]
    if len(f0v) < 10:
        return np.nan, np.nan, np.nan
    T = 1.0 / f0v
    dT = np.abs(np.diff(T))
    jitter = float(np.mean(dT) / np.mean(T) * 100.0)
    f0_std = float(np.std(f0v))
    voiced_pct = float(len(f0v) / len(f0) * 100.0)
    return jitter, f0_std, voiced_pct


def load_or_compute_acoustics(aug0_df, audio_dir, cache_path):
    cache = pd.DataFrame()
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        print('[AUDIO] jitter cache loaded: %d recordings' % len(cache))
    cached = set(cache['recording'].tolist()) if 'recording' in cache.columns else set()
    rows = []
    todo = [(r['subject'], r['task'], r['is_pd']) for _, r in aug0_df.iterrows()
            if r['subject'] + r['task'] not in cached]
    for i, (subject, task, is_pd) in enumerate(todo):
        wav = os.path.join(audio_dir, 'pd' if is_pd else 'hc', subject + task + '.wav')
        if not os.path.exists(wav):
            rows.append({'recording': subject + task, 'jitter_pct': np.nan,
                         'f0_std_audio': np.nan, 'voiced_pct': np.nan})
            continue
        jitter, f0_std, voiced = compute_recording_acoustics(wav)
        rows.append({'recording': subject + task, 'jitter_pct': jitter,
                     'f0_std_audio': f0_std, 'voiced_pct': voiced})
        if (i + 1) % 25 == 0:
            print('[AUDIO] %d / %d recordings analyzed' % (i + 1, len(todo)))
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
        print('[AUDIO] jitter cache saved: %d recordings' % len(cache))
    aug0_df = aug0_df.copy()
    aug0_df['recording'] = aug0_df['subject'] + aug0_df['task']
    aug0_df = aug0_df.merge(cache, on='recording', how='left')
    return aug0_df


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sign(a[:, None] - b[None, :]).mean())


def icc_2_1(x):
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse)))


def wilson_interval(k, n, z=1.959963985):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def make_fig8(subj_table, aug0_df, source_metric, source_label, icc1, icc2, out_path):
    hc = subj_table[subj_table['label'] == 0]
    pdg = subj_table[subj_table['label'] == 1]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    data_a = [hc['alpha1'], pdg['alpha1'], hc['alpha2'], pdg['alpha2']]
    pos = [0, 1, 2, 3]
    vp = ax_a.violinplot(data_a, positions=pos, showmedians=False, showextrema=False, widths=0.8)
    colors_a = [HC_COLOR, PD_COLOR, HC_COLOR, PD_COLOR]
    for body, c in zip(vp['bodies'], colors_a):
        body.set_facecolor(c)
        body.set_alpha(0.45)
    ax_a.boxplot(data_a, positions=pos, widths=0.16, showfliers=False,
                 medianprops=dict(color='black', lw=1.2),
                 boxprops=dict(color='black', lw=0.8),
                 whiskerprops=dict(color='black', lw=0.8),
                 capprops=dict(color='black', lw=0.8))
    u1, p1 = stats.mannwhitneyu(hc['alpha1'], pdg['alpha1'], alternative='two-sided')
    u2, p2 = stats.mannwhitneyu(hc['alpha2'], pdg['alpha2'], alternative='two-sided')
    ax_a.set_xticks(pos)
    ax_a.set_xticklabels([r'$\alpha_1$ HC', r'$\alpha_1$ PD', r'$\alpha_2$ HC', r'$\alpha_2$ PD'])
    ax_a.set_ylabel('Balanced factor (subject mean)')
    ax_a.set_title(r'(a) Group distributions  $p_{\alpha_1}$=%.1e, $p_{\alpha_2}$=%.1e' % (p1, p2), fontsize=11)
    ax_a.grid(axis='y', ls='--', alpha=0.4, lw=0.5)

    ax_b.scatter(hc['alpha1'], hc['alpha2'], c=HC_COLOR, alpha=0.65, s=42, label='HC (n=%d)' % len(hc), edgecolors='white', lw=0.4)
    ax_b.scatter(pdg['alpha1'], pdg['alpha2'], c=PD_COLOR, alpha=0.65, s=42, label='PD (n=%d)' % len(pdg), edgecolors='white', lw=0.4)
    lims = [min(subj_table['alpha1'].min(), subj_table['alpha2'].min()) - 0.02,
            max(subj_table['alpha1'].max(), subj_table['alpha2'].max()) + 0.02]
    ax_b.plot(lims, lims, ls='--', color='gray', lw=1, alpha=0.8, label=r'$\alpha_1=\alpha_2$')
    ax_b.scatter([hc['alpha1'].mean()], [hc['alpha2'].mean()], marker='X', s=140, c=HC_COLOR, edgecolors='black', zorder=5)
    ax_b.scatter([pdg['alpha1'].mean()], [pdg['alpha2'].mean()], marker='X', s=140, c=PD_COLOR, edgecolors='black', zorder=5)
    ax_b.set_xlabel(r'$\alpha_1$ (F0 to MFCC path)')
    ax_b.set_ylabel(r'$\alpha_2$ (MFCC to F0 path)')
    ax_b.set_title('(b) Subject‑level balanced‑factor plane', fontsize=11)
    ax_b.legend(fontsize=9, loc='best')
    ax_b.grid(ls='--', alpha=0.4, lw=0.5)

    valid = aug0_df[[source_metric, 'alpha1', 'alpha2', 'label']].dropna()
    r1 = stats.spearmanr(valid[source_metric], valid['alpha1'])
    r2 = stats.spearmanr(valid[source_metric], valid['alpha2'])
    hc_v = valid[valid['label'] == 0]
    pd_v = valid[valid['label'] == 1]
    ax_c.scatter(hc_v[source_metric], hc_v['alpha1'], c=HC_COLOR, alpha=0.55, s=28, label='HC', edgecolors='none')
    ax_c.scatter(pd_v[source_metric], pd_v['alpha1'], c=PD_COLOR, alpha=0.55, s=28, label='PD', edgecolors='none')
    coef = np.polyfit(valid[source_metric], valid['alpha1'], 1)
    xs = np.linspace(valid[source_metric].min(), valid[source_metric].max(), 50)
    ax_c.plot(xs, np.polyval(coef, xs), color='black', lw=1.4, ls='-')
    ax_c.set_xlabel(source_label)
    ax_c.set_ylabel(r'$\alpha_1$')
    ax_c.set_title(r'(c) $\alpha$ vs F0 variation  $\rho_{\alpha_1}$=%.3f (p=%.1e), $\rho_{\alpha_2}$=%.3f' % (
        r1.statistic, r1.pvalue, r2.statistic), fontsize=11)
    ax_c.legend(fontsize=9)
    ax_c.grid(ls='--', alpha=0.4, lw=0.5)

    tasks = sorted(aug0_df['task'].unique())
    groups = [(0, HC_COLOR, 'HC'), (1, PD_COLOR, 'PD')]
    pos_d, data_d, colors_d, labels_d = [], [], [], []
    p = 0
    for task in tasks:
        for lab, c, gname in groups:
            vals = aug0_df[(aug0_df['task'] == task) & (aug0_df['label'] == lab)]['alpha2'].values
            data_d.append(vals)
            pos_d.append(p)
            colors_d.append(c)
            labels_d.append('%s\n%s' % (task, gname))
            p += 1
    bp = ax_d.boxplot(data_d, positions=pos_d, widths=0.6, patch_artist=True, showfliers=False,
                      medianprops=dict(color='black', lw=1.2))
    for patch, c in zip(bp['boxes'], colors_d):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    ax_d.set_xticks(pos_d)
    ax_d.set_xticklabels(labels_d, fontsize=9)
    ax_d.set_ylabel(r'$\alpha_2$')
    ax_d.set_title(r'(d) Cross‑task consistency  ICC(2,1): $\alpha_1$=%.3f, $\alpha_2$=%.3f' % (icc1, icc2), fontsize=11)
    ax_d.grid(axis='y', ls='--', alpha=0.4, lw=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_analysis(df, train_subj, val_subj, test_subj, mean, scale, model, alpha_model,
                 audio_dir, out_dir, dataset_name, skip_jitter=False):
    """解耦分析函数：可以传入子集df运行，不依赖全局main变量"""
    os.makedirs(out_dir, exist_ok=True)
    Xall_m, Xall_f = split_features(np.stack(df['feature'].values), mean, scale)
    alphas = alpha_model.predict([Xall_m, Xall_f], batch_size=256, verbose=0)
    probs_all = model.predict([Xall_m, Xall_f], batch_size=256, verbose=0).ravel()
    _, mca = build_alpha_model(model)
    probs_without_alpha1, probs_without_alpha2 = predict_path_ablations(
        model, mca, [Xall_m, Xall_f])

    df = df.copy()
    df['alpha1'] = alphas[:, 0]
    df['alpha2'] = alphas[:, 1]
    df['alpha_ratio'] = df['alpha1'] / df['alpha2']
    df['pred_prob'] = probs_all
    df['pred_prob_without_alpha1'] = probs_without_alpha1
    df['pred_prob_without_alpha2'] = probs_without_alpha2
    df['delta_prob_alpha1'] = probs_all - probs_without_alpha1
    df['delta_prob_alpha2'] = probs_all - probs_without_alpha2
    df['abs_delta_prob_alpha1'] = np.abs(df['delta_prob_alpha1'])
    df['abs_delta_prob_alpha2'] = np.abs(df['delta_prob_alpha2'])
    df['split_new'] = df['subject'].map(
        lambda s: 'train' if s in train_subj else ('val' if s in val_subj else 'test'))

    print(f'[ALPHA-{dataset_name}] all samples: alpha1 mean={df["alpha1"].mean():.4f} std={df["alpha1"].std():.4f} | alpha2 mean={df["alpha2"].mean():.4f} std={df["alpha2"].std():.4f}')
    aug0_df = df[df['aug'] == 0].reset_index(drop=True)
    aug0_df['f0_std_feat'] = [f[161] for f in aug0_df['feature']]

    if skip_jitter or not os.path.isdir(audio_dir):
        print('[AUDIO] jitter analysis skipped')
        aug0_df['jitter_pct'] = np.nan
        aug0_df['f0_std_audio'] = np.nan
        aug0_df['voiced_pct'] = np.nan
    else:
        cache_path = os.path.join(out_dir, 'jitter_cache.csv')
        aug0_df = load_or_compute_acoustics(aug0_df, audio_dir, cache_path)

    subj_cols = ['alpha1', 'alpha2', 'alpha_ratio', 'jitter_pct', 'f0_std_feat',
                 'f0_std_audio', 'abs_delta_prob_alpha1', 'abs_delta_prob_alpha2']
    subj_table = aug0_df.groupby(['subject', 'label'])[subj_cols].mean().reset_index()

    table_rows = []
    for metric in ['alpha1', 'alpha2', 'alpha_ratio', 'abs_delta_prob_alpha1',
                   'abs_delta_prob_alpha2', 'jitter_pct', 'f0_std_feat']:
        hc_v = subj_table[subj_table['label'] == 0][metric].dropna().values
        pd_v = subj_table[subj_table['label'] == 1][metric].dropna().values
        if len(hc_v) < 3 or len(pd_v) < 3:
            continue
        u, p = stats.mannwhitneyu(hc_v, pd_v, alternative='two-sided')
        table_rows.append({
            'metric': metric,
            'hc_n': len(hc_v), 'pd_n': len(pd_v),
            'hc_median': np.median(hc_v), 'hc_q1': np.percentile(hc_v, 25), 'hc_q3': np.percentile(hc_v, 75),
            'pd_median': np.median(pd_v), 'pd_q1': np.percentile(pd_v, 25), 'pd_q3': np.percentile(pd_v, 75),
            'mannwhitney_u': u, 'p_value': p, 'cliffs_delta': cliffs_delta(pd_v, hc_v),
        })
    table15 = pd.DataFrame(table_rows)
    table15.to_csv(os.path.join(out_dir, 'table15_group_statistics.csv'), index=False)

    corr_rows = []
    for m in ['jitter_pct', 'f0_std_feat']:
        for a in ['alpha1', 'alpha2', 'alpha_ratio']:
            v = aug0_df[[m, a]].dropna()
            if len(v) < 10:
                continue
            r = stats.spearmanr(v[m], v[a])
            corr_rows.append({'source_metric': m, 'alpha': a, 'n': len(v),
                              'spearman_rho': r.statistic, 'p_value': r.pvalue, 'level': 'sample'})
    for m in ['jitter_pct', 'f0_std_feat']:
        for a in ['alpha1', 'alpha2', 'alpha_ratio']:
            v = subj_table[[m, a]].dropna()
            if len(v) < 10:
                continue
            r = stats.spearmanr(v[m], v[a])
            corr_rows.append({'source_metric': m, 'alpha': a, 'n': len(v),
                              'spearman_rho': r.statistic, 'p_value': r.pvalue, 'level': 'subject'})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(out_dir, 'alpha_source_correlations.csv'), index=False)

    piv1 = aug0_df.pivot_table(index='subject', columns='task', values='alpha1')
    piv2 = aug0_df.pivot_table(index='subject', columns='task', values='alpha2')
    piv1 = piv1.dropna()
    piv2 = piv2.dropna()
    icc1 = icc_2_1(piv1.values) if len(piv1) > 1 else np.nan
    icc2 = icc_2_1(piv2.values) if len(piv2) > 1 else np.nan

    subj_table['f0_to_mfcc_dominant'] = (
        subj_table['abs_delta_prob_alpha1'] > subj_table['abs_delta_prob_alpha2'])
    dom_hc = subj_table[subj_table['label'] == 0]['f0_to_mfcc_dominant']
    dom_pd = subj_table[subj_table['label'] == 1]['f0_to_mfcc_dominant']
    ct = np.array([[int(dom_hc.sum()), int((~dom_hc).sum())],
                   [int(dom_pd.sum()), int((~dom_pd).sum())]])
    chi2 = float('nan')
    p_chi = 1.0
    if not (ct.sum(axis=0).min() == 0 or ct.sum(axis=1).min() == 0):
        try:
            chi2, p_chi, _, _ = stats.chi2_contingency(ct)
        except ValueError:
            _, p_chi = stats.fisher_exact(ct)

    test_aug0 = aug0_df[aug0_df['subject'].isin(test_subj)].copy()
    test_aug0['correct'] = ((test_aug0['pred_prob'] > 0.5).astype(int) == test_aug0['label']).astype(int)
    test_aug0['f0_to_mfcc_dominant'] = (
        test_aug0['abs_delta_prob_alpha1'] > test_aug0['abs_delta_prob_alpha2'])
    dom_acc_rows = []
    for dom, gname in [(True, 'f0_to_mfcc_dominant'), (False, 'mfcc_to_f0_dominant')]:
        sub = test_aug0[test_aug0['f0_to_mfcc_dominant'] == dom]
        if len(sub) == 0:
            continue
        k = int(sub['correct'].sum())
        n = len(sub)
        lo, hi = wilson_interval(k, n)
        dom_acc_rows.append({'group': gname, 'n': n, 'correct': k, 'accuracy': k / n,
                             'ci95_low': lo, 'ci95_high': hi})
    dom_acc = pd.DataFrame(dom_acc_rows)
    dom_acc.to_csv(os.path.join(out_dir, 'dominance_accuracy.csv'), index=False)

    task_dom = aug0_df.copy()
    task_dom['f0_to_mfcc_dominant'] = (
        task_dom['abs_delta_prob_alpha1'] > task_dom['abs_delta_prob_alpha2'])
    task_dom_table = task_dom.groupby(['task', 'label'])['f0_to_mfcc_dominant'].agg(['mean', 'count']).reset_index()
    task_dom_table.to_csv(os.path.join(out_dir, 'dominance_by_task.csv'), index=False)

    has_jitter = aug0_df['jitter_pct'].notna().sum() >= 30
    source_metric = 'jitter_pct' if has_jitter else 'f0_std_feat'
    source_label = 'Jitter (%)' if has_jitter else 'F0 std (Hz)'
    fig8_path = os.path.join(out_dir, 'fig8_alpha_divergence.png')
    make_fig8(subj_table, aug0_df, source_metric, source_label, icc1, icc2, fig8_path)

    out_cols = ['filename', 'subject', 'task', 'aug', 'label', 'orig_split', 'split_new',
                'alpha1', 'alpha2', 'alpha_ratio', 'pred_prob',
                'pred_prob_without_alpha1', 'pred_prob_without_alpha2',
                'delta_prob_alpha1', 'delta_prob_alpha2']
    df[out_cols].to_csv(os.path.join(out_dir, 'alpha_per_sample.csv'), index=False)

    return {
        "df": df,
        "subj_table": subj_table,
        "icc1": icc1,
        "icc2": icc2,
        "p_chi": p_chi,
        "dom_hc": dom_hc,
        "dom_pd": dom_pd,
        "dom_acc": dom_acc,
        "table15": table15,
        "corr_df": corr_df,
        "fig8_path": fig8_path,
        "aug0_df": aug0_df
    }


def build_summary_report(analysis_res, test_acc, test_auc, sens, spec, yte_len, seed, dataset_name):
    lines = []
    lines.append('=' * 72)
    lines.append(f'Sample‑level balanced‑factor divergence analysis | Dataset: {dataset_name}')
    lines.append('=' * 72)
    subj_table = analysis_res['subj_table']
    aug0_df = analysis_res['aug0_df']
    table15 = analysis_res['table15']
    corr_df = analysis_res['corr_df']
    icc1 = analysis_res['icc1']
    icc2 = analysis_res['icc2']
    p_chi = analysis_res['p_chi']
    dom_hc = analysis_res['dom_hc']
    dom_pd = analysis_res['dom_pd']
    dom_acc = analysis_res['dom_acc']

    lines.append(f'Dataset: {dataset_name}, {len(subj_table)} subjects ({(subj_table["label"]==1).sum()} PD / {(subj_table["label"]==0).sum()} HC), {len(aug0_df)} original recordings')
    lines.append(f'Protocol: subject‑level split 70/15/15 (seed {seed}), scaler fit on train subjects')
    lines.append('')
    lines.append(f'Model performance (held‑out test subjects, aug0 recordings, n={yte_len}):')
    lines.append(f'  accuracy={test_acc:.4f}  AUC={test_auc:.4f}  sensitivity={sens:.4f}  specificity={spec:.4f}')
    lines.append('')
    if not table15.empty:
        lines.append('Group statistics (subject level, Mann‑Whitney U):')
        for _, r in table15.iterrows():
            lines.append('  %-12s HC median=%.4f (IQR %.4f-%.4f)  PD median=%.4f (IQR %.4f-%.4f)  U=%.1f p=%.3e d=%.3f' % (
                r['metric'], r['hc_median'], r['hc_q1'], r['hc_q3'], r['pd_median'], r['pd_q1'], r['pd_q3'],
                r['mannwhitney_u'], r['p_value'], r['cliffs_delta']))
        lines.append('')
    lines.append(f'Cross‑task consistency (ICC(2,1), subjects x tasks): alpha1={icc1:.4f}  alpha2={icc2:.4f}')
    lines.append('')
    if not corr_df.empty:
        lines.append('Acoustic-feature correlations (Spearman; association only):')
        for _, r in corr_df.iterrows():
            lines.append('  [%s] %s vs %s: rho=%.3f p=%.3e n=%d' % (
                r['level'], r['source_metric'], r['alpha'], r['spearman_rho'], r['p_value'], r['n']))
        lines.append('')
    lines.append('Path impact (larger absolute prediction change after disabling a path):')
    lines.append('  HC F0-to-MFCC dominant: %.1f%% (%d/%d)   PD F0-to-MFCC dominant: %.1f%% (%d/%d)   chi2 p=%.4f' % (
        dom_hc.mean() * 100, dom_hc.sum(), len(dom_hc),
        dom_pd.mean() * 100, dom_pd.sum(), len(dom_pd), p_chi))
    lines.append('  Test accuracy by dominance group:')
    for _, r in dom_acc.iterrows():
        lines.append('    %s: %.4f (%d/%d, 95%% CI %.3f‑%.3f)' % (
            r['group'], r['accuracy'], r['correct'], r['n'], r['ci95_low'], r['ci95_high']))
    return "\n".join(lines)


def analyze_main_model_csv(csv_path, out_dir):
    data = pd.read_csv(csv_path)
    required = {'split', 'subject', 'label', 'alpha_mfcc_to_f0',
                'alpha_f0_to_mfcc', 'delta_mfcc_to_f0', 'delta_f0_to_mfcc'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError('Main-model balance CSV lacks columns: ' + ', '.join(sorted(missing)))
    if data.empty:
        raise ValueError('No recordings found')
    if not np.allclose(data['alpha_mfcc_to_f0'] + data['alpha_f0_to_mfcc'], 1.0, atol=1e-5):
        raise ValueError('Balance weights must sum to one for each recording')
    os.makedirs(out_dir, exist_ok=True)
    metrics = ['alpha_mfcc_to_f0', 'alpha_f0_to_mfcc',
               'delta_mfcc_to_f0', 'delta_f0_to_mfcc']
    subject_table = data.groupby(['subject', 'label'], as_index=False)[metrics].mean()
    subject_table.to_csv(os.path.join(out_dir, 'subject_balance_statistics.csv'), index=False)
    inference_data = data[data['split'] == 'test'] if (data['split'] == 'test').any() else data
    inference_subjects = inference_data.groupby(['subject', 'label'], as_index=False)[metrics].mean()
    rows = []
    for metric in metrics:
        hc = inference_subjects.loc[inference_subjects.label == 0, metric].to_numpy()
        pd_values = inference_subjects.loc[inference_subjects.label == 1, metric].to_numpy()
        if not len(hc) or not len(pd_values):
            raise ValueError('Analysis requires both HC and PD subjects in the test split')
        u, p = stats.mannwhitneyu(hc, pd_values, alternative='two-sided')
        rows.append({'metric': metric, 'hc_median': float(np.median(hc)),
                     'pd_median': float(np.median(pd_values)),
                     'mannwhitney_u': float(u), 'p_value': float(p),
                     'n_hc_subjects': len(hc), 'n_pd_subjects': len(pd_values)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'group_balance_statistics.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, metrics[:2]):
        values = [inference_subjects.loc[inference_subjects.label == label, metric].to_numpy()
                  for label in (0, 1)]
        ax.boxplot(values, labels=['HC', 'PD'], showfliers=False)
        ax.set_title(metric)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'alpha_divergence.png'), dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(
            axes, metrics[2:], ['MFCC to source', 'Source to MFCC']):
        values = [inference_subjects.loc[inference_subjects.label == label, metric].to_numpy()
                  for label in (0, 1)]
        ax.boxplot(values, labels=['HC', 'PD'], showfliers=False)
        ax.set(title=title, ylabel='Absolute change in PD probability')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'path_effects.png'), dpi=200)
    plt.close(fig)
    source_metrics = ['source_f0_median_hz', 'source_f0_change_hz',
                      'source_voicing_probability']
    if set(source_metrics).issubset(inference_data.columns):
        correlated = inference_data.groupby(['subject', 'label'], as_index=False)[
            source_metrics + metrics].mean()
        correlations = []
        for source_metric in source_metrics:
            for alpha_metric in metrics[:2]:
                x = correlated[source_metric].to_numpy()
                y = correlated[alpha_metric].to_numpy()
                valid = np.isfinite(x) & np.isfinite(y)
                rho, p = stats.spearmanr(x[valid], y[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
                correlations.append({'source_metric': source_metric, 'alpha_metric': alpha_metric,
                                     'spearman_rho': rho, 'p_value': p,
                                     'n_subjects': int(valid.sum())})
        pd.DataFrame(correlations).to_csv(
            os.path.join(out_dir, 'source_alpha_correlations.csv'), index=False)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        source_labels = ['F0 median (Hz)', 'F0 change (Hz)', 'Voicing probability']
        for ax, source_metric, source_label in zip(axes, source_metrics, source_labels):
            for label, name, color in ((0, 'HC', HC_COLOR), (1, 'PD', PD_COLOR)):
                group = correlated[correlated.label == label]
                ax.scatter(group[source_metric], group['alpha_mfcc_to_f0'],
                           label=name, color=color, alpha=0.8)
            ax.set(xlabel=source_label, ylabel='MFCC to source weight')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'source_alpha_relationship.png'), dpi=200)
        plt.close(fig)
    diagnostics = {
        'analysis_split': 'test' if (data['split'] == 'test').any() else 'all',
        'n_subjects': int(len(inference_subjects)),
        'alpha_mfcc_to_f0_std': float(inference_subjects['alpha_mfcc_to_f0'].std()),
        'alpha_f0_to_mfcc_std': float(inference_subjects['alpha_f0_to_mfcc'].std()),
        'mean_abs_delta_mfcc_to_f0': float(inference_subjects['delta_mfcc_to_f0'].mean()),
        'mean_abs_delta_f0_to_mfcc': float(inference_subjects['delta_f0_to_mfcc'].mean()),
    }
    with open(os.path.join(out_dir, 'path_diagnostics.json'), 'w', encoding='utf-8') as stream:
        json.dump(diagnostics, stream, ensure_ascii=False, indent=2)
    print('[MAIN MODEL] subject-level analysis saved to', out_dir)


def main():
    configure_utf8_console(sys.stdout)
    configure_utf8_console(sys.stderr)
    parser = argparse.ArgumentParser(description='Train the dual-branch model and run sample-level balanced-factor (alpha) analysis')
    parser.add_argument('--main-model-csv', type=str,
                        help='Analyze alpha_per_sample.csv exported by train.py, without retraining another model')
    parser.add_argument('--legacy-retrain', action='store_true',
                        help='Run the independent legacy NPZ experiment instead of analyzing the hierarchical model')
    parser.add_argument('--dataset-name', type=str, default="pcgita", help="数据集标识，输出目录会带上这个名字")
    parser.add_argument('--data-dir', type=str, default=os.path.join(BASE_DIR, 'data', 'multimodal_features'),
                        help="npz特征根目录，包含train/val/test子文件夹")
    parser.add_argument('--subject-hc-prefix', type=str, default="AVPEPUDEAC",
                        help="HC健康人subject前缀，用于判断PD/HC，旧PC-GITA默认AVPEPUDEAC")
    parser.add_argument('--filename-regex', type=str, default=r'^(AVPEPUD\w+?\d{4})([a-z]\d)_aug(\d+)$',
                        help="npz文件名解析正则，group1:subject, group2:task, group3:aug")

    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--audio-dir', type=str, default=DEFAULT_AUDIO_DIR)
    parser.add_argument('--out-root', type=str, default=DEFAULT_OUT_ROOT, help="输出根目录，自动拼接dataset-name子文件夹")
    parser.add_argument('--skip-train', action='store_true', help='load model_best.h5 instead of training')
    parser.add_argument('--skip-jitter', action='store_true', help='skip pyin jitter computation (Fig.8c falls back to F0 std)')
    args = parser.parse_args()

    use_default_main_model = args.main_model_csv is None and not args.legacy_retrain
    if use_default_main_model:
        from modules.train_utils import MODEL_VERSION
        args.main_model_csv = os.path.join(
            BASE_DIR, 'results', 'alpha_analysis',
            f'full_model_{MODEL_VERSION}_alpha_per_sample.csv')
    if args.main_model_csv:
        if not os.path.isfile(args.main_model_csv):
            raise FileNotFoundError(
                f'{args.main_model_csv} not found. Run train.py --full_training first.')
        output_name = f'full_model_{MODEL_VERSION}' if use_default_main_model else args.dataset_name
        out_dir = os.path.join(args.out_root, output_name, 'main_model_analysis')
        analyze_main_model_csv(args.main_model_csv, out_dir)
        return

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    out_dir = os.path.join(args.out_root, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    protocol_path = os.path.join(out_dir, 'model_protocol.json')

    df = load_dataset(args.data_dir, args.filename_regex, args.subject_hc_prefix)
    train_subj, val_subj, test_subj = assign_subject_split(df, args.seed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    train_df = df[df['subject'].isin(train_subj)].reset_index(drop=True)
    val_df = df[(df['subject'].isin(val_subj)) & (df['aug'] == 0)].reset_index(drop=True)
    test_df = df[(df['subject'].isin(test_subj)) & (df['aug'] == 0)].reset_index(drop=True)
    aug0_df = df[df['aug'] == 0].reset_index(drop=True)

    print('[DATA] train samples (all augs): %d | val (aug0): %d | test (aug0): %d | aug0 total: %d' % (
        len(train_df), len(val_df), len(test_df), len(aug0_df)))

    X_train = np.stack(train_df['feature'].values)
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    np.savez(os.path.join(out_dir, 'feature_scaler.npz'), mean=mean, scale=scale)

    Xtr_m, Xtr_f = split_features(X_train, mean, scale)
    Xva_m, Xva_f = split_features(np.stack(val_df['feature'].values), mean, scale)
    Xte_m, Xte_f = split_features(np.stack(test_df['feature'].values), mean, scale)
    ytr = train_df['label'].values
    yva = val_df['label'].values
    yte = test_df['label'].values

    model_path = os.path.join(out_dir, 'model_best.h5')
    mca = None
    if args.skip_train:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{model_path} not found, train first')
        require_corrected_model_protocol(protocol_path)
        print(f'[MODEL] loading {model_path}')
        custom_objects = {
            'ModalCrossAttention': ModalCrossAttention,
            'ECAModule': ECAModule,
            'FeatureLevelAttention': FeatureLevelAttention,
            'GlobalLevelAttention': GlobalLevelAttention,
            'AlphaTracker': AlphaTracker,
        }
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    else:
        model, mca = build_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )
        callbacks = [
            EarlyStopping(patience=30, monitor='val_auc', mode='max', min_delta=0.001, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_auc', mode='max'),
            AlphaEpochCurveCallback(mca, os.path.join(out_dir, 'alpha_training_curve.png')),
        ]
        print(f'[TRAIN‑{args.dataset_name}] start: {args.epochs} epochs max, batch {args.batch_size}')
        model.fit([Xtr_m, Xtr_f], ytr,
                  validation_data=([Xva_m, Xva_f], yva),
                  epochs=args.epochs, batch_size=args.batch_size,
                  callbacks=callbacks, verbose=2)
        with open(protocol_path, 'w', encoding='utf-8') as f:
            json.dump({'alpha_path_version': ALPHA_PATH_VERSION, 'seed': args.seed}, f, indent=2)

    probs_te = model.predict([Xte_m, Xte_f], batch_size=256, verbose=0).ravel()
    pred_te = (probs_te > 0.5).astype(int)
    test_acc = float((pred_te == yte).mean())
    test_auc = float(roc_auc_score(yte, probs_te))
    sens = float(((pred_te == 1) & (yte == 1)).sum() / max(1, (yte == 1).sum()))
    spec = float(((pred_te == 0) & (yte == 0)).sum() / max(1, (yte == 0).sum()))
    print(f'[TEST‑{args.dataset_name}] acc={test_acc:.4f} auc={test_auc:.4f} sensitivity={sens:.4f} specificity={spec:.4f} (n={len(yte)})')

    alpha_model, mca_ref = build_alpha_model(model)

    # 解耦分析，可以传df子集，不依赖全局df；也可以单独调用run_analysis跑子集
    analysis_res = run_analysis(df, train_subj, val_subj, test_subj, mean, scale,
                                 model, alpha_model, args.audio_dir, out_dir, args.dataset_name, skip_jitter=args.skip_jitter)

    metrics = {
        'dataset_name': args.dataset_name,
        'alpha_path_version': ALPHA_PATH_VERSION,
        'n_subjects': int(len(analysis_res['subj_table'])),
        'n_recordings_aug0': int(len(aug0_df)),
        'test_n': int(len(yte)),
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'test_sensitivity': sens,
        'test_specificity': spec,
        'icc2_1_alpha1': float(analysis_res['icc1']) if np.isfinite(analysis_res['icc1']) else None,
        'icc2_1_alpha2': float(analysis_res['icc2']) if np.isfinite(analysis_res['icc2']) else None,
        'dominance_chi2_p': float(analysis_res['p_chi']),
        'f0_to_mfcc_dominant_hc': float(analysis_res['dom_hc'].mean()),
        'f0_to_mfcc_dominant_pd': float(analysis_res['dom_pd'].mean()),
        'seed': args.seed,
    }
    with open(os.path.join(out_dir, 'test_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    report_text = build_summary_report(analysis_res, test_acc, test_auc, sens, spec, len(yte), args.seed, args.dataset_name)
    report_path = os.path.join(out_dir, 'summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)


if __name__ == '__main__':
    main()
