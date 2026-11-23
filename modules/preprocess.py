import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.model_selection import StratifiedShuffleSplit


def safe_path(path):
    return os.path.abspath(path)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def apply_augmentation(y, sr, max_augments=3):
    """应用多种数据增强"""
    augmented_samples = []
    augmented_samples.append(y)  # 原始样本

    augmentations = [
        {'noise_scale': 0.006},
        {'time_stretch': 1.0},
        {'pitch_shift': 1},
        {'time_stretch': 1.2, 'pitch_shift': -1},
        {'time_stretch': 0.8, 'pitch_shift': 2},
        {'noise_scale': 0.01, 'time_stretch': 1.1, 'pitch_shift': -2}
    ]

    for aug in augmentations[:max_augments]:
        try:
            augmented = y.copy()
            if 'noise_scale' in aug:
                noise = np.random.normal(0, aug['noise_scale'], len(augmented))
                augmented += noise
            if 'time_stretch' in aug:
                augmented = librosa.effects.time_stretch(augmented, rate=aug['time_stretch'])
            if 'pitch_shift' in aug:
                augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=aug['pitch_shift'])
            augmented = np.pad(augmented, (0, max(0, len(y) - len(augmented))))[:len(y)]
            augmented_samples.append(augmented)
        except Exception as e:
            logging.warning(f"增强失败: {str(e)}")
    return augmented_samples


def validate_audio(file_path: str) -> bool:
    """音频验证"""
    try:
        y, sr = librosa.load(safe_path(file_path), sr=16000, mono=True)
        y, _ = librosa.effects.trim(y)
        if len(y) / sr < 0.005 or np.max(np.abs(y)) < 0.001:
            return False
        return True
    except Exception as e:
        logging.warning(f"无效文件 {file_path}: {str(e)}")
        return False


def frame_signal(signal, frame_length, frame_step, window_type='hamming'):
    """分帧加窗"""
    signal_length = len(signal)
    frame_length = int(frame_length)
    frame_step = int(frame_step)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    if window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'hanning':
        window = np.hanning(frame_length)
    else:
        window = np.ones(frame_length)

    frames = frames * window
    return frames


def extract_features(y: np.ndarray, sr: int, frame_length=0.025, frame_step=0.01, window_type='hamming') -> np.ndarray:
    """提取 MFCC（声道）与 F0（声源）特征，各 160 维。

    - MFCC：直接在原始语音上分帧，避免对拼接后的帧序列二次分帧。
    - F0：在原始语音上做基频跟踪，只保留声源相关特征
      （F0 轮廓、一阶/二阶差分、发声概率），不再混入 mel 频谱统计量。
    """
    try:
        frame_length_samples = int(frame_length * sr)
        frame_step_samples = int(frame_step * sr)
        window = 'hann' if window_type == 'hanning' else window_type

        # 极短音频补齐，避免 n_fft / pyin 帧长越界
        if len(y) < frame_length_samples:
            y = np.pad(y, (0, frame_length_samples - len(y)))

        #  声道特征：MFCC（160维 = 40 × [均值, 标准差, Δ均值, ΔΔ均值]）
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=40, n_mels=40,
            n_fft=frame_length_samples,
            hop_length=frame_step_samples,
            window=window,
        )
        mfcc_features = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            librosa.feature.delta(mfcc, order=1).mean(axis=1),
            librosa.feature.delta(mfcc, order=2).mean(axis=1)
        ])

        #  声源特征：在原始语音上跟踪 F0
        pitch_frame_length = 2048
        y_pitch = y if len(y) >= pitch_frame_length else np.pad(y, (0, pitch_frame_length - len(y)))
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_pitch, fmin=50, fmax=500, sr=sr,
            frame_length=pitch_frame_length,
            hop_length=frame_step_samples,
        )
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        voiced_probs = np.nan_to_num(voiced_probs, nan=0.0, posinf=0.0, neginf=0.0)
        if f0.size == 0:
            f0 = np.zeros(1)
            voiced_probs = np.zeros(1)

        def resample(x, n=40):
            if x.size == 1:
                return np.repeat(x, n)
            return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, x.size), x)

        # 40 个时间点 × [F0, ΔF0, ΔΔF0, 发声概率] = 160 维纯声源特征
        f0_contour = resample(f0)
        f0_delta = np.gradient(f0_contour)
        f0_delta2 = np.gradient(f0_delta)
        voiced_contour = resample(voiced_probs)
        f0_features = np.concatenate([f0_contour, f0_delta, f0_delta2, voiced_contour])

        return np.concatenate([mfcc_features, f0_features]).astype(np.float32)
    except Exception as e:
        logging.error(f"特征提取失败: {str(e)}")
        return None


def process_file(file_path: str, label: int, frame_length=0.025, frame_step=0.01, window_type='hamming') -> list:
    """处理单个文件"""
    samples = []
    try:
        y, sr = librosa.load(safe_path(file_path), sr=16000)
        y = librosa.util.normalize(y)
        augmented_audios = apply_augmentation(y, sr, 2)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        for idx, audio in enumerate(augmented_audios):
            feat = extract_features(audio, sr, frame_length, frame_step, window_type)
            if feat is not None:
                samples.append({
                    'filename': f"{base_name}_aug{idx}.wav",
                    'label': label,
                    'features': feat
                })
    except Exception as e:
        logging.error(f"处理失败 {file_path}: {str(e)}")
    return samples


def convert_wav_to_csv(input_dir: str, output_csv_train: str, output_csv_val: str, output_csv_test: str, test_size=0.2,
                       val_size=0.2, frame_length=0.025, frame_step=0.01, window_type='hamming'):
    """核心转换流程"""
    input_dir = safe_path(input_dir)
    if not os.path.exists(input_dir):
        logging.info("输入文件夹不存在，尝试使用之前提取的特征。")
        try:
            augmented_train_df = pd.read_csv(safe_path(output_csv_train))
            val_df = pd.read_csv(safe_path(output_csv_val))
            test_df = pd.read_csv(safe_path(output_csv_test))
            return augmented_train_df, val_df, test_df
        except FileNotFoundError:
            logging.error("之前提取的特征文件也未找到，请检查路径或重新提取特征。")
            return None, None, None

    data = []
    for root, _, files in os.walk(input_dir):
        current_label = 1 if any(kw in root.lower() for kw in ["pd", "parkinson"]) else 0
        for file in tqdm(files, desc=f"处理 {os.path.basename(root)}"):
            if not file.lower().endswith(('.wav', '.mp3')):
                continue
            file_path = safe_path(os.path.join(root, file))
            if not validate_audio(file_path):
                continue

            for sample in process_file(file_path, current_label, frame_length, frame_step, window_type):
                data.append([sample['filename'], sample['label'], *sample['features']])

    # 使用分层抽样划分训练集、验证集和测试集
    df = pd.DataFrame(data)
    new_feature_dim = len(df.columns) - 2
    columns = ["filename", "label"] + [f"feat_{i}" for i in range(new_feature_dim)]
    df.columns = columns

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_val_idx, test_idx in sss_test.split(df, df["label"]):
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

    remaining_size = val_size / (1 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=remaining_size)
    for train_idx, val_idx in sss_val.split(train_val_df, train_val_df["label"]):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    # 对训练集进行数据增强
    augmented_train_data = []
    for index, row in train_df.iterrows():
        augmented_train_data.append(row.tolist())
        label = row["label"]
        audio_path = os.path.join(input_dir, row["filename"].replace("_aug0", ""))
        if os.path.exists(audio_path):
            augmented_samples = process_file(audio_path, label, frame_length, frame_step, window_type)
            for sample in augmented_samples:
                augmented_train_data.append([sample['filename'], sample['label'], *sample['features']])

    augmented_train_df = pd.DataFrame(augmented_train_data)
    augmented_train_df.columns = columns

    # 数据验证
    def validate_data(df):
        try:
            assert df.shape[1] == new_feature_dim + 2, f"列数异常 ({df.shape[1]} != {new_feature_dim + 2})"
            assert (df['label'].isin([0, 1])).all(), "无效标签"
            logging.info(f"总样本: {len(df)}, 标签分布:\n{df['label'].value_counts()}")
        except AssertionError as e:
            logging.error(f"数据验证失败: {str(e)}")
            raise
        return df

    augmented_train_df = validate_data(augmented_train_df)
    val_df = validate_data(val_df)
    test_df = validate_data(test_df)

    # 保存到 CSV
    augmented_train_df.to_csv(safe_path(output_csv_train), index=False)
    val_df.to_csv(safe_path(output_csv_val), index=False)
    test_df.to_csv(safe_path(output_csv_test), index=False)

    return augmented_train_df, val_df, test_df

    # 4.2
