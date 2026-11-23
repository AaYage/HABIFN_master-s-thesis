"""帧级 MFCC + F0 特征提取与分组划分。

这个模块是现有 ``modules/preprocess.py`` 的并行版本：
- 不覆盖现有 ``data/processed_features/*.csv``；
- 输出 ``framed_train.npz / framed_val.npz / framed_test.npz``；
- 先按录音（或受试者）分组划分，再只对训练集做增强；
- MFCC 输出 ``(T, 120)``：40 MFCC + 40 Δ + 40 ΔΔ；
- F0 输出 ``(T, 4)``：log-F0、Δ、ΔΔ、发声概率。

用法::

    python -m modules.preprocess_framed --data_path "D:\\...\\test" \
        --out_dir "data/processed_features" --seq_len 200 --group base

小样本冒烟测试::

    python -m modules.preprocess_framed --data_path "D:\\...\\test" \
        --out_dir "d:\\tmp\\framed_smoke" --max_files 6 --no_augment --seq_len 128
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from math import gcd
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import resample_poly, savgol_filter
from sklearn.model_selection import train_test_split

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

try:
    import parselmouth
except ImportError:  # pragma: no cover
    parselmouth = None

_LIBROSA_OK = librosa is not None
_PARSELMOUTH_OK = parselmouth is not None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
SUBJECT_RE = re.compile(r"^([A-Za-z]+\d+)", re.IGNORECASE)
AUG_RE = re.compile(r"_aug\d+$", re.IGNORECASE)


def safe_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """读取音频并重采样到 ``sr``。

    优先使用 scipy（避免 libsndfile 依赖）；失败时回退到 librosa。
    """
    try:
        fs, data = wavfile.read(safe_path(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / max(float(np.iinfo(data.dtype).max), 1.0)
        elif data.dtype.kind == "u":
            data = (data.astype(np.float32) - 32768.0) / 32768.0
        else:
            data = data.astype(np.float32)
        if fs != sr:
            g = gcd(int(sr), int(fs))
            data = resample_poly(data, int(sr) // g, int(fs) // g).astype(np.float32)
    except Exception:
        if librosa is None:
            raise
        data, _ = librosa.load(safe_path(path), sr=sr, mono=True)
        data = data.astype(np.float32)

    if data.size == 0:
        return data
    peak = float(np.max(np.abs(data)))
    if peak > 1e-8:
        data = data / peak
    return data


def parse_label(rel_path: str) -> int | None:
    """按相对路径判断 PD/HC，避免父目录名误判。"""
    parts = [p.lower() for p in Path(rel_path).parts]
    if any(p == "pd" or "parkinson" in p for p in parts):
        return 1
    if any(p in {"hc", "healthy", "control", "controls"} for p in parts):
        return 0
    return None


def parse_subject(filename: str) -> str:
    """从文件名提取受试者 ID，例如 AVPEPUDEA0001a1 -> AVPEPUDEA0001。"""
    stem = AUG_RE.sub("", Path(filename).stem)
    match = SUBJECT_RE.match(stem)
    return match.group(1) if match else stem


def parse_base(filename: str) -> str:
    """录音级 base ID（去掉 _augN）。"""
    return AUG_RE.sub("", Path(filename).stem)


def collect_records(data_path: str, max_files: int = 0) -> List[Dict]:
    data_path = safe_path(data_path)
    records: List[Dict] = []
    for root, _, files in os.walk(data_path):
        for name in files:
            if Path(name).suffix.lower() not in AUDIO_EXTS:
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, data_path)
            label = parse_label(rel)
            if label is None:
                logging.warning("无法判断标签，跳过: %s", rel)
                continue
            records.append(
                {
                    "path": path,
                    "rel_path": rel,
                    "filename": name,
                    "label": int(label),
                    "subject": parse_subject(name),
                    "base": parse_base(name),
                }
            )

    records.sort(key=lambda item: (item["label"], item["path"]))
    if max_files and max_files > 0:
        kept: List[Dict] = []
        for label in (0, 1):
            group = [item for item in records if item["label"] == label][:max_files]
            kept.extend(group)
        records = sorted(kept, key=lambda item: (item["label"], item["path"]))

    if not records:
        raise RuntimeError(f"在 {data_path} 下没有找到可用的 wav/mp3 文件")
    return records


def _safe_train_test_split(keys: List[str], labels: List[int], test_size: float, seed: int):
    try:
        return train_test_split(keys, test_size=test_size, stratify=labels, random_state=seed)
    except ValueError as exc:
        logging.warning("分层划分失败，改为普通划分: %s", exc)
        return train_test_split(keys, test_size=test_size, random_state=seed)


def split_records(
    records: List[Dict],
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    group_by: str = "base",
) -> Dict[str, List[Dict]]:
    """先按 base/subject 分组，再划分 train/val/test，保证同组不跨集合。"""
    group_key = "subject" if group_by == "subject" else "base"
    groups: Dict[str, List[Dict]] = {}
    for record in records:
        groups.setdefault(record[group_key], []).append(record)

    keys: List[str] = []
    labels: List[int] = []
    for key, items in groups.items():
        values = {item["label"] for item in items}
        if len(values) != 1:
            raise ValueError(f"{group_key}={key} 同时包含多个标签: {sorted(values)}")
        keys.append(key)
        labels.append(items[0]["label"])

    train_val_keys, test_keys = _safe_train_test_split(keys, labels, test_size, seed)
    train_val_labels = [next(iter({r["label"] for r in groups[k]})) for k in train_val_keys]
    val_ratio = val_size / max(1e-8, 1.0 - test_size)
    train_keys, val_keys = _safe_train_test_split(train_val_keys, train_val_labels, val_ratio, seed)

    def gather(selected: List[str]) -> List[Dict]:
        out: List[Dict] = []
        for key in selected:
            out.extend(groups[key])
        return out

    return {
        "train": gather(train_keys),
        "val": gather(val_keys),
        "test": gather(test_keys),
        "group_key": group_key,
        "groups": {key: [item["filename"] for item in items] for key, items in groups.items()},
    }


def _fix_length(array: np.ndarray, seq_len: int) -> Tuple[np.ndarray, int]:
    length = int(array.shape[0])
    if length >= seq_len:
        start = (length - seq_len) // 2
        return array[start : start + seq_len], seq_len
    pad = seq_len - length
    return np.pad(array, ((0, pad), (0, 0)), mode="constant"), length


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if values.size == 1:
        return np.repeat(values, target_len)
    return np.interp(
        np.linspace(0.0, 1.0, target_len),
        np.linspace(0.0, 1.0, values.size),
        values,
    ).astype(np.float32)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel2hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fmax = min(float(fmax), sr / 2.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def _numpy_mfcc_seq(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    cmvn: bool,
) -> np.ndarray:
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    n_frames = 1 + (y.size - n_fft) // hop_length
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = y[idx] * np.hanning(n_fft)[None, :]
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels, 20.0, sr / 2.0)
    logmel = np.log(power @ fb.T + 1e-10)
    mfcc = dct(logmel, type=2, axis=1, norm="ortho")[:, :n_mfcc].astype(np.float32)
    width = min(9, n_frames if n_frames % 2 else n_frames - 1)
    if width >= 3:
        delta1 = savgol_filter(mfcc, width, 1, axis=0, mode="interp")
        delta2 = savgol_filter(mfcc, width, 2, axis=0, mode="interp")
    else:
        delta1 = np.gradient(mfcc, axis=0)
        delta2 = np.gradient(delta1, axis=0)
    seq = np.concatenate([mfcc, delta1, delta2], axis=1).astype(np.float32)
    if cmvn:
        seq = (seq - seq.mean(axis=0, keepdims=True)) / (seq.std(axis=0, keepdims=True) + 1e-6)
    return seq.astype(np.float32)


def _librosa_mfcc_seq(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    cmvn: bool,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
    )
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    seq = np.concatenate([mfcc, delta1, delta2], axis=0).T.astype(np.float32)
    if cmvn:
        seq = (seq - seq.mean(axis=0, keepdims=True)) / (seq.std(axis=0, keepdims=True) + 1e-6)
    return seq.astype(np.float32)


def _parselmouth_f0(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=hop_length / float(sr), pitch_floor=fmin, pitch_ceiling=fmax)
    selected = pitch.selected_array
    if isinstance(selected, dict):
        f0 = np.asarray(selected.get("frequency", []), dtype=np.float32)
        strength = np.asarray(selected.get("strength", (f0 > 0).astype(np.float32)), dtype=np.float32)
    else:
        f0 = np.asarray(selected["frequency"], dtype=np.float32)
        names = selected.dtype.names or ()
        if "strength" in names:
            strength = np.asarray(selected["strength"], dtype=np.float32)
        else:
            strength = (f0 > 0).astype(np.float32)
    return f0, strength


def _librosa_f0(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pitch_frame_length = 2048
    y_pitch = y if y.size >= pitch_frame_length else np.pad(y, (0, pitch_frame_length - y.size))
    f0, _, voiced_prob = librosa.pyin(
        y_pitch,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=pitch_frame_length,
        hop_length=hop_length,
    )
    return np.asarray(f0, dtype=np.float32), np.asarray(voiced_prob, dtype=np.float32)


def extract_frame_features(
    y: np.ndarray,
    sr: int = 16000,
    seq_len: int = 200,
    n_mfcc: int = 40,
    n_mels: int = 40,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 500.0,
    cmvn: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """返回 ``(mfcc_seq, f0_seq, valid_len)``。"""
    global _LIBROSA_OK, _PARSELMOUTH_OK

    min_len = int(n_fft + 8 * hop_length)
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))

    mfcc_seq = None
    if _LIBROSA_OK:
        try:
            mfcc_seq = _librosa_mfcc_seq(y, sr, n_mfcc, n_mels, n_fft, hop_length, cmvn)
        except Exception as exc:
            _LIBROSA_OK = False
            logging.warning("librosa MFCC 不可用，改用 numpy 实现: %s", exc)
    if mfcc_seq is None:
        mfcc_seq = _numpy_mfcc_seq(y, sr, n_mfcc, n_mels, n_fft, hop_length, cmvn)

    f0 = voiced_prob = None
    if _LIBROSA_OK:
        try:
            f0, voiced_prob = _librosa_f0(y, sr, hop_length, fmin, fmax)
        except Exception as exc:
            _LIBROSA_OK = False
            logging.warning("librosa pyin 不可用，改用 Praat: %s", exc)
    if f0 is None and _PARSELMOUTH_OK:
        try:
            f0, voiced_prob = _parselmouth_f0(y, sr, hop_length, fmin, fmax)
        except Exception as exc:
            _PARSELMOUTH_OK = False
            logging.warning("Praat F0 不可用: %s", exc)
    if f0 is None:
        f0 = np.zeros(1, dtype=np.float32)
        voiced_prob = np.zeros(1, dtype=np.float32)

    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if f0.size == 0:
        f0 = np.zeros(1, dtype=np.float32)
        voiced_prob = np.zeros(1, dtype=np.float32)

    target_len = mfcc_seq.shape[0]
    log_f0 = _resample_1d(np.log1p(f0), target_len)
    voiced = _resample_1d(voiced_prob, target_len)
    f0_delta = np.gradient(log_f0).astype(np.float32) if target_len > 1 else np.zeros_like(log_f0)
    f0_delta2 = np.gradient(f0_delta).astype(np.float32) if target_len > 1 else np.zeros_like(log_f0)
    f0_seq = np.stack([log_f0, f0_delta, f0_delta2, voiced], axis=1).astype(np.float32)

    mfcc_seq, valid_len = _fix_length(mfcc_seq, seq_len)
    f0_seq, _ = _fix_length(f0_seq, seq_len)
    mfcc_seq = np.nan_to_num(mfcc_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    f0_seq = np.nan_to_num(f0_seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return mfcc_seq, f0_seq, valid_len


def augment_audio(y: np.ndarray, sr: int, aug_index: int, seed: int) -> np.ndarray:
    """训练集增强：0=原始，1=加噪，2=升降调。失败时返回原音频。"""
    if aug_index == 0:
        return y
    rng = np.random.default_rng(seed + 1000 * aug_index)
    try:
        if aug_index % 2 == 1:
            rms = float(np.sqrt(np.mean(y**2)) + 1e-8)
            noise_rms = rms / (10 ** (25.0 / 20.0))
            return (y + rng.normal(0.0, noise_rms, size=y.shape)).astype(np.float32)
        steps = 1.0 if aug_index % 4 == 2 else -1.0
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps).astype(np.float32)
    except Exception as exc:  # pragma: no cover
        logging.warning("增强失败，使用原音频: %s", exc)
        return y


def build_split(
    records: List[Dict],
    split: str,
    seq_len: int,
    augments: int,
    seed: int,
    feature_kwargs: Dict,
) -> Dict[str, np.ndarray]:
    mfcc_list: List[np.ndarray] = []
    f0_list: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[str] = []
    bases: List[str] = []
    filenames: List[str] = []
    aug_ids: List[int] = []
    lengths: List[int] = []

    for record in records:
        y = load_audio(record["path"], sr=feature_kwargs.get("sr", 16000))
        if y.size == 0:
            continue
        n_aug = augments if split == "train" else 0
        for aug_index in range(n_aug + 1):
            audio = augment_audio(y, feature_kwargs.get("sr", 16000), aug_index, seed) if aug_index else y
            mfcc_seq, f0_seq, valid_len = extract_frame_features(audio, **feature_kwargs)
            mfcc_list.append(mfcc_seq)
            f0_list.append(f0_seq)
            labels.append(record["label"])
            subjects.append(record["subject"])
            bases.append(record["base"])
            filenames.append(record["filename"])
            aug_ids.append(aug_index)
            lengths.append(valid_len)

    if not mfcc_list:
        raise RuntimeError(f"{split} 没有生成任何样本")

    return {
        "mfcc": np.stack(mfcc_list).astype(np.float32),
        "f0": np.stack(f0_list).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int32),
        "subject": np.asarray(subjects, dtype="U"),
        "base": np.asarray(bases, dtype="U"),
        "filename": np.asarray(filenames, dtype="U"),
        "aug": np.asarray(aug_ids, dtype=np.int32),
        "length": np.asarray(lengths, dtype=np.int32),
        "split": np.asarray([split] * len(labels), dtype="U"),
    }


def fit_scaler(train_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mfcc = train_data["mfcc"]
    f0 = train_data["f0"]
    lengths = train_data["length"]
    mfcc_valid = np.concatenate([mfcc[i, : int(lengths[i])] for i in range(len(lengths))], axis=0)
    f0_valid = np.concatenate([f0[i, : int(lengths[i])] for i in range(len(lengths))], axis=0)
    return {
        "mfcc_mean": mfcc_valid.mean(axis=0, keepdims=True).astype(np.float32),
        "mfcc_std": (mfcc_valid.std(axis=0, keepdims=True) + 1e-6).astype(np.float32),
        "f0_mean": f0_valid.mean(axis=0, keepdims=True).astype(np.float32),
        "f0_std": (f0_valid.std(axis=0, keepdims=True) + 1e-6).astype(np.float32),
    }


def apply_scaler(data: Dict[str, np.ndarray], scaler: Dict[str, np.ndarray]) -> None:
    data["mfcc"] = ((data["mfcc"] - scaler["mfcc_mean"]) / scaler["mfcc_std"]).astype(np.float32)
    data["f0"] = ((data["f0"] - scaler["f0_mean"]) / scaler["f0_std"]).astype(np.float32)


def save_split(data: Dict[str, np.ndarray], path: str) -> None:
    np.savez_compressed(
        path,
        mfcc=data["mfcc"],
        f0=data["f0"],
        y=data["y"],
        subject=data["subject"],
        base=data["base"],
        filename=data["filename"],
        aug=data["aug"],
        length=data["length"],
        split=data["split"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="帧级 MFCC+F0 特征提取与分组划分")
    parser.add_argument("--data_path", required=True, help="原始音频根目录（包含 hc/pd 子目录）")
    parser.add_argument("--out_dir", default="data/processed_features", help="输出目录")
    parser.add_argument("--seq_len", type=int, default=200, help="统一帧数")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group", choices=["base", "subject"], default="base", help="划分单位")
    parser.add_argument("--augments", type=int, default=2, help="训练集每个录音的增强份数")
    parser.add_argument("--no_augment", action="store_true", help="不生成增强样本")
    parser.add_argument("--infer_only", action="store_true", help="不划分不增强，全部样本处理后存 framed_all.npz（用于跨库分析）")
    parser.add_argument("--scaler_path", default="", help="infer_only 时复用已有标准化参数，如 PC-GITA 的 framed_scaler.npz")
    parser.add_argument("--max_files", type=int, default=0, help="每类最多使用多少个文件（0=全部，用于冒烟测试）")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--n_fft", type=int, default=400)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=500.0)
    parser.add_argument("--cmvn", action="store_true", default=True, help="每条语音 MFCC 做 CMVN（默认开启）")
    parser.add_argument("--no_cmvn", action="store_false", dest="cmvn")
    args = parser.parse_args()

    out_dir = safe_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    records = collect_records(args.data_path, max_files=args.max_files)
    logging.info("找到 %d 条录音，受试者 %d 个", len(records), len({r["subject"] for r in records}))

    feature_kwargs = {
        "sr": args.sr,
        "seq_len": args.seq_len,
        "n_mfcc": args.n_mfcc,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "fmin": args.fmin,
        "fmax": args.fmax,
        "cmvn": args.cmvn,
    }

    if args.infer_only:
        logging.info("推理模式：提取全部 %d 条录音（不划分、不增强）", len(records))
        data = build_split(records, "all", args.seq_len, 0, args.seed, feature_kwargs)
        if args.scaler_path:
            with np.load(safe_path(args.scaler_path)) as sc:
                scaler = {k: sc[k] for k in ("mfcc_mean", "mfcc_std", "f0_mean", "f0_std")}
        else:
            scaler = fit_scaler(data)
        apply_scaler(data, scaler)
        save_split(data, os.path.join(out_dir, "framed_all.npz"))
        pd.DataFrame({
            "filename": [str(v) for v in data["filename"]],
            "subject": [str(v) for v in data["subject"]],
            "label": [int(v) for v in data["y"]],
            "length": [int(v) for v in data["length"]],
        }).to_csv(os.path.join(out_dir, "framed_all_index.csv"), index=False)
        logging.info("已保存 %s", os.path.join(out_dir, "framed_all.npz"))
        return

    split_data = split_records(records, args.test_size, args.val_size, args.seed, args.group)
    for split in ("train", "val", "test"):
        logging.info("%s: %d 条录音", split, len(split_data[split]))

    augments = 0 if args.no_augment else max(0, args.augments)

    arrays = {}
    for split in ("train", "val", "test"):
        logging.info("提取 %s 特征 ...", split)
        arrays[split] = build_split(split_data[split], split, args.seq_len, augments, args.seed, feature_kwargs)
        logging.info(
            "%s: mfcc=%s f0=%s 标签=%s",
            split,
            arrays[split]["mfcc"].shape,
            arrays[split]["f0"].shape,
            np.bincount(arrays[split]["y"]).tolist(),
        )

    scaler = fit_scaler(arrays["train"])
    for split in ("train", "val", "test"):
        apply_scaler(arrays[split], scaler)
        save_split(arrays[split], os.path.join(out_dir, f"framed_{split}.npz"))

    np.savez_compressed(
        os.path.join(out_dir, "framed_scaler.npz"),
        seq_len=np.asarray(args.seq_len),
        **scaler,
    )

    index_rows = []
    for split in ("train", "val", "test"):
        data = arrays[split]
        for i in range(len(data["y"])):
            index_rows.append(
                {
                    "split": split,
                    "filename": str(data["filename"][i]),
                    "subject": str(data["subject"][i]),
                    "base": str(data["base"][i]),
                    "label": int(data["y"][i]),
                    "aug": int(data["aug"][i]),
                    "length": int(data["length"][i]),
                }
            )
    pd.DataFrame(index_rows).to_csv(os.path.join(out_dir, "framed_index.csv"), index=False)

    metadata = {
        "data_path": safe_path(args.data_path),
        "out_dir": out_dir,
        "seq_len": args.seq_len,
        "group": args.group,
        "seed": args.seed,
        "augments": augments,
        "num_records": len(records),
        "splits": {split: len(split_data[split]) for split in ("train", "val", "test")},
        "shapes": {split: {"mfcc": list(arrays[split]["mfcc"].shape), "f0": list(arrays[split]["f0"].shape)} for split in arrays},
    }
    with open(os.path.join(out_dir, "framed_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    logging.info("完成，输出目录: %s", out_dir)


if __name__ == "__main__":
    main()
