from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile


def _to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    if data.dtype == np.int16:
        return (data.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
    if data.dtype == np.int32:
        return (data.astype(np.float32) / 2147483648.0).clip(-1.0, 1.0)
    if data.dtype == np.uint8:
        return ((data.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    max_abs = max(abs(np.iinfo(data.dtype).min), abs(np.iinfo(data.dtype).max))
    return (data.astype(np.float32) / float(max_abs)).clip(-1.0, 1.0)


def read(file: str | Path) -> Tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(file))
    return _to_float32(np.asarray(data)), int(sr)


def write(file: str | Path, data: np.ndarray, samplerate: int) -> None:
    audio = np.asarray(data, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    int_audio = (audio * 32767.0).astype(np.int16)
    wavfile.write(str(file), int(samplerate), int_audio)
