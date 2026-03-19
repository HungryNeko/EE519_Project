import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
import whisper
import whisperx

from functions import SpeakerFeatureExtractor


# =========================
# 模型（与训练保持一致）
# =========================
class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SwitchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        l = self.enc(x[:, 0].unsqueeze(1))
        r = self.enc(x[:, 1].unsqueeze(1))
        d = torch.abs(l - r)
        return self.fc(torch.cat([l, r, d], dim=1))


# =========================
# 音频与窗口（与训练一致：固定长度+补零）
# =========================
def load_audio(path: Path, sr=16000):
    wav, s = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if s != sr:
        wav = librosa.resample(wav, orig_sr=s, target_sr=sr)
    return wav.astype(np.float32), sr


def extract_window(wav: np.ndarray, sr: int, start_time: float, end_time: float):
    start_i = int(round(start_time * sr))
    end_i = int(round(end_time * sr))
    length = max(1, end_i - start_i)
    out = np.zeros(length, dtype=np.float32)

    src_start = max(0, start_i)
    src_end = min(len(wav), end_i)
    if src_end <= src_start:
        return out

    dst_start = src_start - start_i
    dst_end = dst_start + (src_end - src_start)
    out[dst_start:dst_end] = wav[src_start:src_end]
    return out


# =========================
# 语言检测（字符级）
# =========================
def detect_lang(text: str):
    if not text:
        return "unk"
    zh = any('\u4e00' <= c <= '\u9fff' for c in text)
    hi = any('\u0900' <= c <= '\u097f' for c in text)
    en = any(c.isascii() and c.isalpha() for c in text)

    if zh:
        return "zh"
    if hi:
        return "hi"
    if en:
        return "en"
    return "unk"


# =========================
# 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path.resolve()}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ===== 加载模型 =====
    model = SwitchCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ===== 加载音频 =====
    wav, sr = load_audio(audio_path, sr=16000)

    # ===== Whisper 转写 =====
    whisper_model = whisper.load_model("base", device=device)
    asr_res = whisper_model.transcribe(str(audio_path))
    full_text = asr_res.get("text", "").strip()

    print("\n=== ASR TEXT ===")
    print(full_text if full_text else "(empty)")

    # ===== WhisperX 对齐（word-level）=====
    align_model, metadata = whisperx.load_align_model(
        language_code=asr_res["language"], device=device
    )

    aligned = whisperx.align(
        asr_res["segments"],
        align_model,
        metadata,
        str(audio_path),
        device
    )

    words = aligned.get("word_segments", [])

    # ===== 找语言切换：用“前一个 word 的 end” =====
    prev_lang = None
    prev_word = None
    switch_time = None

    for w in words:
        word = w.get("word", "")
        if not word:
            continue

        lang = detect_lang(word)

        if prev_lang is None:
            prev_lang = lang
            prev_word = w
            continue

        if lang != prev_lang:
            # 关键：用前一个词的 end（对齐训练逻辑）
            switch_time = prev_word["end"]
            break

        prev_lang = lang
        prev_word = w

    if switch_time is None:
        print("\nNo switch detected → 没有检测到切换")
        print("Result: mix")
        return

    print(f"\nSwitch detected at: {switch_time:.3f} sec")

    # ===== 切窗口（固定1秒+补零，与训练一致）=====
    seg1 = extract_window(wav, sr, switch_time - 1.0, switch_time)
    seg2 = extract_window(wav, sr, switch_time, switch_time + 1.0)

    # ===== embedding（完全复用你的提取器）=====
    extractor = SpeakerFeatureExtractor(sr=16000)
    emb1 = extractor.extract_embedding(seg1)
    emb2 = extractor.extract_embedding(seg2)

    # 避免 slow warning：先 stack 再转 tensor
    pair = np.stack([emb1, emb2], axis=0)  # [2,512]
    x = torch.from_numpy(pair).unsqueeze(0).float().to(device)  # [1,2,512]

    # ===== 预测 =====
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    label = "code_switch" if pred == 1 else "mix"
    print(f"Prediction: {label}")
    print(f"Probabilities: mix={probs[1]:.4f}, code_switch={probs[0]:.4f}")


if __name__ == "__main__":
    main()