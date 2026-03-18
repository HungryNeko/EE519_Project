import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 🔴 彻底禁用GPU

import torch
import whisper
import numpy as np

print("=== DEVICE CHECK ===")
print("CUDA available:", torch.cuda.is_available())  # 应该是 False

# =========================
# 1. LOAD MODEL（强制CPU）
# =========================
print("\n=== LOAD MODEL ===")
model = whisper.load_model("base", device="cpu")
print("Model device:", next(model.parameters()).device)

# =========================
# 2. AUDIO
# =========================
sr = 16000
t = np.linspace(0, 1, sr, endpoint=False)
audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)

# =========================
# 3. ENCODER
# =========================
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to("cpu")  # 🔴 强制

with torch.no_grad():
    enc = model.encoder(mel.unsqueeze(0))

emb = enc.mean(dim=1).squeeze().cpu().numpy()

print("Embedding shape:", emb.shape)

# =========================
# 4. ASR
# =========================
result = model.decode(mel, whisper.DecodingOptions(language="en"))
print("Decoded:", result.text)

print("\n=== DONE ===")