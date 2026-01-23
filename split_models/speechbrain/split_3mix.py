import os
import torch
import torchaudio
import torchaudio.functional as F
from speechbrain.inference import SepformerSeparation

# =========================
# 路径
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MIX_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../preprocess/mixed.wav")
)

OUT_DIR = BASE_DIR

assert os.path.exists(MIX_PATH), f"找不到音频文件: {MIX_PATH}"

# =========================
# 读取音频
# =========================
mix, sr = torchaudio.load(MIX_PATH)

# 强制 mono
if mix.size(0) > 1:
    mix = mix.mean(dim=0, keepdim=True)

# 重采样到 16k（非常关键）
if sr != 16000:
    mix = F.resample(mix, sr, 16000)
    sr = 16000

# RMS 归一化（稳定模型输入）
mix = mix / (mix.pow(2).mean().sqrt() + 1e-8)

# =========================
# 加载 SepFormer libri3mix
# =========================
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-libri3mix",
    savedir=os.path.join(BASE_DIR, "hf_models", "sepformer_l3")
)

# =========================
# 分离
# 输出: [1, T, 3]
# =========================
with torch.no_grad():
    est_sources = model.separate_batch(mix)

# =========================
# 后处理 + 全部保存
# =========================
def safe_norm(wav):
    return wav / (wav.abs().max() + 1e-8)

for i in range(est_sources.shape[-1]):
    spk = est_sources[0, :, i]
    spk = safe_norm(spk).unsqueeze(0)

    out_path = os.path.join(OUT_DIR, f"spk{i+1}.wav")
    torchaudio.save(out_path, spk, sr)

    print(f"输出: {out_path}")
