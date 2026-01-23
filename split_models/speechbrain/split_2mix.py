import os
import torch
import torchaudio
import torchaudio.functional as F
from speechbrain.inference import SepformerSeparation

# =====================
# 路径
# =====================
base_dir = os.path.dirname(__file__)
mix_path = os.path.join(base_dir, "../../preprocess/mixed.wav")

# =====================
# 读音频
# =====================
mix, sr = torchaudio.load(mix_path)

# 强制 mono
if mix.size(0) > 1:
    mix = mix.mean(dim=0, keepdim=True)

# 重采样到 16k（非常关键）
if sr != 16000:
    mix = F.resample(mix, sr, 16000)
    sr = 16000

# RMS 归一化（抑制串音）
mix = mix / (mix.pow(2).mean().sqrt() + 1e-8)

# =====================
# 加载模型（本地）
# =====================
model = SepformerSeparation.from_hparams(
    source=os.path.join(base_dir, "pretrained_models", "sepformer"),
    savedir=os.path.join(base_dir, "pretrained_models", "sepformer")
)

# =====================
# 分离
# =====================
est_sources = model.separate_batch(mix)
spk1 = est_sources[0, :, 0]
spk2 = est_sources[0, :, 1]

# =====================
# 后处理（杀串音关键）
# =====================

# aggressive norm
def aggressive_norm(wav, q=0.95):
    scale = torch.quantile(wav.abs(), q)
    return wav / (scale + 1e-8)

spk1 = aggressive_norm(spk1)
spk2 = aggressive_norm(spk2)

# 高通，去低频串音
spk1 = F.highpass_biquad(spk1.unsqueeze(0), sr, cutoff_freq=80)
spk2 = F.highpass_biquad(spk2.unsqueeze(0), sr, cutoff_freq=80)

# 能量门限
def energy_gate(wav, thr=0.02):
    return wav * (wav.abs() > thr)

spk1 = energy_gate(spk1)
spk2 = energy_gate(spk2)

# =====================
# 保存
# =====================
torchaudio.save(os.path.join(base_dir, "spk1.wav"), spk1, sr)
torchaudio.save(os.path.join(base_dir, "spk2.wav"), spk2, sr)
