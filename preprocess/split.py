from speechbrain.pretrained import SepformerSeparation
import torchaudio

model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer"
)

# 读取混合音频
mix, sr = torchaudio.load("preprocess\mixed.wav")

# 分离
est_sources = model.separate_batch(mix)

# 保存结果
torchaudio.save("spk1.wav", est_sources[0, :, 0].detach(), sr)
torchaudio.save("spk2.wav", est_sources[0, :, 1].detach(), sr)
