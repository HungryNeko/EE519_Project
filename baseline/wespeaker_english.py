from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from wespeaker_nuaazs import load_model

from baseline.common import BaseSpeakerBaseline, PredictionResult


class WeSpeakerEnglishBaseline(BaseSpeakerBaseline):
    model_name = "wespeaker_english"
    target_sample_rate = 16000

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Path | None = None,
        threshold: float = 0.75,
    ):
        cpu_device = "cpu"
        super().__init__(device=cpu_device, cache_dir=cache_dir)
        self.threshold = threshold
        self.speaker = load_model("english")
        self.speaker.set_gpu(-1)

    def _embed(self, audio, sample_rate: int) -> torch.Tensor:
        wav = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        if sample_rate != self.speaker.resample_rate:
            wav = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.speaker.resample_rate,
            )(wav)

        feats = self.speaker.compute_fbank(
            wav,
            sample_rate=self.speaker.resample_rate,
            cmn=True,
        )
        feats = feats.unsqueeze(0).to(self.speaker.device)

        self.speaker.model.eval()
        with torch.no_grad():
            outputs = self.speaker.model(feats)
            outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
        return outputs[0].detach().cpu()

    def predict(self, left_audio, right_audio, sample_rate: int) -> PredictionResult:
        emb1 = self._embed(left_audio, sample_rate)
        emb2 = self._embed(right_audio, sample_rate)

        cosine_score = torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2) + 1e-8)
        same_speaker_score = float((cosine_score + 1.0) / 2.0)
        raw_score = float(cosine_score.item())
        prediction = int(same_speaker_score >= self.threshold)
        return PredictionResult(
            prediction=prediction,
            same_speaker_score=same_speaker_score,
            raw_score=raw_score,
        )
