from __future__ import annotations

from pathlib import Path

import torch
from speechbrain.inference.speaker import SpeakerRecognition

from baseline.common import BaseSpeakerBaseline, PredictionResult


class SpeechBrainXVectorBaseline(BaseSpeakerBaseline):
    model_name = "speechbrain_xvector"
    target_sample_rate = 16000

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Path | None = None,
        threshold: float = 0.75,
    ):
        cpu_device = "cpu"
        super().__init__(device=cpu_device, cache_dir=cache_dir)
        self.threshold = float(threshold)
        savedir = None
        if cache_dir is not None:
            savedir = str(cache_dir / self.model_name)

        self.verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=savedir,
            run_opts={"device": cpu_device},
        )

    def predict(self, left_audio, right_audio, sample_rate: int) -> PredictionResult:
        wav1 = torch.tensor(left_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        wav2 = torch.tensor(right_audio, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score, _ = self.verifier.verify_batch(wav1, wav2)

        raw_score = float(score.squeeze().detach().cpu().item())
        same_speaker_score = (raw_score + 1.0) / 2.0
        prediction = int(same_speaker_score >= self.threshold)
        return PredictionResult(
            prediction=prediction,
            same_speaker_score=same_speaker_score,
            raw_score=raw_score,
        )
