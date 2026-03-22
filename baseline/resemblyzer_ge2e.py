from __future__ import annotations

from pathlib import Path

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

from baseline.common import BaseSpeakerBaseline, PredictionResult, cosine_similarity


class ResemblyzerGE2EBaseline(BaseSpeakerBaseline):
    model_name = "resemblyzer_ge2e"
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
        self.encoder = VoiceEncoder(device=cpu_device)

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        processed = preprocess_wav(audio, source_sr=sample_rate)
        return self.encoder.embed_utterance(processed)

    def predict(self, left_audio, right_audio, sample_rate: int) -> PredictionResult:
        emb1 = self._embed(left_audio, sample_rate)
        emb2 = self._embed(right_audio, sample_rate)

        raw_score = cosine_similarity(emb1, emb2)
        same_speaker_score = (raw_score + 1.0) / 2.0
        prediction = int(same_speaker_score >= self.threshold)
        return PredictionResult(
            prediction=prediction,
            same_speaker_score=same_speaker_score,
            raw_score=raw_score,
        )
