from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from baseline.common import BaseSpeakerBaseline, PredictionResult, cosine_similarity


class PyannoteWeSpeakerVoxCelebResnet34LMBaseline(BaseSpeakerBaseline):
    model_name = "pyannote_wespeaker_voxceleb_resnet34_lm"
    hf_model_id = "pyannote/wespeaker-voxceleb-resnet34-LM"
    target_sample_rate = 16000

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Path | None = None,
        threshold: float = 0.5,
        hf_token: str | None = None,
    ):
        resolved_device = device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"

        super().__init__(device=resolved_device, cache_dir=cache_dir)
        self.threshold = threshold

        try:
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for "
                f"{self.hf_model_id}. Install it before running this baseline."
            ) from exc

        model_cache_dir = None
        if cache_dir is not None:
            model_cache_dir = cache_dir / self.model_name
            model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = Model.from_pretrained(
            self.hf_model_id,
            use_auth_token=hf_token,
            cache_dir=str(model_cache_dir) if model_cache_dir is not None else None,
        )
        self.inference = Inference(
            self.model,
            window="whole",
            device=torch.device(self.device),
        )

    def _embed(self, audio, sample_rate: int) -> np.ndarray:
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        embedding = self.inference(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        )
        return np.asarray(embedding, dtype=np.float32).reshape(-1)

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
