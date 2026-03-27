from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMForXVector

from baseline.common import BaseSpeakerBaseline, PredictionResult


class MicrosoftWavLMBasePlusSVBaseline(BaseSpeakerBaseline):
    model_name = "microsoft_wavlm_base_plus_sv"
    hf_model_id = "microsoft/wavlm-base-plus-sv"
    target_sample_rate = 16000

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: Path | None = None,
        threshold: float = 0.5,
    ):
        resolved_device = device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            resolved_device = "cpu"

        super().__init__(device=resolved_device, cache_dir=cache_dir)
        self.threshold = threshold

        model_cache_dir = None
        if cache_dir is not None:
            model_cache_dir = str(cache_dir / self.model_name)

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.hf_model_id,
            cache_dir=model_cache_dir,
        )
        self.model = WavLMForXVector.from_pretrained(
            self.hf_model_id,
            cache_dir=model_cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()

    def _embed(self, audio, sample_rate: int) -> torch.Tensor:
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = F.normalize(outputs.embeddings, dim=-1)
        return embeddings.squeeze(0).detach().cpu()

    def predict(self, left_audio, right_audio, sample_rate: int) -> PredictionResult:
        emb1 = self._embed(left_audio, sample_rate)
        emb2 = self._embed(right_audio, sample_rate)

        raw_score = float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())
        same_speaker_score = (raw_score + 1.0) / 2.0
        prediction = int(same_speaker_score >= self.threshold)
        return PredictionResult(
            prediction=prediction,
            same_speaker_score=same_speaker_score,
            raw_score=raw_score,
        )
