from __future__ import annotations

from pathlib import Path

from baseline.common import BaseSpeakerBaseline, PredictionResult
from dl_model.old.predict import MLPWhisperSpeakerPredictor


class ProjectMLPWhisperBaseline(BaseSpeakerBaseline):
    model_name = "project_mlp_whisper"
    target_sample_rate = 16000

    def __init__(self, device: str = "cpu", cache_dir: Path | None = None):
        cpu_device = "cpu"
        super().__init__(device=cpu_device, cache_dir=cache_dir)
        repo_root = Path(__file__).resolve().parents[1]
        model_path = repo_root / "dl_model" / "checkpoints" / "MLPModel1_best.pth"
        self.predictor = MLPWhisperSpeakerPredictor(
            model_path=model_path,
            device=cpu_device,
            sample_rate=self.target_sample_rate,
        )

    def predict(self, left_audio, right_audio, sample_rate: int) -> PredictionResult:
        result = self.predictor.predict_pair(left_audio, right_audio, sample_rate=sample_rate)
        return PredictionResult(
            prediction=int(result["prediction"]),
            same_speaker_score=float(result["same_speaker_score"]),
            raw_score=float(result["raw_score"]),
        )
