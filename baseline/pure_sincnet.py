from __future__ import annotations

from pathlib import Path

import torch

from baseline.common import BaseSpeakerBaseline, PredictionResult
from dl_model.old.speechbrain_ablation.shared import SincNetPairStudent


class PureSincNetBaseline(BaseSpeakerBaseline):
    model_name = "pure_sincnet"
    target_sample_rate = 16000

    def __init__(self, device: str = "cpu", cache_dir: Path | None = None):
        cpu_device = "cpu"
        super().__init__(device=cpu_device, cache_dir=cache_dir)
        repo_root = Path(__file__).resolve().parents[1]
        checkpoint_path = repo_root / "dl_model" / "speechbrain_ablation" / "checkpoints" / "pure_sincnet_best_acc.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        saved_args = checkpoint.get("args", {})
        self.model = SincNetPairStudent(
            sample_rate=int(saved_args.get("sr", self.target_sample_rate)),
            emb_dim=int(saved_args.get("emb_dim", 192)),
            dropout=float(saved_args.get("dropout", 0.15)),
            sinc_channels=64,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, left_audio, right_audio, sample_rate: int) -> PredictionResult:
        left = torch.tensor(left_audio, dtype=torch.float32, device=self.device).unsqueeze(0)
        right = torch.tensor(right_audio, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(left, right)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        same_speaker_score = float(probs[1].detach().cpu().item())
        raw_score = float((logits[0, 1] - logits[0, 0]).detach().cpu().item())
        prediction = int(torch.argmax(probs).detach().cpu().item())
        return PredictionResult(
            prediction=prediction,
            same_speaker_score=same_speaker_score,
            raw_score=raw_score,
        )
