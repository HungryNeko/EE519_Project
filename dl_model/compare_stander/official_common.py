import sys
import types

import torch
import torch.nn as nn
from speechbrain.lobes.features import Fbank
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, TDNNBlock
from speechbrain.lobes.models.ResNet import ResNet


def _sanitize_speechbrain_lazy_modules():
    """
    SpeechBrain registers optional integrations as LazyModule in sys.modules.
    Some Torch internals call inspect.getmodule(), which probes module.__file__
    and can accidentally trigger LazyModule import side effects (for k2/spacy/flair).
    We set a synthetic __file__ to prevent __getattr__ from firing during inspect,
    and replace k2 lazy module with a harmless stub unless user explicitly imports it.
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:
        return

    for name, module in list(sys.modules.items()):
        if not isinstance(module, LazyModule):
            continue

        if "__file__" not in module.__dict__:
            target = getattr(module, "target", name)
            module.__dict__["__file__"] = f"<lazy:{target}>"

        if getattr(module, "target", None) == "speechbrain.integrations.k2_fsa":
            stub = types.ModuleType(name)
            stub.__file__ = module.__dict__.get("__file__", "<lazy:speechbrain.integrations.k2_fsa>")
            sys.modules[name] = stub


_sanitize_speechbrain_lazy_modules()


class PairClassifierHead(nn.Module):
    def __init__(self, emb_dim, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 2),
        )

    def forward(self, left_emb, right_emb):
        pair = torch.cat(
            [
                0.5 * (left_emb + right_emb),
                torch.abs(left_emb - right_emb),
                left_emb * right_emb,
            ],
            dim=1,
        )
        return self.net(pair)


class _SpeechBrainPairBase(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=40):
        super().__init__()
        self.fbank = Fbank(sample_rate=sample_rate, n_mels=n_mels)

    def _features(self, wav):
        feats = self.fbank(wav)
        feats = feats - feats.mean(dim=1, keepdim=True)
        feats = feats / (feats.std(dim=1, keepdim=True) + 1e-5)
        return feats


class SpeechBrainECAPAPair(_SpeechBrainPairBase):
    def __init__(self, sample_rate=16000, n_mels=40, emb_dim=192, dropout=0.15):
        super().__init__(sample_rate=sample_rate, n_mels=n_mels)
        self.encoder = ECAPA_TDNN(
            input_size=n_mels,
            lin_neurons=emb_dim,
        )
        self.classifier = PairClassifierHead(emb_dim=emb_dim, dropout=dropout)

    def encode(self, wav):
        feats = self._features(wav)
        emb = self.encoder(feats)
        if emb.dim() == 3 and emb.size(1) == 1:
            emb = emb.squeeze(1)
        return emb

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(left_emb, right_emb)


class SpeechBrainResNetPair(_SpeechBrainPairBase):
    def __init__(self, sample_rate=16000, n_mels=40, emb_dim=192, dropout=0.15):
        super().__init__(sample_rate=sample_rate, n_mels=n_mels)
        self.encoder = ResNet(
            input_size=n_mels,
            lin_neurons=emb_dim,
        )
        self.classifier = PairClassifierHead(emb_dim=emb_dim, dropout=dropout)

    def encode(self, wav):
        feats = self._features(wav)
        return self.encoder(feats)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(left_emb, right_emb)


class SpeechBrainTDNNPair(_SpeechBrainPairBase):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=40,
        channels=(128, 192, 256, 256),
        emb_dim=192,
        dropout=0.15,
    ):
        super().__init__(sample_rate=sample_rate, n_mels=n_mels)
        c1, c2, c3, c4 = channels
        self.tdnn = nn.Sequential(
            TDNNBlock(n_mels, c1, kernel_size=5, dilation=1, dropout=dropout),
            TDNNBlock(c1, c2, kernel_size=3, dilation=2, dropout=dropout),
            TDNNBlock(c2, c3, kernel_size=3, dilation=3, dropout=dropout),
            TDNNBlock(c3, c4, kernel_size=1, dilation=1, dropout=dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(c4 * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_dim),
        )
        self.classifier = PairClassifierHead(emb_dim=emb_dim, dropout=dropout)

    def _pool(self, x):
        mean = x.mean(dim=2)
        std = torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)
        return torch.cat([mean, std], dim=1)

    def encode(self, wav):
        feats = self._features(wav).transpose(1, 2)
        x = self.tdnn(feats)
        return self.proj(self._pool(x))

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(left_emb, right_emb)
