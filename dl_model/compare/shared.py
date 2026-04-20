import math

import torch
import torch.nn as nn

from dl_model.old.speechbrain_ablation.shared import (
    CNNPairStudent,
    EmbeddingClassifierStudent,
    LogMelFeatureExtractor,
    MLPPairStudent,
    PairModelBase,
    ResNetPairStudent,
    SincConv1d,
    SincNetPairStudent as _BaseSincNetPairStudent,
    SincTDNNPairStudent as _BaseSincTDNNPairStudent,
    TDNNPairStudent,
    TransformerPairStudent,
    augment_waveforms,
    benchmark_student,
    evaluate_student,
    load_soft_labels,
    set_seed,
    soft_distill_loss,
)


def _assert_cuda_tensor(tensor, tensor_name):
    if tensor.device.type != "cuda":
        raise RuntimeError(
            f"{tensor_name} is on {tensor.device.type}, but compare SincNet models require CUDA input. "
            "Please run with --student-device cuda."
        )


def _assert_cuda_module(module):
    param = next(module.parameters(), None)
    if param is not None and param.device.type != "cuda":
        raise RuntimeError(
            f"{module.__class__.__name__} parameters are on {param.device.type}. "
            "Please move model to CUDA."
        )


class SincNetPairStudent(_BaseSincNetPairStudent):
    def forward(self, left_audio, right_audio):
        _assert_cuda_tensor(left_audio, "left_audio")
        _assert_cuda_tensor(right_audio, "right_audio")
        _assert_cuda_module(self)
        return super().forward(left_audio, right_audio)


class SincTDNNPairStudent(_BaseSincTDNNPairStudent):
    def forward(self, left_audio, right_audio):
        _assert_cuda_tensor(left_audio, "left_audio")
        _assert_cuda_tensor(right_audio, "right_audio")
        _assert_cuda_module(self)
        return super().forward(left_audio, right_audio)


class StatsPooling(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=2)
        std = torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)
        return torch.cat([mean, std], dim=1)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 3:
            scale = self.pool1d(x)
            scale = self.conv1(scale)
            scale = self.relu(scale)
            scale = self.conv2(scale)
            scale = self.sigmoid(scale)
            return x * scale
        if x.dim() == 4:
            scale = self.pool2d(x).squeeze(-1)
            scale = self.conv1(scale)
            scale = self.relu(scale)
            scale = self.conv2(scale)
            scale = self.sigmoid(scale).unsqueeze(-1)
            return x * scale
        raise ValueError(f"SEModule expects 3D or 4D input, got shape {tuple(x.shape)}")


class Res2Block1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8):
        super().__init__()
        if channels % scale != 0:
            raise ValueError("channels must be divisible by scale")
        self.width = channels // scale
        self.scale = scale
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        self.width,
                        self.width,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=((kernel_size - 1) // 2) * dilation,
                    ),
                    nn.BatchNorm1d(self.width),
                    nn.ReLU(),
                )
                for _ in range(scale - 1)
            ]
        )

    def forward(self, x):
        splits = torch.split(x, self.width, dim=1)
        outputs = [splits[0]]
        running = splits[0]
        for idx, block in enumerate(self.blocks, start=1):
            running = block(splits[idx] + running)
            outputs.append(running)
        return torch.cat(outputs, dim=1)


class ECAPAResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, scale=8, dropout=0.1):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.res2 = Res2Block1D(channels, kernel_size=kernel_size, dilation=dilation, scale=scale)
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.se = SEModule(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.pre(x)
        x = self.res2(x)
        x = self.post(x)
        x = self.se(x)
        x = self.dropout(x)
        return self.relu(x + residual)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        weights = self.attn(x)
        mean = torch.sum(x * weights, dim=2)
        var = torch.sum(((x - mean.unsqueeze(2)) ** 2) * weights, dim=2).clamp_min(1e-5)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


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


class ECAPAPairStudent(PairModelBase):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        channels=256,
        emb_dim=192,
        dropout=0.15,
        time_mask_max=12,
        freq_mask_max=8,
        use_specaugment=True,
    ):
        super().__init__(sample_rate, n_mels, time_mask_max, freq_mask_max, use_specaugment)
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ECAPAResidualBlock(channels, dilation=2, dropout=dropout),
                ECAPAResidualBlock(channels, dilation=3, dropout=dropout),
                ECAPAResidualBlock(channels, dilation=4, dropout=dropout),
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv1d(channels * len(self.blocks), channels * 2, kernel_size=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
        )
        self.pool = AttentiveStatsPooling(channels * 2)
        self.embedding = nn.Sequential(
            nn.Linear(channels * 4, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_dim),
        )
        self.classifier = PairClassifierHead(emb_dim, dropout=dropout)

    def encode(self, wav):
        feats = self.normalize_feats(wav).transpose(1, 2)
        x = self.stem(feats)
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        x = self.merge(torch.cat(block_outputs, dim=1))
        return self.embedding(self.pool(x))

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(left_emb, right_emb)


class FrequencyChannelMixer(nn.Module):
    def __init__(self, channels, dropout=0.15):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class ReDimNetPairStudent(PairModelBase):
    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        base_channels=48,
        emb_dim=192,
        dropout=0.15,
        time_mask_max=12,
        freq_mask_max=8,
        use_specaugment=True,
    ):
        super().__init__(sample_rate, n_mels, time_mask_max, freq_mask_max, use_specaugment)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            FrequencyChannelMixer(base_channels, dropout=dropout),
            FrequencyChannelMixer(base_channels, dropout=dropout),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            FrequencyChannelMixer(base_channels * 2, dropout=dropout),
            FrequencyChannelMixer(base_channels * 2, dropout=dropout),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 3),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            FrequencyChannelMixer(base_channels * 3, dropout=dropout),
            SEModule(base_channels * 3, reduction=6),
        )
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 3, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(emb_dim),
        )
        self.classifier = PairClassifierHead(emb_dim, dropout=dropout)

    def encode(self, wav):
        feats = self.normalize_feats(wav).transpose(1, 2).unsqueeze(1)
        x = self.stem(feats)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        return self.proj(x)

    def forward(self, left_audio, right_audio):
        left_emb = self.encode(left_audio)
        right_emb = self.encode(right_audio)
        return self.classifier(left_emb, right_emb)
