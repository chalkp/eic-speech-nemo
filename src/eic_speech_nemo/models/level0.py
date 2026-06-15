from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """Causal Conv2d: pads left = kernel-1, right = stride-1 on both dims"""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, bias=True):
        super().__init__()
        self._left = kernel_size - 1
        self._right = stride - 1
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=0, groups=groups, bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self._left, self._right, self._left, self._right))
        return self.conv(x)


class FeedForward(nn.Module):
    """SiLU FF: Linear -> SiLU -> Linear"""

    def __init__(self, d_model: int, expansion: int):
        super().__init__()
        d_ff = d_model * expansion
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class ConvModule(nn.Module):
    """ Conformer Convolution Module
    pointwise -> GLU -> depthwise -> LN -> SiLU -> pointwise.

    State dict:
        pointwise_conv1.weight : [2*d, d, 1]
        depthwise_conv.weight  : [d, 1, K]
        batch_norm.weight      : [d]
        batch_norm.bias        : [d]
        pointwise_conv2.weight : [d, d, 1]
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)
        self.pad_size = kernel_size - 1
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            groups=d_model, bias=False,
        )
        # NeMo names this batch_norm but actually uses LayerNorm
        self.batch_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x.transpose(1, 2) # [B, D, T]
        x = self.pointwise_conv1(x) # [B, 2D, T]
        x = F.glu(x, dim=1) # [B, D, T]
        x = F.pad(x, (self.pad_size, 0)) # Causal padding
        x = self.depthwise_conv(x) # [B, D, T]
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2) # [B, T, D]
        x = F.silu(x)
        x = self.pointwise_conv2(x) # [B, D, T]
        return x.transpose(1, 2) # [B, T, D]



def sinusoidal_pos_emb(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length - 1, -(length), -1, dtype=torch.float32, device=device)
    dim = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
    div = torch.exp(-dim * (math.log(10000.0) / d_model))

    pe = torch.zeros(2 * length - 1, d_model, device=device)
    pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div.unsqueeze(0))
    pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div.unsqueeze(0))
    return pe.unsqueeze(0)
