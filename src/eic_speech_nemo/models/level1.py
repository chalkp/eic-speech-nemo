from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eic_speech_nemo.models.level0 import CausalConv2d


class MelSpectrogram(nn.Module):
    """ Mel spectrogram

    State dict keys:
        preprocessor.featurizer.window : [400]
        preprocessor.featurizer.fb     : [1,128,257]
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_mels: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.featurizer = nn.Module()
        self.featurizer.register_buffer(
            "window", torch.hann_window(win_length)
        ) # hann window
        self.featurizer.register_buffer(
            "fb", torch.zeros(1, n_mels, n_fft // 2 + 1)
        ) # mel filterbank

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor):
        # STFT
        x = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.featurizer.window,
            center=True,
            return_complex=True,
        )
        x = x.abs().pow(2) # Power spectrum [B, n_fft//2+1, T]
        x = torch.matmul(self.featurizer.fb, x) # Mel filterbank [B, n_mels, T]
        x = torch.log(x + 2**-24)

        feat_len = (lengths + self.hop_length) // self.hop_length
        feat_len = feat_len.clamp(max=x.size(2))

        return x, feat_len


class ConvSubsampling(nn.Module):
    """
    State dict keys (encoder.pre_encode):
        conv.0  : CausalConv2D(1, 256, 3, stride=2)
        conv.1  : ReLU
        conv.2  : CausalConv2D(256, 256, 3, stride=2, groups=256) — dw
        conv.3  : Conv2d(256, 256, 1)                — pointwise
        conv.4  : ReLU
        conv.5  : CausalConv2D(256, 256, 3, stride=2, groups=256) — dw
        conv.6  : Conv2d(256, 256, 1)                — pointwise
        conv.7  : ReLU
        out     : Linear(256 * 17, 1024) = Linear(4352, 1024)
    """

    def __init__(self, n_mels: int, subsampling_channels: int, d_model: int):
        super().__init__()
        C = subsampling_channels # 256

        self.conv = nn.Sequential(
            CausalConv2d(1, C, 3, stride=2),                            # 0
            nn.ReLU(),                                                  # 1
            CausalConv2d(C, C, 3, stride=2, groups=C),                  # 2
            nn.Conv2d(C, C, 1, bias=True),                              # 3
            nn.ReLU(),                                                  # 4
            CausalConv2d(C, C, 3, stride=2, groups=C),                  # 5
            nn.Conv2d(C, C, 1, bias=True),                              # 6
            nn.ReLU(),                                                  # 7
        )
        """
        output = (input + left + right - kernel) // stride + 1
                = (input + 2 + 1 - 3) // 2 + 1 = input // 2 + 1
        """
        # Causal pad: left=2, right=1, kernel=3, stride=2
        freq = n_mels
        for _ in range(3):
            freq = freq // 2 + 1 # 128 -> 65 -> 33 -> 17
        self.out = nn.Linear(C * freq, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: [B, n_mels, T]
        x = x.transpose(1, 2) # [B, T, F]
        x = x.unsqueeze(1) # [B, 1, T, F]
        x = self.conv(x) # [B, C, T_sub, F_sub]
        B, C, T, F = x.shape
        # [B, T, C, F] -> [B, T, C * F]
        x = x.transpose(1, 2).reshape(B, T, C * F)
        x = self.out(x) # [B, T, d_model]

        for _ in range(3):
            lengths = lengths // 2 + 1
        lengths = lengths.clamp(max=T)

        return x, lengths


class RelPosMultiHeadAttention(nn.Module):
    """
    State dict keys (self_attn):
        pos_bias_u     : [n_heads, head_dim]
        pos_bias_v     : [n_heads, head_dim]
        linear_q.weight: [d, d]
        linear_k.weight: [d, d]
        linear_v.weight: [d, d]
        linear_out.weight: [d, d]
        linear_pos.weight: [d, d]
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)

        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.head_dim))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, 2T-1] -> [B, H, T, T]
        B, H, T, L = x.shape
        # Pad left col
        x = F.pad(x, (1, 0)) # [B, H, T, L+1]
        x = x.reshape(B, H, L + 1, T) # reshape
        x = x[:, :, 1:, :] # drop first row
        x = x.reshape(B, H, T, L)
        return x[:, :, :, :T] # trim to [B, H, T, T]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # pos_emb: [1, 2T-1, D] (sin)
        B, T, D = x.shape
        H, d = self.n_heads, self.head_dim

        q = self.linear_q(x).reshape(B, T, H, d).permute(0, 2, 1, 3) # [B, H, T, d]
        k = self.linear_k(x).reshape(B, T, H, d).permute(0, 2, 1, 3)
        v = self.linear_v(x).reshape(B, T, H, d).permute(0, 2, 1, 3)

        # Pos Embed
        p = self.linear_pos(pos_emb) # [1, 2T-1, D]
        p = p.reshape(1, -1, H, d).permute(0, 2, 1, 3) # [1, H, 2T-1, d]

        # Content attention: (q + bias_u) @ k^T
        q_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2) # [B,H,T,d]
        content_score = torch.matmul(q_u, k.transpose(-2, -1)) # [B,H,T,T]

        # Position attention: (q + bias_v) @ p^T
        q_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        pos_score = torch.matmul(q_v, p.transpose(-2, -1)) # [B,H,T,2T-1]
        pos_score = self._rel_shift(pos_score) # [B,H,T,T]

        scale = math.sqrt(d)
        attn = (content_score + pos_score) / scale # [B,H,T,T]
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v) # [B,H,T,d]
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        return self.linear_out(out)
