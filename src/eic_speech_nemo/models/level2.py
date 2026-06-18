from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from eic_speech_nemo.models.level0 import FeedForward, ConvModule, sinusoidal_pos_emb
from eic_speech_nemo.models.level1 import ConvSubsampling, RelPosMultiHeadAttention


class ConformerLayer(nn.Module):
    """ Single Conformer layer

    Order: 1/2 FF1 -> SelfAttn -> ConvModule -> 1/2 FF2 -> LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, ff_expansion: int, conv_kernel: int):
        super().__init__()
        # FF1
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForward(d_model, ff_expansion)

        # Self Attn
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPosMultiHeadAttention(d_model, n_heads)

        # Convolution
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, conv_kernel)

        # FF2
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForward(d_model, ff_expansion)

        # Layer norm
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        # 1/2 FF1
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        # Self Attn
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        # Convolution
        x = x + self.conv(self.norm_conv(x))
        # 1/2 FF2
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        # layer norm
        x = self.norm_out(x)
        return x


class ConformerEncoder(nn.Module):
    """FastConformer encoder + subsampling"""

    def __init__(
        self,
        n_mels: int,
        subsampling_channels: int,
        d_model: int,
        n_heads: int,
        ff_expansion: int,
        conv_kernel_size: int,
        encoder_layers: int,
    ):
        super().__init__()
        self.pre_encode = ConvSubsampling(n_mels, subsampling_channels, d_model)
        self.layers = nn.ModuleList([
            ConformerLayer(d_model, n_heads, ff_expansion, conv_kernel_size)
            for _ in range(encoder_layers)
        ])
        self.d_model = d_model

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        # audio_signal: [B, n_mels, T] (log-mel)
        # length [B]
        # Subsampling: [B, n_mels, T] -> [B, T_sub, d_model]
        x, length = self.pre_encode(audio_signal, length)

        # Pos embed
        T = x.size(1)
        pos_emb = sinusoidal_pos_emb(T, self.d_model, x.device).to(x.dtype)

        # Conformer
        for layer in self.layers:
            x = layer(x, pos_emb)

        # [B, T, D] -> [B, D, T]
        return x.transpose(1, 2), length


class PredictionNetwork(nn.Module):
    """ LSTM for RNNT

    State dict keys (decoder.prediction):
        embed.weight                    : [1025, 640]
        dec_rnn.lstm.weight_ih_l{0,1}   : [2560, 640]
        dec_rnn.lstm.weight_hh_l{0,1}   : [2560, 640]
        dec_rnn.lstm.bias_ih_l{0,1}     : [2560]
        dec_rnn.lstm.bias_hh_l{0,1}     : [2560]
    """

    def __init__(self, vocab_size: int, blank_id: int, pred_hidden: int, pred_rnn_layers: int):
        super().__init__()
        self.blank_id = blank_id
        self.hidden_size = pred_hidden

        self.embed = nn.Embedding(
            vocab_size + 1, pred_hidden, padding_idx=blank_id
        )
        self.dec_rnn = nn.Module()
        self.dec_rnn.lstm = nn.LSTM(
            input_size=pred_hidden,
            hidden_size=pred_hidden,
            num_layers=pred_rnn_layers,
            batch_first=True,
        )

    def forward(
        self,
        targets: Optional[torch.Tensor] = None,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        add_sos: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # targets: [B, U] (token IDs)
        # state: (h, c) (LSTM state)
        if targets is None:
            # SOS = zero vector
            B = 1
            y = torch.zeros(B, 1, self.hidden_size, device=self._device())
        else:
            y = self.embed(targets) # [B, U, H]

        if add_sos:
            B = y.size(0)
            sos = torch.zeros(B, 1, self.hidden_size, device=y.device)
            y = torch.cat([sos, y], dim=1)

        y, state = self.dec_rnn.lstm(y, state)
        return y, state

    def _device(self):
        return self.embed.weight.device


class RNNTDecoder(nn.Module):
    """Wrapper for decoder.prediction"""

    def __init__(self, vocab_size: int, blank_id: int, pred_hidden: int, pred_rnn_layers: int):
        super().__init__()
        self.prediction = PredictionNetwork(vocab_size, blank_id, pred_hidden, pred_rnn_layers)


class RNNTJoint(nn.Module):
    """ Joint network: enc_proj(f) + pred_proj(g) -> ReLU -> linear -> logit

    State dict keys:
        joint.enc.weight        : [640, 1024]
        joint.enc.bias          : [640]
        joint.pred.weight       : [640, 640]
        joint.pred.bias         : [640]
        joint.joint_net.2.weight: [1025, 640]
        joint.joint_net.2.bias  : [1025]
    """

    def __init__(self, d_model: int, pred_hidden: int, joint_hidden: int, vocab_size: int):
        super().__init__()
        self.enc = nn.Linear(d_model, joint_hidden)
        self.pred = nn.Linear(pred_hidden, joint_hidden)
        # joint_net:
        # 0 = ReLU
        # 1 = Dropout
        # 2 = Linear
        self.joint_net = nn.Sequential(
            nn.ReLU(),
            nn.Identity(),  # dropout
            nn.Linear(joint_hidden, vocab_size + 1),
        )

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # f: [B, T, D_enc] encoder output
        # g: [B, U, D_pred] decoder output
        f = self.enc(f).unsqueeze(2) # [B, T, 1, H]
        g = self.pred(g).unsqueeze(1) # [B, 1, U, H]
        h = f + g # [B, T, U, H]
        return self.joint_net(h) # [B, T, U, V+1]
