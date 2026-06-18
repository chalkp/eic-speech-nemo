"""
NemotronASR
- https://arxiv.org/pdf/2312.17279
- https://arxiv.org/pdf/2305.05084

Architecture: FastConformer Encoder -> LSTM Prediction Network -> RNNT Joint
Decoding: Greedy RNNT (token-by-token, argmax)

Ported from https://github.com/NVIDIA-NeMo/NeMo.git
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from eic_speech_nemo.models.level1 import MelSpectrogram
from eic_speech_nemo.models.level2 import ConformerEncoder, RNNTDecoder, RNNTJoint


@dataclass
class ASRConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    window_size: int = 400 # 0.025 * 16000
    hop_length: int = 160 # 0.01  * 16000
    n_mels: int = 128

    encoder_layers: int = 24
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion: int = 4
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_channels: int = 256

    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    joint_hidden: int = 640

    vocab_size: int = 1024 # BPE tokens
    blank_id: int = 1024

    max_symbols_per_step: int = 10


class NemotronASR(nn.Module):
    """Nemotron-Speech-Streaming ASR model"""

    def __init__(self, cfg: ASRConfig, vocabulary: list[str] | None = None):
        super().__init__()
        self.cfg = cfg
        self.vocabulary = vocabulary or []

        self.preprocessor = MelSpectrogram(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.window_size,
            n_mels=cfg.n_mels,
        )
        self.encoder = ConformerEncoder(
            n_mels=cfg.n_mels,
            subsampling_channels=cfg.subsampling_channels,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ff_expansion=cfg.ff_expansion,
            conv_kernel_size=cfg.conv_kernel_size,
            encoder_layers=cfg.encoder_layers,
        )
        self.decoder = RNNTDecoder(
            vocab_size=cfg.vocab_size,
            blank_id=cfg.blank_id,
            pred_hidden=cfg.pred_hidden,
            pred_rnn_layers=cfg.pred_rnn_layers,
        )
        self.joint = RNNTJoint(
            d_model=cfg.d_model,
            pred_hidden=cfg.pred_hidden,
            joint_hidden=cfg.joint_hidden,
            vocab_size=cfg.vocab_size,
        )

    # Greedy RNNT Decoding
    @torch.no_grad()
    def _greedy_decode(
        self, encoder_out: torch.Tensor, enc_len: torch.Tensor
    ) -> list[list[int]]:
        # encoder_out: [B, D, T]
        # enc_len: [B]
        B = encoder_out.size(0)
        results = []

        for b in range(B):
            T = int(enc_len[b].item())
            enc = encoder_out[b: b + 1, :, :T] # [1, D, T]
            enc = enc.transpose(1, 2) # [1, T, D]

            tokens: list[int] = []
            state = None
            last_token = self.cfg.blank_id

            for t in range(T):
                f = enc[:, t: t + 1, :] # [1, 1, D]

                for _ in range(self.cfg.max_symbols_per_step):
                    label = torch.tensor(
                        [[last_token]], device=enc.device, dtype=torch.long
                    )
                    g, new_state = self.decoder.prediction(label, state)
                    # g: [1, 1, H]

                    # Joint step
                    logits = self.joint(f, g) # [1, 1, 1, V+1]
                    logp = logits[0, 0, 0] # [V+1]
                    k = logp.argmax().item()

                    if k == self.cfg.blank_id:
                        break
                    else:
                        tokens.append(k)
                        last_token = k
                        state = new_state

            results.append(tokens)
        return results

    def _tokens_to_text(self, token_ids: list[int]) -> str:
        """Decode BPE token ID to text using the spm vocab"""
        if not self.vocabulary:
            return str(token_ids)

        pieces = []
        for tid in token_ids:
            if 0 <= tid < len(self.vocabulary):
                pieces.append(self.vocabulary[tid])

        text = "".join(pieces)
        text = text.replace("▁", " ")
        return text.strip()

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor) -> str:
        # audio: [T_samples] / [1, T_samples]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        device = next(self.parameters()).device
        # Preprocessor (STFT) always runs in FP32 — cuFFT doesn't support BF16
        audio = audio.to(device=device, dtype=torch.float32)
        lengths = torch.tensor([audio.size(1)], device=device, dtype=torch.long)

        # Mel spectrogram (FP32)
        features, feat_len = self.preprocessor(audio, lengths)

        # Cast to encoder dtype (BF16 when using reduced precision)
        features = features.to(self.encoder.pre_encode.out.weight.dtype)

        # Encoder
        encoded, enc_len = self.encoder(features, feat_len)

        # Greedy RNNT decode
        token_ids = self._greedy_decode(encoded, enc_len)

        # Tokens -> text
        return self._tokens_to_text(token_ids[0])

    @torch.no_grad()
    def transcribe_batch(self, audios: list[torch.Tensor]) -> list[str]:
        """Transcribe a batch of audio waveforms"""
        return [self.transcribe(a) for a in audios]

    @staticmethod
    def load_from_pt(
        path: str, device: str = "cpu", precision: str = "fp32",
    ) -> "NemotronASR":
        """
        The .pt file contains:
            state_dict : OrderedDict of all model weights
            vocabulary : list of 1024 BPE token strings
            config : dict with architecture params

        Args:
            precision: "fp32", "bf16", or "fp8" (fp8 requires torchao + Hopper GPU).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Build config
        cfg_dict = checkpoint.get("config", {})
        cfg = ASRConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in ASRConfig.__dataclass_fields__
        })

        # Build model
        vocab = checkpoint.get("vocabulary", [])
        model = NemotronASR(cfg, vocabulary=vocab)

        # Remap NeMo keys to match CausalConv2d wrapper structure
        # encoder.pre_encode.conv.{0,2,5}.weight -> encoder.pre_encode.conv.{0,2,5}.conv.weight
        sd = {}
        for k, v in checkpoint["state_dict"].items():
            new_k = k
            for idx in ("0", "2", "5"):
                prefix = f"encoder.pre_encode.conv.{idx}."
                if k.startswith(prefix) and not k.startswith(prefix + "conv."):
                    suffix = k[len(prefix):]
                    new_k = f"{prefix}conv.{suffix}"
                    break
            sd[new_k] = v

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # Debug
        real_missing = [k for k in missing if "running_" not in k and "num_batches" not in k]
        if real_missing:
            print(f"[ASR] Warning: {len(real_missing)} missing keys: {real_missing[:5]}")
        if unexpected:
            print(f"[ASR] Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}")

        model.eval()

        if precision in ("bf16", "fp8"):
            model = model.to(torch.bfloat16)
            # cuFFT doesn't support BF16, keep preprocessor in FP32 (wow, new knowledge af)
            model.preprocessor = model.preprocessor.to(torch.float32)

        if device != "cpu":
            model = model.to(device)

        if precision == "fp8":
            try:
                import torchao
                torchao.quantize_(model, torchao.float8_weight_only())
            except ImportError:
                raise ImportError(
                    "FP8 quantization requires torchao: pip install torchao"
                )

        return model
