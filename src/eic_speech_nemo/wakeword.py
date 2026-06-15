"""Pure-PyTorch streaming wake word detector.

Replaces openwakeword + onnxruntime with a single torch-only module.
Loads the combined .pt archive produced by ``convert_wakeword.py``.

Pipeline per chunk:
    raw PCM (int16→float32) → STFT → mel filterbank → log-dB normalize
    → sliding 76-frame window → embedding CNN → sliding 16-frame buffer
    → wakeword classifier → score ∈ [0, 1]
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WakeWordDetector:
    """Streaming wake word detector — pure PyTorch, no ONNX runtime.

    Drop-in replacement for ``openwakeword.model.Model``.
    """

    def __init__(self, model_dir: str = "weights", prefix: str = "wakeword",
                 threshold: float = 0.5):
        """
        Args:
            model_dir: Directory containing the wakeword model files.
            prefix: Filename prefix (default "wakeword"), expects:
                    {prefix}_mel.pt, {prefix}_embed.pt, {prefix}_classifier.pt
            threshold: Detection threshold.
        """
        import os
        self.threshold = threshold

        mel_path = os.path.join(model_dir, f"{prefix}_mel.pt")
        embed_path = os.path.join(model_dir, f"{prefix}_embed.pt")
        ww_path = os.path.join(model_dir, f"{prefix}_classifier.pt")

        # Load mel spectrogram weights (plain tensors)
        mel_data = torch.load(mel_path, map_location="cpu", weights_only=True)
        self._stft_real: torch.Tensor = mel_data["stft_real"]       # (257, 1, 512)
        self._stft_imag: torch.Tensor = mel_data["stft_imag"]       # (257, 1, 512)
        self._mel_fb: torch.Tensor = mel_data["mel_filterbank"]     # (257, 32)

        # Load TorchScript models (self-contained, no onnx2torch needed)
        self._embed = torch.jit.load(embed_path, map_location="cpu")
        self._embed.eval()
        self._ww = torch.jit.load(ww_path, map_location="cpu")
        self._ww.eval()

        # ── Streaming state (mirrors openwakeword.utils.AudioFeatures) ──
        self._sr = 16000
        self._raw_buffer: deque = deque(maxlen=self._sr * 10)
        # mel buffer: starts as (76, 32) of ones (like openwakeword)
        self._mel_buffer = np.ones((76, 32), dtype=np.float32)
        self._mel_max_len = 10 * 97  # ~10s of mel frames
        # embedding/feature buffer: starts blank
        self._feature_buffer = self._compute_blank_features()
        self._feature_max_len = 120  # ~10s of feature history
        self._accumulated_samples = 0

    # ── Mel spectrogram (replaces melspectrogram.onnx) ──────────────

    @torch.no_grad()
    def _get_melspectrogram(self, raw_samples: list) -> np.ndarray:
        """Compute mel spectrogram from raw audio samples.

        Exactly replicates the ONNX melspectrogram graph:
            Conv1d(real) + Conv1d(imag) → power → matmul(melW)
            → log → *10/ln(10) → dB clipping (max-80) → /10+2
        """
        x = np.array(raw_samples, dtype=np.int16)
        x = x.astype(np.float32)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, N)

        real = F.conv1d(x_t, self._stft_real, stride=160)
        imag = F.conv1d(x_t, self._stft_imag, stride=160)
        power = (real ** 2 + imag ** 2).squeeze(0).T  # (T, 257)
        mel = power @ self._mel_fb  # (T, 32)

        # Log-dB normalization (matching ONNX graph constants)
        mel_log = torch.log(mel.clamp(min=1e-10))
        mel_db = mel_log * (10.0 / 2.3025851249694824)
        mel_max = mel_db.max()
        mel_db = mel_db.clamp(min=mel_max - 80.0)

        spec = mel_db.numpy()
        # openwakeword transform: spec / 10 + 2
        return spec / 10.0 + 2.0

    @torch.no_grad()
    def _compute_embedding(self, mel_window: np.ndarray) -> np.ndarray:
        """Embedding CNN: (76, 32) → (96,)."""
        x = torch.from_numpy(mel_window).float().unsqueeze(0).unsqueeze(-1)  # (1,76,32,1)
        return self._embed(x).squeeze().numpy()

    def _compute_blank_features(self) -> np.ndarray:
        """Initialize feature buffer with blank audio embeddings (mirrors OWW init)."""
        blank_audio = np.zeros(160000, dtype=np.int16).tolist()
        spec = self._get_melspectrogram(blank_audio)
        windows = []
        for i in range(0, spec.shape[0], 8):
            w = spec[i:i + 76]
            if w.shape[0] == 76:
                windows.append(w)
        if not windows:
            return np.zeros((1, 96), dtype=np.float32)
        batch = np.array(windows, dtype=np.float32)
        embeddings = []
        for w in batch:
            embeddings.append(self._compute_embedding(w))
        return np.vstack(embeddings)

    # ── Streaming interface ─────────────────────────────────────────

    def predict(self, audio_chunk: np.ndarray) -> float:
        """Process one audio chunk and return the wake word score.

        Args:
            audio_chunk: Flat int16 PCM array (recommend 1280 samples = 80ms).

        Returns:
            Wake word confidence score in [0, 1].
        """
        if len(audio_chunk) < 400:
            return 0.0

        # Buffer raw audio
        self._raw_buffer.extend(audio_chunk.tolist())
        self._accumulated_samples += len(audio_chunk)

        # Only process every ~80ms (1280 samples)
        if self._accumulated_samples < 1280:
            return 0.0

        n_chunks = self._accumulated_samples // 1280

        # ── Update mel spectrogram (matches _streaming_melspectrogram) ──
        raw_list = list(self._raw_buffer)
        lookback = self._accumulated_samples + 160 * 3
        recent = raw_list[-lookback:]
        new_mel = self._get_melspectrogram(recent)
        self._mel_buffer = np.vstack([self._mel_buffer, new_mel])
        if self._mel_buffer.shape[0] > self._mel_max_len:
            self._mel_buffer = self._mel_buffer[-self._mel_max_len:]

        # ── Update embedding buffer (matches _streaming_features) ──
        for i in np.arange(n_chunks - 1, -1, -1):
            ndx = -8 * int(i)
            ndx = ndx if ndx != 0 else len(self._mel_buffer)
            start = -76 + ndx
            end = ndx
            window = self._mel_buffer[start:end].astype(np.float32)
            if window.shape[0] == 76:
                emb = self._compute_embedding(window)
                self._feature_buffer = np.vstack([self._feature_buffer, emb])

        if self._feature_buffer.shape[0] > self._feature_max_len:
            self._feature_buffer = self._feature_buffer[-self._feature_max_len:]

        self._accumulated_samples = 0

        # ── Wakeword classification ─────────────────────────────────
        features = self._feature_buffer[-16:]  # (16, 96)
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)  # (1,16,96)
        with torch.no_grad():
            score = self._ww(x)
        return float(score.squeeze())

    def reset(self) -> None:
        """Clear all internal buffers (call after wake word triggers)."""
        self._raw_buffer.clear()
        self._mel_buffer = np.ones((76, 32), dtype=np.float32)
        self._feature_buffer = self._compute_blank_features()
        self._accumulated_samples = 0
