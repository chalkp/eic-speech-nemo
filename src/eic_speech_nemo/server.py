from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# Enable TF32 for faster matmuls on Ampere+ GPUs (used by torch.compile)
torch.set_float32_matmul_precision("high")

from eic_speech_nemo.audio import AudioSource, MicrophoneSource
from eic_speech_nemo.state_machine import Event, State, StateMachine, TransitionResult

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    """Server configuration — all tunables in one place."""

    # Audio
    sample_rate: int = 16000
    chunk_size: int = 512 # Silero VAD

    # Wake word
    wakeword_model: str = "wakeword.pt"
    wakeword_threshold: float = 0.5

    # VAD
    silence_timeout: float = 1.2
    min_utterance_sec: float = 0.8

    # Compute
    precision: str = "fp32"  # "fp32" / "bf16" / "fp8"
    device: str = "auto"
    torch_compile: bool = True

    # Paths
    model_dir: str = "weights"
    vad_model: str = "silero_vad.pt"
    asr_model: str = "nemotron_asr.pt"


@dataclass
class TranscriptionResult:
    """Emitted when ASR finishes processing an utterance."""
    text: str
    audio_duration: float # s
    processing_time: float # s
    rtf: float # real time factor
    timestamp: float = field(default_factory=time.time)


TranscriptionCallback = Callable[[TranscriptionResult], None]


class ASRServer:
    def __init__(self, config: ASRConfig | None = None, **kwargs):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = ASRConfig(**kwargs)

        if self.cfg.device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(self.cfg.device)

        self._sm = StateMachine()

        self._ww = None # OpenWakeWord
        self._vad = None # Silero VAD
        self._asr = None # NemotronASR

        self._audio_source: AudioSource | None = None

        self._utterance: list[np.ndarray] = []
        self._silence_frames: int = 0

        self._transcription_cbs: list[TranscriptionCallback] = []
        self._cb_lock = threading.Lock()

        self._running = False
        self._loop_thread: threading.Thread | None = None
        self._callbacks_registered = False

    @property
    def state(self) -> State:
        """Current pipeline state."""
        return self._sm.state

    @property
    def device(self) -> torch.device:
        """Compute device in use."""
        return self._device

    @property
    def is_running(self) -> bool:
        return self._running

    def subscribe_audio_source(self, source: AudioSource) -> None:
        """Attach a custom audio source (replaces the default microphone).

        Must be called before ``start()``.

        Args:
            source: Any object implementing the ``AudioSource`` protocol.
        """
        if self._running:
            raise RuntimeError("Cannot change audio source while running")
        self._audio_source = source

    def on_transcription(self, cb: TranscriptionCallback) -> None:
        with self._cb_lock:
            self._transcription_cbs.append(cb)

    def remove_transcription_callback(self, cb: TranscriptionCallback) -> None:
        """Remove a previously registered transcription callback."""
        with self._cb_lock:
            self._transcription_cbs.remove(cb)

    def _resolve_path(self, filename: str) -> str:
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.cfg.model_dir, filename)

    def _load_wakeword(self) -> None:
        try:
            from eic_speech_nemo.wakeword import WakeWordDetector
            self._ww = WakeWordDetector(
                model_dir=self.cfg.model_dir,
                threshold=self.cfg.wakeword_threshold,
            )
            logger.info("Wake word loaded from: %s", self.cfg.model_dir)
        except Exception as e:
            logger.warning("Wake word unavailable: %s", e)

    def _load_vad(self) -> None:
        path = self._resolve_path(self.cfg.vad_model)
        try:
            self._vad = torch.jit.load(path)
            self._vad.eval()
            if self.cfg.torch_compile and self._device.type == "cuda":
                try:
                    self._vad = torch.compile(self._vad, mode="reduce-overhead")
                    logger.info("VAD compiled (reduce-overhead)")
                except Exception as e:
                    logger.warning("VAD torch.compile failed (using eager): %s", e)
            logger.info("VAD loaded: %s", path)
        except Exception as e:
            logger.warning("VAD unavailable: %s", e)

    def _load_asr(self) -> None:
        path = self._resolve_path(self.cfg.asr_model)
        try:
            from eic_speech_nemo.models.nemo import NemotronASR
            self._asr = NemotronASR.load_from_pt(
                path, device=str(self._device), precision=self.cfg.precision,
            )
            self._asr.eval()
            if self.cfg.torch_compile:
                self._compile_asr()
            logger.info("ASR loaded: %s (%s)", path, self._device)
        except Exception as e:
            logger.warning("ASR unavailable: %s", e)

    def _compile_asr(self) -> None:
        mode = "reduce-overhead" if self._device.type == "cuda" else "default"
        compiled = []
        try:
            self._asr.encoder = torch.compile(self._asr.encoder, mode=mode, dynamic=True)
            compiled.append("encoder")
        except Exception as e:
            logger.warning("torch.compile encoder failed: %s", e)
        try:
            self._asr.decoder = torch.compile(self._asr.decoder, mode=mode, dynamic=True)
            compiled.append("decoder")
        except Exception as e:
            logger.warning("torch.compile decoder failed: %s", e)
        try:
            self._asr.joint = torch.compile(self._asr.joint, mode=mode, dynamic=True)
            compiled.append("joint")
        except Exception as e:
            logger.warning("torch.compile joint failed: %s", e)
        try:
            self._asr.preprocessor = torch.compile(self._asr.preprocessor, mode=mode, dynamic=True)
            compiled.append("preprocessor")
        except Exception as e:
            logger.warning("torch.compile preprocessor failed: %s", e)
        if compiled:
            logger.info("ASR compiled (%s): %s", mode, ", ".join(compiled))

    def load_models(self) -> None:
        """Load all models.  Called automatically by ``start()``."""
        self._load_wakeword()
        self._load_vad()
        self._load_asr()

    def unload_models(self) -> None:
        """Unload all models and free GPU memory."""
        self._ww = None
        self._vad = None
        if self._asr is not None:
            self._asr.cpu()
            self._asr = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Models unloaded")

    def _vad_prob(self, chunk: np.ndarray) -> float:
        if self._vad is None:
            return 0.0
        t = torch.from_numpy(chunk.flatten()).float()
        return self._vad(t, self.cfg.sample_rate).item()

    def _reset_vad(self) -> None:
        if self._vad is not None:
            self._vad.reset_states()

    def _check_wakeword(self, chunk: np.ndarray) -> bool:
        if self._ww is None:
            return False
        pcm16 = (chunk * 32767).astype(np.int16).flatten()
        score = self._ww.predict(pcm16)
        return score > self.cfg.wakeword_threshold

    def _reset_wakeword(self) -> None:
        if self._ww is not None:
            self._ww.reset()

    def _transcribe(self, buffers: list[np.ndarray]) -> TranscriptionResult:
        audio = np.concatenate(buffers).flatten().astype(np.float32)
        duration = len(audio) / self.cfg.sample_rate

        if self._asr is None:
            return TranscriptionResult(
                text="[ASR unavailable]",
                audio_duration=duration,
                processing_time=0.0,
                rtf=0.0,
            )

        audio_t = torch.from_numpy(audio).to(self._device)
        t0 = time.perf_counter()
        with torch.no_grad():
            text = self._asr.transcribe(audio_t)
        elapsed = time.perf_counter() - t0

        return TranscriptionResult(
            text=text.strip() if text else "",
            audio_duration=duration,
            processing_time=elapsed,
            rtf=elapsed / max(duration, 0.01),
        )

    def _emit_transcription(self, result: TranscriptionResult) -> None:
        with self._cb_lock:
            for cb in self._transcription_cbs:
                try:
                    cb(result)
                except Exception:
                    logger.exception("Error in transcription callback")

    def _on_chunk_wake(self, chunk: np.ndarray) -> None:
        """WAKE state: check for wake word."""
        if self._check_wakeword(chunk):
            self._sm.handle_event(Event.WAKE_DETECTED)
            self._utterance = [chunk]
            self._silence_frames = 0
            logger.debug("Wake word detected")

    def _on_chunk_recording(self, chunk: np.ndarray) -> None:
        """RECORDING state: accumulate audio, check VAD for end of speech."""
        self._utterance.append(chunk)

        if self._vad_prob(chunk) > 0.5:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        timeout_frames = self.cfg.silence_timeout * self.cfg.sample_rate / self.cfg.chunk_size
        if self._silence_frames > timeout_frames:
            min_frames = self.cfg.min_utterance_sec * self.cfg.sample_rate / self.cfg.chunk_size
            if len(self._utterance) > min_frames:
                self._sm.handle_event(Event.SPEECH_END)
            else:
                self._sm.handle_event(Event.UTTERANCE_SHORT)
                self._utterance.clear()

    def _on_enter_processing(self, _: TransitionResult) -> None:
        """PROCESSING state: run ASR and emit result."""
        result = self._transcribe(self._utterance)
        self._utterance.clear()
        self._emit_transcription(result)
        # Transition back to WAKE
        if self._sm.state == State.PROCESSING:
            self._sm.handle_event(Event.TRANSCRIPTION)

    def _on_enter_wake(self, _: TransitionResult) -> None:
        """Reset models when entering WAKE state."""
        self._reset_wakeword()
        self._reset_vad()
        self._silence_frames = 0
        if self._audio_source is not None:
            self._audio_source.drain()

    def start(self) -> None:
        if self._running:
            return

        # load model
        if self._ww is None and self._vad is None and self._asr is None:
            self.load_models()

        # audio source
        if self._audio_source is None:
            self._audio_source = MicrophoneSource(
                sample_rate=self.cfg.sample_rate,
                chunk_size=self.cfg.chunk_size,
            )

        # state machine
        if not self._callbacks_registered:
            self._sm.on_enter(State.PROCESSING, self._on_enter_processing)
            self._sm.on_enter(State.WAKE, self._on_enter_wake)
            self._callbacks_registered = True

        # start audio
        self._audio_source.start()

        # WAKE transition
        self._sm.handle_event(Event.START)

        self._running = True
        logger.info("ASR server started (device=%s)", self._device)

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        # stop audio
        if self._audio_source is not None:
            self._audio_source.stop()

        # force to IDLE
        if self._sm.can_handle(Event.STOP):
            self._sm.handle_event(Event.STOP)
        else:
            self._sm.force_state(State.IDLE)

        # unload model
        self.unload_models()

        logger.info("ASR server stopped")

    def change_state(self, event: Event) -> TransitionResult:
        return self._sm.handle_event(event)

    def cancel(self) -> None:
        if self._sm.can_handle(Event.CANCEL):
            self._utterance.clear()
            self._sm.handle_event(Event.CANCEL)

    def enable_transcription(self) -> None:
        if self._sm.state == State.IDLE:
            self._sm.handle_event(Event.START)

    def disable_transcription(self) -> None:
        if self._sm.can_handle(Event.STOP):
            self._sm.handle_event(Event.STOP)

    # main loop
    def step(self) -> None:
        if self._audio_source is None:
            return

        chunk = self._audio_source.read(timeout=0.05)
        if chunk is None:
            return

        state = self._sm.state

        if state == State.WAKE:
            self._on_chunk_wake(chunk)
        elif state == State.RECORDING:
            self._on_chunk_recording(chunk)

    def run_forever(self) -> None:
        try:
            while self._running:
                self.step()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def run_in_background(self) -> None:
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(
            target=self.run_forever, daemon=True, name="asr-server"
        )
        self._loop_thread.start()

    def __enter__(self) -> "ASRServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
