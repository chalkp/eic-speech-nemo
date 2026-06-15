from __future__ import annotations

import queue
import threading
from typing import Callable, Protocol

import numpy as np
import sounddevice as sd


class AudioSource(Protocol):
    sample_rate: int
    chunk_size: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, timeout: float = 0.1) -> np.ndarray | None: ...
    def subscribe(self, cb: Callable[[np.ndarray], None]) -> None: ...
    def unsubscribe(self, cb: Callable[[np.ndarray], None]) -> None: ...
    def drain(self) -> None: ...


class MicrophoneSource:
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._subscribers: list[Callable[[np.ndarray], None]] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata.copy()
        self._queue.put(chunk)
        with self._lock:
            for cb in self._subscribers:
                cb(chunk)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def subscribe(self, cb: Callable[[np.ndarray], None]) -> None:
        with self._lock:
            if cb not in self._subscribers:
                self._subscribers.append(cb)

    def unsubscribe(self, cb: Callable[[np.ndarray], None]) -> None:
        with self._lock:
            self._subscribers.remove(cb)

    def drain(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
