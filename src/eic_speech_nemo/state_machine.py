from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable


class State(enum.Enum):
    """Pipeline states."""
    IDLE       = "IDLE"          # Server running but not listening
    WAKE       = "WAKE"          # Listening for wake word
    RECORDING  = "RECORDING"     # Recording user speech (VAD active)
    PROCESSING = "PROCESSING"    # Running ASR on captured audio
    ERROR      = "ERROR"         # Recoverable error state


class Event(enum.Enum):
    """Events that drive state transitions."""
    START           = "START"           # Begin listening for wake word
    WAKE_DETECTED   = "WAKE_DETECTED"   # Wake word triggered
    SPEECH_END      = "SPEECH_END"      # VAD detected end of speech
    UTTERANCE_SHORT = "UTTERANCE_SHORT" # Utterance too short to process
    TRANSCRIPTION   = "TRANSCRIPTION"   # ASR finished
    CANCEL          = "CANCEL"          # ESC key or explicit cancel
    STOP            = "STOP"            # Stop pipeline
    ERROR           = "ERROR"           # Error occurred


@dataclass
class TransitionResult:
    """Returned by StateMachine.handle_event()."""
    previous: State
    current: State
    event: Event
    timestamp: float = field(default_factory=time.time)


# FSM
_TRANSITIONS: dict[tuple[State, Event], State] = {
    # IDLE
    (State.IDLE, Event.START):           State.WAKE,
    (State.IDLE, Event.STOP):            State.IDLE,

    # WAKE
    (State.WAKE, Event.WAKE_DETECTED):   State.RECORDING,
    (State.WAKE, Event.CANCEL):          State.IDLE,
    (State.WAKE, Event.STOP):            State.IDLE,

    # RECORDING
    (State.RECORDING, Event.SPEECH_END):      State.PROCESSING,
    (State.RECORDING, Event.UTTERANCE_SHORT): State.WAKE,
    (State.RECORDING, Event.CANCEL):          State.WAKE,
    (State.RECORDING, Event.STOP):            State.IDLE,

    # PROCESSING
    (State.PROCESSING, Event.TRANSCRIPTION):  State.WAKE,
    (State.PROCESSING, Event.CANCEL):         State.WAKE,
    (State.PROCESSING, Event.STOP):           State.IDLE,

    # ERROR
    (State.ERROR, Event.START):          State.WAKE,
    (State.ERROR, Event.STOP):           State.IDLE,
    (State.ERROR, Event.CANCEL):         State.IDLE,
}

# Type alias for callbacks
StateCallback = Callable[[TransitionResult], None]


class StateMachine:
    def __init__(self):
        self._state = State.IDLE
        self._enter_cbs: dict[State, list[StateCallback]] = {}
        self._exit_cbs: dict[State, list[StateCallback]] = {}
        self._any_cbs: list[StateCallback] = []

    @property
    def state(self) -> State:
        return self._state

    def on_enter(self, state: State, cb: StateCallback) -> None:
        self._enter_cbs.setdefault(state, []).append(cb)

    def on_exit(self, state: State, cb: StateCallback) -> None:
        self._exit_cbs.setdefault(state, []).append(cb)

    def on_transition(self, cb: StateCallback) -> None:
        self._any_cbs.append(cb)

    def handle_event(self, event: Event) -> TransitionResult:
        key = (self._state, event)
        if key not in _TRANSITIONS:
            raise ValueError(
                f"No transition for ({self._state.value}, {event.value})"
            )

        prev = self._state
        nxt = _TRANSITIONS[key]

        result = TransitionResult(previous=prev, current=nxt, event=event)

        for cb in self._exit_cbs.get(prev, []):
            cb(result)

        self._state = nxt

        for cb in self._enter_cbs.get(nxt, []):
            cb(result)

        for cb in self._any_cbs:
            cb(result)

        return result

    def force_state(self, state: State) -> None:
        self._state = state

    def can_handle(self, event: Event) -> bool:
        return (self._state, event) in _TRANSITIONS
