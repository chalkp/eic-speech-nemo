"""CLI entry point for eic-speech-nemo: Wake Word + VAD + ASR pipeline."""

from __future__ import annotations

import os
os.environ["TORCH_LOGS"] = "-all"

import sys
import queue
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch

from eic_speech_nemo.audio import MicrophoneSource
from eic_speech_nemo.models.nemo import NemotronASR
from eic_speech_nemo.wakeword import WakeWordDetector


def main():
    parser = argparse.ArgumentParser(description="Voice Pipeline — Wake Word + VAD + ASR")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for speedup (takes longer to initialize)")
    parser.add_argument("--weights", default="weights",
                        help="Directory containing model weight files (default: weights)")
    args = parser.parse_args()

    print("Loading models...")

    # Load ASR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ASR device: {device}")
    asr_model = NemotronASR.load_from_pt(
        os.path.join(args.weights, "nemotron_asr.pt"), device=device
    )

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        if args.compile:
            asr_model.encoder = torch.compile(
                asr_model.encoder, mode="reduce-overhead", dynamic=True
            )
            print("Warming up ASR (compiling Triton kernels, takes ~60s)...")
            dummy_audio = torch.zeros(16000 * 3, dtype=torch.float32)
            asr_model.transcribe(dummy_audio)
            print("ASR loaded and warmed up.")
        else:
            print("ASR loaded (Compilation disabled).")
    else:
        print("ASR loaded.")

    # Load VAD
    vad_model = torch.jit.load(os.path.join(args.weights, "silero_vad.pt"))
    vad_model.eval()
    print("VAD model loaded.")

    # Load Wake Word (pure PyTorch — no openwakeword/onnxruntime needed)
    ww_model = WakeWordDetector(args.weights)
    print("Wake Word model loaded.")

    # Audio source (16kHz) — Silero VAD expects 512 samples per chunk (32ms)
    mic = MicrophoneSource(sample_rate=16000, chunk_size=512)
    audio_queue: queue.Queue = queue.Queue()
    mic.subscribe(audio_queue.put)
    mic.start()

    print("\nListening... Say 'Robot' to wake!")

    try:
        while True:
            # 1. Wake Word Loop
            chunk = audio_queue.get()

            # WakeWordDetector expects flat int16 arrays
            chunk_int16 = (chunk * 32767).astype(np.int16).flatten()
            score = ww_model.predict(chunk_int16)

            if score > 0.5:
                print(f"\n[!] WAKE WORD DETECTED (score: {score:.2f})")
                print(">>> Recording speech... (Speak now, will stop when you pause)")

                # Reset VAD state for a fresh utterance
                vad_model.reset_states()

                # 2. Record Speech until silence
                frames = []
                silence_frames = 0
                # 1.2 seconds of silence = 1.2 / 0.032 = ~38 chunks
                max_silence_frames = int(1.2 / (512 / 16000))

                while True:
                    c = audio_queue.get()
                    frames.append(c)

                    # Calculate VAD probability
                    t = torch.from_numpy(c.flatten()).float()
                    prob = vad_model(t, 16000).item()

                    if prob < 0.3:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    if silence_frames >= max_silence_frames:
                        break

                audio_data = np.concatenate(frames).flatten()
                print(">>> Processing speech...")

                # 3. Transcribe
                tensor = torch.from_numpy(audio_data.astype(np.float32))
                text = asr_model.transcribe(tensor)
                print(f"\n   YOU SAID: '{text}'\n")

                print("Listening... Say 'Robot' to wake!")
                # Flush mic buffer and reset wake word state
                mic.drain()
                while not audio_queue.empty():
                    audio_queue.get()
                ww_model.reset()

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
