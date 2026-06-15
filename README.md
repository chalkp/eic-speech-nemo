# eic-speech-nemo

NeMo-based speech pipeline with Wake Word detection, VAD, and ASR.
It requires very bare minimum dependencies, and I like it this way.
Completely useless outside our settings.

## dependencies

- **PortAudio** C lib (sudo apt-get install portaudio19-dev)
- **torch** >= 2.7
- **numpy**
- **sounddevice** Clean API, callback-based, low-latency, better than pyaudio from my experience with less clunkier API

## download weights

**Put your weights under ```weights/``` directory.**

- [nemotron_asr.pt](https://huggingface.co/chalkp/eic-speech-nemo/resolve/main/nemotron_asr.pt)
- [silero_vad.pt](https://huggingface.co/chalkp/eic-speech-nemo/resolve/main/silero_vad.pt)
- [wakeword_mel.pt](https://huggingface.co/chalkp/eic-speech-nemo/resolve/main/wakeword_mel.pt)
- [wakeword_embed.pt](https://huggingface.co/chalkp/eic-speech-nemo/resolve/main/wakeword_embed.pt)
- [wakeword_classifier.pt](https://huggingface.co/chalkp/eic-speech-nemo/resolve/main/wakeword_classifier.pt)

## demo

```bash
bash setup.sh
source .venv/bin/activate
eic-speech-nemo
```

## Installation

(Install PortAudio beforehands)

```bash
# sudo apt-get install portaudio19-dev
pip install -e .
```

## Usage

```python
from eic_speech_nemo import ASRServer, State
from eic_speech_nemo.server import ASRConfig

server = ASRServer(config=ASRConfig(model_dir="./models"))
server.start()
server.run_forever()
```
