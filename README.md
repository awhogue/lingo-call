# Lingo-Call

> **Work in Progress** - This project is under active development.

Multilingual voice conversation application using STT, LLM, and TTS.

## Overview

Lingo-Call enables natural voice conversations in 23 languages by combining:

- **STT**: [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) (Facebook Research) - supports 1600+ languages
- **LLM**: [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - runs Llama/Qwen models with MPS/CUDA support
- **TTS**: [Chatterbox](https://github.com/resemble-ai/chatterbox) - voice cloning in 23 languages

## Supported Languages

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

## Installation

```bash
# Install base package (audio recording/playback only)
pip install -e .

# Install with all ML components
pip install -e ".[full]"

# Or install individual components
pip install -e ".[stt]"   # omnilingual-asr
pip install -e ".[llm]"   # llama-cpp-python
pip install -e ".[tts]"   # chatterbox-tts
```

### Requirements

- **Python 3.10-3.13** (Python 3.14 not yet supported due to dependency constraints)
- A GGUF model file (Llama 3.x or Qwen 2.x recommended)
- macOS with Apple Silicon (MPS) or NVIDIA GPU (CUDA) recommended

## Usage

```bash
# List supported languages
python cli.py --list-languages

# Basic conversation in English
python cli.py --llm-model ./models/llama-3.1-8b-q4.gguf

# Spanish conversation with voice cloning
python cli.py \
    --language es \
    --voice reference_speaker.wav \
    --llm-model ./models/llama-3.1-8b-q4.gguf

# Test all components
python cli.py --test --llm-model ./models/llama.gguf

# Use voice activity detection instead of push-to-talk
python cli.py --mode vad --llm-model ./models/llama.gguf
```

## Recording Modes

- **push_to_talk** (default): Hold SPACE to record, release to process
- **vad**: Automatic voice activity detection - stops after silence
- **fixed**: Records for a fixed duration

## Project Structure

```
src/lingo_call/
├── stt/           # Speech-to-text (omnilingual-asr)
├── llm/           # Language model (llama-cpp-python)
├── tts/           # Text-to-speech (Chatterbox)
├── audio/         # Recording and playback
├── config.py      # Configuration
├── conversation.py # History management
└── pipeline.py    # Main orchestrator
```

## Hardware Recommendations

Optimized for MacBook Pro with Apple Silicon (24GB+ RAM):

| Component | Memory | Device |
|-----------|--------|--------|
| STT (CTC 1B) | ~4GB | MPS/CPU |
| LLM (7B Q4) | ~4-5GB | MPS |
| TTS | ~2GB | MPS |

## Configuration

Save/load configuration via YAML:

```bash
# Save current config
python cli.py --language es --save-config config.yaml

# Load config
python cli.py --config config.yaml
```

## License

MIT
