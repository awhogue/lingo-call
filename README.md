# Lingo-Call

> **Work in Progress** - This project is under active development.

Multilingual voice conversation application using STT, LLM, and TTS.

## Overview

Lingo-Call enables natural voice conversations in 23 languages by combining:

- **STT**: [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) (Facebook Research) - supports 1600+ languages
- **LLM**: [HuggingFace Transformers](https://huggingface.co/docs/transformers) - runs Llama, Qwen, Mistral, and other models with MPS/CUDA support
- **TTS**: [Chatterbox](https://github.com/resemble-ai/chatterbox) - voice cloning in 23 languages

## Supported Languages

Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

## Installation

### Python Setup (pyenv recommended)

```bash
# Install pyenv if you don't have it
brew install pyenv

# Add to your shell (add these to ~/.zshrc or ~/.bashrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart shell, then install Python 3.11 (best compatibility)
pyenv install 3.11.11
pyenv local 3.11.11

# Create virtual environment
python -m venv venv
source venv/bin/activate
```

### Package Installation

```bash
# Install build dependencies first
pip install --upgrade pip setuptools wheel numpy

# Install TTS (has tricky dependencies)
pip install --no-build-isolation chatterbox-tts

# Install remaining components
pip install -e ".[llm,stt]"
pip install -e .
```

### Requirements

- **Python 3.11** recommended (3.10-3.11 supported; 3.12+ has dependency issues with chatterbox-tts)
- HuggingFace account with access to gated models (e.g., Llama)
- macOS with Apple Silicon (MPS) or NVIDIA GPU (CUDA) recommended

## Usage

```bash
# List supported languages
python cli.py --list-languages

# Basic conversation with default model (Llama-3.2-3B-Instruct)
python cli.py

# Use a different HuggingFace model
python cli.py --llm-model Qwen/Qwen2.5-3B-Instruct

# Spanish conversation with voice cloning
python cli.py \
    --language es \
    --voice reference_speaker.wav

# Test all components
python cli.py --test

# Use voice activity detection instead of push-to-talk
python cli.py --mode vad

# Disable 4-bit quantization (uses more memory)
python cli.py --no-4bit
```

## Recording Modes

- **push_to_talk** (default): Hold SPACE to record, release to process
- **vad**: Automatic voice activity detection - stops after silence
- **fixed**: Records for a fixed duration

## Project Structure

```
src/lingo_call/
├── stt/           # Speech-to-text (omnilingual-asr)
├── llm/           # Language model (HuggingFace transformers)
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
| LLM (3B 4-bit) | ~2-3GB | MPS/CUDA |
| LLM (7B 4-bit) | ~4-5GB | MPS/CUDA |
| TTS | ~2GB | MPS |

**Recommended models:**
- `meta-llama/Llama-3.2-3B-Instruct` (default, requires HF access)
- `Qwen/Qwen2.5-3B-Instruct` (no access required)
- `microsoft/Phi-3-mini-4k-instruct`

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
