#!/usr/bin/env python3
"""Command-line interface for Lingo-Call."""

import argparse
import logging
import sys
from pathlib import Path

from lingo_call.config import (
    AppConfig,
    AudioConfig,
    ConversationConfig,
    LLMConfig,
    STTConfig,
    TTSConfig,
    SUPPORTED_LANGUAGES,
)
from lingo_call.pipeline import ConversationPipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def list_languages() -> None:
    """Print supported languages."""
    print("\nSupported Languages:")
    print("-" * 50)
    for code, (stt_lang, tts_lang, display_name) in sorted(SUPPORTED_LANGUAGES.items()):
        print(f"  {code:4s} - {display_name:12s} (STT: {stt_lang}, TTS: {tts_lang})")
    print()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Lingo-Call: Multilingual voice conversation app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with English
  python cli.py --llm-model ./models/llama-3.1-8b-q4.gguf

  # Spanish conversation with voice cloning
  python cli.py --language es --voice speaker.wav --llm-model ./models/llama.gguf

  # List supported languages
  python cli.py --list-languages

  # Test all components
  python cli.py --test --llm-model ./models/llama.gguf

  # Use VAD mode instead of push-to-talk
  python cli.py --mode vad --llm-model ./models/llama.gguf
""",
    )

    # Language options
    lang_group = parser.add_argument_group("Language")
    lang_group.add_argument(
        "-l", "--language",
        type=str,
        default="en",
        help="Language code (e.g., 'en', 'es', 'fr'). Use --list-languages to see all.",
    )
    lang_group.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported languages and exit",
    )

    # STT options
    stt_group = parser.add_argument_group("Speech-to-Text")
    stt_group.add_argument(
        "--stt-model",
        type=str,
        default="omniASR_CTC_1B_v2",
        choices=[
            "omniASR_CTC_300M_v2",
            "omniASR_CTC_1B_v2",
            "omniASR_CTC_7B_v2",
            "omniASR_LLM_3B_v2",
        ],
        help="STT model card (default: omniASR_CTC_1B_v2)",
    )
    stt_group.add_argument(
        "--stt-device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device for STT inference (default: mps)",
    )

    # LLM options
    llm_group = parser.add_argument_group("Language Model")
    llm_group.add_argument(
        "--llm-model",
        type=str,
        required=False,
        help="Path to GGUF model file (required for conversation)",
    )
    llm_group.add_argument(
        "--llm-family",
        type=str,
        default="llama",
        choices=["llama", "qwen"],
        help="Model family for chat template (default: llama)",
    )
    llm_group.add_argument(
        "--llm-ctx",
        type=int,
        default=4096,
        help="Context window size (default: 4096)",
    )
    llm_group.add_argument(
        "--llm-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, default: -1)",
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )

    # TTS options
    tts_group = parser.add_argument_group("Text-to-Speech")
    tts_group.add_argument(
        "-v", "--voice",
        type=str,
        default="",
        help="Reference audio file for voice cloning",
    )
    tts_group.add_argument(
        "--tts-model",
        type=str,
        default="multilingual",
        choices=["turbo", "multilingual"],
        help="TTS model type (default: multilingual)",
    )
    tts_group.add_argument(
        "--tts-device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device for TTS inference (default: mps)",
    )
    tts_group.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="TTS CFG weight (default: 0.5)",
    )
    tts_group.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="TTS exaggeration (default: 0.5)",
    )

    # Audio options
    audio_group = parser.add_argument_group("Audio")
    audio_group.add_argument(
        "--mode",
        type=str,
        default="push_to_talk",
        choices=["push_to_talk", "vad", "fixed"],
        help="Recording mode (default: push_to_talk)",
    )
    audio_group.add_argument(
        "--ptt-key",
        type=str,
        default="space",
        choices=["space", "ctrl", "shift", "alt"],
        help="Push-to-talk key (default: space)",
    )

    # General options
    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--config",
        type=str,
        help="Load configuration from YAML file",
    )
    general_group.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to YAML file and exit",
    )
    general_group.add_argument(
        "--test",
        action="store_true",
        help="Test all components and exit",
    )
    general_group.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose logging",
    )
    general_group.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt (use {language} for language name)",
    )
    general_group.add_argument(
        "--max-history",
        type=int,
        default=20,
        help="Maximum conversation history (default: 20)",
    )

    return parser


def build_config(args: argparse.Namespace) -> AppConfig:
    """Build configuration from command-line arguments."""
    # Load from file if specified
    if args.config:
        config = AppConfig.from_yaml(args.config)
    else:
        config = AppConfig(language=args.language)

    # Override with CLI arguments
    config.language = args.language

    # STT config
    config.stt.model_card = args.stt_model
    config.stt.device = args.stt_device

    # LLM config
    if args.llm_model:
        config.llm.model_path = args.llm_model
    config.llm.model_family = args.llm_family
    config.llm.n_ctx = args.llm_ctx
    config.llm.n_gpu_layers = args.llm_gpu_layers
    config.llm.temperature = args.temperature

    # TTS config
    config.tts.model_type = args.tts_model
    config.tts.device = args.tts_device
    config.tts.voice_file = args.voice
    config.tts.cfg_weight = args.cfg_weight
    config.tts.exaggeration = args.exaggeration

    # Audio config
    config.audio.push_to_talk_key = args.ptt_key

    # Conversation config
    config.conversation.max_history = args.max_history
    if args.system_prompt:
        config.conversation.system_prompt = args.system_prompt

    return config


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle --list-languages
    if args.list_languages:
        list_languages()
        return 0

    # Setup logging
    setup_logging(args.verbose)

    # Build configuration
    try:
        config = build_config(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Handle --save-config
    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return 0

    # Check LLM model is provided for conversation/test
    if not config.llm.model_path:
        print("Error: --llm-model is required", file=sys.stderr)
        print("Provide a path to a GGUF model file.", file=sys.stderr)
        return 1

    if not Path(config.llm.model_path).exists():
        print(f"Error: LLM model not found: {config.llm.model_path}", file=sys.stderr)
        return 1

    # Create pipeline
    print(f"Initializing Lingo-Call for {config.display_language}...")
    pipeline = ConversationPipeline(config)

    try:
        # Handle --test
        if args.test:
            print("\nTesting components...\n")
            results = pipeline.test_components()
            print("\nResults:")
            for component, success in results.items():
                status = "PASS" if success else "FAIL"
                print(f"  {component}: {status}")
            return 0 if all(results.values()) else 1

        # Define callbacks for transcription and response display
        def on_transcription(text: str) -> None:
            print(f"\nYou: {text}")

        def on_response(text: str) -> None:
            print(f"\nAssistant: {text}")

        # Start interactive conversation
        pipeline.start_interactive(
            mode=args.mode,
            on_transcription=on_transcription,
            on_response=on_response,
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pipeline.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
