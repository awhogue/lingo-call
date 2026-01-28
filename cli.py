#!/usr/bin/env python3
"""Command-line interface for Lingo-Call."""

import argparse
import logging
import sys

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
  # Basic usage with default model (Llama-3.2-3B)
  python cli.py

  # Text mode (type input, skip STT)
  python cli.py --text-mode

  # Use a different HuggingFace model
  python cli.py --llm-model Qwen/Qwen2.5-3B-Instruct

  # Spanish conversation with voice cloning
  python cli.py --language es --voice speaker.wav

  # List supported languages
  python cli.py --list-languages

  # Test all components
  python cli.py --test

  # Use VAD mode instead of push-to-talk
  python cli.py --mode vad
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
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model name (default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    llm_group.add_argument(
        "--llm-device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device for LLM inference (default: auto)",
    )
    llm_group.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (uses more memory)",
    )
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    llm_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace (required for some models)",
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
        choices=["turbo", "multilingual", "xtts"],
        help="TTS model: turbo, multilingual (Chatterbox), or xtts (Coqui XTTS-v2)",
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

    # Audio/Input options
    audio_group = parser.add_argument_group("Audio/Input")
    audio_group.add_argument(
        "--text-mode",
        action="store_true",
        help="Text input mode: type instead of speak (skips STT)",
    )
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
    config.llm.model_name = args.llm_model
    config.llm.device = args.llm_device
    config.llm.use_4bit = not args.no_4bit
    config.llm.temperature = args.temperature
    config.llm.trust_remote_code = args.trust_remote_code

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

    # Create pipeline
    text_mode = args.text_mode
    mode_str = "text mode" if text_mode else "voice mode"
    print(f"Initializing Lingo-Call for {config.display_language} ({mode_str})...")
    pipeline = ConversationPipeline(config, text_mode=text_mode)

    try:
        # Handle --test
        if args.test:
            print("\nTesting components...\n")
            results = pipeline.test_components()
            print("\nResults:")
            for component, success in results.items():
                if success is None:
                    status = "SKIPPED"
                elif success:
                    status = "PASS"
                else:
                    status = "FAIL"
                print(f"  {component}: {status}")
            # Only fail if any component explicitly failed (not skipped)
            failed = any(v is False for v in results.values())
            return 1 if failed else 0

        # Define callback for response display
        def on_response(text: str) -> None:
            print(f"\nAssistant: {text}")

        if text_mode:
            # Text input mode
            pipeline.start_text_interactive(on_response=on_response)
        else:
            # Voice input mode
            def on_transcription(text: str) -> None:
                print(f"\nYou: {text}")

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
