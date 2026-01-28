"""Main conversation pipeline orchestrator."""

import logging
import signal
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from lingo_call.audio.player import AudioPlayer
from lingo_call.audio.recorder import AudioRecorder
from lingo_call.config import AppConfig
from lingo_call.conversation import ConversationManager
from lingo_call.llm.base import LLMProvider
from lingo_call.llm.transformers_llm import TransformersLLM
from lingo_call.stt.base import STTProvider
from lingo_call.stt.omnilingual import OmnilingualSTT
from lingo_call.tts.base import TTSProvider
from lingo_call.tts.chatterbox import ChatterboxTTS
from lingo_call.tts.xtts import XTTSv2TTS

logger = logging.getLogger(__name__)


class ConversationPipeline:
    """Main orchestrator for voice conversations.

    Coordinates STT, LLM, and TTS providers to enable
    natural voice conversations in multiple languages.
    """

    def __init__(
        self,
        config: AppConfig,
        stt: STTProvider | None = None,
        llm: LLMProvider | None = None,
        tts: TTSProvider | None = None,
        text_mode: bool = False,
    ) -> None:
        """Initialize the conversation pipeline.

        Args:
            config: Application configuration
            stt: Optional custom STT provider (creates OmnilingualSTT if None)
            llm: Optional custom LLM provider (creates TransformersLLM if None)
            tts: Optional custom TTS provider (creates ChatterboxTTS if None)
            text_mode: If True, skip STT initialization (user types input)
        """
        self.config = config
        self.text_mode = text_mode

        # Initialize providers
        if text_mode:
            self.stt = None
            self.recorder = None
        else:
            self.stt = stt or OmnilingualSTT(config.stt, config.stt_language)
            self.recorder = AudioRecorder(config.audio)

        # Initialize LLM eagerly (load model now, not on first use)
        self.llm = llm or TransformersLLM(config.llm, lazy=False)
        if tts is not None:
            self.tts = tts
        elif config.tts.model_type == "xtts":
            self.tts = XTTSv2TTS(config.tts, config.tts_language)
        else:
            self.tts = ChatterboxTTS(config.tts, config.tts_language)

        # Initialize audio player (needed for TTS output)
        self.player = AudioPlayer()

        # Initialize conversation manager
        self.conversation = ConversationManager(max_history=config.conversation.max_history)

        # Running state
        self._running = False

    def run_turn(
        self,
        audio: np.ndarray,
        on_transcription: Callable[[str], None] | None = None,
        on_response: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Process a single conversation turn.

        Args:
            audio: Input audio from user
            on_transcription: Callback when transcription is ready
            on_response: Callback when LLM response is ready

        Returns:
            Synthesized audio response
        """
        # Step 1: Speech-to-text
        logger.info("Transcribing audio...")
        transcription = self.stt.transcribe(audio, self.config.audio.sample_rate)

        if not transcription.strip():
            logger.warning("Empty transcription")
            return np.array([], dtype=np.float32)

        logger.info(f"Transcription: {transcription}")
        if on_transcription:
            on_transcription(transcription)

        # Add user message to conversation
        self.conversation.add_user_message(transcription)

        # Step 2: Generate LLM response
        logger.info("Generating response...")
        response = self.llm.generate(
            messages=self.conversation.get_messages(),
            system_prompt=self.config.system_prompt,
        )

        if not response.strip():
            logger.warning("Empty LLM response")
            return np.array([], dtype=np.float32)

        logger.info(f"Response: {response}")
        if on_response:
            on_response(response)

        # Add assistant message to conversation
        self.conversation.add_assistant_message(response)

        # Step 3: Text-to-speech
        logger.info("Synthesizing speech...")
        audio_response, self._last_sample_rate = self.tts.synthesize(response)

        # Optionally save raw TTS audio for debugging
        self._save_tts_debug(audio_response, self._last_sample_rate)

        return audio_response

    def run_text_turn(
        self,
        text: str,
        on_response: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Process a single conversation turn from text input.

        Args:
            text: User's text input
            on_response: Callback when LLM response is ready

        Returns:
            Synthesized audio response
        """
        if not text.strip():
            logger.warning("Empty input")
            return np.array([], dtype=np.float32)

        # Add user message to conversation
        self.conversation.add_user_message(text)

        # Generate LLM response
        logger.info("Generating response...")
        response = self.llm.generate(
            messages=self.conversation.get_messages(),
            system_prompt=self.config.system_prompt,
        )

        if not response.strip():
            logger.warning("Empty LLM response")
            return np.array([], dtype=np.float32)

        logger.info(f"Response: {response}")
        if on_response:
            on_response(response)

        # Add assistant message to conversation
        self.conversation.add_assistant_message(response)

        # Text-to-speech
        logger.info("Synthesizing speech...")
        audio_response, self._last_sample_rate = self.tts.synthesize(response)

        # Optionally save raw TTS audio for debugging
        self._save_tts_debug(audio_response, self._last_sample_rate)

        return audio_response

    def start_interactive(
        self,
        mode: str = "push_to_talk",
        on_transcription: Callable[[str], None] | None = None,
        on_response: Callable[[str], None] | None = None,
    ) -> None:
        """Start an interactive conversation loop.

        Args:
            mode: Recording mode - "push_to_talk", "vad", or "fixed"
            on_transcription: Callback when transcription is ready
            on_response: Callback when LLM response is ready
        """
        self._running = True

        # Set up signal handler for graceful exit
        def signal_handler(sig, frame):
            print("\nExiting...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)

        print(f"\n{'='*50}")
        print(f"Lingo-Call - Voice Conversation in {self.config.display_language}")
        print(f"{'='*50}")
        print(f"STT Model: {self.config.stt.model_card}")
        print(f"LLM Model: {self.config.llm.model_name}")
        print(f"TTS Model: {self.config.tts.model_type}")
        print(f"{'='*50}")
        print("Press Ctrl+C to exit\n")

        while self._running:
            try:
                # Record audio based on mode
                if mode == "push_to_talk":
                    audio = self.recorder.record_push_to_talk(
                        on_start=lambda: print("Recording..."),
                        on_stop=lambda: print("Processing..."),
                    )
                elif mode == "vad":
                    print("Listening... (speak now)")
                    audio = self.recorder.record_with_vad()
                else:  # fixed duration
                    print("Recording...")
                    audio = self.recorder.record_blocking(duration=5.0)
                    print("Processing...")

                if len(audio) == 0:
                    continue

                # Process the turn
                response_audio = self.run_turn(
                    audio,
                    on_transcription=on_transcription,
                    on_response=on_response,
                )

                if len(response_audio) > 0:
                    # Play the response using the actual TTS sample rate if available
                    sample_rate = getattr(
                        self, "_last_sample_rate", self.config.tts.sample_rate
                    )
                    self.player.play(response_audio, sample_rate)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in conversation turn: {e}")
                print(f"Error: {e}")
                continue

        print("\nConversation ended.")
        print(f"Total turns: {self.conversation.turn_count}")

    def start_text_interactive(
        self,
        on_response: Callable[[str], None] | None = None,
    ) -> None:
        """Start an interactive text-based conversation loop.

        User types input, LLM generates response, TTS plays audio.

        Args:
            on_response: Callback when LLM response is ready
        """
        self._running = True

        # No signal handler needed - input() raises KeyboardInterrupt on Ctrl+C

        print(f"\n{'='*50}")
        print(f"Lingo-Call - Text Conversation in {self.config.display_language}")
        print(f"{'='*50}")
        print(f"LLM Model: {self.config.llm.model_name}")
        print(f"TTS Model: {self.config.tts.model_type}")
        print(f"{'='*50}")
        print("Type your message and press Enter. Ctrl+C to exit.\n")

        while self._running:
            try:
                # Get text input from user
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ("exit", "quit", "q"):
                    break

                print("Processing...")

                # Process the turn
                response_audio = self.run_text_turn(
                    user_input,
                    on_response=on_response,
                )

                if len(response_audio) > 0:
                    sample_rate = getattr(self, "_last_sample_rate", self.config.tts.sample_rate)
                    self.player.play(response_audio, sample_rate)

            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in conversation turn: {e}")
                print(f"Error: {e}")
                continue

        print("Conversation ended.")
        print(f"Total turns: {self.conversation.turn_count}")

    def _save_tts_debug(self, audio: np.ndarray, sample_rate: int) -> None:
        """Save raw TTS audio to disk for debugging if configured."""
        debug_dir = getattr(self.config.tts, "debug_save_dir", "")

        if not debug_dir or len(audio) == 0:
            return

        try:
            out_dir = Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Use turn count to generate a simple, monotonic filename
            turn = getattr(self.conversation, "turn_count", 0)
            file_path = out_dir / f"tts_turn_{turn:04d}.wav"

            self.player.save(audio, sample_rate, str(file_path))
            logger.info(f"Saved TTS debug audio to {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save TTS debug audio: {e}")

    def test_components(self) -> dict[str, bool]:
        """Test each component of the pipeline.

        Returns:
            Dictionary of component names to success status
        """
        results = {}

        # Test STT (skip in text mode)
        if self.stt is not None:
            print("Testing STT...")
            try:
                # Create a short silent audio for testing
                test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                self.stt.transcribe(test_audio, 16000)
                results["stt"] = True
                print("  STT: OK")
            except Exception as e:
                results["stt"] = False
                print(f"  STT: FAILED - {e}")
        else:
            print("Testing STT... SKIPPED (text mode)")
            results["stt"] = None

        # Test LLM
        print("Testing LLM...")
        try:
            response = self.llm.generate(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="Respond briefly.",
            )
            results["llm"] = bool(response)
            print(f"  LLM: OK - Response: {response[:50]}...")
        except Exception as e:
            results["llm"] = False
            print(f"  LLM: FAILED - {e}")

        # Test TTS
        print("Testing TTS...")
        try:
            audio, sr = self.tts.synthesize("Hello, this is a test.")
            results["tts"] = len(audio) > 0
            print(f"  TTS: OK - Generated {len(audio)/sr:.2f}s of audio")
        except Exception as e:
            results["tts"] = False
            print(f"  TTS: FAILED - {e}")

        # Test audio recording (skip in text mode)
        if self.recorder is not None:
            print("Testing audio recording...")
            try:
                audio = self.recorder.record_blocking(duration=0.5)
                results["audio_recording"] = len(audio) > 0
                print(f"  Audio recording: OK - Recorded {len(audio)} samples")
            except Exception as e:
                results["audio_recording"] = False
                print(f"  Audio recording: FAILED - {e}")
        else:
            print("Testing audio recording... SKIPPED (text mode)")
            results["audio_recording"] = None

        # Test audio playback
        print("Testing audio playback...")
        try:
            test_audio = np.sin(
                2 * np.pi * 440 * np.arange(24000) / 24000
            ).astype(np.float32)
            self.player.play(test_audio, 24000, blocking=True)
            results["audio_playback"] = True
            print("  Audio playback: OK")
        except Exception as e:
            results["audio_playback"] = False
            print(f"  Audio playback: FAILED - {e}")

        return results

    def close(self) -> None:
        """Clean up all resources."""
        if self.stt is not None:
            self.stt.close()
        self.llm.close()
        self.tts.close()
        if self.recorder is not None:
            self.recorder.close()
        logger.info("Pipeline closed")
