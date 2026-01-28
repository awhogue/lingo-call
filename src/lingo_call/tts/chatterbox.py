"""Chatterbox TTS implementation for text-to-speech."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lingo_call.config import TTSConfig
from lingo_call.tts.base import TTSProvider

logger = logging.getLogger(__name__)


class ChatterboxTTS(TTSProvider):
    """Text-to-speech using Resemble AI's Chatterbox.

    Supports voice cloning and 23 languages:
    ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
    """

    def __init__(self, config: TTSConfig, language: str) -> None:
        """Initialize the Chatterbox TTS provider.

        Args:
            config: TTS configuration
            language: Language code (e.g., 'es' for Spanish)
        """
        self.config = config
        self._language = language
        self._voice_file = config.voice_file
        self._model: Any = None
        self._device: str | None = None

    def _load_model(self) -> None:
        """Load the TTS model lazily."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is not installed") from e

        try:
            if self.config.model_type == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS as TTS
            else:
                from chatterbox.tts import ChatterboxTTS as TTS
        except ImportError as e:
            raise ImportError(
                "chatterbox-tts is not installed. Install with: pip install chatterbox-tts"
            ) from e

        device = self.config.device

        # Determine the actual device to use
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        # Patch torch.load to handle CUDA tensors on non-CUDA machines
        original_torch_load = torch.load

        def patched_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = device
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = patched_load
            logger.info(f"Loading Chatterbox {self.config.model_type} on {device}...")
            self._model = TTS.from_pretrained(device=device)
            self._device = device
            logger.info(f"Successfully loaded TTS model on {device}")
        except Exception as e:
            if self.config.fallback_to_cpu and device != "cpu":
                logger.warning(f"Failed to load on {device}: {e}. Falling back to CPU...")
                try:
                    # Update map_location for CPU fallback
                    def cpu_load(*args, **kwargs):
                        kwargs["map_location"] = "cpu"
                        return original_torch_load(*args, **kwargs)

                    torch.load = cpu_load
                    self._model = TTS.from_pretrained(device="cpu")
                    self._device = "cpu"
                    logger.info("Successfully loaded TTS model on CPU")
                finally:
                    torch.load = original_torch_load
            else:
                raise
        finally:
            torch.load = original_torch_load

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._load_model()

        if not text.strip():
            # Return empty audio for empty text
            return np.array([], dtype=np.float32), self.config.sample_rate

        try:
            # Build generation kwargs
            kwargs: dict[str, Any] = {
                "text": text,
                "cfg_weight": self.config.cfg_weight,
                "exaggeration": self.config.exaggeration,
            }

            # Add voice cloning reference if available
            if self._voice_file and Path(self._voice_file).exists():
                kwargs["audio_prompt_path"] = self._voice_file
            elif self._voice_file:
                logger.warning(f"Voice file not found: {self._voice_file}")

            # Add language for multilingual model
            if self.config.model_type == "multilingual":
                kwargs["language_id"] = self._language

            # Generate audio
            wav = self._model.generate(**kwargs)

            # Get sample rate from model if available
            if hasattr(self._model, "sr"):
                sample_rate = self._model.sr
            elif hasattr(self._model, "sample_rate"):
                sample_rate = self._model.sample_rate
            else:
                sample_rate = self.config.sample_rate

            # Convert to numpy array
            if hasattr(wav, "numpy"):
                # PyTorch tensor
                audio = wav.squeeze().cpu().numpy()
            elif hasattr(wav, "cpu"):
                audio = wav.squeeze().cpu().numpy()
            else:
                audio = np.asarray(wav).squeeze()

            audio = audio.astype(np.float32)

            # Normalize if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            return audio, sample_rate

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    def set_language(self, language: str) -> None:
        """Set the language for synthesis.

        Args:
            language: Language code (e.g., 'es')
        """
        self._language = language
        logger.info(f"TTS language set to: {language}")

    def set_voice(self, voice_file: str) -> None:
        """Set the reference voice for cloning.

        Args:
            voice_file: Path to reference audio file
        """
        if voice_file and not Path(voice_file).exists():
            logger.warning(f"Voice file not found: {voice_file}")
        self._voice_file = voice_file
        logger.info(f"TTS voice file set to: {voice_file}")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    @property
    def device(self) -> str | None:
        """Get the device the model is running on."""
        return self._device

    def close(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            self._model = None
            self._device = None
            logger.info("TTS model unloaded")
