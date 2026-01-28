"""XTTS-v2 TTS implementation using Coqui TTS (coqui/XTTS-v2)."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lingo_call.config import TTSConfig
from lingo_call.tts.base import TTSProvider

logger = logging.getLogger(__name__)

# XTTS-v2 sample rate (fixed)
XTTS_SAMPLE_RATE = 24000

# Map our short language codes to XTTS language codes where they differ
XTTS_LANGUAGE_MAP: dict[str, str] = {
    "zh": "zh-cn",
}


class XTTSv2TTS(TTSProvider):
    """Text-to-speech using Coqui XTTS-v2 (coqui/XTTS-v2).

    Supports voice cloning and 17 languages:
    en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
    """

    def __init__(self, config: TTSConfig, language: str) -> None:
        """Initialize the XTTS-v2 TTS provider.

        Args:
            config: TTS configuration
            language: Language code (e.g., 'es' for Spanish)
        """
        self.config = config
        self._language = self._to_xtts_lang(language)
        self._voice_file = config.voice_file
        self._model: Any = None
        self._device: str | None = None

    @staticmethod
    def _to_xtts_lang(code: str) -> str:
        """Map internal language code to XTTS language code."""
        return XTTS_LANGUAGE_MAP.get(code, code)

    def _load_model(self) -> None:
        """Load the XTTS-v2 model lazily."""
        if self._model is not None:
            return

        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is not installed") from e

        try:
            from TTS.api import TTS
        except ImportError as e:
            raise ImportError(
                "Coqui TTS is not installed. Install with: pip install TTS"
            ) from e

        device = self.config.device

        if device == "mps" and (
            not getattr(torch.backends, "mps", None)
            or not torch.backends.mps.is_available()
        ):
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        try:
            logger.info(f"Loading XTTS-v2 on {device}...")
            # Model ID for Coqui TTS API (same as coqui/XTTS-v2 on Hugging Face)
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                device
            )
            self._device = device
            logger.info(f"Successfully loaded XTTS-v2 on {device}")
        except Exception as e:
            if self.config.fallback_to_cpu and device != "cpu":
                logger.warning(
                    f"Failed to load XTTS-v2 on {device}: {e}. Falling back to CPU..."
                )
                try:
                    self._model = TTS(
                        "tts_models/multilingual/multi-dataset/xtts_v2"
                    ).to("cpu")
                    self._device = "cpu"
                    logger.info("Successfully loaded XTTS-v2 on CPU")
                except Exception as e2:
                    raise e2 from e
            else:
                raise

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._load_model()

        if not text.strip():
            return np.array([], dtype=np.float32), XTTS_SAMPLE_RATE

        speaker_wav = None
        if self._voice_file and Path(self._voice_file).exists():
            speaker_wav = self._voice_file
        elif self._voice_file:
            logger.warning(f"Voice file not found: {self._voice_file}")
        else:
            logger.warning(
                "XTTS-v2 requires a reference voice for cloning. Set --voice to a WAV file."
            )

        if speaker_wav is None:
            raise ValueError(
                "XTTS-v2 requires a reference voice file (--voice). "
                "Provide a path to a short (e.g. 6 second) WAV clip for voice cloning."
            )

        try:
            # tts() returns list of amplitude values (or numpy array)
            wav = self._model.tts(
                text=text,
                speaker_wav=speaker_wav,
                language=self._language,
            )

            audio = np.asarray(wav).squeeze().astype(np.float32)

            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            return audio, XTTS_SAMPLE_RATE

        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}")
            raise

    def set_language(self, language: str) -> None:
        """Set the language for synthesis.

        Args:
            language: Language code (e.g., 'es')
        """
        self._language = self._to_xtts_lang(language)
        logger.info(f"XTTS language set to: {self._language}")

    def set_voice(self, voice_file: str) -> None:
        """Set the reference voice for cloning.

        Args:
            voice_file: Path to reference audio file (WAV, ~6 seconds recommended)
        """
        if voice_file and not Path(voice_file).exists():
            logger.warning(f"Voice file not found: {voice_file}")
        self._voice_file = voice_file
        logger.info(f"XTTS voice file set to: {voice_file}")

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
            logger.info("XTTS-v2 model unloaded")
