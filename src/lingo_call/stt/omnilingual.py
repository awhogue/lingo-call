"""Omnilingual ASR implementation for speech-to-text."""

import logging
from typing import Any

import numpy as np

from lingo_call.config import STTConfig
from lingo_call.stt.base import STTProvider

logger = logging.getLogger(__name__)


class OmnilingualSTT(STTProvider):
    """Speech-to-text using Facebook's omnilingual-asr.

    Supports 1600+ languages with models of varying sizes:
    - omniASR_CTC_300M_v2: 300M params, ~1.2GB, fastest
    - omniASR_CTC_1B_v2: 1B params, ~4GB, fast
    - omniASR_CTC_7B_v2: 6.5B params, ~25GB, medium speed
    - omniASR_LLM_3B_v2: 4.4B params, ~17GB, slower but better quality
    """

    def __init__(self, config: STTConfig, language: str) -> None:
        """Initialize the omnilingual ASR provider.

        Args:
            config: STT configuration
            language: Language code (e.g., 'spa_Latn' for Spanish)
        """
        self.config = config
        self._language = language
        self._pipeline: Any = None
        self._device: str | None = None

    def _load_model(self) -> None:
        """Load the ASR model lazily."""
        if self._pipeline is not None:
            return

        try:
            from omni_asr import ASRInferencePipeline
        except ImportError as e:
            raise ImportError(
                "omnilingual-asr is not installed. Install with: pip install omnilingual-asr"
            ) from e

        device = self.config.device

        # Try to load on requested device
        try:
            logger.info(f"Loading {self.config.model_card} on {device}...")
            self._pipeline = ASRInferencePipeline(
                model_card=self.config.model_card,
                device=device,
            )
            self._device = device
            logger.info(f"Successfully loaded model on {device}")
        except Exception as e:
            if self.config.fallback_to_cpu and device != "cpu":
                logger.warning(f"Failed to load on {device}: {e}. Falling back to CPU...")
                self._pipeline = ASRInferencePipeline(
                    model_card=self.config.model_card,
                    device="cpu",
                )
                self._device = "cpu"
                logger.info("Successfully loaded model on CPU")
            else:
                raise

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio in Hz

        Returns:
            Transcribed text
        """
        self._load_model()

        # Ensure audio is the right shape and type
        if audio.ndim > 1:
            # Convert to mono by averaging channels
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)

        # Normalize if needed (should be in range [-1, 1])
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Resample if needed (omnilingual-asr expects 16kHz)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
            sample_rate = 16000

        # Run inference
        try:
            result = self._pipeline(
                audio=audio,
                sample_rate=sample_rate,
                lang=self._language,
            )

            # Extract text from result
            if isinstance(result, dict):
                return result.get("text", "").strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                # Handle list of results
                if hasattr(result, "__iter__"):
                    texts = [r.get("text", "") if isinstance(r, dict) else str(r) for r in result]
                    return " ".join(texts).strip()
                return str(result).strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            audio: Input audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio array
        """
        try:
            import soundfile as sf
            import io

            # Write to buffer and read back at new sample rate
            buffer = io.BytesIO()
            sf.write(buffer, audio, orig_sr, format="WAV")
            buffer.seek(0)
            resampled, _ = sf.read(buffer, samplerate=target_sr)
            return resampled.astype(np.float32)
        except Exception:
            # Fallback: simple linear interpolation
            duration = len(audio) / orig_sr
            target_length = int(duration * target_sr)
            indices = np.linspace(0, len(audio) - 1, target_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def set_language(self, language: str) -> None:
        """Set the language for transcription.

        Args:
            language: Language code in omnilingual format (e.g., 'spa_Latn')
        """
        self._language = language
        logger.info(f"STT language set to: {language}")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._pipeline is not None

    @property
    def device(self) -> str | None:
        """Get the device the model is running on."""
        return self._device

    def close(self) -> None:
        """Clean up resources."""
        if self._pipeline is not None:
            # Clear the pipeline to free memory
            self._pipeline = None
            self._device = None
            logger.info("STT model unloaded")
