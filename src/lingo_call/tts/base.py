"""Abstract base class for text-to-speech providers."""

from abc import ABC, abstractmethod

import numpy as np


class TTSProvider(ABC):
    """Abstract interface for text-to-speech providers."""

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (audio_array, sample_rate)
            - audio_array: numpy array of audio samples (float32)
            - sample_rate: sample rate in Hz
        """
        pass

    @abstractmethod
    def set_language(self, language: str) -> None:
        """Set the language for synthesis.

        Args:
            language: Language code (provider-specific format)
        """
        pass

    @abstractmethod
    def set_voice(self, voice_file: str) -> None:
        """Set the reference voice for cloning.

        Args:
            voice_file: Path to reference audio file
        """
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        pass

    def close(self) -> None:
        """Clean up resources. Override if needed."""
        pass
