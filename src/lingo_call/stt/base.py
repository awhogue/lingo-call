"""Abstract base class for speech-to-text providers."""

from abc import ABC, abstractmethod

import numpy as np


class STTProvider(ABC):
    """Abstract interface for speech-to-text providers."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32, mono)
            sample_rate: Sample rate of the audio in Hz

        Returns:
            Transcribed text
        """
        pass

    @abstractmethod
    def set_language(self, language: str) -> None:
        """Set the language for transcription.

        Args:
            language: Language code (provider-specific format)
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
