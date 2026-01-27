"""Audio playback using sounddevice."""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Audio playback using sounddevice."""

    def __init__(self) -> None:
        """Initialize the audio player."""
        self._playing = False
        self._lock = threading.Lock()

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        """Play audio.

        Args:
            audio: Audio data as numpy array (float32)
            sample_rate: Sample rate in Hz
            blocking: If True, wait for playback to complete
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from e

        if len(audio) == 0:
            logger.warning("Attempted to play empty audio")
            return

        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        with self._lock:
            self._playing = True

        try:
            logger.debug(f"Playing audio: {len(audio)/sample_rate:.2f}s at {sample_rate}Hz")
            sd.play(audio, sample_rate)

            if blocking:
                sd.wait()
        finally:
            with self._lock:
                self._playing = False

    def play_file(self, file_path: str, blocking: bool = True) -> None:
        """Play audio from a file.

        Args:
            file_path: Path to audio file
            blocking: If True, wait for playback to complete
        """
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is not installed. Install with: pip install soundfile"
            ) from e

        audio, sample_rate = sf.read(file_path, dtype=np.float32)

        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        self.play(audio, sample_rate, blocking)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        try:
            import sounddevice as sd

            sd.stop()
        except ImportError:
            pass

        with self._lock:
            self._playing = False

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            return self._playing

    def save(self, audio: np.ndarray, sample_rate: int, file_path: str) -> None:
        """Save audio to a file.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            file_path: Path to save the audio file
        """
        try:
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "soundfile is not installed. Install with: pip install soundfile"
            ) from e

        # Ensure float32 and normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        sf.write(file_path, audio, sample_rate)
        logger.info(f"Saved audio to {file_path}")
