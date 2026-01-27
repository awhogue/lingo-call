"""Audio recording with push-to-talk support."""

import logging
import threading
import time
from typing import Callable

import numpy as np

from lingo_call.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Microphone audio recorder with push-to-talk support.

    Uses sounddevice for audio capture and pynput for keyboard detection.
    """

    def __init__(self, config: AudioConfig) -> None:
        """Initialize the audio recorder.

        Args:
            config: Audio configuration
        """
        self.config = config
        self._recording = False
        self._audio_buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self._key_listener = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status
    ) -> None:
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self._recording:
            with self._lock:
                self._audio_buffer.append(indata.copy())

    def record_blocking(self, duration: float | None = None) -> np.ndarray:
        """Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds. If None, uses max_duration.

        Returns:
            Recorded audio as numpy array
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from e

        duration = duration or self.config.max_duration

        logger.info(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.config.sample_rate),
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=np.float32,
        )
        sd.wait()

        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32)

    def record_push_to_talk(
        self,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Record audio while a key is held (push-to-talk).

        Args:
            on_start: Callback when recording starts
            on_stop: Callback when recording stops

        Returns:
            Recorded audio as numpy array
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from e

        try:
            from pynput import keyboard
        except ImportError as e:
            raise ImportError(
                "pynput is not installed. Install with: pip install pynput"
            ) from e

        # Map key name to pynput key
        key_map = {
            "space": keyboard.Key.space,
            "ctrl": keyboard.Key.ctrl,
            "shift": keyboard.Key.shift,
            "alt": keyboard.Key.alt,
        }
        target_key = key_map.get(
            self.config.push_to_talk_key.lower(), keyboard.Key.space
        )

        self._audio_buffer = []
        self._recording = False
        start_time = 0.0

        def on_press(key):
            nonlocal start_time
            if key == target_key and not self._recording:
                self._recording = True
                start_time = time.time()
                if on_start:
                    on_start()
                logger.debug("Recording started (key pressed)")

        def on_release(key):
            if key == target_key and self._recording:
                self._recording = False
                if on_stop:
                    on_stop()
                logger.debug("Recording stopped (key released)")
                return False  # Stop listener

        # Start audio stream
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=np.float32,
            callback=self._audio_callback,
        )

        print(f"Hold [{self.config.push_to_talk_key.upper()}] to speak...")

        with self._stream:
            # Start keyboard listener
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()

        # Combine audio chunks
        with self._lock:
            if self._audio_buffer:
                audio = np.concatenate(self._audio_buffer)
            else:
                audio = np.array([], dtype=np.float32)

        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Check minimum duration
        duration = len(audio) / self.config.sample_rate
        if duration < self.config.min_duration:
            logger.warning(
                f"Recording too short ({duration:.2f}s < {self.config.min_duration}s)"
            )
            return np.array([], dtype=np.float32)

        logger.info(f"Recorded {duration:.2f} seconds of audio")
        return audio.astype(np.float32)

    def record_with_vad(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float | None = None,
    ) -> np.ndarray:
        """Record audio with voice activity detection.

        Stops recording after detecting silence for a specified duration.

        Args:
            silence_threshold: RMS threshold for silence detection
            silence_duration: Duration of silence (seconds) before stopping
            max_duration: Maximum recording duration

        Returns:
            Recorded audio as numpy array
        """
        try:
            import sounddevice as sd
        except ImportError as e:
            raise ImportError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            ) from e

        max_duration = max_duration or self.config.max_duration
        chunk_duration = 0.1  # 100ms chunks for VAD
        chunk_samples = int(chunk_duration * self.config.sample_rate)

        audio_chunks = []
        silence_samples = 0
        total_samples = 0
        has_started = False

        logger.info("Recording... (speak now, will stop after silence)")

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=np.float32,
            blocksize=chunk_samples,
        ) as stream:
            while total_samples < max_duration * self.config.sample_rate:
                chunk, overflowed = stream.read(chunk_samples)

                if overflowed:
                    logger.warning("Audio buffer overflow")

                # Calculate RMS for VAD
                rms = np.sqrt(np.mean(chunk**2))

                if rms > silence_threshold:
                    has_started = True
                    silence_samples = 0
                    audio_chunks.append(chunk.copy())
                elif has_started:
                    silence_samples += len(chunk)
                    audio_chunks.append(chunk.copy())

                    if silence_samples >= silence_duration * self.config.sample_rate:
                        logger.debug("Silence detected, stopping recording")
                        break

                total_samples += len(chunk)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(audio_chunks)

        # Convert to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / self.config.sample_rate
        logger.info(f"Recorded {duration:.2f} seconds of audio")

        return audio.astype(np.float32)

    def close(self) -> None:
        """Clean up resources."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
