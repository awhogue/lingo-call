"""Text-to-speech providers."""

from lingo_call.tts.base import TTSProvider
from lingo_call.tts.chatterbox import ChatterboxTTS

__all__ = ["TTSProvider", "ChatterboxTTS"]
