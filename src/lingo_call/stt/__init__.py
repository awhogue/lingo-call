"""Speech-to-text providers."""

from lingo_call.stt.base import STTProvider
from lingo_call.stt.omnilingual import OmnilingualSTT

__all__ = ["STTProvider", "OmnilingualSTT"]
