"""Configuration dataclasses for Lingo-Call."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Supported languages: intersection of Chatterbox TTS and omnilingual-asr
# Format: code -> (stt_lang, tts_lang, display_name)
SUPPORTED_LANGUAGES: dict[str, tuple[str, str, str]] = {
    "ar": ("arb_Arab", "ar", "Arabic"),
    "da": ("dan_Latn", "da", "Danish"),
    "de": ("deu_Latn", "de", "German"),
    "el": ("ell_Grek", "el", "Greek"),
    "en": ("eng_Latn", "en", "English"),
    "es": ("spa_Latn", "es", "Spanish"),
    "fi": ("fin_Latn", "fi", "Finnish"),
    "fr": ("fra_Latn", "fr", "French"),
    "he": ("heb_Hebr", "he", "Hebrew"),
    "hi": ("hin_Deva", "hi", "Hindi"),
    "it": ("ita_Latn", "it", "Italian"),
    "ja": ("jpn_Jpan", "ja", "Japanese"),
    "ko": ("kor_Hang", "ko", "Korean"),
    "ms": ("zsm_Latn", "ms", "Malay"),
    "nl": ("nld_Latn", "nl", "Dutch"),
    "no": ("nob_Latn", "no", "Norwegian"),
    "pl": ("pol_Latn", "pl", "Polish"),
    "pt": ("por_Latn", "pt", "Portuguese"),
    "ru": ("rus_Cyrl", "ru", "Russian"),
    "sv": ("swe_Latn", "sv", "Swedish"),
    "sw": ("swh_Latn", "sw", "Swahili"),
    "tr": ("tur_Latn", "tr", "Turkish"),
    "zh": ("cmn_Hans", "zh", "Chinese"),
}


def get_language_info(code: str) -> tuple[str, str, str]:
    """Get language info by code or STT language code.

    Args:
        code: Either a short code (e.g., 'es') or STT code (e.g., 'spa_Latn')

    Returns:
        Tuple of (stt_lang, tts_lang, display_name)

    Raises:
        ValueError: If language code is not supported
    """
    # Check if it's a short code
    if code in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[code]

    # Check if it's an STT language code
    for short_code, (stt_lang, tts_lang, display_name) in SUPPORTED_LANGUAGES.items():
        if code == stt_lang:
            return (stt_lang, tts_lang, display_name)

    supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
    raise ValueError(f"Unsupported language: {code}. Supported: {supported}")


@dataclass
class STTConfig:
    """Configuration for speech-to-text."""

    # Model card for omnilingual-asr
    # Options: omniASR_CTC_300M_v2, omniASR_CTC_1B_v2, omniASR_CTC_7B_v2, omniASR_LLM_3B_v2
    model_card: str = "omniASR_CTC_1B_v2"

    # Device for inference: "mps", "cuda", or "cpu"
    device: str = "mps"

    # Fallback to CPU if device not available
    fallback_to_cpu: bool = True


@dataclass
class LLMConfig:
    """Configuration for LLM."""

    # HuggingFace model name or path
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Device for inference: "auto", "mps", "cuda", or "cpu"
    device: str = "auto"

    # Whether to use 4-bit quantization (reduces memory usage)
    use_4bit: bool = True

    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Whether to trust remote code (required for some models)
    trust_remote_code: bool = False


@dataclass
class TTSConfig:
    """Configuration for text-to-speech."""

    # Model type: "turbo" (English only), "multilingual" (Chatterbox), or "xtts" (Coqui XTTS-v2)
    model_type: Literal["turbo", "multilingual", "xtts"] = "multilingual"

    # Path to reference audio file for voice cloning
    voice_file: str = ""

    # Device for inference
    device: str = "mps"

    # Fallback to CPU if device not available
    fallback_to_cpu: bool = True

    # Generation parameters
    cfg_weight: float = 0.5
    exaggeration: float = 0.5

    # Audio output sample rate
    sample_rate: int = 24000


@dataclass
class AudioConfig:
    """Configuration for audio recording and playback."""

    # Recording sample rate (16kHz recommended for STT)
    sample_rate: int = 16000

    # Number of audio channels (1 = mono)
    channels: int = 1

    # Push-to-talk key (keyboard key name)
    push_to_talk_key: str = "space"

    # Minimum recording duration in seconds
    min_duration: float = 0.5

    # Maximum recording duration in seconds
    max_duration: float = 30.0


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""

    # Maximum number of messages to keep in history
    max_history: int = 20

    # System prompt template (use {language} placeholder)
    system_prompt: str = field(
        default="You are a helpful assistant. Respond in {language}. "
        "Keep your responses concise and natural for spoken conversation."
    )


@dataclass
class AppConfig:
    """Main application configuration."""

    # Language code (e.g., "es", "spa_Latn")
    language: str = "en"

    # Component configurations
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate language
        self._language_info = get_language_info(self.language)

    @property
    def stt_language(self) -> str:
        """Get the STT language code."""
        return self._language_info[0]

    @property
    def tts_language(self) -> str:
        """Get the TTS language code."""
        return self._language_info[1]

    @property
    def display_language(self) -> str:
        """Get the display name of the language."""
        return self._language_info[2]

    @property
    def system_prompt(self) -> str:
        """Get the formatted system prompt."""
        return self.conversation.system_prompt.format(language=self.display_language)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Build nested configs
        stt = STTConfig(**data.pop("stt", {}))
        llm = LLMConfig(**data.pop("llm", {}))
        tts = TTSConfig(**data.pop("tts", {}))
        audio = AudioConfig(**data.pop("audio", {}))
        conversation = ConversationConfig(**data.pop("conversation", {}))

        return cls(
            stt=stt,
            llm=llm,
            tts=tts,
            audio=audio,
            conversation=conversation,
            **data,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import yaml
        from dataclasses import asdict

        # Convert to dict, excluding private attributes
        data = {
            "language": self.language,
            "stt": asdict(self.stt),
            "llm": asdict(self.llm),
            "tts": asdict(self.tts),
            "audio": asdict(self.audio),
            "conversation": asdict(self.conversation),
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
