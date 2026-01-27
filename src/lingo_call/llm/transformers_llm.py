"""LLM implementation using HuggingFace transformers."""

import logging
from typing import Any

from lingo_call.config import LLMConfig
from lingo_call.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class TransformersLLM(LLMProvider):
    """LLM provider using HuggingFace transformers.

    Supports any causal language model from HuggingFace Hub including:
    - Llama 3.x series (meta-llama/Llama-3.2-3B-Instruct, etc.)
    - Qwen 2.x series (Qwen/Qwen2.5-7B-Instruct, etc.)
    - Mistral, Phi, and other compatible models
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the transformers provider.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    def _load_model(self) -> None:
        """Load the LLM model lazily."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Install with: "
                "pip install transformers torch accelerate"
            ) from e

        if not self.config.model_name:
            raise ValueError("LLM model_name is required but not set")

        logger.info(f"Loading LLM: {self.config.model_name}...")

        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = self.config.device

        logger.info(f"Using device: {self._device}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Ensure pad token is set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Configure quantization if requested
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }

        if self.config.use_4bit and self._device in ("cuda", "mps"):
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
                logger.info("Using 4-bit quantization")
            except Exception as e:
                logger.warning(f"4-bit quantization not available: {e}. Loading without quantization.")
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
        else:
            if self._device == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
            else:
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        logger.info("LLM loaded successfully")

    def generate(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        self._load_model()

        try:
            import torch
        except ImportError as e:
            raise ImportError("torch is not installed") from e

        # Build messages list with system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Apply chat template
        try:
            prompt = self._tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}. Using fallback format.")
            prompt = self._format_fallback(full_messages)

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        # Move to device if not using device_map
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        response = self._tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return response.strip()

    def _format_fallback(self, messages: list[dict[str, str]]) -> str:
        """Fallback message formatting when chat template fails."""
        parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

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
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("LLM model unloaded")
