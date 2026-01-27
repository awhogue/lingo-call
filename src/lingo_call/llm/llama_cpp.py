"""LLM implementation using llama-cpp-python."""

import logging
from typing import Any

from lingo_call.config import LLMConfig
from lingo_call.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class LlamaCppLLM(LLMProvider):
    """LLM provider using llama-cpp-python with MPS/CUDA support.

    Supports GGUF models including:
    - Llama 3.x series
    - Qwen 2.x series
    - Other compatible models
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the llama-cpp-python provider.

        Args:
            config: LLM configuration
        """
        self.config = config
        self._model: Any = None

    def _load_model(self) -> None:
        """Load the LLM model lazily."""
        if self._model is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. Install with: "
                "pip install llama-cpp-python"
            ) from e

        if not self.config.model_path:
            raise ValueError("LLM model_path is required but not set")

        logger.info(f"Loading LLM from {self.config.model_path}...")

        self._model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=self.config.n_threads,
            verbose=False,
        )

        logger.info("LLM loaded successfully")

    def _format_messages(
        self, messages: list[dict[str, str]], system_prompt: str | None
    ) -> str:
        """Format messages according to the model family's chat template.

        Args:
            messages: List of message dicts
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if self.config.model_family == "llama":
            return self._format_llama(messages, system_prompt)
        elif self.config.model_family == "qwen":
            return self._format_qwen(messages, system_prompt)
        else:
            # Generic format
            return self._format_generic(messages, system_prompt)

    def _format_llama(
        self, messages: list[dict[str, str]], system_prompt: str | None
    ) -> str:
        """Format messages for Llama 3.x models."""
        parts = ["<|begin_of_text|>"]

        if system_prompt:
            parts.append(
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            )

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(
                f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            )

        # Add assistant header for generation
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(parts)

    def _format_qwen(
        self, messages: list[dict[str, str]], system_prompt: str | None
    ) -> str:
        """Format messages for Qwen 2.x models."""
        parts = []

        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Add assistant start for generation
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

    def _format_generic(
        self, messages: list[dict[str, str]], system_prompt: str | None
    ) -> str:
        """Generic chat format as fallback."""
        parts = []

        if system_prompt:
            parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            parts.append(f"{role}: {content}")

        parts.append("Assistant:")

        return "\n".join(parts)

    def _get_stop_tokens(self) -> list[str]:
        """Get stop tokens for the model family."""
        if self.config.model_family == "llama":
            return ["<|eot_id|>", "<|end_of_text|>"]
        elif self.config.model_family == "qwen":
            return ["<|im_end|>", "<|endoftext|>"]
        else:
            return ["\nUser:", "\nHuman:", "\n\n"]

    def generate(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        self._load_model()

        prompt = self._format_messages(messages, system_prompt)
        stop_tokens = self._get_stop_tokens()

        try:
            output = self._model(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=stop_tokens,
                echo=False,
            )

            # Extract text from output
            if isinstance(output, dict):
                choices = output.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
                    return text.strip()
            return ""

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None

    def close(self) -> None:
        """Clean up resources."""
        if self._model is not None:
            # llama-cpp-python handles cleanup automatically
            self._model = None
            logger.info("LLM model unloaded")
