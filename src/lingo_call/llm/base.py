"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, messages: list[dict[str, str]], system_prompt: str | None = None) -> str:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are typically 'user' and 'assistant'.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Generated response text
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
