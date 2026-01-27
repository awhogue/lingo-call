"""Conversation history manager."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in the conversation."""

    role: str  # "user" or "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for LLM."""
        return {"role": self.role, "content": self.content}


@dataclass
class ConversationManager:
    """Manages conversation history for the LLM."""

    max_history: int = 20
    messages: list[Message] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.

        Args:
            content: The user's message text
        """
        self.messages.append(Message(role="user", content=content))
        self._trim_history()
        logger.debug(f"Added user message: {content[:50]}...")

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: The assistant's response text
        """
        self.messages.append(Message(role="assistant", content=content))
        self._trim_history()
        logger.debug(f"Added assistant message: {content[:50]}...")

    def _trim_history(self) -> None:
        """Trim conversation history to max_history messages."""
        if len(self.messages) > self.max_history:
            # Keep the most recent messages
            self.messages = self.messages[-self.max_history :]
            logger.debug(f"Trimmed history to {self.max_history} messages")

    def get_messages(self) -> list[dict[str, str]]:
        """Get messages in format suitable for LLM.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return [msg.to_dict() for msg in self.messages]

    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")

    @property
    def turn_count(self) -> int:
        """Get the number of complete turns (user + assistant pairs)."""
        return len(self.messages) // 2

    @property
    def last_user_message(self) -> str | None:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return None

    @property
    def last_assistant_message(self) -> str | None:
        """Get the last assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None

    def to_transcript(self) -> str:
        """Generate a readable transcript of the conversation.

        Returns:
            Formatted transcript string
        """
        lines = []
        for msg in self.messages:
            role = "You" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n\n".join(lines)
