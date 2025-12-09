"""
Conversation Manager Service

A provider-agnostic service for managing conversation history and context.
Can be used with any LLM provider (Bedrock, OpenAI, Anthropic, etc.)

Features token-aware history management to optimize costs and prevent context overflow.
"""

import json
import os
from typing import List, Dict, Optional, Literal
from datetime import datetime


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text (rough approximation).
    Uses ~4 characters per token for English text.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class ConversationManager:
    """
    Manages conversation history and context for LLM interactions.
    
    This service is provider-agnostic and can be used with any LLM client.
    It maintains a conversation history and provides methods to manage it.
    
    Features:
    - Token-aware trimming to optimize costs
    - Multiple trimming strategies (sliding window, token-based, smart)
    - Automatic history management
    - Configurable limits
    """
    
    def __init__(
        self,
        max_history: Optional[int] = None,
        max_tokens: Optional[int] = None,
        trim_strategy: Literal["sliding_window", "token_based", "smart"] = "smart",
        keep_system_messages: bool = True,
        reserve_tokens: int = 1000  # Reserve tokens for current request + response
    ):
        """
        Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of messages to keep (sliding window).
                        If None, no message limit is applied.
            max_tokens: Maximum total tokens in history (token-based trimming).
                       If None, no token limit is applied.
                       Recommended: 80% of model context window (e.g., 160K for 200K context)
            trim_strategy: Trimming strategy:
                          - "sliding_window": Keep last N messages
                          - "token_based": Trim when exceeding max_tokens
                          - "smart": Keep system messages + recent messages, trim middle
            keep_system_messages: If True, system messages are never trimmed (smart strategy)
            reserve_tokens: Tokens to reserve for current request + response (default: 1000)
        """
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.trim_strategy = trim_strategy
        self.keep_system_messages = keep_system_messages
        self.reserve_tokens = reserve_tokens
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def _estimate_message_tokens(self, message: Dict) -> int:
        """Estimate tokens for a message (including role and formatting overhead)."""
        content = message.get("content", "")
        # Estimate: content tokens + ~5 tokens for role/formatting overhead
        return estimate_tokens(content) + 5
    
    def _get_total_tokens(self) -> int:
        """Get total estimated tokens in conversation history."""
        return sum(self._estimate_message_tokens(msg) for msg in self.conversation_history)
    
    def _trim_history(self) -> int:
        """
        Trim conversation history based on configured strategy.
        
        Returns:
            int: Number of messages removed
        """
        initial_count = len(self.conversation_history)
        
        if self.trim_strategy == "sliding_window":
            if self.max_history is not None and len(self.conversation_history) > self.max_history:
                self.conversation_history = self.conversation_history[-self.max_history:]
        
        elif self.trim_strategy == "token_based":
            if self.max_tokens is not None:
                # Trim from oldest messages until under limit
                while self._get_total_tokens() > (self.max_tokens - self.reserve_tokens):
                    if len(self.conversation_history) <= 1:
                        break  # Keep at least one message
                    self.conversation_history.pop(0)
        
        elif self.trim_strategy == "smart":
            # Smart strategy: Keep system messages + recent messages, trim middle
            if self.max_tokens is not None:
                total_tokens = self._get_total_tokens()
                target_tokens = self.max_tokens - self.reserve_tokens
                
                if total_tokens > target_tokens:
                    # Separate system messages from others
                    system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
                    other_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]
                    
                    # Keep all system messages
                    # Keep recent messages (from end), trim oldest (from start)
                    tokens_used = sum(self._estimate_message_tokens(msg) for msg in system_messages)
                    
                    # Keep recent messages that fit
                    kept_messages = []
                    for msg in reversed(other_messages):
                        msg_tokens = self._estimate_message_tokens(msg)
                        if tokens_used + msg_tokens <= target_tokens:
                            kept_messages.insert(0, msg)
                            tokens_used += msg_tokens
                        else:
                            break
                    
                    # Reconstruct: system messages + kept recent messages
                    self.conversation_history = system_messages + kept_messages
            elif self.max_history is not None:
                # Fallback to sliding window if max_history is set
                if len(self.conversation_history) > self.max_history:
                    # Keep system messages + recent messages
                    system_messages = [msg for msg in self.conversation_history if msg["role"] == "system"]
                    other_messages = [msg for msg in self.conversation_history if msg["role"] != "system"]
                    
                    # Keep last (max_history - system_count) other messages
                    keep_count = self.max_history - len(system_messages)
                    if keep_count > 0:
                        kept_messages = other_messages[-keep_count:]
                    else:
                        kept_messages = []
                    
                    self.conversation_history = system_messages + kept_messages
        
        self.last_updated = datetime.now()
        return initial_count - len(self.conversation_history)
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Automatically trims history if limits are exceeded.
        
        Args:
            role: Message role ('user', 'assistant', 'system', etc.)
            content: Message content/text
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        self.last_updated = datetime.now()
        
        # Auto-trim if needed
        removed = self._trim_history()
        if removed > 0:
            # Log trimming (could be enhanced with proper logging)
            pass
    
    def add_user_message(self, content: str) -> None:
        """Convenience method to add a user message."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Convenience method to add an assistant message."""
        self.add_message("assistant", content)
    
    def add_system_message(self, content: str) -> None:
        """Convenience method to add a system message."""
        self.add_message("system", content)
    
    def clear(self) -> int:
        """
        Clear all conversation history.
        
        Returns:
            int: Number of messages that were cleared
        """
        count = len(self.conversation_history)
        self.conversation_history = []
        self.last_updated = datetime.now()
        return count
    
    def get_history(self, include_timestamps: bool = False) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Args:
            include_timestamps: If True, includes timestamp in returned dicts
        
        Returns:
            List of message dictionaries. Each dict has 'role' and 'content'.
            If include_timestamps=True, also includes 'timestamp'.
        """
        if include_timestamps:
            return self.conversation_history.copy()
        else:
            # Return only role and content
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.conversation_history
            ]
    
    def get_messages_for_provider(
        self, 
        provider: str = "anthropic",
        include_current_user_message: Optional[str] = None
    ) -> List[Dict]:
        """
        Get messages formatted for a specific LLM provider.
        
        Args:
            provider: Provider name ('anthropic', 'openai', 'bedrock', etc.)
            include_current_user_message: Optional current user message to include
        
        Returns:
            List of messages formatted for the specified provider
        """
        messages = []
        
        # Add conversation history
        for msg in self.conversation_history:
            if provider in ["anthropic", "bedrock"]:
                # Anthropic/Bedrock format
                messages.append({
                    "role": msg["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": msg["content"]
                        }
                    ]
                })
            elif provider == "openai":
                # OpenAI format
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            else:
                # Generic format (same as OpenAI)
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message if provided
        if include_current_user_message:
            if provider in ["anthropic", "bedrock"]:
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": include_current_user_message
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": include_current_user_message
                })
        
        return messages
    
    def get_legacy_format(self, include_current_prompt: Optional[str] = None) -> str:
        """
        Get conversation history in legacy format (Human/Assistant format).
        Useful for older Claude models or other providers that use text-based prompts.
        
        Args:
            include_current_prompt: Optional current prompt to append
        
        Returns:
            Formatted string with conversation history
        """
        history_text = ""
        for msg in self.conversation_history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            history_text += f"\n\n{role}: {msg['content']}"
        
        if include_current_prompt:
            history_text += f"\n\nHuman: {include_current_prompt}\n\nAssistant:"
        
        return history_text
    
    def get_message_count(self) -> int:
        """Get the number of messages in the conversation history."""
        return len(self.conversation_history)
    
    def get_stats(self) -> Dict:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary with conversation stats including token information
        """
        user_count = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        assistant_count = sum(1 for msg in self.conversation_history if msg["role"] == "assistant")
        system_count = sum(1 for msg in self.conversation_history if msg["role"] == "system")
        total_tokens = self._get_total_tokens()
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "system_messages": system_count,
            "estimated_tokens": total_tokens,
            "max_tokens": self.max_tokens,
            "max_history": self.max_history,
            "trim_strategy": self.trim_strategy,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    def trim_history(self, keep_last: int) -> int:
        """
        Trim conversation history to keep only the last N messages.
        
        Args:
            keep_last: Number of messages to keep
        
        Returns:
            Number of messages removed
        """
        if keep_last < 0:
            keep_last = 0
        
        removed = max(0, len(self.conversation_history) - keep_last)
        if removed > 0:
            self.conversation_history = self.conversation_history[-keep_last:]
            self.last_updated = datetime.now()
        
        return removed
    
    def to_dict(self) -> Dict:
        """
        Convert conversation manager state to dictionary for serialization.
        
        Returns:
            Dictionary containing all conversation data
        """
        return {
            "conversation_history": self.conversation_history,
            "max_history": self.max_history,
            "max_tokens": self.max_tokens,
            "trim_strategy": self.trim_strategy,
            "keep_system_messages": self.keep_system_messages,
            "reserve_tokens": self.reserve_tokens,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationManager':
        """
        Create ConversationManager instance from dictionary.
        
        Args:
            data: Dictionary containing conversation data
        
        Returns:
            ConversationManager instance
        """
        manager = cls(
            max_history=data.get("max_history"),
            max_tokens=data.get("max_tokens"),
            trim_strategy=data.get("trim_strategy", "smart"),
            keep_system_messages=data.get("keep_system_messages", True),
            reserve_tokens=data.get("reserve_tokens", 1000)
        )
        
        manager.conversation_history = data.get("conversation_history", [])
        
        # Restore timestamps
        if "created_at" in data:
            manager.created_at = datetime.fromisoformat(data["created_at"])
        if "last_updated" in data:
            manager.last_updated = datetime.fromisoformat(data["last_updated"])
        
        return manager
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save conversation manager state to JSON file.
        
        Args:
            filepath: Path to JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['ConversationManager']:
        """
        Load conversation manager state from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            ConversationManager instance or None if file doesn't exist
        """
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            # Return None if file is corrupted or can't be read
            return None

