"""Lightweight token usage collector shared across LLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import contextvars


@dataclass
class TokenUsageCollector:
    """Accumulates prompt/completion tokens across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int | None = None, completion: int | None = None) -> None:
        if prompt:
            self.prompt_tokens += int(prompt)
        if completion:
            self.completion_tokens += int(completion)

    def snapshot(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    @staticmethod
    def diff(newer: dict, older: dict) -> dict:
        """Compute token delta between two snapshots."""
        return {
            "prompt_tokens": (newer.get("prompt_tokens", 0) - older.get("prompt_tokens", 0)),
            "completion_tokens": (newer.get("completion_tokens", 0) - older.get("completion_tokens", 0)),
            "total_tokens": (newer.get("total_tokens", 0) - older.get("total_tokens", 0)),
        }


_collector_var: contextvars.ContextVar[Optional[TokenUsageCollector]] = contextvars.ContextVar(
    "token_usage_collector", default=None
)


def set_usage_collector(collector: TokenUsageCollector) -> contextvars.Token:
    """Set the active collector for the current context."""
    return _collector_var.set(collector)


def clear_usage_collector(token: contextvars.Token) -> None:
    """Restore prior collector (typically after a run completes)."""
    _collector_var.reset(token)


def get_usage_collector() -> Optional[TokenUsageCollector]:
    """Return current collector, if any."""
    return _collector_var.get()
