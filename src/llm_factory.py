"""Factory for the GPT model used by the agent backend."""

from __future__ import annotations

from functools import lru_cache

from .gpt_llm import OpenAIChatModel


GPT_MODEL_NAME = "gpt-5"
GPT_TEMPERATURE: float | None = None


def get_primary_model_name() -> str:
    """Return the GPT model name used by the primary agent."""
    return GPT_MODEL_NAME


def get_primary_model_dirname() -> str:
    """Filesystem-safe dirname for logging outputs."""
    return get_primary_model_name().split("/")[-1]


@lru_cache(maxsize=1)
def get_primary_llm() -> OpenAIChatModel:
    """Return the GPT LLM instance to use for agent/clarify/reflection."""
    return OpenAIChatModel(
        model=GPT_MODEL_NAME,
        temperature=GPT_TEMPERATURE,
    )
