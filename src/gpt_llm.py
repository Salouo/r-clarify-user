"""Lightweight wrapper around OpenAI's official chat completions API."""

from __future__ import annotations

import os
from typing import Iterable, List

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from openai import BadRequestError, OpenAI

from .token_usage import get_usage_collector

load_dotenv(".env", override=False)


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _convert_messages(messages: Iterable[BaseMessage]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
        payload.append({"role": role, "content": _content_to_text(message.content)})
    return payload


class OpenAIChatModel:
    """Minimal chat model wrapper around OpenAI Chat Completions."""

    _client: OpenAI | None = None

    def __init__(
        self,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if OpenAIChatModel._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            OpenAIChatModel._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def client(self) -> OpenAI:
        assert OpenAIChatModel._client is not None
        return OpenAIChatModel._client

    def invoke(self, messages: List[BaseMessage]):
        payload = _convert_messages(messages)
        request_kwargs = {
            "model": self.model,
            "messages": payload,
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as err:
            err_msg = str(err).lower()
            if "temperature" in err_msg and self.temperature is not None:
                request_kwargs.pop("temperature", None)
                response = self.client.chat.completions.create(**request_kwargs)
            else:
                raise
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        collector = get_usage_collector()
        if collector and (prompt_tokens or completion_tokens):
            collector.add(prompt=prompt_tokens, completion=completion_tokens)

        content = choice.message.content or ""
        if isinstance(content, list):
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            text = str(content)

        add_kwargs = {}
        if prompt_tokens is not None or completion_tokens is not None:
            add_kwargs["token_usage"] = {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
                "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
            }

        return AIMessage(content=text, additional_kwargs=add_kwargs)
