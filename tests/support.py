"""Test helpers shared across modules (fixtures live in ``conftest.py``)."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, ClassVar

import httpx
from fastapi import FastAPI
from pydantic import SecretStr

from ai_inference_gateway.api import GatewaySettings

DUMMY_KEY = "sk-test-0123456789abcdef-DUMMY"
# Key-shaped fixture for redaction tests, assembled at runtime so the source never
# contains a credential-shaped literal (keeps the pre-commit secret scanner strict).
FAKE_ANTHROPIC_KEY = "-".join(("sk", "ant", "api03", "abcdefghijklmnop"))

type Handler = Callable[[httpx.Request], httpx.Response]
type AppFactory = Callable[..., FastAPI]


def openai_sse(*texts: str, done: bool = True, model: str = "mistral-small-latest") -> bytes:
    """OpenAI-format stream: one chunk per text, then finish + usage + ``[DONE]``."""
    chunks = [
        {"model": model, "choices": [{"delta": {"content": text}, "finish_reason": None}]}
        for text in texts
    ]
    body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
    if done:
        tail = {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 12}}
        body += f"data: {json.dumps(tail)}\n\ndata: [DONE]\n\n".encode()
    return body


def anthropic_sse(*texts: str, model: str = "claude-sonnet-5-5", stop: bool = True) -> bytes:
    """Anthropic Messages stream: message_start, deltas, message_delta, message_stop."""
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {"model": model, "usage": {"input_tokens": 20, "output_tokens": 1}},
            },
        ),
        ("ping", {"type": "ping"}),
    ]
    events += [
        (
            "content_block_delta",
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": t}},
        )
        for t in texts
    ]
    events.append(("message_delta", {"type": "message_delta", "usage": {"output_tokens": 9}}))
    if stop:
        events.append(("message_stop", {"type": "message_stop"}))
    return b"".join(f"event: {e}\ndata: {json.dumps(d)}\n\n".encode() for e, d in events)


class Upstream:
    """Scriptable fake for every provider endpoint (OpenAI-format and Anthropic)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.override: Callable[[httpx.Request, dict[str, Any]], httpx.Response] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        self.calls.append((request.url.path, body))
        if self.override is not None:
            return self.override(request, body)
        stream = bool(body.get("stream"))
        if request.url.path == "/v1/messages":
            if stream:
                return httpx.Response(200, content=anthropic_sse("Deep ", "answer"))
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "content": [{"type": "text", "text": "Deep answer"}],
                    "usage": {"input_tokens": 20, "output_tokens": 9},
                },
            )
        if stream:
            return httpx.Response(200, content=openai_sse("Par", "is"))
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": "Paris"}}],
                "usage": {"total_tokens": 12},
            },
        )


class NetworkBlockedError(ConnectionError):
    """Raised by the test network guard; an OSError so stacks map it like a real failure."""


class RecordingQdrant:
    """Stand-in for a remote ``AsyncQdrantClient`` (the real one probes the server on init)."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        RecordingQdrant.calls.append(kwargs)


def mock_client(handler: Handler) -> httpx.AsyncClient:
    """httpx client whose transport is ``handler`` (no sockets involved)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def make_settings(**overrides: Any) -> GatewaySettings:
    """Hermetic settings: Mistral + Anthropic with dummy keys, in-memory Qdrant, no Redis."""
    base: dict[str, Any] = {
        "mistral_api_key": SecretStr(DUMMY_KEY),
        "anthropic_api_key": SecretStr(DUMMY_KEY),
        "qdrant_url": ":memory:",
        "redis_url": None,
    }
    return GatewaySettings(**{**base, **overrides})


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """Run the app lifespan and yield an in-process client (ASGITransport)."""
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client,
    ):
        yield client
