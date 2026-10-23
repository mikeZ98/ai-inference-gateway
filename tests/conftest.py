"""Shared hermetic fixtures.

Every test runs with:
- a socket-level network guard (only loopback / Unix sockets may connect, and
  only loopback names resolve), so an accidental real call fails loudly;
- provider / gateway / Qdrant / Redis environment variables removed, so a
  developer's ``.env`` can never leak real keys or endpoints into a test.
"""

import ipaddress
import os
import socket
from collections.abc import AsyncIterator
from typing import Any

import fakeredis
import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from ai_inference_gateway.api import create_app
from ai_inference_gateway.schemas import InferenceRequest
from tests.support import DUMMY_KEY, AppFactory, NetworkBlockedError, Upstream, make_settings

ENV_PREFIXES = ("GATEWAY_", "QDRANT_", "REDIS_")
ENV_SUFFIXES = ("_API_KEY",)


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _is_local(address: object) -> bool:
    if isinstance(address, str | bytes):  # AF_UNIX path
        return True
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            host = host.decode()
        if host in {"localhost", ""}:
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False
    return False


def _guarded_connect(self: socket.socket, address: Any) -> None:
    if not _is_local(address):
        raise NetworkBlockedError(f"network access blocked in tests: {address!r}")
    _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> int:
    if not _is_local(address):
        raise NetworkBlockedError(f"network access blocked in tests: {address!r}")
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    if host is not None and not _is_local((host,)):
        raise NetworkBlockedError(f"DNS resolution blocked in tests: {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    for name in list(os.environ):
        if name.startswith(ENV_PREFIXES) or name.endswith(ENV_SUFFIXES):
            monkeypatch.delenv(name)


@pytest.fixture
def api_key() -> SecretStr:
    return SecretStr(DUMMY_KEY)


@pytest.fixture
def simple_request() -> InferenceRequest:
    return InferenceRequest(prompt="What is the capital of France?")


@pytest.fixture
def complex_request() -> InferenceRequest:
    return InferenceRequest(
        prompt="Debug this and explain why it fails:\n```python\ndef f(x):\n    return x / 0\n```"
    )


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


@pytest.fixture
async def http_client(upstream: Upstream) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        yield client


@pytest.fixture
async def qdrant() -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def redis_server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
async def redis(redis_server: fakeredis.FakeServer) -> AsyncIterator[fakeredis.FakeAsyncRedis]:
    client = fakeredis.FakeAsyncRedis(server=redis_server)
    yield client
    await client.aclose()


@pytest.fixture
def app_factory(http_client: httpx.AsyncClient) -> AppFactory:
    """Build an app with hermetic settings and the mocked upstream client."""

    def factory(**kwargs: Any) -> FastAPI:
        settings = kwargs.pop("settings", None) or make_settings()
        kwargs.setdefault("http_client", http_client)
        return create_app(settings, **kwargs)

    return factory
