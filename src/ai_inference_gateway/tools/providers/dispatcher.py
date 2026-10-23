"""Registry-backed implementation of the core ``ProviderDispatcher`` port."""

from collections.abc import AsyncGenerator, Iterable
from contextlib import aclosing

from ai_inference_gateway.core.errors import RoutingError
from ai_inference_gateway.core.ports import ProviderCompletion, ProviderStreamEvent
from ai_inference_gateway.schemas import InferenceRequest, ModelProvider, RouteTarget
from ai_inference_gateway.tools.providers.base import BaseProvider, normalize_base_url


class DynamicProviderDispatcher:
    """Resolves an adapter per target and delegates the completion to it.

    Targets carrying an ``endpoint`` are matched against adapters' base URLs;
    otherwise the first adapter registered for ``target.provider`` is used.
    A target nothing is registered for raises ``RoutingError`` (configuration
    problem); adapter failures propagate as ``ProviderError``.
    """

    def __init__(self, adapters: Iterable[BaseProvider] = ()) -> None:
        """Register ``adapters`` in order (first per provider is its default)."""
        self._by_provider: dict[ModelProvider, BaseProvider] = {}
        self._by_endpoint: dict[str, BaseProvider] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: BaseProvider) -> None:
        """Add ``adapter``; base URLs must be unique across the registry."""
        if adapter.base_url in self._by_endpoint:
            raise ValueError(f"an adapter is already registered for {adapter.base_url!r}")
        self._by_endpoint[adapter.base_url] = adapter
        self._by_provider.setdefault(adapter.provider, adapter)

    @property
    def providers(self) -> frozenset[ModelProvider]:
        """Providers with at least one registered adapter."""
        return frozenset(self._by_provider)

    def resolve(self, target: RouteTarget) -> BaseProvider:
        """Return the adapter for ``target``. Raises ``RoutingError``."""
        if target.endpoint is not None:
            adapter = self._by_endpoint.get(normalize_base_url(target.endpoint))
            if adapter is None:
                raise RoutingError(f"no adapter registered for endpoint {target.endpoint!r}")
            if adapter.provider is not target.provider:
                raise RoutingError(
                    f"endpoint {target.endpoint!r} belongs to {adapter.provider.value!r}, "
                    f"not {target.provider.value!r}"
                )
            return adapter
        adapter = self._by_provider.get(target.provider)
        if adapter is None:
            raise RoutingError(f"no adapter registered for provider {target.provider.value!r}")
        return adapter

    async def complete(self, target: RouteTarget, request: InferenceRequest) -> ProviderCompletion:
        """Dispatch ``request`` to the adapter resolved for ``target``."""
        return await self.resolve(target).complete(target, request)

    async def stream(
        self, target: RouteTarget, request: InferenceRequest
    ) -> AsyncGenerator[ProviderStreamEvent]:
        """Stream ``request`` through the adapter resolved for ``target``."""
        adapter = self.resolve(target)
        async with aclosing(adapter.stream(target, request)) as events:
            async for event in events:
                yield event
