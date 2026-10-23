"""Environment-driven configuration (pydantic-settings).

Provider keys use their conventional variable names (``ANTHROPIC_API_KEY``,
``MISTRAL_API_KEY``, ...); gateway options use the ``GATEWAY_`` prefix. Only
providers with credentials (or, for local servers, a base URL) are enabled.
"""

from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_inference_gateway.schemas import DEFAULT_SIMILARITY_THRESHOLD, ModelProvider


def _key(name: str) -> SecretStr | None:
    return Field(default=None, validation_alias=AliasChoices(name, f"GATEWAY_{name}"))


class GatewaySettings(BaseSettings):
    """All runtime configuration; immutable once loaded."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )

    # --- Upstream providers ---------------------------------------------------
    openrouter_api_key: SecretStr | None = _key("OPENROUTER_API_KEY")
    mistral_api_key: SecretStr | None = _key("MISTRAL_API_KEY")
    deepseek_api_key: SecretStr | None = _key("DEEPSEEK_API_KEY")
    groq_api_key: SecretStr | None = _key("GROQ_API_KEY")
    together_api_key: SecretStr | None = _key("TOGETHER_API_KEY")
    openai_api_key: SecretStr | None = _key("OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = _key("ANTHROPIC_API_KEY")
    vllm_base_url: str | None = None
    ollama_base_url: str | None = None
    provider_timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)

    # --- Routing --------------------------------------------------------------
    tier1_provider: ModelProvider = ModelProvider.MISTRAL
    tier1_model: str = "mistral-small-latest"
    tier2_model: str = "claude-sonnet-5-5"
    tier3_model: str = "claude-opus-5-5"
    tier3_min_score: float = Field(default=0.8, gt=0.0, le=1.0)

    # --- Semantic cache -------------------------------------------------------
    qdrant_url: str = Field(
        default=":memory:",
        validation_alias=AliasChoices("QDRANT_URL", "GATEWAY_QDRANT_URL"),
        description="Qdrant server URL, or ':memory:' for a process-local cache.",
    )
    qdrant_api_key: SecretStr | None = _key("QDRANT_API_KEY")
    qdrant_collection: str = "semantic_cache"
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, ge=0.95, lt=1.0)

    # --- Embeddings -----------------------------------------------------------
    embedding_backend: Literal["hashing", "openai_compatible"] = "hashing"
    embedding_dimension: int = Field(default=384, ge=1, le=8192)
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _embedding_complete(self) -> Self:
        if self.embedding_backend == "openai_compatible" and not (
            self.embedding_base_url and self.embedding_model
        ):
            raise ValueError(
                "embedding_backend=openai_compatible requires embedding_base_url and "
                "embedding_model"
            )
        return self
