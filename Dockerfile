# syntax=docker/dockerfile:1
#
# AI Inference Gateway: production image.
#
#   builder: resolves the locked production dependency set with uv and installs
#            the project as a regular (non-editable) wheel into /opt/venv.
#   runner:  python:3.13-slim + /opt/venv only. No uv, no sources, no tests, no
#            dev dependencies, no build caches. Runs as uid 10001 and works on a
#            read-only root filesystem (bytecode is precompiled at build time).
#
# Base images are pinned by digest; bump tag and digest together.

ARG PYTHON_IMAGE=python:3.13-slim@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_FROZEN=1

WORKDIR /build

# Dependencies first (cached until pyproject.toml / uv.lock change).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --no-dev --no-install-project

# Then the project itself, as a wheel (not editable): /build is not needed at runtime.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable

# --------------------------------------------------------------------------- #
FROM ${PYTHON_IMAGE} AS runner

LABEL org.opencontainers.image.title="ai-inference-gateway" \
      org.opencontainers.image.description="Async FastAPI inference gateway: semantic cache, complexity routing, rate limiting."

RUN groupadd --system --gid 10001 appuser \
 && useradd --system --uid 10001 --gid appuser --no-create-home \
            --home-dir /nonexistent --shell /usr/sbin/nologin appuser

# Owned by root and not writable by appuser: the app cannot modify its own code.
COPY --from=builder --chown=root:root /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TMPDIR=/tmp

WORKDIR /nonexistent
USER 10001:10001
EXPOSE 8000

# No curl/wget in the image: probe /health with the stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]

# --no-proxy-headers: client IPs are resolved by the gateway's own guardrail
# (GATEWAY_TRUSTED_PROXIES), so uvicorn must not rewrite them first.
# --no-server-header: do not advertise the server implementation.
CMD ["python", "-m", "uvicorn", "ai_inference_gateway.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--no-proxy-headers", "--no-server-header", \
     "--timeout-graceful-shutdown", "20"]
