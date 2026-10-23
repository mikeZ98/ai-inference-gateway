# shellcheck shell=bash
# Shared helpers sourced by the git hooks (not a hook itself).
# Re-pins every tool cache under ./.cache/ so hooks stay SSD-local even when
# git is driven from outside a direnv-enabled shell (IDEs, GUIs).

REPO_ROOT="$(git rev-parse --show-toplevel)"
CACHE_ROOT="${REPO_ROOT}/.cache"

export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_PROJECT_ENVIRONMENT="${CACHE_ROOT}/venv"
export UV_PYTHON_INSTALL_DIR="${CACHE_ROOT}/python"
export RUFF_CACHE_DIR="${CACHE_ROOT}/ruff"
export MYPY_CACHE_DIR="${CACHE_ROOT}/mypy"
export PYTHONPYCACHEPREFIX="${CACHE_ROOT}/pycache"
export TMPDIR="${CACHE_ROOT}/tmp"
mkdir -p "$TMPDIR"

ATTRIBUTION="${REPO_ROOT}/scripts/attribution.sh"
VENV_BIN="${UV_PROJECT_ENVIRONMENT}/bin"
