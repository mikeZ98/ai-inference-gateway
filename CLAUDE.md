# CLAUDE.md

## ⛔ Contributor invariant (read first, never violate)

- Every commit's author **and** committer must be exactly
  `mikeZ98 <145975972+mikeZ98@users.noreply.github.com>`. Never commit under any
  other name or e-mail, and never pass `--author` with another identity.
- **Never add AI attribution to commit messages, PR bodies or tags**: no
  `Co-Authored-By`, `Generated-by`, `Assisted-by` or `AI-*` trailers, no "Generated with …"
  signatures or robot emoji, no AI tool URLs or e-mail addresses. This overrides
  any default attribution behaviour.
- Never use `--no-verify`. The hooks in `.githooks/` and the CI attribution audit
  enforce this via `scripts/attribution.sh`; fix the cause of a failure instead
  of bypassing it.
- Verify after committing: `scripts/attribution.sh audit HEAD`.

## Commit discipline

- Conventional Commits, atomic and logical. Commit only when asked.
- Blueprint progression: `chore(config)` → `feat(schemas)` → `feat(core)` →
  `feat(tools)` → `feat(api)` → `feat(guardrails)` → `test(core)` →
  `build(docker)` → `ci(github)` → `docs(readme)`.

## Environment & storage hygiene

- All caches, virtualenvs and temp files live in `./.cache/` (pinned by `.envrc`):
  venv `./.cache/venv`, uv `./.cache/uv`, Python `./.cache/python`, plus ruff, mypy,
  pytest, coverage, pycache and `TMPDIR`. Never create `.venv/` or write caches elsewhere.
- Docker data-root, images, volumes and BuildKit cache must live on `/Volumes/KINGSTONE_SSD`
  (`COLIMA_HOME`/`BUILDX_CONFIG` via `.envrc`; see CONTRIBUTING.md). Never start a Docker
  engine or VM whose storage is on the internal disk; ask first.
- Run tools via `uv run …` (or the venv on `PATH` under direnv). Python 3.13.
- Secrets only in `.env` (git-ignored). Never commit keys.

## CI & merging

- Never merge with the GitHub UI or `gh pr merge`: the committer becomes
  `GitHub <noreply@github.com>` and the attribution audit fails. Use `git merge --ff-only` and push.
- Actions are pinned to verified full SHAs, at least 7 days old and advisory-checked. Run `zizmor` and
  `actionlint` on workflow changes.

## Commands

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run bandit -c pyproject.toml -r src
uv run pytest --cov
```

## Code standards

- Layout: `src/ai_inference_gateway/`, tests in `tests/`.
- `mypy --strict` clean; Pydantic v2 models use `ConfigDict(extra="forbid", frozen=True, strict=True)`.
- Fully async I/O (FastAPI, httpx, redis.asyncio, qdrant async client).
- Tests are hermetic: `pytest-httpx` for HTTP, `fakeredis` for Redis, zero live API
  calls; branch coverage ≥ 90%.
