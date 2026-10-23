# ai-inference-gateway

Async FastAPI inference gateway with a Qdrant semantic cache, complexity-based
multi-model routing and Redis-backed rate limiting.

Full documentation lands with the `docs(readme)` milestone. Until then see
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, CI and the merge policy.

## Storage isolation

All tooling state stays on the external SSD (`/Volumes/KINGSTONE_SSD`), never on
the host's internal disk:

- **Python tooling:** virtualenv, uv, ruff, mypy, pytest, coverage, pycache and
  `TMPDIR` live under `./.cache/` (pinned by `.envrc`).
- **Docker:** the data-root, images, volumes and BuildKit build cache live in the
  container VM's disk, which must be on the SSD. With Colima, `.envrc` sets
  `COLIMA_HOME=/Volumes/KINGSTONE_SSD/.colima` and `BUILDX_CONFIG=./.cache/buildx`.
  With Docker Desktop, set *Disk image location* to `/Volumes/KINGSTONE_SSD/DockerDesktop`.
  See [CONTRIBUTING.md](CONTRIBUTING.md#docker-storage-isolation-ssd-invariant).
