# Contributing

> [!IMPORTANT]
> **Sole-contributor invariant: no exceptions.**
>
> Every commit's **author and committer** must be exactly:
>
> | Field | Value                                         |
> | ----- | --------------------------------------------- |
> | Name  | `mikeZ98`                                     |
> | Email | `145975972+mikeZ98@users.noreply.github.com`  |
>
> No other names, machine usernames or local/personal e-mail addresses.
> Commit messages must not contain AI attribution of any kind: no
> `Co-Authored-By`, `Generated-by`, `Assisted-by`, `AI-*` trailers, no
> "Generated with …" signatures, no AI tool URLs and no foreign e-mail
> addresses. `Signed-off-by` is allowed only for the identity above.
>
> Every enforcement point runs the same `scripts/attribution.sh`:
>
> 1. `.githooks/pre-commit` checks the pending author and committer identity.
> 2. `.githooks/commit-msg` checks identity, forbidden trailers/signatures and the Conventional Commits format.
> 3. `.githooks/pre-push` audits every outgoing commit, including ones made with `--no-verify`.
> 4. The CI attribution audit job re-checks every commit on remote branches.

## Setup

Requirements: [`uv`](https://docs.astral.sh/uv/) and [`direnv`](https://direnv.net/).

```sh
direnv allow   # pins caches to ./.cache/, installs hooks, sets repo-local identity
uv sync        # creates ./.cache/venv with runtime + dev dependencies
```

`.envrc` puts every cache, virtualenv, Python install and temp directory
under `./.cache/` (uv, ruff, mypy, pytest, coverage, pycache, `TMPDIR`), so nothing
is written to the host disk. It also runs:

```sh
git config --local core.hooksPath .githooks
git config --local user.name  mikeZ98
git config --local user.email 145975972+mikeZ98@users.noreply.github.com
```

If you don't use direnv, run those three commands by hand.

Secrets go in `.env` (git-ignored, auto-loaded by direnv); see `.env.example`.

## Docker storage isolation (SSD invariant)

> [!IMPORTANT]
> Docker's **data-root, image layers, volumes and BuildKit build cache must live
> on the external SSD** (`/Volumes/KINGSTONE_SSD`), never on the host's internal
> disk. Do not start a Docker engine whose storage is anywhere else.

On macOS the data-root (`/var/lib/docker`, including the BuildKit cache) lives
inside the container VM's disk image, so the rule means the **VM disk goes on
the SSD**.

**Colima (recommended).** `.envrc` exports:

| Variable | Value | Holds |
| --- | --- | --- |
| `COLIMA_HOME` | `/Volumes/KINGSTONE_SSD/.colima` | VM disk images (data-root and build cache), Lima state |
| `BUILDX_CONFIG` | `./.cache/buildx` | buildx builder state |
| `DOCKER_HOST` | `unix://$COLIMA_HOME/default/docker.sock` | set only once that socket exists |

```sh
cd /Volumes/KINGSTONE_SSD/Projects/ai-inference-gateway   # direnv loads the variables
colima start --cpu 4 --memory 8 --disk 60                 # VM is created under $COLIMA_HOME
direnv reload                                             # picks up DOCKER_HOST
```

**Docker Desktop (alternative).** Go to Settings → Resources → Advanced →
*Disk image location*, choose `/Volumes/KINGSTONE_SSD/DockerDesktop`, then
*Apply & restart*. This moves the whole VM disk, including data-root and build
cache.

**Verify before building:**

```sh
echo "$COLIMA_HOME"                      # /Volumes/KINGSTONE_SSD/.colima
du -sh "$COLIMA_HOME"                    # the VM disk grows here
du -sh ~/.colima 2>/dev/null             # must be absent or empty
docker buildx du                         # build cache (inside the SSD-backed VM)
```

Never export a build cache outside the repo's `.cache/`. Use
`--cache-to type=local,dest=./.cache/buildx-cache` if you need a local cache.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull
request targeting `main`. Superseded runs are cancelled.

| Job | What it enforces |
| --- | --- |
| `attribution-audit` | `scripts/attribution.sh audit` over the pushed or PR commit range (full history on first push or force-push) |
| `quality-gates` | `ruff check`, `ruff format --check`, `mypy --strict src/ tests/`, `bandit`, `pytest` with ≥ 90% branch coverage (Python 3.13, locked deps via uv) |
| `container-scan` | builds the `runner` image (buildx + GHA cache) and fails on fixable CRITICAL/HIGH CVEs (Trivy) |
| `publish` | `main` pushes only: pushes the exact scanned image to `ghcr.io/<owner>/ai-inference-gateway` as `latest` and `<sha>` |

**Merge policy.** Merging through the GitHub UI or `gh pr merge` (merge,
squash or rebase) makes `GitHub <noreply@github.com>` the committer, and the
attribution audit rejects it. Merge locally instead and push:

```sh
git switch main && git pull --ff-only
git merge --ff-only <branch>   # rebase the branch onto main first if needed
git push
```

**Action pinning policy.** Every action is pinned to a full commit SHA, with
its release tag in a comment. A new pin must be:

- resolved from the release tag;
- GitHub-verified and reachable from the action's default branch;
- at least 7 days old;
- free of open GitHub security advisories.

Trivy is pinned explicitly, both the action (`>= 0.35.0`, after
GHSA-69fq-xp46-6x23) and the scanner binary (`>= 0.72.0`, after GHSA-mcj4, -q3fv
and -8rc5). Before committing workflow changes, run them through `zizmor` and
`actionlint`:

```sh
uvx zizmor --offline --persona=auditor .github/workflows/
uvx --with shellcheck-py --from actionlint-py actionlint
```

## Quality gates

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run bandit -c pyproject.toml -r src
uv run pytest --cov --cov-report=term-missing   # >= 90% branch coverage, no live API calls
```

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), atomic and
logical: `<type>(<scope>)?: <description>` where type is one of
`feat fix docs style refactor perf test build ci chore revert`.

To audit history manually:

```sh
scripts/attribution.sh audit HEAD          # whole branch
scripts/attribution.sh audit origin/main..HEAD
```

If a commit violates the invariant, rewrite it (`git commit --amend
--reset-author`, or an interactive rebase) before pushing.
