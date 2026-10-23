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
