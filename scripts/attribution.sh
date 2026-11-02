#!/usr/bin/env bash
# Single source of truth for the sole-contributor identity invariant: every commit
# is authored and committed by one canonical identity. Message trailers are not
# restricted (AI-assistance Co-Authored-By trailers are allowed).
# Used by .githooks/{pre-commit,commit-msg,pre-push} and by CI.
#
# Usage:
#   scripts/attribution.sh identity             # check pending author + committer ident
#   scripts/attribution.sh audit <rev-range>... # audit existing commits (git rev-list args)
#
# Bash 3.2 compatible (macOS default shell).
set -euo pipefail

readonly ALLOWED_NAME="mikeZ98"
readonly ALLOWED_EMAIL="145975972+mikeZ98@users.noreply.github.com"

red() { printf '\033[31m%s\033[0m\n' "$*" >&2; }

# Reports a violation; callers record failure so every violation gets listed.
fail() {
  red "attribution: $*"
}

# check_ident <role> <name> <email>
check_ident() {
  local role="$1" name="$2" email="$3"
  if [ "$name" != "$ALLOWED_NAME" ] || [ "$email" != "$ALLOWED_EMAIL" ]; then
    fail "${role} identity '${name} <${email}>' is not the sole allowed contributor '${ALLOWED_NAME} <${ALLOWED_EMAIL}>'"
    return 1
  fi
}

# split_ident "<Name> <email> <ts> <tz>" -> sets IDENT_NAME / IDENT_EMAIL
split_ident() {
  local ident="$1"
  IDENT_NAME="${ident%% <*}"
  IDENT_EMAIL="${ident#*<}"
  IDENT_EMAIL="${IDENT_EMAIL%%>*}"
}

cmd_identity() {
  local rc=0
  split_ident "$(git var GIT_AUTHOR_IDENT)"
  check_ident "author" "$IDENT_NAME" "$IDENT_EMAIL" || rc=1
  split_ident "$(git var GIT_COMMITTER_IDENT)"
  check_ident "committer" "$IDENT_NAME" "$IDENT_EMAIL" || rc=1
  if [ "$rc" -ne 0 ]; then
    red "attribution: fix with: git config user.name '${ALLOWED_NAME}' && git config user.email '${ALLOWED_EMAIL}'"
  fi
  return "$rc"
}

cmd_audit() {
  [ "$#" -gt 0 ] || { red "usage: attribution.sh audit <rev-list args>"; return 2; }
  local rc=0 sha an ae cn ce count=0
  # Unit separator (0x1f) is safe: it cannot appear in idents.
  while IFS=$'\x1f' read -r sha an ae cn ce; do
    [ -n "$sha" ] || continue
    count=$((count + 1))
    check_ident "author of ${sha:0:12}" "$an" "$ae" || rc=1
    check_ident "committer of ${sha:0:12}" "$cn" "$ce" || rc=1
  done < <(git log --format='%H%x1f%an%x1f%ae%x1f%cn%x1f%ce' "$@")
  if [ "$rc" -eq 0 ]; then
    printf 'attribution: %d commit(s) audited, all clean\n' "$count" >&2
  fi
  return "$rc"
}

main() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    identity) cmd_identity ;;
    audit) cmd_audit "$@" ;;
    *) red "usage: $0 {identity|audit <rev-range>...}"; return 2 ;;
  esac
}

main "$@"
