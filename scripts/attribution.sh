#!/usr/bin/env bash
# Single source of truth for the sole-contributor / attribution invariant.
# Used by .githooks/{pre-commit,commit-msg,pre-push} and by CI.
#
# Usage:
#   scripts/attribution.sh identity             # check pending author + committer ident
#   scripts/attribution.sh message <file>       # check a commit message file
#   scripts/attribution.sh audit <rev-range>... # audit existing commits (git rev-list args)
#
# Bash 3.2 compatible (macOS default shell).
set -euo pipefail

readonly ALLOWED_NAME="mikeZ98"
readonly ALLOWED_EMAIL="145975972+mikeZ98@users.noreply.github.com"

# Vendor / tool names that must never appear in trailers or signature lines.
readonly AI_VENDORS='claude|anthropic|chatgpt|openai|gpt-?[0-9]|copilot|gemini|bard|codeium|cursor|tabnine|aider|devin|llm'

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

# Strip git comment lines and everything below the scissors line (commit -v).
clean_message() {
  sed -e '/^# -\{24\} >8 -\{24\}$/,$d' -e '/^#/d'
}

# check_message_text <label>  (message on stdin)
check_message_text() {
  local label="$1" msg body rc=0 emails email
  msg="$(clean_message)"
  body="$(printf '%s\n' "$msg" | sed '1d')"

  # 1. Co-authorship of any kind violates the sole-contributor invariant.
  if printf '%s\n' "$msg" | grep -qiE '^[[:space:]]*co-?authored-?by[[:space:]]*:'; then
    fail "${label}: 'Co-Authored-By' trailers are forbidden"; rc=1
  fi

  # 2. Generation / assistance trailers (Generated-by:, Assisted-by:, AI-Assisted:, ...).
  if printf '%s\n' "$msg" | grep -qiE '^[[:space:]]*((generated|assisted|created|written|authored|produced|made|powered|suggested)[-_ ]?(by|with|using)|ai[-_][a-z-]+)[[:space:]]*:'; then
    fail "${label}: generation/assistance trailers are forbidden"; rc=1
  fi

  # 3. Any git-style trailer (Hyphenated-Key: value) in the body naming an AI vendor.
  if printf '%s\n' "$body" | grep -iE '^[[:space:]]*[a-z]+(-[a-z]+)+[[:space:]]*:' | grep -qiE "$AI_VENDORS"; then
    fail "${label}: trailers referencing AI tools/vendors are forbidden"; rc=1
  fi

  # 4. Signature lines: "Generated with Claude Code", "written by ChatGPT", robot emoji, vendor URLs.
  if printf '%s\n' "$msg" | grep -qiE "(generated|created|written|authored|produced|assisted|made|powered|drafted)[[:space:]]+(entirely[[:space:]]+|partially[[:space:]]+)?(by|with|using|via)[[:space:]]+(an?[[:space:]]+)?(\[)?(${AI_VENDORS}|ai[[:space:]]|ai$)"; then
    fail "${label}: AI generation signatures are forbidden"; rc=1
  fi
  if printf '%s\n' "$msg" | grep -qiE 'claude[[:space:]]*code|claude\.(ai|com)|anthropic\.com|openai\.com|chatgpt'; then
    fail "${label}: AI tool references/URLs are forbidden"; rc=1
  fi
  if printf '%s\n' "$msg" | grep -qF "$(printf '\360\237\244\226')"; then
    fail "${label}: robot-emoji signatures are forbidden"; rc=1
  fi

  # 5. Signed-off-by is only permitted for the allowed identity.
  if printf '%s\n' "$msg" | grep -iE '^[[:space:]]*signed-off-by[[:space:]]*:' \
      | grep -vqF "Signed-off-by: ${ALLOWED_NAME} <${ALLOWED_EMAIL}>"; then
    fail "${label}: Signed-off-by is only allowed as '${ALLOWED_NAME} <${ALLOWED_EMAIL}>'"; rc=1
  fi

  # 6. No foreign e-mail addresses anywhere in the message.
  emails="$(printf '%s\n' "$msg" | grep -oE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' || true)"
  for email in $emails; do
    if [ "$email" != "$ALLOWED_EMAIL" ]; then
      fail "${label}: foreign e-mail address '${email}' in message"; rc=1
    fi
  done

  return "$rc"
}

cmd_message() {
  local file="${1:?usage: attribution.sh message <file>}"
  check_message_text "commit message" < "$file"
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
    git log -1 --format=%B "$sha" | check_message_text "message of ${sha:0:12}" || rc=1
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
    message) cmd_message "$@" ;;
    audit) cmd_audit "$@" ;;
    *) red "usage: $0 {identity|message <file>|audit <rev-range>...}"; return 2 ;;
  esac
}

main "$@"
