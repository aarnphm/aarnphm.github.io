#!/usr/bin/env bash
# agents.sh - install aarnphm's claude code & codex agent configuration
# usage: curl -fsSL https://aarnphm.xyz/agents.sh | bash
set -euo pipefail

REPO_SSH="git@github.com:aarnphm/agents.git"
REPO_HTTPS="https://github.com/aarnphm/agents.git"
AGENTS_DIR="${HOME}/workspace/agents"
CLAUDE_DIR="${HOME}/.claude"
CODEX_DIR="${HOME}/.codex"
SYNC_ONLY=false

info() { printf '\033[0;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
err() {
  printf '\033[0;31m[-]\033[0m %s\n' "$*" >&2
  exit 1
}

for arg in "$@"; do
  case "$arg" in
  --sync-only) SYNC_ONLY=true ;;
  --help | -h)
    echo "usage: agents.sh [--sync-only]"
    echo "  --sync-only  skip clone/pull, just distribute files"
    exit 0
    ;;
  *) warn "unknown flag: $arg" ;;
  esac
done

command -v git >/dev/null 2>&1 || err "git is required"
command -v rsync >/dev/null 2>&1 || err "rsync is required"

if [ "$SYNC_ONLY" = false ]; then
  if [ -d "$AGENTS_DIR/.git" ]; then
    info "pulling latest in $AGENTS_DIR"
    if ! git -C "$AGENTS_DIR" pull --ff-only 2>/dev/null; then
      warn "pull --ff-only failed (diverged history?), trying rebase"
      git -C "$AGENTS_DIR" pull --rebase || err "pull failed, resolve manually in $AGENTS_DIR"
    fi
  else
    if [ -d "$AGENTS_DIR" ]; then
      warn "backing up existing $AGENTS_DIR"
      mv "$AGENTS_DIR" "${AGENTS_DIR}.bak.$(date +%Y%m%d%H%M%S)"
    fi
    mkdir -p "$(dirname "$AGENTS_DIR")"
    info "cloning agents repo to $AGENTS_DIR"
    if ssh -o BatchMode=yes -o ConnectTimeout=5 -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
      git clone "$REPO_SSH" "$AGENTS_DIR"
    else
      info "ssh auth not available, falling back to https"
      git clone "$REPO_HTTPS" "$AGENTS_DIR"
    fi
  fi
fi

[ -d "$AGENTS_DIR/.git" ] || err "$AGENTS_DIR is not a git repo, run without --sync-only first"

sync_back() {
  local source="$1"
  shift
  for path in "$@"; do
    if [ -d "$source/$path" ] && [ -d "$AGENTS_DIR/$path" ]; then
      rsync -a --update "$source/$path/" "$AGENTS_DIR/$path/"
    elif [ -f "$source/$path" ]; then
      if [ ! -f "$AGENTS_DIR/$path" ] || [ "$source/$path" -nt "$AGENTS_DIR/$path" ]; then
        mkdir -p "$(dirname "$AGENTS_DIR/$path")"
        cp -f "$source/$path" "$AGENTS_DIR/$path"
      fi
    fi
  done
}

sync_agents_back() {
  local source_dir="$1"
  local ext="$2"
  [ -d "$source_dir" ] || return 0
  for f in "$source_dir"/*."$ext"; do
    [ -f "$f" ] || continue
    local base="$(basename "$f")"
    if [ ! -f "$AGENTS_DIR/agents/$base" ] || [ "$f" -nt "$AGENTS_DIR/agents/$base" ]; then
      cp -f "$f" "$AGENTS_DIR/agents/$base"
    fi
  done
}

sync_files() {
  local target="$1"
  shift
  for path in "$@"; do
    if [ -d "$AGENTS_DIR/$path" ]; then
      rsync -a --delete "$AGENTS_DIR/$path/" "$target/$path/"
    elif [ -f "$AGENTS_DIR/$path" ]; then
      mkdir -p "$(dirname "$target/$path")"
      cp -f "$AGENTS_DIR/$path" "$target/$path"
    fi
  done
}

sync_agents() {
  local target_dir="$1"
  local ext="$2"
  mkdir -p "$target_dir"
  rm -f "$target_dir"/*.md "$target_dir"/*.toml 2>/dev/null || true
  for f in "$AGENTS_DIR/agents"/*."$ext"; do
    [ -f "$f" ] && cp -f "$f" "$target_dir/"
  done
}

info "checking for newer files in $CLAUDE_DIR"
sync_back "$CLAUDE_DIR" \
  CLAUDE.md \
  settings.json \
  statusline-command.sh \
  commands \
  skills
sync_agents_back "$CLAUDE_DIR/agents" "md"

info "checking for newer files in $CODEX_DIR"
sync_back "$CODEX_DIR" \
  AGENTS.md \
  config.toml \
  skills
sync_agents_back "$CODEX_DIR/agents" "toml"

if ! git -C "$AGENTS_DIR" diff --quiet 2>/dev/null ||
  [ -n "$(git -C "$AGENTS_DIR" ls-files --others --exclude-standard 2>/dev/null)" ]; then
  info "sync-back found newer files, committing to repo"
  git -C "$AGENTS_DIR" add -A
  git -C "$AGENTS_DIR" commit -m "sync: pull newer configs from ~/.claude and ~/.codex" --no-gpg-sign 2>/dev/null || true
fi

# claude code: *.md agents, commands/, skills/
info "syncing claude code files to $CLAUDE_DIR"
mkdir -p "$CLAUDE_DIR"
sync_files "$CLAUDE_DIR" \
  CLAUDE.md \
  settings.json \
  statusline-command.sh \
  commands \
  skills
sync_agents "$CLAUDE_DIR/agents" "md"

# codex: *.toml agents, prompts/, skills/
info "syncing codex files to $CODEX_DIR"
mkdir -p "$CODEX_DIR"
sync_files "$CODEX_DIR" \
  AGENTS.md \
  config.toml \
  prompts \
  skills
sync_agents "$CODEX_DIR/agents" "toml"

HOOK_FILE="$AGENTS_DIR/.git/hooks/post-commit"
HOOK_MARKER="# managed by agents.sh"

if [ ! -f "$HOOK_FILE" ] || ! grep -q "$HOOK_MARKER" "$HOOK_FILE" 2>/dev/null; then
  info "installing post-commit hook for auto-sync"
  mkdir -p "$(dirname "$HOOK_FILE")"
  cat >"$HOOK_FILE" <<'HOOK'
#!/usr/bin/env bash
# managed by agents.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

sync_files() {
  local target="$1"; shift
  for path in "$@"; do
    if [ -d "$path" ]; then
      rsync -a --delete "$path/" "$target/$path/"
    elif [ -f "$path" ]; then
      mkdir -p "$(dirname "$target/$path")"
      cp -f "$path" "$target/$path"
    fi
  done
}

sync_agents() {
  local target_dir="$1"; local ext="$2"
  mkdir -p "$target_dir"
  rm -f "$target_dir"/*.md "$target_dir"/*.toml 2>/dev/null || true
  for f in agents/*."$ext"; do
    [ -f "$f" ] && cp -f "$f" "$target_dir/"
  done
}

sync_files "$HOME/.claude" \
  CLAUDE.md settings.json statusline-command.sh \
  commands skills
sync_agents "$HOME/.claude/agents" "md"

sync_files "$HOME/.codex" \
  AGENTS.md config.toml \
  prompts skills
sync_agents "$HOME/.codex/agents" "toml"
HOOK
  chmod +x "$HOOK_FILE"
fi

echo ""
info "done. files distributed:"
echo ""
echo "  repo:        $AGENTS_DIR"
echo "  claude code: $CLAUDE_DIR"
echo "    <- CLAUDE.md, settings.json, statusline-command.sh"
echo "    <- agents/*.md, commands/, skills/"
echo "  codex:       $CODEX_DIR"
echo "    <- AGENTS.md, config.toml"
echo "    <- agents/*.toml, prompts/, skills/"
echo ""
info "post-commit hook installed: changes auto-sync on commit"
warn "plugins install on first claude code run via enabledPlugins in settings.json"
