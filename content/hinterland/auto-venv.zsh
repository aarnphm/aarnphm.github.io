#!/usr/bin/env zsh
# Auto-activate Python virtual environments when entering directories
# Add to ~/.zshrc: source /path/to/auto-venv.zsh

# Track currently activated venv to avoid redundant operations
typeset -g _VENV_ACTIVE_PATH=""

_auto_venv_activate() {
  # Find .venv in current dir or parents
  local venv_path=""
  local search_dir="$PWD"

  while [[ "$search_dir" != "/" ]]; do
    if [[ -f "$search_dir/.venv/bin/activate" ]]; then
      venv_path="$search_dir/.venv"
      break
    fi
    search_dir="${search_dir:h}"
  done

  # Same venv already active, do nothing
  [[ "$venv_path" == "$_VENV_ACTIVE_PATH" ]] && return

  # Deactivate current venv if switching or leaving
  if [[ -n "$_VENV_ACTIVE_PATH" ]] && (( $+functions[deactivate] )); then
    deactivate
    _VENV_ACTIVE_PATH=""
  fi

  # Activate new venv if found
  if [[ -n "$venv_path" ]]; then
    source "$venv_path/bin/activate"
    _VENV_ACTIVE_PATH="$venv_path"
  fi
}

# Hook into directory changes
autoload -U add-zsh-hook
add-zsh-hook chpwd _auto_venv_activate

# Activate on shell startup
_auto_venv_activate
