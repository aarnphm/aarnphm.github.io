#!/bin/sh

set -eu

script_name=${0##*/}
cargo_home=${CARGO_HOME:-$HOME/.cargo}
cargo_bin=${cargo_home}/bin
nix_profile_bin=/nix/var/nix/profiles/default/bin
daemon_bin=/nix/var/nix/profiles/default/bin/nix-daemon
rustup_installer_url=${DETACHTOOLS_RUSTUP_INSTALLER_URL:-https://sh.rustup.rs}
installer_url=${DETACHTOOLS_INSTALLER_URL:-https://install.determinate.systems/nix}
trusted_users=${DETACHTOOLS_TRUSTED_USERS:-root ubuntu}
bootstrap_ref=${DETACHTOOLS_BOOTSTRAP_REF:-github:aarnphm/detachtools/main#bootstrap}
bootstrap_system=${DETACHTOOLS_BOOTSTRAP_SYSTEM:-linux}
bootstrap_target=${DETACHTOOLS_BOOTSTRAP_TARGET:-ubuntu}
daemon_log=${DETACHTOOLS_DAEMON_LOG:-/tmp/nix-daemon.log}
self_url=${DETACHTOOLS_SELF_URL:-https://aarnphm.xyz/detachtools.sh}
fallback_self_url=${DETACHTOOLS_FALLBACK_SELF_URL:-https://raw.githubusercontent.com/aarnphm/aarnphm.github.io/main/quartz/static/detachtools.sh}
rustup_toolchain=${DETACHTOOLS_RUSTUP_TOOLCHAIN:-nightly}
rustup_profile=${DETACHTOOLS_RUSTUP_PROFILE:-complete}
cargo_path_ready=${DETACHTOOLS_CARGO_PATH_READY:-0}
cargo_path_injected=0
rustup_installer_script=
nix_installer_script=
reexec_script=

say() {
  printf '%s\n' "[$script_name] $*"
}

die() {
  say "error: $*"
  exit 1
}

cleanup() {
  if [ -n "${rustup_installer_script}" ] && [ -f "${rustup_installer_script}" ]; then
    rm -f "${rustup_installer_script}"
  fi
  if [ -n "${nix_installer_script}" ] && [ -f "${nix_installer_script}" ]; then
    rm -f "${nix_installer_script}"
  fi
  if [ -n "${reexec_script}" ] && [ -f "${reexec_script}" ]; then
    rm -f "${reexec_script}"
  fi
}

trap cleanup EXIT INT TERM

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

prepend_path() {
  case ":$PATH:" in
  *":$1:"*) ;;
  *)
    PATH="$1:$PATH"
    export PATH
    if [ "$1" = "${cargo_bin}" ]; then
      cargo_path_injected=1
    fi
    ;;
  esac
}

refresh_rust_path() {
  if command -v rustup >/dev/null 2>&1; then
    return 0
  fi

  if [ -d "${cargo_bin}" ]; then
    prepend_path "${cargo_bin}"
  fi
}

have_rustup() {
  refresh_rust_path
  command -v rustup >/dev/null 2>&1
}

verify_rustup() {
  have_rustup || die "rustup is not available"
  rustup --version >/dev/null 2>&1 || die "rustup is on path but not responding"
}

rust_toolchain_installed() {
  verify_rustup
  rustup toolchain list 2>/dev/null | grep -q "^${rustup_toolchain}"
}

install_rustup() {
  if have_rustup; then
    say "rustup already available"
    return 0
  fi

  say "installing rustup with no path mutation"
  rustup_installer_script=$(mktemp "${TMPDIR:-/tmp}/detachtools-rustup-installer.XXXXXX")
  curl --proto '=https' --tlsv1.2 -sSf "${rustup_installer_url}" -o "${rustup_installer_script}"
  sh "${rustup_installer_script}" -y --default-toolchain none --no-modify-path
  rm -f "${rustup_installer_script}"
  rustup_installer_script=
  verify_rustup
}

ensure_rust_toolchain() {
  if rust_toolchain_installed; then
    say "rust toolchain ${rustup_toolchain} already available"
  else
    say "installing rust toolchain ${rustup_toolchain} with profile ${rustup_profile}"
    if ! rustup toolchain install "${rustup_toolchain}" --profile "${rustup_profile}"; then
      die "failed to install rust toolchain ${rustup_toolchain} with profile ${rustup_profile}; rustup docs warn that profile complete almost always fails"
    fi
  fi

  say "setting default rust toolchain to ${rustup_toolchain}"
  rustup default "${rustup_toolchain}" >/dev/null 2>&1 || die "failed to set default rust toolchain"
  rustup run "${rustup_toolchain}" rustc --version >/dev/null 2>&1 || die "rust toolchain is not usable"
}

script_can_reexec_from_file() {
  case "$0" in
  sh | -sh | bash | -bash | dash | -dash | zsh | -zsh | /bin/sh | /bin/bash | /bin/dash | /bin/zsh)
    return 1
    ;;
  esac

  [ -f "$0" ]
}

reexec_with_cargo_path() {
  refresh_rust_path
  if [ "${cargo_path_ready}" = "1" ]; then
    return 0
  fi

  [ "${cargo_path_injected}" = "1" ] || return 0

  say "restarting script with rustup path in process env"
  if script_can_reexec_from_file; then
    exec env DETACHTOOLS_CARGO_PATH_READY=1 PATH="${cargo_bin}:$PATH" "$0" "$@"
  fi

  reexec_script=$(mktemp "${TMPDIR:-/tmp}/detachtools-reexec.XXXXXX")
  if curl --proto '=https' --tlsv1.2 -sSf "${self_url}" -o "${reexec_script}"; then
    :
  elif curl --proto '=https' --tlsv1.2 -sSf "${fallback_self_url}" -o "${reexec_script}"; then
    say "primary self url failed, fell back to raw github for re-exec"
  else
    die "failed to fetch script for re-exec from both ${self_url} and ${fallback_self_url}"
  fi

  chmod +x "${reexec_script}"
  exec env DETACHTOOLS_CARGO_PATH_READY=1 PATH="${cargo_bin}:$PATH" \
    DETACHTOOLS_SELF_URL="${self_url}" \
    DETACHTOOLS_FALLBACK_SELF_URL="${fallback_self_url}" \
    "${reexec_script}" "$@"
}

refresh_nix_path() {
  if command -v nix >/dev/null 2>&1; then
    return 0
  fi

  if [ -d "${nix_profile_bin}" ]; then
    PATH="${nix_profile_bin}:$PATH"
    export PATH
  fi

  if command -v nix >/dev/null 2>&1; then
    return 0
  fi

  if [ -r /etc/profile.d/nix.sh ]; then
    . /etc/profile.d/nix.sh
  fi

  if command -v nix >/dev/null 2>&1; then
    return 0
  fi

  if [ -r /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  fi
}

have_nix() {
  refresh_nix_path
  command -v nix >/dev/null 2>&1
}

verify_nix() {
  have_nix || die "nix is not available"
  nix --version >/dev/null 2>&1 || die "nix is on path but not responding"
}

daemon_running() {
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -x nix-daemon >/dev/null 2>&1 && return 0
  fi

  ps -eo comm= 2>/dev/null | grep -x 'nix-daemon' >/dev/null 2>&1
}

wait_for_daemon() {
  attempt=1
  while [ "${attempt}" -le 15 ]; do
    if daemon_running; then
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  die "nix-daemon did not start successfully"
}

install_nix() {
  if have_nix; then
    say "nix already available"
    return 0
  fi

  say "installing nix via determinate systems"
  nix_installer_script=$(mktemp "${TMPDIR:-/tmp}/detachtools-nix-installer.XXXXXX")
  curl -fsSL "${installer_url}" -o "${nix_installer_script}"
  sh "${nix_installer_script}" install linux --init none --no-confirm --extra-conf "trusted-users = ${trusted_users}"
  rm -f "${nix_installer_script}"
  nix_installer_script=
  verify_nix
}

start_daemon() {
  if daemon_running; then
    say "nix-daemon already running"
    return 0
  fi

  [ -x "${daemon_bin}" ] || die "expected nix-daemon at ${daemon_bin}"
  say "starting nix-daemon"
  sudo env DAEMON_BIN="${daemon_bin}" DAEMON_LOG="${daemon_log}" sh -c 'nohup "$DAEMON_BIN" >>"$DAEMON_LOG" 2>&1 </dev/null &'
  wait_for_daemon
}

run_bootstrap() {
  say "running detachtools bootstrap for ${bootstrap_system} ${bootstrap_target}"
  nix run "${bootstrap_ref}" -- "${bootstrap_system}" "${bootstrap_target}"
}

main() {
  require_command curl
  require_command sudo

  [ "$(uname -s)" = "Linux" ] || die "this script only supports linux"

  install_rustup
  verify_rustup
  ensure_rust_toolchain
  reexec_with_cargo_path "$@"
  install_nix
  verify_nix
  start_daemon
  verify_nix
  daemon_running || die "nix-daemon is not running"
  run_bootstrap
  verify_nix
  daemon_running || die "nix-daemon is not running after bootstrap"
  say "rustup available at $(command -v rustup)"
  say "nix available at $(command -v nix)"
  say "nix-daemon is running"
}

main "$@"
