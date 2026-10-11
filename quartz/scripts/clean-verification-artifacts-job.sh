#!/bin/sh
set -u

if [ "$#" -lt 3 ]; then
  printf 'Usage: %s REPOSITORY PNPM LOG_DIRECTORY [cleanup options]\n' "$0" >&2
  exit 64
fi

repository=$1
pnpm=$2
log_directory=$3
shift 3

if output=$( (cd "$repository" && "$pnpm" artifacts:clean "$@") 2>&1); then
  exit 0
else
  cleanup_status=$?
fi

failure_log="$log_directory/failure.log"
if mkdir -p "$log_directory" && {
  /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf 'repository: %s\ncommand: %s artifacts:clean\nexit status: %s\n\n%s\n' \
    "$repository" "$pnpm" "$cleanup_status" "$output"
} > "$failure_log"; then
  if [ "${GARDEN_ARTIFACT_CLEANUP_NOTIFY:-1}" != '0' ]; then
    if ! /usr/bin/osascript - "$cleanup_status" "$failure_log" <<'APPLESCRIPT' >/dev/null 2>>"$failure_log"
on run argv
  display notification ("Cleanup failed (exit " & item 1 of argv & "). Details: " & item 2 of argv) with title "Garden artifact cleanup failed"
end run
APPLESCRIPT
    then
      printf '\nThe macOS notification command failed.\n' >> "$failure_log"
    fi
  fi
else
  printf 'Garden artifact cleanup failed (exit %s); could not write %s.\n%s\n' \
    "$cleanup_status" "$failure_log" "$output" >&2
fi

exit "$cleanup_status"
