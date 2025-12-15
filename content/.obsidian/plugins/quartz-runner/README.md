# obsidian quartz runner

starts `tsx quartz/scripts/dev.ts --retry <retryLimit> --bg` at the git root, so quartz runs outside obsidian’s process and writes:

- `.dev.pid`
- `.dev.log`
- `.dev.err.log`

## commands

- `quartz: stop dev server`
- `quartz: start dev server`
- `quartz: follow dev logs` (opens a new terminal and tails `.dev.log` + `.dev.err.log`)
- `quartz: start dev server and follow logs`

## settings

- `retry limit`: forwarded to `quartz/scripts/dev.ts --retry`
- `tail lines`: forwarded to `tail -n <tailLines> -F`
