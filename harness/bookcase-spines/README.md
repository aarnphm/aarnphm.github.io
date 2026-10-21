# Bookcase spine harness

Python coordinates a resumable queue. `rsvg-convert` supplies the [Rust SVG renderer](https://gnome.pages.gitlab.gnome.org/librsvg/devel-docs/architecture.html);
`cwebp` and `dwebp` encode and validate the raster outputs. The coordinator uses the
Python standard library and adds no package dependency or Node entry point.
Optional model CLIs retain their own installed runtimes.

Every visible region, including lettering, photography and texture, must be vector
geometry in the SVG master. No bitmap, external resource or live font is allowed.
The browser-facing exports keep the current shelf (96 × 320, q80) and detail
(256 × 1280, q84) ceilings and never exceed reference dimensions.

| Role                       | Default                                           |
| -------------------------- | ------------------------------------------------- |
| Trace and repair           | Gemini 4 Argon High                               |
| Independent visual review  | GPT-6 Astra and Claude Opus 5.5, both high effort |
| Additional required review | Caw, installed Cursor model, ask mode             |
| Coordinator                | One Python process; sequential model calls        |

The defaults live in [config.json](config.json). Astra, Opus and Caw must each pass
the exact candidate. The implementation provider cannot review its own trace.
Sol Medium remains selectable: in a separate config set `trace` to
`{"provider":"sol","model":"gpt-6.1-sol","effort":"medium"}`, then pass that
file to `plan --config`. `prepare` follows the planned role when `--provider` is
omitted. No provider substitution or paid usage override happens on account errors.
Version 2 policy requires a fresh plan; earlier run directories remain historical
evidence and are never rewritten to satisfy the new reviewer requirements.

## Invoke

From the repository root:

```sh
bash harness/bookcase-spines/scripts/run.sh doctor
bash harness/bookcase-spines/scripts/run.sh plan --run spines-v1
bash harness/bookcase-spines/scripts/run.sh status --run spines-v1
bash harness/bookcase-spines/scripts/run.sh run --run spines-v1 --dry-run
bash harness/bookcase-spines/scripts/run.sh run --run spines-v1 --max-jobs 1
bash harness/bookcase-spines/scripts/run.sh resume --run spines-v1 --until-done
bash harness/bookcase-spines/scripts/run.sh review --run spines-v1 --job ID --provider opus
bash harness/bookcase-spines/scripts/run.sh export --run spines-v1
```

`run.sh` uses `uv run --no-project --offline python`. An installed Python 3.12+
is sufficient; no project environment sync or package install occurs. Run IDs
resolve under `.quartz-cache/bookcase-spines/`. An explicit path selects a different
root. Existing run directories cannot be replaced by `plan`.

`--until-done` drains runnable work within a four-minute invocation and a 24-call
ceiling. Each model call has a maximum 180-second deadline. All owned subprocesses
share the invocation deadline. Each job gets at most three trace attempts.
Successes and review receipts survive explicit `resume`; unresolved source gaps,
review ambiguity and exhausted attempts require attention. Repeated invocation
never implies an unattended scheduler. Inspect `stopReason`, `counts` and
`complete`; a queue containing blocked jobs is incomplete.

`resume` retries transport failures. It retains passing reviews for the same
candidate. A repair replaces the candidate only after deterministic checks pass,
then clears reviews of the old candidate. A failed replacement leaves those bytes
and their evidence intact. A changed front/reference, catalogue, ledger,
configuration, implementation or prompt requires a fresh plan.

## Admission and export

The default plan reads `quartz/util/base/bookcase/jackets.json` and the current
handoff at `docs/verification/bookcase-spine-art/research/round-22/open-ledger.json`.
That ledger path is stable across sourcing rounds. Counts must agree, and every
available front/reference hash is checked. References are the already admitted
detail artwork. Original source URLs/hashes and edition evidence remain provenance;
the harness neither fetches new sources nor changes edition decisions or fronts.

A custom `--input input.json` accepts `{ "version": 1, "jobs": [...] }`. Each job
has `id`, `title`, `notes`, `source`, `reference`, `front`, and `blockedReason`.
`source` is `{ "url", "sha256", "evidence" }`; file pins are `{ "file", "sha256" }`.
The last four fields may be null, causing an explicit source block when admission
is incomplete. Relative file paths resolve beside the input file. Repeated shelf
placements can share one job's `notes`; duplicate note assignments are rejected.
Default catalogue identities use the pinned poster keys, while unresolved ledger
entries retain their recorded note paths.

The SVG validator rejects unsupported XML, resources, styles, fonts, image nodes,
cycles and excess size/depth/coordinates. Native rasterization is bounded, both
WebPs are decoded, and `verify --job ID` rerenders and compares exact hashes.
These checks establish format and reproducibility. Fresh independent reviewers
inspect source and both raster tiers to establish visual fidelity.

`export` requires passing receipts from every required provider and revalidates
all accepted jobs. It writes a content-addressed bundle and manifest under the run.
It never publishes to `jackets.json` or modifies the browser renderer. Partial
exports state their numerator, denominator and incomplete status. Canvas drawing
and frame-time acceptance remain a separate experiment in the
[renderer design](../../docs/verification/bookcase-spine-harness/renderer.md).

## Evidence and ownership

State is replaced atomically, with a journal in `events.jsonl`. One coordinator
holds an OS file lock and ownership marker. After a crash, `recover --run ID`
refuses a live owner or unresolved provider marker; inspect those processes before
removing any stale active marker. It does not kill an unrelated process.

Raw provider assignments and capped logs live in
`work/{codex,claude,argon,cursor}/bookcase-spines/INVOCATION/`. Request, transport and
review receipts stay in the run. Argon and Caw responses are inspected with their
skills' native inspection helpers. A zero exit without a complete terminal result
cannot satisfy review. Model usage is preserved when reported.

Sol and Astra use read-only sandbox mode with user config disabled. Caw uses ask mode,
without `--force` or blanket MCP approval. Argon receives a read-only assignment;
that prompt does not create an OS sandbox. All providers receive explicit instructions
to avoid browsers, edits and further delegation. Fresh invocation directories and
bounded process groups separate their output. Account access is checked only by a
live call; `doctor` checks executable/helper availability.

Opus uses the installed Claude CLI in bare, restricted mode, with only the `Read`
tool and no MCP servers, shell or edits. Its adapter requires `CLAUDE_API_KEY`
from the process environment or repository `.env`, maps it to the child process's
`ANTHROPIC_API_KEY`, and never serializes credentials into requests or receipts.
An optional `CLAUDE_WORKSPACE_ID` (or `ANTHROPIC_WORKSPACE_ID`) sets the
`anthropic-workspace-id` header for an unscoped API key. Missing credentials or API
errors stop the review without subscription fallback. Native completion, the
observed model and successful reads of all three images are checked before its
review can count.

An organization-scoped key still targets one workspace per model request. To find
the Default workspace, the read-only List Workspaces request needs
`include_default=true`; the default is omitted otherwise. See Anthropic's
[workspace scoping](https://platform.claude.com/docs/en/manage-claude/workspaces)
and [Admin API authentication](https://platform.claude.com/docs/en/manage-claude/admin-api).

Use durable reference paths for long runs. Expiring verification fixtures are
appropriate for smoke tests; deleting or changing a pinned input makes its run
stale. This runner does not hydrate missing original source archives.

## Check and measure

```sh
bash harness/bookcase-spines/scripts/run.sh test
uv run --no-project --offline python harness/bookcase-spines/scripts/benchmark.py --samples 5
.venv/bin/ruff check harness/bookcase-spines
.venv/bin/ruff format --check harness/bookcase-spines
MYPYPATH=harness/bookcase-spines .venv/bin/mypy --follow-imports=silent --cache-dir docs/verification/bookcase-spine-harness/artifacts/mypy harness/bookcase-spines/*.py
```

E2E exercises use real subprocesses, PNG/WebP bytes, SVG rendering, filesystem
state and the public CLI. They do not invoke models. Exact commands/results and
native process records are saved under
`docs/verification/bookcase-spine-harness/artifacts/`. The benchmark measures a
100-path geometric fixture; it makes no inference about model latency or browser
frame rate. See [FAILURES.md](FAILURES.md) for the planned failure inventory and
the [verification report](../../docs/verification/bookcase-spine-harness/review-status.md)
for executed coverage and limits.
