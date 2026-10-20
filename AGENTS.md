# aarnphm's garden

This repository powers a Quartz digital garden with custom plugins and a Cloudflare worker. Other languages and independent projects live under `content/`.

## Implementation

- Fix the owning boundary with the simplest implementation that meets current requirements. Reuse established helpers and dependencies before introducing abstractions or packages.
- Remove obsolete code when its callers no longer need it. Preserve required provider recovery, external API contracts, and persisted data. Add compatibility or migration code only for an identified current requirement.
- Check `docs` for any additional notes on design patterns as well as relevant information. If possible, always use 80% of ASD-STE100 for description et al.
- Place course files by the routing in [courses directory](docs/courses.md): vendored files come from `pnpm vendor:ocw`; authored files live in `lectures/`, `psets/` and `statements/`.
- Keep components modular. Use Preact and the existing SCSS tokens for Quartz UI. NO `box-shadow` and decorative `border-left` styling, unless specified so.
- Use the `garden-design-system` skill for any UI, chart, figure or mock, and for UI review. It holds the tokens, the reading and tool registers, layout modes, component contracts and the review checklist. Its `references/` are generated from the Claude-app design system; regenerate them after that changes, and never edit them by hand.
- Keep utility filenames kebab-case. Share a helper in `quartz/util` when multiple owners need it; keep a script-specific helper with its script.
- Use comments for non-obvious constraints or decisions; omit comments that repeat the code.
- Keep filesystem access out of Quartz transformers. Emitters own filesystem output.
- Register browser listeners within `document.addEventListener('nav', () => { ... })` and register their `window.addCleanup` callbacks inside that handler. Keep transient state scoped to the page; persist only state that should survive navigation or reloads.
- Use native buttons for actions and links for navigation. Preserve keyboard operation and visible focus when styling controls.
- Bookcase keyboard focus uses fill and text changes, without outline rings on its controls, viewport label or preview dialog. Category initials select and focus the matching topic from the bookcase; repeated initials cycle matching categories. Keep typing in search fields and preview navigation independent.
- Keep secrets in local `.env` files or Cloudflare Secrets. Never add secret values to tracked configuration.
- Use LaTeX math syntax in Markdown and preserve existing note paths and source citations.

## Workflow and checks

- Inspect scoped Git state and the existing watcher before editing. Preserve staging, unrelated work, and user-owned processes.
- Use `pnpm`, `oxlint`, `oxfmt`, and `tsgo` for the Quartz project. Nested projects may define their own package manager and build checks in a closer AGENTS.md.
- The pre-commit hook (`harness/token-budget/check.py`, a uv script) rejects staged files over 5000 tokens (bytes / 4). Split the file as its message describes; do not bypass the hook or mark authored code exempt in `.gitattributes`. A fresh clone installs it as `.git/hooks/pre-commit` containing `exec "$(git rev-parse --show-toplevel)/harness/token-budget/check.py"`. The [token-budget skill](.agents/skills/token-budget/SKILL.md) runs the refactor harness that splits over-budget files into deep modules.
- Routine targeted local checks are authorized with the task. See [development commands](docs/agent-development.md) for command effects, and read the current package script before an unfamiliar invocation.
- Use the running `quartz/scripts/dev.ts` watcher for browser evidence. Do not launch a full Quartz bundle/build or restart an existing process just to verify an edit. Wait for the matching new `build:ready`, confirm HTTP availability, and inspect the rendered page. Report the actual blocker if that path fails.
- Save browser evidence and generated verification fixtures under `docs/verification/<task>/artifacts/`, with the command, inputs, and results. Generated files expire 72 hours after their last modification through `pnpm artifacts:clean`, scheduled daily by a local macOS LaunchAgent. Keep replay scripts and durable task documentation outside `artifacts/`. Reserve `.quartz-cache` for operational caches, authentication, and sync state; preserve existing Claude design-system files there.
- You can use both Opus/Fable and Argon/Gemini via `claude` (or `argon --model <model-id>`) and `agy` respectively. Feel free to strategize with Argon for quality controls and performance-based tasks, as well as Fable/Opus for design inspiration and creative direction. Especially whenever we deal with design pattern, we definitely want to use Claude's inputs on it. If you need to use Claude, make sure to use all of the credits we have via API with the envvar `CLAUDE_API_KEY` here before going into our quota.
- Keep provider identity and per-field provenance explicit. A computed estimate is distinct from a native measurement. A verified replacement and separate deletion approval are required before removing source activities.
- Preserve existing flashcard text and identities unless the requested correction requires a change; edits can affect scheduling.

## Decision checkpoints

- You also have access to the Skill(decision) when a worker handoff, completion claim, or proposed follow-up leaves the next action uncertain. Use `decisions.decide` through MCP or `agent-decide` through the CLI. Routine steps with a clear outcome can proceed directly.
- Supply explicit acceptance checks and fresh evidence, distinguishing observed results from worker claims. For proposed items, state the requirement or gap each item addresses and the evidence needed to resolve it; keep suggestions distinct from authorized work.
- Run deterministic checks first. Use the optional Composer assessor only for unresolved checks with evidence to interpret, following the skill's subscription-only billing requirements. Treat the result as advisory: the coordinating agent performs the next action within the existing task scope. A recommendation does not grant permission or establish that supplied evidence is current.

## Testing

- NEVER write unit tests after you write code.
- Highly prefer E2E tests as the sole testing mechanism. Use them to verify that complex features work through their real entry points and observable outcomes.
- At the end of E2E tests, produce a verifiable and repeatable artifact. Record the exact command, inputs or fixtures, and results alongside the relevant trace, screenshot, response, or generated output so another person can inspect the evidence and repeat the run.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code. Keep an isolated test only when it catches a concrete failure that the E2E tests miss.
- Do not add tests that mirror the implementation, copy constants, assert mocked call plumbing, or regex source text to prove an edit. Assert behavior at the owning system boundary.
- Run the relevant existing tests for a change. Preserve distinct regression coverage when pruning tests, identify the surviving coverage for deleted duplicates, and report any uncovered failure modes or baseline failures.
