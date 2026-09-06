# aarnphm's garden

This repository powers a Quartz digital garden with custom plugins and a Cloudflare worker. Other languages and independent projects live under `content/`.

## Implementation

- Fix the owning boundary with the simplest implementation that meets current requirements. Reuse established helpers and dependencies before introducing abstractions or packages.
- Remove obsolete code when its callers no longer need it. Preserve required provider recovery, external API contracts, and persisted data. Add compatibility or migration code only for an identified current requirement.
- Keep components modular. Use Preact and the existing SCSS tokens for Quartz UI. Avoid `box-shadow` and decorative `border-left` styling.
- Keep utility filenames kebab-case. Share a helper in `quartz/util` when multiple owners need it; keep a script-specific helper with its script.
- Use comments for non-obvious constraints or decisions; omit comments that repeat the code.
- Keep filesystem access out of Quartz transformers. Emitters own filesystem output.
- Register browser listeners within `document.addEventListener('nav', () => { ... })` and register their `window.addCleanup` callbacks inside that handler. Keep transient state scoped to the page; persist only state that should survive navigation or reloads.
- Use native buttons for actions and links for navigation. Preserve keyboard operation and visible focus when styling controls.
- Keep secrets in local `.env` files or Cloudflare Secrets. Never add secret values to tracked configuration.
- Use LaTeX math syntax in Markdown and preserve existing note paths and source citations.

## Workflow and checks

- Inspect scoped Git state and the existing watcher before editing. Preserve staging, unrelated work, and user-owned processes.
- Use `pnpm`, `oxlint`, `oxfmt`, and `tsgo` for the Quartz project. Nested projects may define their own package manager and build checks in a closer AGENTS.md.
- Run relevant existing tests and tests for changed behavior. File fixtures and generated-output assertions are valid; tests that regex source text to prove an implementation edit are not.
- Routine targeted local checks are authorized with the task. See [development commands](docs/agent-development.md) for command effects, and read the current package script before an unfamiliar invocation.
- Use the running `quartz/scripts/dev.ts` watcher for browser evidence. Do not launch a full Quartz bundle/build or restart an existing process just to verify an edit. Wait for the matching new `build:ready`, confirm HTTP availability, and inspect the rendered page. Report the actual blocker if that path fails.
- Keep provider identity and per-field provenance explicit. A computed estimate is distinct from a native measurement. A verified replacement and separate deletion approval are required before removing source activities.
- Preserve existing flashcard text and identities unless the requested correction requires a change; edits can affect scheduling.
