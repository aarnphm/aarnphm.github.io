# Development commands for agents

Run commands from the owning package. Read its current `package.json` or task runner before relying on these descriptions.

| Command                           | Effect and scope                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `pnpm test <test-file>`           | Runs the selected Node tests through `tsx --test`. Use affected existing tests as well as new tests.                           |
| `pnpm exec oxlint <files>`        | Lints selected files without applying fixes.                                                                                   |
| `pnpm exec oxfmt --check <files>` | Checks formatting without rewriting files.                                                                                     |
| `pnpm exec tsgo --noEmit`         | Checks the repository TypeScript project without emitting code. Distinguish unrelated baseline errors.                         |
| `pnpm check`                      | Runs repository-wide formatting, lint, type, and test checks. Broader than a focused change usually needs.                     |
| `pnpm dev`                        | Starts the development container workflow. Inspect the existing process before starting another instance.                      |
| `pnpm swarm`                      | Starts `quartz/scripts/dev.ts`. Preserve an existing watcher.                                                                  |
| `pnpm prod`, `pnpm bundle`        | Builds the full Quartz site. Use the current watcher for ordinary verification.                                                |
| `pnpm format`                     | Rewrites BibTeX and source formatting, applies lint fixes, and formats Python. Avoid it for scoped edits.                      |
| `pnpm db:migrate:local`           | Mutates local development databases.                                                                                           |
| `pnpm db:migrate`                 | Applies migrations to remote Cloudflare D1 databases. Requires authorization for that remote change.                           |
| `pnpm cf:deploy`                  | Runs broad checks, remote database migrations, and a Worker deployment. Requires authorization covering those effects.         |
| Provider sync or backfill scripts | May contact providers, replace caches, or update remote activities. Read the specific command and its flags before running it. |

Do not delete generated files as a routine step after starting the watcher. Let the owning emitter handle its output. For browser checks, use the matching `build:ready` and successful HTTP response before inspecting source-rendered markup and interactions. DOM injection proves only a prototype unless the served implementation is separately verified.
