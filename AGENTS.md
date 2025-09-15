# Repository Guidelines

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker, and a lightweight Go dev server. Follow these guidelines to keep changes consistent and easy to review.
Under content, there are additional tools/implementation both in Rust, Python, C, C++. Make sure to use the best practices for best performance.

## Project Structure & Module Organization

- `quartz/` TypeScript source (CLI, plugins, components). Tests live beside utils: `quartz/util/*.test.ts`.
- `content/` Markdown/notes and assets; built into the static site.
- `public/` Build output (served locally and deployed).
- `worker/` Cloudflare Worker TypeScript.
- `server.go` Local static server for `public/`.
- `.github/` CI, docs; `quartz.config.ts` site config; `dist/` transient build artifacts.

## Build, Test, and Development Commands

- `pnpm dev` Run Quartz in development with live rebuild and preview.
- `pnpm bundle` Build the site into `public/` (use `pnpm prod` for production flags).
- `pnpm test` Run TypeScript tests via `tsx --test`.
- `pnpm check` Type-check, format check, and run tests.
- `pnpm cf:dev` Build and run the Worker locally; `pnpm cf:deploy` to deploy.
- `go run server.go -dir public -port 8080` Serve the static site.

## Coding Style & Naming Conventions

- TypeScript/TSX: 2-space indent, ES modules. Format with Prettier (`pnpm format`).
- Components: `PascalCase.tsx` (e.g., `ExplorerNode.tsx`). Utilities: lowercase or camel file names (e.g., `path.ts`, `fileTrie.ts`).
- Variables/functions: `camelCase`; types/interfaces: `PascalCase`.
- Python (optional tools in `pyproject.toml`): ruff and mypy with 2-space indent; keep notebooks and scripts minimal.
- Markdown files should be use wikilinks and absolute internal-links when reference with based from `content`. For example.
  - [[thoughts/optimization#momentum]], [[thoughts/Attention]]
- These markdown will be consumed with Obsidian. Any content under `content/` should also be indexed accordingly.
- All math equation should be written with LaTeX in markdown.

## Commit & Pull Request Guidelines

- Use Conventional Commits: `feat:`, `fix:`, `perf:`, `chore:`, `docs:`; add scope when useful (e.g., `chore(deps-dev): ...`).
- PRs: concise description, linked issues, screenshots for UI changes, and notes on testing. Keep diffs focused; update `quartz.config.ts` if adding/removing plugins.

## Security & Configuration Tips

- Do not commit secrets; use `.env` locally and `wrangler.toml`/CF secrets for Worker.
- Large binaries go through Git LFS; keep `public/` reproducible via `pnpm bundle`.
- Node >= 22, pnpm 9; Python 3.11 for optional tooling.
