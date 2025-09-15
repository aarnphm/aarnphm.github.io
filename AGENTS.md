# Repository Guidelines

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker. Follow these guidelines to keep changes consistent and easy to review.
Under `content`, there are additional tools/implementation both in Rust, Python, C, C++. Make sure to use the best practices for best performance.

## Project Structure & Module Organization

- `quartz/` TypeScript source (CLI, plugins, components). Tests live beside utils: `quartz/util/*.test.ts`.
- `content/` Markdown/notes and assets; built into the static site.
- `public/` Build output (served locally and deployed).
- `worker/` Cloudflare Worker TypeScript.
- `.github/` CI, docs; `quartz.config.ts` site config; `dist/` transient build artifacts.

## Coding Style & Naming Conventions

- TypeScript/TSX: 2-space indent, ES modules. Format with Prettier (`pnpm format`).
- Components: `PascalCase.tsx` (e.g., `ExplorerNode.tsx`). Utilities: lowercase or camel file names (e.g., `path.ts`, `fileTrie.ts`).
- Variables/functions: `camelCase`; types/interfaces: `PascalCase`.
- Python (optional tools in `pyproject.toml`): ruff and mypy with 2-space indent; keep notebooks and scripts minimal.
- Markdown files should be use wikilinks and absolute internal-links when reference with based from `content`.
  - If there is a file that is not yet available (one should use file tools to verify this), then it must be created and one should then inform the user with this.
- Markdown files will be consumed with Obsidian. Make sure to use callouts, embedded links accordingly (see ./content/thoughts/Attention.md for example.)
- All math equation should be written with LaTeX in markdown.

## Security & Configuration Tips

- Do not commit secrets; use `.env` locally and `wrangler.toml`/CF secrets for Worker.
- Large binaries go through Git LFS; keep `public/` reproducible via `pnpm bundle`.
- Node >= 22, pnpm 9; Python 3.11 for optional tooling.
