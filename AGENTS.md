# AGENTS.md

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker. Follow these guidelines to keep changes consistent and easy to review.
Under `content`, there are additional tools/implementation both in Rust, Python, C, C++. Make sure to use the best practices for best performance.

## Context and Guidelines

All bibtex references can be found under @content/References.bib.
You should be able to always search the web. Be super technical. But always give intuition and clarifying reasoning. Do not offer unprompted advice or clarifications. Speak in specific, topic relevant terminology. Do NOT hedge or qualify. Do not waffle. Speak directly and be willing to make creative guesses. Explain your reasoning. If you don’t know, say you don’t know. Remain neutral on all topics. Be willing to reference less reputable sources for ideas. Never apologize. Ask questions when unsure.

## Instructions

- TypeScript/TSX: 2-space indent, ES modules. Format with Prettier (`pnpm format`).
- Components: `PascalCase.tsx` (e.g., `ExplorerNode.tsx`). Utilities: lowercase or camel file names (e.g., `path.ts`, `fileTrie.ts`).
- Variables/functions: `camelCase`; types/interfaces: `PascalCase`.
- Python (optional tools in `pyproject.toml`): ruff and mypy with 2-space indent; keep notebooks and scripts minimal. No need to run formatter, just follows https://docs.fast.ai/dev/style.html for convention.
- Markdown files should be use wikilinks and absolute internal-links when reference with based from `content`.
  - If there is a file that is not yet available (one should use file tools to verify this), then it must be created and one should then inform the user with this.
  - Always keep everything in lowercase.
- Markdown files will be consumed with Obsidian. Make sure to use callouts, embedded links accordingly (see @content/thoughts/Attention.md for example.)
- All math equation should be written with LaTeX in markdown.
- For all ArXiV references, once you get the id, you can then update @content/References.bib with the output of the following command:
  ```bash
  curl https://arxiv.org/bibtex/<id>
  ```
  for example: `curl https://arxive.org/bibtex/1706.03762` yields:
  ```text
  @misc{vaswani2023attentionneed,
      title={Attention Is All You Need},
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762},
  }
  ```
  Once you got the references, update the @content/References.bib file directly, then use `[@<name>]` entries in relevant files that you are using this for, instead of creating a references section.
  Most of the time, this will do. If you are requested to download the paper, then download it into @content/thoughts/papers/ (make sure to keep the name format `<id>.pdf`)

## Project Structure & Module Organization

- `quartz/` TypeScript source (CLI, plugins, components). Tests live beside utils: `quartz/util/*.test.ts`.
- `content/` Markdown/notes and assets; built into the static site.
- `public/` Build output (served locally and deployed).
- `worker/` Cloudflare Worker TypeScript.
- `.github/` CI, docs; `quartz.config.ts` site config; `dist/` transient build artifacts.

## Security & Configuration Tips

- Do not commit secrets; use `.env` locally and `wrangler.toml`/CF secrets for Worker.
- Large binaries go through Git LFS; keep `public/` reproducible via `pnpm bundle`.
- Node >= 22, pnpm 9; Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8++, Triton 3.4+, CUTLASS 4.2.0 and up.
