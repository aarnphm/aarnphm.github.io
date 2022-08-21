# GEMINI.md

This file provides instructional context for Gemini.

## Project Overview

This is a personal website and digital garden, built with [Quartz](https://quartz.jzhao.xyz/), a static site generator for digital gardens. It's heavily modified and uses a combination of TypeScript and Python. The content is written in Markdown and is located in the `content` directory.

The site is deployed to Cloudflare Pages, as indicated by the `wrangler.toml` file and the `cf:deploy` script in `package.json`. It also seems to use some machine learning components, with references to `onnxruntime-web`, `@huggingface/transformers`, and a `semanticSearch` configuration in `quartz.config.ts`.

## Architecture Overview

**Core Structure:**

- `quartz/` - TypeScript source for CLI, plugins, and components, built on top of remark/rehype/unist, mdast,hast ecosystem and best practices
- `content/` - Markdown notes, academic papers, assets, library implementations, tool monorepo.
  - `content/hinterland` - Special projects
- `worker/` - Cloudflare Worker TypeScript
- `public/` - Build output directory

**Stack:**

- **Static Site Generator:** Quartz 4.0 with extensive customization
- **Runtime:** Node.js >= 22, pnpm 9 package manager
- **Deployment:** Cloudflare Worker
- **Content:** Markdown with Obsidian-flavored syntax, LaTeX math, citations
- **Additional Languages:** Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8+, CUTLASS 4.0, CuTeDSL Python.

## Building and Running

- **Install dependencies**: `pnpm install`
- **Run in development mode**: `pnpm swarm`
- **Build the site**: `pnpm bundle`
- **Deploy to Cloudflare**: `pnpm cf:deploy`

## Development Conventions

### Code Quality

- `pnpm check` - Complete validation pipeline
- `pnpm format` - Format code with Prettier and organize References.bib with bibtex-tidy
- `tsc --noEmit` - TypeScript type checking without emitting files
- `tsx --test` - TypeScript test

### Language Guidelines

- **TypeScript/TSX**:
  - 2-space indent, ES modules. Format with Prettier (`pnpm format`).
  - Components: `PascalCase.tsx` (e.g., `ExplorerNode.tsx`). Utilities: lowercase or camel file names (e.g., `path.ts`, `fileTrie.ts`).
  - Variables/functions: `camelCase`; types/interfaces: `PascalCase`.
- **Python**:
  - ruff and mypy with 2-space indent; keep notebooks and scripts minimal.
  - Follows https://docs.fast.ai/dev/style.html for convention.
  - Dependencies can be installed with `uv pip install <dependencies>`
- **Markdown**:
  - Always keep everything in lowercase.
  - files should be use [[wikilink]] and absolute internal-links when reference with based from @content
  - If you think any sentences are more suitable as sidenotes, then use the following syntax:
    ```markdown
    {{sidenotes[<some_labels>]: <some_text_here>}}
    ```
  - If there is a file that is not yet available, then it must be created.
  - Markdown files will be consumed with Obsidian.
    - Make sure to use callouts, embedded links accordingly. For example:

      ```markdown
      > [!important] This is a callout
      > And some content under here

      And this is a [[thoughts/Attention|Attention]] as a internal wikilinks.
      ```

  - When parsing frontmatter, if there is an entry `gemini`, make sure to also consider it for additional instructions of any given files
  - All math equation should be written with LaTeX, with KaTeX flavor
    - For block-form, it should be formatted with `$$` with new lines.
    - for inline `$\text{hello}$` should work

- **arXiv references**:
  - Use `curl https://arxiv.org/bibtex/<id>` to get metadata.
  - Update the @content/References.bib file directly, then use `[@<name>]` entries in relevant files.
  - If requested to download the paper, then download it into @content/thoughts/papers/ with the name format `<id>.pdf`.

### Project Structure & Module Organization

- `quartz/` TypeScript source (CLI, plugins, components). Tests live beside utils: `quartz/util/*.test.ts`.
- `content/` Markdown/notes and assets; built into the static site. Additional tools, kernels written in Rust, C, C++, Python, Go.
  - `content/hinterland`: Special projects
- `public/` Build output (served locally and deployed).
- `worker/` Cloudflare Worker TypeScript.
- `.github/` CI, docs; `quartz.config.ts` site config; `dist/` transient build artifacts.

### Security & Configuration Tips

- Do not commit secrets; use `.env` locally and `wrangler.toml`/CF secrets for Worker.
- Large binaries go through Git LFS; keep `public/` reproducible via `pnpm bundle`.
- Node >= 22, pnpm 9; Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8++, Lean4, Triton 3.4+, CUTLASS 4.2.0+, CuTeDSL in Python.
- When thinking hard about a problem, make sure to use sequential-thinking and uses at least around minimum of seven or more steps.
