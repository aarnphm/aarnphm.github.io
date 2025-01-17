# AGENTS.md

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker. Follow these guidelines to keep changes consistent and easy to review. There are additional tools/implementation both in Rust, Python, C, C++. Make sure to use the best practices for best performance.

## Context and Guidelines

**Build and Development**:

- `pnpm dev` - run dev (concurrency 8, bundleInfo, verbose)
  - After this, run `fd --glob "*.[pdf|ddl]" public -x rm` to mae it compatible with `wrangler`
- `pnpm prod` - Production build with NODE_ENV=production
- `pnpm cf:deploy` - Deploy to Cloudflare (runs check first)

**Code Quality**:

- `pnpm check` - Complete validation pipeline
- `pnpm format` - Format code with Prettier and organize References.bib with bibtex-tidy
- `tsc --noEmit` - TypeScript type checking without emitting files
- `tsc --test` - TypeScript test

**Language Guidelines**:

- TypeScript/TSX:
  - 2-space indent, ES modules. Format with Prettier (`pnpm format`).
  - Components: `PascalCase.tsx` (e.g., `ExplorerNode.tsx`). Utilities: lowercase or camel file names (e.g., `path.ts`, `fileTrie.ts`).
    - If you are writing buttons, most case prefer `span[type="button"]` over button. But make sure to ask for confirmation.
  - Variables/functions: `camelCase`; types/interfaces: `PascalCase`.
- Python:
  - ruff and mypy with 2-space indent; keep notebooks and scripts minimal.
  - No need to run formatter
  - just follows https://docs.fast.ai/dev/style.html for convention.
  - No need to do gated imports. Just assume dependencies are available, and can be installed with `uv pip install <dependencies>`
- Markdown:
  - Always keep everything in lowercase.
  - files should be use [[wikilink]] and absolute internal-links when reference with based from @content
    - `content/hinterland` - Special projects
  - If you think any sentences are more suitable as sidenotes, then use the following syntax:
    ```markdown
    {{sidenotes[<some_labels>]: <some_text_here>}}
    ```
    See @content/thoughts/love.md#L54 for example.
  - If there is a file that is not yet available, then it must be created.
  - Markdown files will be consumed with Obsidian.
    - Make sure to use callouts, embedded links accordingly. For example:

      ```markdown
      > [!important] This is a callout
      > And some content under here

      And this is a [[thoughts/Attention|Attention]] as a internal wikilinks.
      ```

  - When parsing frontmatter, if there is an entry `claude`, make sure to also consider it for additional instructions of any given files
  - All math equation should be written with LaTeX, with KaTeX flavor
    - For block-form, it should be formatted with `$$` with new lines. For example:
      ```markdown
      $$
      f(y)\ge f(x)+\langle\nabla f(x),y- x\rangle+\frac{\mu}{2}\|y- x\|^2
      $$
      ```
    - for inline `$\text{hello}$` should work

- arXiv references:
  - Use the following command to get metadata for any given arXiv id:
    ```bash
    curl https://arxiv.org/bibtex/<id>
    ```
  - for example: `curl https://arxive.org/bibtex/1706.03762` yields:
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
  - Once you got the references, update the @content/References.bib file directly, then use `[@<name>]` entries in relevant files that you are using this for, instead of creating a references section.
  - Most of the time, this will do. If you are requested to download the paper, then download it into @content/thoughts/papers/ (make sure to keep the name format `<id>.pdf`)

## Project Structure & Module Organization

- `quartz/` TypeScript source (CLI, plugins, components). Tests live beside utils: `quartz/util/*.test.ts`.
- `content/` Markdown/notes and assets; built into the static site. Additional tools, kernels written in Rust, C, C++, Python, Go.
  - `content/hinterland`: Special projects 😃
- `public/` Build output (served locally and deployed).
- `worker/` Cloudflare Worker TypeScript.
- `.github/` CI, docs; `quartz.config.ts` site config; `dist/` transient build artifacts.

## Security & Configuration Tips

- Do not commit secrets; use `.env` locally and `wrangler.toml`/CF secrets for Worker.
- Large binaries go through Git LFS; keep `public/` reproducible via `pnpm bundle`.
- Node >= 22, pnpm 9; Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8++, Lean4, Triton 3.4+, CUTLASS 4.2.0+, CuTeDSL in Python.
