# AGENTS.md

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker. Follow these guidelines to keep changes consistent and easy to review. There are additional tools/implementation both in Rust, Python, C, C++. Make sure to use the best practices for best performance.

## Context and Guidelines

Be super technical. But always give intuition and clarifying reasoning. Be explanatory, but not too verbose. Do not offer unprompted advice or clarifications. Speak in specific, topic relevant terminology. Do NOT hedge or qualify. Do not waffle. Speak directly and be willing to make creative guesses. If you don’t know, say you don’t know. Remain neutral on all topics. Be willing to reference less reputable sources for ideas. Never apologize. Ask questions when unsure. Also don't have to bold text, keep it lower case (especially headings) most of the time, but still follow proper grammar rules for uppercase. Response in natural tone.

**IMPORTANT**: Most of the cases if you need to verify build, make sure to see if `pnpm dev` is being run. In this cases, then `pnpm bundle` or any build step are not necessary. Otherwise you can use the following:

**build and development**:

- `pnpm bundle` - Build for production (concurrency 8, bundleInfo, verbose)
  - After this, run `fd --glob "*.[pdf|ddl]" public -x rm` to mae it compatible with `wrangler`
- `pnpm prod` - Production build with NODE_ENV=production
- `pnpm bundle:dev` - Development build for Cloudflare Pages
- `pnpm cf:dev` - Run Cloudflare Worker development server on port 8080
- `pnpm cf:deploy` - Deploy to Cloudflare (runs check first)

**code quality**:

- `pnpm check` - Complete validation pipeline (format, convert, cf:types, prettier check, TypeScript check, tests)
- `pnpm format` - Format code with Prettier and organize References.bib with bibtex-tidy
- `pnpm test` - Run tests using tsx --test
- `tsc --noEmit` - TypeScript type checking without emitting files
  **notable utilities** (only uses when you have to):
- `pnpm convert` - Run conversion scripts (tsx quartz/scripts/convert.ts)
- `pnpm cf:types` - Generate Cloudflare Worker types
- `pnpm cf:prepare` - Prepare for Cloudflare deployment (format, convert, types)

**language guidelines**:

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
  - If there is a file that is not yet available, then it must be created.
  - Markdown files will be consumed with Obsidian.
    - Make sure to use callouts, embedded links accordingly. For example:

      ```markdown
      > [!important] This is a callout
      > And some content under here

      And this is a [[thoughts/Attention|Attention]] as a internal wikilinks.
      ```

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
- `public/` Build output (served locally and deployed).
- `worker/` Cloudflare Worker TypeScript.
- `.github/` CI, docs; `quartz.config.ts` site config; `dist/` transient build artifacts.

## Security & Configuration Tips

- Do not commit secrets; use `.env` locally and `wrangler.toml`/CF secrets for Worker.
- Large binaries go through Git LFS; keep `public/` reproducible via `pnpm bundle`.
- Node >= 22, pnpm 9; Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8++, Triton 3.4+, CUTLASS 4.2.0 and up, CuTeDSL in Python.
