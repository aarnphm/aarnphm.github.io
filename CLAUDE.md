# CLAUDE.md

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker. Follow these guidelines to keep changes consistent and easy to review. There are additional tools/implementation both in Rust, Python, C, C++ under @content/. Make sure to use the best practices for best performance system-level wise.

## Development Commands

**Build and Development:**

- `pnpm dev` - Running dev script (you should run this in the background, if port 8080 is empty)
  - After this, run `fd --glob "*.[pdf|ddl]" public -x rm` to mae it compatible with `wrangler`
- `pnpm cf:deploy` - Deploy to Cloudflare (runs check first)

**Code Quality:**

- `pnpm check` - Complete validation pipeline
- `pnpm format` - Format code with Prettier and organize References.bib with bibtex-tidy
- `tsc --noEmit` - TypeScript type checking without emitting files
- `tsc --test` - TypeScript test

## Architecture Overview

**Core Structure:**

- `quartz/` - TypeScript source for CLI, plugins, and components, built on top of remark/rehype/unist, mdast,hast ecosystem and best practices
- `content/` - Markdown notes, academic papers, assets, library implementations, tool monorepo.
  - `content/hinterland` - Special projects
- `worker/` - Cloudflare Worker TypeScript
- `public/` - Build output directory

**Key Technologies:**

- **Static Site Generator:** Quartz 4.0 with extensive customization
- **Runtime:** Node.js >= 22, pnpm 9 package manager
- **Deployment:** Cloudflare Pages + Worker
- **Content:** Markdown with Obsidian-flavored syntax, LaTeX math, citations
- **Additional Languages:** Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8+, CUTLASS 4.0, CuTeDSL Python.

**Plugin System:**

The site uses custom Quartz plugins (quartz/plugins/):

- Academic citations with References.bib integration
- TikZ diagram rendering via TikzJax
- Pseudocode rendering
- Telescopic text expansion
- Code file transclusions
- Twitter embeds
- Slides generation
- Custom syntax highlighting (rose-pine themes)

**Component Architecture:**

React-like components in `quartz/components/` using Preact:

- PascalCase.tsx naming (e.g., ExplorerNode.tsx)
- Utilities use camelCase (e.g., path.ts, fileTrie.ts)
- 2-space indentation, ES modules
- If you are writing buttons, most case prefer `span[type="button"]` over button. But make sure to ask for confirmation.

## Content Guidelines

**Markdown Conventions:**

- Use wikilinks and absolute internal links based from `content/`
- Obsidian-style callouts and embedded links
- All math equations in LaTeX format
- Citations via `[@reference]` syntax linking to References.bib
- All headings must be in lowercase.
- Make sure to use callouts, embedded links accordingly. For example:

  ```markdown
  > [!important] This is a callout
  > And some content under here

  And this is a [[thoughts/Attention|Attention]] as a internal wikilinks.
  ```

- When parsing frontmatter, if there is an entry `claude`, make sure to also consider it for additional instructions of any given files
- All math equation should be written with LaTeX, with KaTeX flavor
- For block-form, it should be formatted with `$$` with new lines. For inline `$<content>$` will do

**Academic References:**
For arxiv papers, fetch BibTeX entries:

```bash
curl https://arxiv.org/bibtex/<id>
```

Then update `content/References.bib` and reference as `[@entryname]` in markdown.

**File Organization:**

- Academic papers: `content/thoughts/papers/<id>.pdf`
- Notes follow hierarchical structure under `content/thoughts/`
- Code examples and tools can be in multiple languages under `content/`

## Code Style

**TypeScript/TSX:**

- 2-space indentation, ES modules
- Variables/functions: camelCase
- Types/interfaces: PascalCase
- Format with Prettier via `pnpm format`

**Python (optional tools):**

- ruff and mypy with 2-space indentation
- Follow https://docs.fast.ai/dev/style.html conventions
- Keep notebooks and scripts minimal

**Other Languages:**

- Rust: Use cargo conventions
- Go: Use gofmt conventions
- C/C++: Modern C++21 standards
- CUDA: Compatible with 12.8+ and Triton 3.4+
- If you need to write any lean prof, do it under @content/provers/ and compile with lean4 accordingly.

## Testing and Quality Assurance

- Tests live beside utilities: `quartz/util/*.test.ts`
- Run full validation with `pnpm check` before commits
- TypeScript strict mode enabled with `throwOnError: true` for KaTeX
- No secrets in commits; use `.env` locally and Cloudflare secrets for Worker

## Security Notes

- Large binaries managed through Git LFS
- Keep `public/` directory reproducible via `pnpm bundle`
- Use `wrangler.toml` and Cloudflare secrets for Worker configuration
- Please never run `pnpm bundle`. Just run `pnpm exec tsx quartz/scripts/dev.ts` instead
