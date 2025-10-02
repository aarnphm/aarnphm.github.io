# CLAUDE.md

This repository powers a Quartz-based digital garden with custom plugins, a Cloudflare Worker. Follow these guidelines to keep changes consistent and easy to review.
Under `content`, there are additional tools/implementation both in Rust, Python, C, C++. Make sure to use the best practices for best performance.
Also don't have to bold text, keep it causal.

## Development Commands

**IMPORTANT**: Most of the cases if you need to verify build, make sure to see if `pnpm dev` is being run. In this cases, then `pnpm bundle` or any build step are not necessary.

**Build and Development:**

- `pnpm dev` - Start development server with hot reload (concurrency 8, verbose output)
- `pnpm bundle` - Build for production (concurrency 8, bundleInfo, verbose)
- `pnpm prod` - Production build with NODE_ENV=production
- `pnpm bundle:dev` - Development build for Cloudflare Pages
- `pnpm cf:dev` - Run Cloudflare Worker development server on port 8080
- `pnpm cf:deploy` - Deploy to Cloudflare (runs check first)

**Code Quality:**

- `pnpm check` - Complete validation pipeline (format, convert, cf:types, prettier check, TypeScript check, tests)
- `pnpm format` - Format code with Prettier and organize References.bib with bibtex-tidy
- `pnpm test` - Run tests using tsx --test
- `tsc --noEmit` - TypeScript type checking without emitting files

**Utilities:**

- `pnpm convert` - Run conversion scripts (tsx quartz/scripts/convert.ts)
- `pnpm cf:types` - Generate Cloudflare Worker types
- `pnpm cf:prepare` - Prepare for Cloudflare deployment (format, convert, types)

## Architecture Overview

This is a heavily modified Quartz-based digital garden with custom plugins and Cloudflare Worker integration.

**Core Structure:**

- `quartz/` - TypeScript source for CLI, plugins, and components
- `content/` - Markdown notes, academic papers, and assets (builds into static site)
- `worker/` - Cloudflare Worker TypeScript code
- `public/` - Build output directory

**Key Technologies:**

- **Static Site Generator:** Quartz 4.0 with extensive customization
- **Runtime:** Node.js >= 22, pnpm 9 package manager
- **Deployment:** Cloudflare Pages + Worker
- **Content:** Markdown with Obsidian-flavored syntax, LaTeX math, citations
- **Additional Languages:** Python 3.11, Rust nightly, Go 1.23, C++21, CUDA 12.8+

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

## Testing and Quality Assurance

- Tests live beside utilities: `quartz/util/*.test.ts`
- Run full validation with `pnpm check` before commits
- TypeScript strict mode enabled with `throwOnError: true` for KaTeX
- No secrets in commits; use `.env` locally and Cloudflare secrets for Worker

## Security Notes

- Large binaries managed through Git LFS
- Keep `public/` directory reproducible via `pnpm bundle`
- Use `wrangler.toml` and Cloudflare secrets for Worker configuration
