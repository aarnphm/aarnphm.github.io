# CLAUDE.md

This repository powers a Quartz-based digital garden with custom plugins and a Cloudflare Worker. Follow these guidelines to keep changes consistent and maintainable. Additional tools and implementations in Rust, Python, C, C++, and other languages live under `content/`. Use best practices for optimal system-level performance.

## Development Commands

**Build and Development:**

- `pnpm cf:deploy` - Deploy to Cloudflare (runs check first)

**Code Quality:**

- `pnpm check` - Complete validation pipeline
- `pnpm format` - Format code with Prettier and organize References.bib with bibtex-tidy
- `tsc --noEmit` - TypeScript type checking without emitting files
- `tsc --test` - TypeScript test

## Architecture Overview

**Core Structure:**

- `quartz/` - TypeScript source for CLI, plugins, and components, built on remark/rehype/unist, mdast, and hast ecosystems
  - For all `.inline.ts` scripts: avoid `const` arguments (esbuild strips them). Keep functions and variables stateless. Use `localStorage` or `sessionStorage` for persistent state.
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
- Custom sidenotes structured inspired by Tuffe CSS

**Component Architecture:**

React-like components in `quartz/components/` using Preact:

- PascalCase.tsx naming (e.g., ExplorerNode.tsx)
- Utilities use camelCase (e.g., path.ts, fileTrie.ts)
- 2-space indentation, ES modules
- For interactive buttons, prefer `<span role="button">` over `<button>`. Ask for confirmation before implementing.
- Never write a unist processor within `markdownPlugins` or `htmlPlugins` to avoid recursion of nested processors. Use existing parsing structure from Quartz.

## Content Guidelines

**Markdown Conventions:**

- Use wikilinks and absolute internal links based from `content/`
- Obsidian-style callouts and embedded links
- All math equations in LaTeX format
- Citations via `[@reference]` syntax linking to References.bib
- All headings must be in lowercase
- For sidenotes, use: `Main text {{sidenotes[label]: sidenote content}}`
  - Example: See `content/thoughts/love.md:58`
- Make sure to use callouts, embedded links accordingly. For example:

  ```markdown
  > [!important] This is a callout
  > And some content under here

  And this is a [[thoughts/Attention|Attention]] as a internal wikilinks.
  ```

- All math equation should be written with LaTeX, with KaTeX flavor
- For block-form, it should be formatted with `$$\n<content>\n$$`. For inline `$<content>$`

**Academic References:**

- Fetch arXiv BibTeX: `curl https://arxiv.org/bibtex/<id>`
- Add entries to `content/References.bib`
- Reference in markdown: `[@entryname]`

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
- If you need to write any Lean proofs, do it under `content/provers/` and compile with lean4 accordingly.

## Testing and Quality Assurance

- Tests live beside utilities: `quartz/util/*.test.ts`
- Run full validation with `pnpm check` before commits
- TypeScript strict mode enabled
- KaTeX configured with `throwOnError: true`

## Build and Deployment Constraints

- **Never run** `pnpm bundle` or `pnpm build` directly - the dev server (`dev.ts`) is always running. Inspect the running process instead.
- Large binaries are managed through Git LFS
- Keep `public/` directory reproducible via `pnpm bundle`
- Use `wrangler.toml` and Cloudflare secrets for Worker configuration
- No secrets in commits; use `.env` locally

## Plugin Development Constraints

- **NEVER** use `fs` in `quartz/plugins/transformers/` - transformers should ONLY transform hast/ast/mdast trees
- For file I/O operations, use `renderPage.tsx` or `quartz/plugins/emitters/` phase
- Avoid recursion by not creating unist processors within `markdownPlugins` or `htmlPlugins`

## Skills Integration

This repository supports Claude Code skills for specialized workflows. Skills live in `.claude/skills/` and provide domain-specific functionality.

### Available Skills (Planned)

**Content Management:**
- `citation-manager` - Manage academic citations and References.bib
  - Add citations from arXiv, DOI, or BibTeX
  - Format and validate References.bib
  - Insert citation references in markdown files

- `content-creator` - Create properly formatted markdown notes
  - Generate new notes with correct frontmatter
  - Apply consistent formatting (lowercase headings, wikilinks, etc.)
  - Add callouts, sidenotes, and math equations

**Development:**
- `plugin-dev` - Develop and test Quartz plugins
  - Scaffold new transformers, emitters, or filters
  - Test plugins with sample content
  - Validate against existing plugin patterns

- `deployment-helper` - Manage Cloudflare Worker deployments
  - Run validation pipeline (`pnpm check`)
  - Deploy to Cloudflare with `wrangler`
  - Verify deployment status

**Academic Workflow:**
- `academic-paper` - Fetch and organize academic papers
  - Download papers from arXiv
  - Extract and add BibTeX entries
  - Organize PDFs in `content/thoughts/papers/`
  - Create summary notes with proper citations

### Creating New Skills

Skills are stored in `.claude/skills/<skill-name>/skill.md` and follow this structure:

```markdown
# Skill Name

Brief description of what this skill does.

## When to Use

- Use case 1
- Use case 2

## Instructions

1. Step-by-step instructions
2. Commands to run
3. Files to check

## Examples

Example workflows
```

### Skill Best Practices

- Keep skills focused on single domains (citations, content, deployment)
- Use existing tools and commands defined in `package.json`
- Follow the code style guidelines in this document
- Test skills with representative examples
- Document any dependencies or prerequisites
