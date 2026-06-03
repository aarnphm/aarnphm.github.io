# aarnphm's garden

this repo powers a Quartz-based digital garden with custom plugins and a Cloudflare worker. It also contains tools and implementations in Rust, Python, Go, C, C++, OCaml.

This means: no fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code that at all times demonstrates a maniacal obsession with elegant minimalism.

## non-negotiables

- Ship minimal production code that fixes the owning boundary.
- Do not write comments.
- Use `pnpm`, `oxlint`, `oxfmt`, and `tsgo`.
- Inspect with `fd` and `rg`.
- Keep new files in `quartz/util` kebab-case.
- Reuse shared guards and helpers from the owning util module; do not copy `isRecord`, JSON readers, or tiny support functions into call sites.
- Do not run bundle or build. Inspect the running `dev.ts` process when runtime evidence is needed.
- Keep secrets in `.env` locally and Cloudflare Secrets in production.
- Keep filesystem access out of `@quartz/plugins/transformers`.
- Write markdown math with LaTeX blocks.
- Skip shims and backward compatibility unless aarnphm asks for them.
- When you write test NEVER USE `readFile` and create unit tests that regex the actual changes for regression, that is fucking stupid.
- Make sure that `window.addCleanup` must always written within `document.addEventListener('nav', () => {})` here
- no `box-shadow`, `border-left` and any sloppy styling that you might do for frontend components
