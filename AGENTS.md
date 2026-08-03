# aarnphm's garden

this repo powers a Quartz-based digital garden with custom plugins and a Cloudflare worker. It also contains tools and implementations in Rust, Python, Go, C, C++, OCaml.

This means: no fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code that at all times demonstrates a maniacal obsession with elegant minimalism.

## non-negotiables

- Do not preserve backward compatibility. Remove obsolete paths instead of adding compatibility layers, fallbacks, or migrations.
- Choose the simplest implementation that fully meets the current requirements. Avoid speculative abstractions, configuration, and indirection.
- Grow the system in layers. Start from the smallest version that works end to end, and add each new capability on top of a product that already works. Never trade a working product for unfinished complexity.
- Keep components modular and concerns clearly separated.
- Prefer established, well-maintained libraries when they reduce overall complexity or improve reliability. Do not reimplement common functionality without a clear reason.
- Lean on the dependencies already in the project before writing your own implementation or adding packages. Do not assume a library lacks a capability without checking its documentation and types.
- Make architectural decisions for the long term. Do not accept a stopgap that only works for now and is meant to be replaced later.
- Ship minimal production code that fixes the owning boundary.
- Do not write comments.
- Use `pnpm`, `oxlint`, `oxfmt`, and `tsgo`.
- Inspect with `fd` and `rg`.
- Keep new files in `quartz/util` kebab-case.
- Reuse shared guards and helpers from the owning util module; do not copy `isRecord`, JSON readers, or tiny support functions into call sites. If there are scripts available to be self-contained no need to make a util modules. ONLY uses `quartz/util` when it is universally used in a lot of places.
- Do not run bundle or build. Inspect the running `dev.ts` process when runtime evidence is needed. Oftentimes we will pipe the outputs to `/tmp/quartz-dev.log` for easier inspection.
- Keep secrets in `.env` locally and Cloudflare Secrets in production.
- Keep filesystem access out of `@quartz/plugins/transformers`.
- Write markdown math with LaTeX blocks.
- Skip shims and backward compatibility unless aarnphm asks for them.
- When you write test NEVER USE `readFile` and create unit tests that regex the actual changes for regression, that is fucking stupid.
- Make sure that `window.addCleanup` must always written within `document.addEventListener('nav', () => {})`
- no `box-shadow`, `border-left` and any sloppy styling that you might do for frontend components
