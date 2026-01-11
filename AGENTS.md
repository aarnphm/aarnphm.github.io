# aarnphm's garden

this repo powers a Quartz-based digital garden with custom plugins and a Cloudflare worker. It also contains tools and implementations in Rust, Python, Go, C, C++, OCaml.

This means: no fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code that at all times demonstrates a maniacal obsession with elegant minimalism.

## non-negotiables

- do not write comments
- use `pnpm` by default
- do not run bundle or build; assume the user runs `dev.ts` and inspect the running process instead of spawning your own
- do not commit secrets; use `.env` locally and Cloudflare Secrets for the worker
- transformers under @quartz/plugins/transformers/ MUST NOT use filesystem access
- when thinking hard about a problem, use extended thinking as much as possible to reason hard about a problem.
