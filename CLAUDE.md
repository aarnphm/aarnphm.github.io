# CLAUDE.md

this repo powers a quartz-based digital garden with custom plugins and a cloudflare worker. it also contains tools and implementations in rust, python, go, c, and c++ under `content/` directory.

## non-negotiables

- do not write comments
- do not run bundle or build; assume the user runs `dev.ts` and inspect the running process instead of spawning your own
- do not commit secrets; use `.env` locally and Cloudflare Secrets for the worker
- transformers under @quartz/plugins/transformers/ must not use filesystem access

## navigations

available skills:

- `core`: repo-wide workflow constraints and safety rules
- `markdown`: obsidian markdown conventions, callouts/sidenotes, katex, citations
- `quartz-plugins`: quartz plugin boundaries and repo patterns
- `worker`: worker layout, deployment, and secrets rules

when thinking hard about a problem, use extended thinking with at least seven explicit steps before proposing a change.
