---
title: 'Agent tooling'
---

# #sprint-skills + agent-friendliness: reading map

_vllm stopped treating coding agents as a contributor-quality problem around september and started treating them as a throughput source that needs plumbing._

sources, as of 2026-10-11: vllm `main` @ `c059bee` (cloned, history back to 2026-05-01), github PRs/issues/RFCs, and the [SIG Model Performance notes](https://docs.google.com/document/d/1A1TOuCTPQ_72npOSzhWKfYOmDIF7OoagLGaL8gHe1TQ/edit) (read through drive, entries 2026-04-28 → 2026-10-06). **Slack (#sprint-skills, #sig-model-performance) was unavailable through the connector**, so information shared only there is missing. the gaps section lists those sources.

For verification and throughput governance, see [agent review](agent-review.md).

---

## 0. the one number

share of commits on `main` that carry an AI `Co-authored-by:` trailer (claude / codex / cursor / gemini / copilot):

| month          | commits | AI-trailed |   share |
| -------------- | ------: | ---------: | ------: |
| 2026-05        |     868 |        123 |     14% |
| 2026-06        |    1189 |        145 |     12% |
| 2026-07        |    1100 |        253 |     23% |
| 2026-08        |    1204 |        302 |     25% |
| 2026-09        |    1650 |        652 | **40%** |
| 2026-10 (1–11) |     517 |        230 | **44%** |

commit volume went up 1.9x from may to september while AI-trailed commits went up 5.3x, so AI-trailed commits account for 529 of the 782-commit monthly increase (68%). this is a lower bound: trailers are self-reported and nothing enforces them. (per-commit regex over `git log --format=%B` for `Co-authored-by:` lines naming claude, codex, openai, cursor, gemini, copilot or gpt.)

everything below is a response to that curve. three layers:

1. **context**: what an agent reads (`AGENTS.md`, domain guides, skills).
2. **verification**: what catches an agent mechanically (pre-commit lint on agent files, runtime checkers, the PR template, the new-PR bot).
3. **throughput governance**: how humans keep up (fast-track RFC, agent-assisted review, the `vllm-agent` CI shepherd, the CI selector).

---

## 1. read these first (in order)

| #   | what                                                                                                                                                                                         | why it matters                                                                                                                          | time   |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 1   | [RFC #59665: Fast-Track Merging for Model Optimization PRs](https://github.com/vllm-project/vllm/issues/59665) (benchislett) + all 11 comments                                               | the policy discussion; the feedback deadline listed was oct 13                                                                          | 20 min |
| 2   | [`docs/contributing/editing-agent-instructions.md`](https://github.com/vllm-project/vllm/blob/main/docs/contributing/editing-agent-instructions.md)                                          | the design philosophy for every agent file: token budgets, "test with a prompt first", prefer a lint over a rule                        | 5 min  |
| 3   | [`.agents/skills/pr-checklist/SKILL.md`](https://github.com/vllm-project/vllm/blob/main/.agents/skills/pr-checklist/SKILL.md) + [PR #57084](https://github.com/vllm-project/vllm/pull/57084) | the biggest skill (18.7 KB), now mandatory via `AGENTS.md`; it is distilled from 342 review comments on 97 PRs by benchislett and mgoin | 15 min |
| 4   | SIG Model Performance notes, entries **2026-09-29** ("Agentic readiness and fast track") and **2026-10-06** ("hyper-specialization for low-latency inference")                               | where the fast-track idea came from, and the decision to go in-tree with per-concurrency and per-SKU flat models                        | 10 min |
| 5   | [Q4 2026 roadmap #61024](https://github.com/vllm-project/vllm/issues/61024), the SIG CI and SIG Model Performance sections                                                                   | "agent-assisted reviewers and CI shepherds, with checks for unit-test quality" is now a roadmap line                                    | 5 min  |
| 6   | [RFC #60904: a standard for MonoKernels](https://github.com/vllm-project/vllm/issues/60904) (dllehr-amd)                                                                                     | the first concrete, contract-shaped answer to "how do we let agents over-specialize safely"                                             | 20 min |

the rest is reference.

---

## 2. context layer: what agents read

### 2.1 `AGENTS.md` (root, 156 of 200 allowed lines)

[`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md). the key moves:

- **"Pure code-agent PRs are not allowed."** a human must defend every line. AI-assisted PRs must state why they aren't duplicates, the test commands run, model evals when output changes, and that AI was used.
- **fail-closed**: if the work is duplicate or busywork, the agent is told not to proceed.
- **run the `pr-checklist` skill before opening any PR, drafts included** ([#60235](https://github.com/vllm-project/vllm/pull/60235), ZJY0516, oct 6).
- `uv` only, never system python; `VLLM_USE_PRECOMPILED=1` for editable installs.
- test guidance from simon-mo ([#47784](https://github.com/vllm-project/vllm/pull/47784)): design before writing, reuse fixtures, one behavior per test, no kernel benchmarks in `tests/`, run evals on model-affecting changes without waiting to be asked.
- commit trailers: `Co-authored-by: <agent>`.

### 2.2 domain guides

`vllm/parser/AGENTS.md`, `vllm/tool_parsers/AGENTS.md`, `vllm/reasoning/AGENTS.md` (Flora Feng, [#56883](https://github.com/vllm-project/vllm/pull/56883)), plus `rust/AGENTS.md`. each one has a 300-line budget. the tool-parsers guide is a good example of the genre: "the `ToolParser` interface is legacy, kept for out-of-tree plugins; new parsers go in `vllm/parser/` as a unified `Parser`, exposed here as a thin adapter." that is exactly the knowledge you can't infer from the code.

### 2.3 the rules for writing agent files

[`editing-agent-instructions.md`](https://github.com/vllm-project/vllm/blob/main/docs/contributing/editing-agent-instructions.md) reads like it was written by someone who watched `CLAUDE.md` files rot:

- `AGENTS.md` loads on every request, so it gets <200 lines. domain guides <300. `SKILL.md` <500, with detail pushed into linked files.
- **don't add a rule if agents already do it.** test with a prompt first.
- **one-off incident? fix it in code (lint, CI check, assert), not with a new rule.**
- "search for X" instead of hardcoded paths. show the command that fetches a value, never the value ("config snapshots" is a named anti-pattern).
- every addition should be offset by a removal.

### 2.4 skills inventory (`.agents/skills/`, symlinked into `.claude/skills/`)

| skill                   | added              | author                                                                  | what it encodes                                                                                                                                                                                           | note                                                                                                                                                                                                  |
| ----------------------- | ------------------ | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ci-fails-buildkite`    | 06-20, moved 09-01 | Vadim Gimpelson; mgoin                                                  | `ci-fetch-log.sh --pr N` fetches every failed job log, stripped of ANSI                                                                                                                                   | first skill; moved `.claude/` → `.agents/` in [#54806](https://github.com/vllm-project/vllm/pull/54806) to share across agents                                                                        |
| `kernel-microbenchmark` | 08-25              | Thien Tran ([#53688](https://github.com/vllm-project/vllm/pull/53688))  | CUPTI timing with cold L2 by default, SOL sanity table (BF16 attn on B200 ≈ 1.6 PF/s), multi-GPU timing = median of per-sample MAX across ranks                                                           | ROCm path added in [#58646](https://github.com/vllm-project/vllm/pull/58646)                                                                                                                          |
| `debug-ima`             | 08-26              | Thien Tran ([#53702](https://github.com/vllm-project/vllm/pull/53702))  | one link to the CUDA core-dump blog                                                                                                                                                                       | 288 bytes. a pointer, not a skill                                                                                                                                                                     |
| `triton-kernel-writing` | 09-02              | Woosuk ([#55019](https://github.com/vllm-project/vllm/pull/55019))      | grid[1,2] ≤ 65535, int64 offsets for KV addressing, `do_not_specialize` for 0/1 scalars, alignment-driven JIT variants                                                                                    | a TRT-LLM import ([#55009](https://github.com/vllm-project/vllm/pull/55009)) was closed the same day and rewritten from scratch                                                                       |
| `pr-checklist`          | 09-30              | benchislett ([#57084](https://github.com/vllm-project/vllm/pull/57084)) | design fit, LOC complexity bands (<50 / 50–250 / >250), no CPU-GPU sync on the hot path (`VLLM_GPU_SYNC_CHECK=error`), MRV1 deprecation, evidence tables                                                  | the PR template was rebuilt around it (Overview / Claims / Validation / Details). NIXL/PD flow-diagram guidance added by Nicolò Lucchesi in [#60111](https://github.com/vllm-project/vllm/pull/60111) |
| `check-api-compat`      | 09-30              | Flora Feng ([#58003](https://github.com/vllm-project/vllm/pull/58003))  | "vLLM's own code is the thing under test, not evidence." compare name, type, default and semantics against the openai openapi yaml or the anthropic docs; classify as match, extension or incompatibility | the best-written skill in the set, in my read                                                                                                                                                         |
| `model-recipes`         | 10-06              | Roy Wang ([#57786](https://github.com/vllm-project/vllm/pull/57786))    | read recipes.vllm.ai JSON, not the YAML; copy commands verbatim, never splice flags                                                                                                                       | ties into the roadmap proposal to move `vllm-recipes` in-tree                                                                                                                                         |
| `transformers-upgrade`  | 10-09              | hmellor ([#60623](https://github.com/vllm-project/vllm/pull/60623))     | bump the pin vs. raise the floor; the unvendoring pass; dump before/after configs and meta-device models, then diff                                                                                       | tells the agent to "fan candidates out to parallel subagents". the only skill that assumes a multi-agent harness                                                                                      |

6 of the 8 ship an `agents/openai.yaml` (a codex UI card: display name, short description, default prompt). `pr-checklist` and `ci-fails-buildkite` don't.

---
