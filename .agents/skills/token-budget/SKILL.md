---
name: token-budget
description: 'Run a token budget pass: split Garden files over 5000 tokens into deep modules with the harness in harness/token-budget (inventory, Sol and Argon boundary proposals, validation, decisions, then a w worktree, move, deepen, verify and land as a staged patch). Use it when asked to "run a pass on token-budget", "create deep modules", split an over-budget file, or when the pre-commit hook rejects a file.'
---

# Token budget pass

The pre-commit hook `harness/token-budget/check.py` rejects agent-authored files over 5000 tokens (tokens = bytes / 4, so 20000 bytes). A split that cuts a file in source order gives many files and no better modules. A pass instead asks two workers for a module boundary, checks both proposals with deterministic code, and implements the one that survives in a `w` worktree. The design and the pilot results are in `docs/plans/token-budget-refactor.md` (local only); read it before you change the harness.

## Words

Use these words in briefs, reports and reviews:

- **Module**: the original path (the entry) plus a private folder `<path without extension>/` that holds the implementation. Code outside the module imports only the entry.
- **Interface**: the names the entry exports, with their invariants, ordering rules and error modes. The interface is the test surface.
- **Depth**: a lot of behaviour behind a small interface. A deep module gives leverage to each caller.
- **Seam**: a place where behaviour can change without edits at the call site. One adapter makes a hypothetical seam; two adapters make a real one.
- **Deletion test**: if you delete a name from the interface, does complexity move into the callers? If no, hide it.

## Run the pass (stages 0 to 4)

From the main checkout:

```sh
harness/token-budget/pass.sh --top 3          # the 3 files with the most tokens x churn
harness/token-budget/pass.sh <path>...        # the named files
TB_WORKERS= harness/token-budget/pass.sh ...  # baseline only, no model calls
```

A pass with both workers takes 10 to 20 minutes. Run it in the background and do other work. It writes `work/refactor/pass-<date>/` (ignored by Git) and prints, per target, each proposal's status, its `agent-decide` action and its metrics, and the agreement (ARI) between proposals.

What the stages do:

| Stage       | Command                                                      | Output                                                                        |
| ----------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| 0 inventory | `check.py --all --json`, `inventory.py --top N` or `--paths` | `inventory.json`, `graph.json` (tree-sitter blocks, importers, interface use) |
| 1 brief     | `inventory.py --briefs <run>`                                | `<slug>/brief.md`                                                             |
| 2 propose   | `dispatch.sh <run> sol`, `dispatch.sh <run> argon`           | `<slug>/<worker>/` with the final message                                     |
| 3 validate  | `collect.sh <run>` (adds `baseline.py`, runs `validate.py`)  | `<slug>/validation.json`, `<slug>/decisions/*.decision-request.json`          |
| 4 decide    | `agent-decide < request` (assessor none)                     | `<slug>/decisions/*.decision.json`                                            |

Sol runs through `codex exec -s read-only`; it minimises the interface. Argon runs through `argon --browser none`; it keeps the most common caller simple. Both are read-only. A watchdog stops an Argon run 30 seconds after its result event, because agy can keep running after it answers.

## Choose a proposal

1. A proposal is eligible only when its `validation.json` status is `pass` and its decision is `verify`. Any failed check (blob-current, partition, budget, acyclic, entry-only, interface-kept, seam-adapters, paths, scss-order) rules it out.
2. Two eligible proposals with ARI 0.8 or more agree. Take the one with fewer exports after the split and a smaller `cutShare`.
3. With ARI under 0.8, arbitrate. Read both proposals against the four semantic checks in the decision request (deletion test, one concept per file, complete interface, tests at the interface). Write down why you chose one in `<slug>/arbitration.md`.
4. The baseline (contiguous cuts) is a null model. If it beats both workers, say so in the report and stop for that target.
5. When every proposal fails `acyclic` and the warnings say `closure state crosses A -> B`, do stage 4a first.

### Stage 4a: hoist closure state

The target is one large closure whose statements share `let` locals. Before any move, in the worktree, move the shared locals into one state object that the closure owns, with no new files. Check it like a deepen step, then run the pass again on that path from the worktree (`TB_WORKSPACE=<worktree>`). The blocks then reference the state object, and the move is mechanical.

## Implement (stages 5 to 7)

One target at a time, one worktree per target:

```sh
w garden tb-<slug>          # ~/workspace/worktrees/garden/tb-<slug>, branch aarnphm/tb-<slug>, from main
```

The worktree starts from HEAD. If the target's index or working-tree copy differs from HEAD, wait until aarnphm commits it. Compare the worktree blob (`git -C <wt> rev-parse HEAD:<path>`) with the `blob` in `graph.json` before you start.

1. **Move.** Create the files the proposal names and move blocks into them verbatim. Fix only imports and exports. Do not rename, reformat or rewrite. Check the move: the non-import lines of the original and of the new files must match.

   ```sh
   strip() { rg -v '^\s*(import\b|export \{|export \* from|\} from )' | sd '^export (default )?' '' | sort; }
   git -C <wt> show main:<path> | strip > before.txt
   cat <wt>/<path> <wt>/<folder>/* | strip > after.txt
   diff before.txt after.txt
   ```

2. **Deepen.** Hide the exports the proposal hides. Update importers to use the entry. Move tests that import internals to the entry interface, and delete tests of internals when tests at the interface cover the same behaviour. Keep every behaviour that a test covers.

3. **Verify.** In the worktree: `pnpm exec tsgo --noEmit`, `pnpm exec oxlint <changed files>`, the affected tests, and the hook on the new files: `git -C <wt> add <module paths>`, then `(cd <wt> && ~/workspace/garden/harness/token-budget/check.py --paths <module paths>)`. The hook reads the index of the directory it runs in, so it measures the worktree. For client scripts, land first, then use the running `quartz/scripts/dev.ts` watcher in the main checkout: wait for the next `build:ready` and check the rendered page. Then ask Argon, in a new conversation, to review the diff with the proposal as acceptance criteria.

4. **Land.** From the main checkout:

   ```sh
   git -C <wt> diff --binary main -- <module paths> > <run>/<slug>/land.patch
   git apply --index --3way <run>/<slug>/land.patch
   ```

   This stages only the module's paths. Do not commit. aarnphm commits, then `w --rm garden tb-<slug>` removes the worktree.

## Rules

- Workers are read-only. They do not edit, stage, move or delete files, and they do not run builds, tests, installs or browsers.
- Never enable paid usage for Cursor, caw or any provider. If caw reports a billing block, use Codex or Argon.
- `agent-decide` uses assessor `none`. Use the Composer assessor only with a current billing receipt, as the `decision` skill describes.
- Other agents work in this tree. Stage only your own paths, never commit, never rewrite history, and stop only processes that you started.
- Do not mark agent-authored code exempt in `.gitattributes` to pass the hook. Split it.
- When a target's blob changes during a pass, its results are stale. Run the pass again.

## Report

For each target, give aarnphm: the chosen proposal (or none) and why, the files and their token counts, exports before and after, the checks you ran with their results, the path of `land.patch`, and anything you could not verify. Record progress in `progress.md` at the workspace root.

## Files

| File                                                    | Role                                                                                            |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `check.py`                                              | Pre-commit hook; `--all`, `--paths`, `--json`                                                   |
| `common.py`                                             | Budget constants, `cuts`, Git helpers                                                           |
| `blocks.py`                                             | Tree-sitter (ast-grep-py) blocks for scripts, line blocks for styles, references between blocks |
| `inventory.py`                                          | Graph, importers, interface use, `--top`, briefs                                                |
| `brief.py`                                              | The brief the workers read                                                                      |
| `dispatch.sh`                                           | Stage 2 workers                                                                                 |
| `collect.sh`, `baseline.py`, `validate.py`, `checks.py` | Stage 3                                                                                         |
| `pass.sh`                                               | Stages 0 to 4 in one command                                                                    |
