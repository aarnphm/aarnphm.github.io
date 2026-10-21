# Spine harness failure cases

Exercise the public CLI before implementing its owners. Use real local images,
SVG parsing, rasterization, filesystem state and subprocesses. Model transport
fixtures do not establish model quality; live provider smoke runs are recorded
separately.

- Planning must preserve the full scope, including books without references.
- A changed front, reference, catalogue or policy must invalidate dependent work.
- Planning over an existing run must preserve its state and refuse replacement.
- Repeated book placements must share an edition job while keeping note identities.
- An image embedded in SVG, external resource, live font, malformed XML, invalid
  coordinate, cyclic resource or oversized path must fail before rasterization.
- A failed replacement must preserve the last valid candidate and its evidence.
- Raster output must decode, retain the reference aspect and obey both export and
  native-reference dimensions. Repeating the same recipe must reproduce hashes.
- Deterministic validation alone cannot accept artwork or publish it.
- Review must bind exact candidate, input, rubric, rendered images and provider
  invocation. Missing/partial/error results cannot satisfy a review.
- Argon High implements by default; Sol Medium remains selectable. The default
  CLI route must follow the planned worker. A worker cannot approve its own trace.
- Astra, Opus and Caw each supply a required independent receipt. Omitting any
  required role must fail policy validation before work starts.
- Opus must use the configured API-credit key in the child environment only.
  Missing credentials must fail without subscription fallback. Neither request
  files nor process receipts may contain key values. Claude tools are limited to
  reading the supplied images; no shell, edits, MCP or further delegation.
- Claude errors, denied tools, absent image reads or missing structured results
  must never be counted as a completed visual review.
- Sol runs read-only; Caw uses ask mode without force or blanket MCP approval.
  Argon receives a read-only assignment. Browser tools are disabled by instruction.
  Provider process modes are recorded honestly, without claiming OS isolation for
  Argon or remote MCP servers.
- Concurrent coordinators must fail ownership checks. Dead locks need an explicit
  recovery command and recorded evidence; a live process cannot be recovered.
- Timeouts, output overflow and cancellation stop only owned processes and preserve
  raw logs. Rate-limit and billing failures never enable paid overage or switch models.
- Resume must skip completed stages and preserve successful independent reviews.
- Repeated rejected candidates must eventually require attention instead of looping.
- Export must require every configured reviewer to pass and revalidate all bytes.
  Export writes a candidate bundle under the run, never the production catalogue.
- A run with blocked jobs is incomplete even when all remaining jobs pass.

Run `bash harness/bookcase-spines/scripts/run.sh test`. Evidence is written under
`docs/verification/bookcase-spine-harness/artifacts/e2e/` with exact CLI inputs and
outputs. Provider transport evidence lives in each isolated run's `invocations/` directory;
candidate pixels live in `attempts/`. This is a failure inventory, not a claim
that every listed case has an executed regression. The verification report lists
the checks actually run.
