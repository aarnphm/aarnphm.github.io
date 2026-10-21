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
