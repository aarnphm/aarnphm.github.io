# Spine harness failure cases

Exercise the public CLI before implementing its owners. Use real local images,
SVG parsing, rasterization, filesystem state and subprocesses. Model transport
fixtures do not establish model quality; live provider smoke runs are recorded
separately.

- Planning must preserve the full scope, including books without references.
- A catalogue entry delivered as SVG must keep its retained native reference for
  future trace requests. Missing native-reference metadata must fail admission;
  the delivered SVG cannot become the source image for another trace.
- A changed front, reference, catalogue or policy must invalidate dependent work.
- Planning over an existing run must preserve its state and refuse replacement.
- Repeated book placements must share an edition job while keeping note identities.
- An image embedded in SVG, external resource, live font, malformed XML, invalid
  coordinate, cyclic resource or oversized path must fail before rasterization.
- A failed replacement must preserve the last valid candidate and its evidence.
- Raster output must decode, retain the reference aspect and obey both export and
  native-reference dimensions. Repeating the same recipe must reproduce hashes.
- SVG delivery requires a zero-origin viewBox with width and height equal to the
  native reference. A shifted or scaled coordinate frame must preserve the prior
  candidate on rejection, because delivery metadata cannot represent that frame.
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
- Sol tracing may write inside its invocation workspace; reviews run read-only.
  Caw uses ask mode without force or blanket MCP approval. Argon and Caw tracing
  assignments permit only invocation-local output; review permits no edits.
  Browser tools are disabled by instruction.
  Provider process modes are recorded honestly, without claiming OS isolation for
  Argon or remote MCP servers.
- Concurrent coordinators must fail ownership checks. Dead locks need an explicit
  recovery command and recorded evidence; a live process cannot be recovered.
- Timeouts, output overflow and cancellation stop only owned processes and preserve
  raw logs. Rate-limit and billing failures never enable paid overage or switch models.
- A configured model budget above the former four-minute ceiling must reach the
  provider without truncation. The invocation deadline still bounds every child,
  including when its worker budget is longer. Short diagnostic timeouts must work.
- Resume must skip completed stages and preserve successful independent reviews.
- Repeated rejected candidates must eventually require attention instead of looping.
- Export must require every configured reviewer to pass and revalidate all bytes.
- Grok tracing must pin Extra High and 500K context. Its Caw implementation cannot
  review itself; Astra, Opus and Sol must inspect that candidate independently.
- File-backed traces must remain inside the bound invocation, match their hash
  and byte budget, and pass the same SVG validation as inline traces. Escaped
  paths, changed bytes and mismatched requests must preserve the last candidate.
- Inspection for SVG delivery must preserve rendered RGB values. Lossy WebP chroma
  subsampling can turn thin navy strokes on red into dark red, even at quality 80.
  Review image encoding must be explicit and reproducible.
- A completed Grok trace can append prose to its final message. Its exact JSON
  result file must be bound to the invocation and validated independently of the
  native completion event; the raw terminal message must remain available.
  Export writes a candidate bundle under the run, never the production catalogue.
- A run with blocked jobs is incomplete even when all remaining jobs pass.
- A pass without explicit source correspondence and full-face coverage at both
  tiers must fail. A supplied-raster pass cannot certify served browser placement.
- A review with changed invocation identity, job binding or image manifest fails
  even when the returned response repeats the expected candidate hash.
- Duplicate JSON keys must fail before dataclass decoding can overwrite a binding.
- Native model provenance must remain distinct from requested configuration.
- Post-terminal events, unfinished tools, changed conversation IDs and stale
  assistant responses after later activity must not satisfy transport admission.
- Coordinator calls must disable detached automatic naming jobs.

Run `bash harness/bookcase-spines/scripts/run.sh test`. Evidence is written under
`docs/verification/bookcase-spine-harness/artifacts/e2e/` with exact CLI inputs and
outputs. Provider transport evidence lives in each isolated run's `invocations/` directory;
candidate pixels live in `attempts/`. This is a failure inventory, not a claim
that every listed case has an executed regression. The verification report lists
the checks actually run.
