<!-- markdownlint-disable -->

# RFC-NNNN: {{TITLE}}

| Field       | Value      |
| ----------- | ---------- |
| **Author**  | {{AUTHOR}} |
| **Date**    | {{DATE}}   |
| **Status**  | Draft      |
| **Area**    | {{AREA}}   |
| **Tracker** | {{TICKET}} |

## Summary

<!-- 1-2 paragraphs. What is this RFC proposing, at a level readable by
someone unfamiliar with the specific subsystem? -->

## Motivation

<!-- Why is this change necessary? What problem does it solve?
Include concrete data: latency numbers, memory usage, throughput metrics,
user-reported issues, or scaling limitations.
Link to Linear tickets, GitHub issues, or Slack threads where relevant. -->

## Design

<!-- The technical meat. Include:
- Architecture diagrams (Mermaid or ASCII)
- Data flow descriptions
- API surface changes (Python, Mojo, C++ as applicable)
- Memory layout and lifecycle
- Interaction with existing subsystems (scheduler, KV cache, graph compiler, etc.)
- Configuration knobs and their defaults

Use code blocks for API sketches. Be specific about types, ownership,
and threading model. -->

### API Changes

<!-- Show the public-facing API diff. For Python APIs, show the function
signatures. For Mojo/C++ APIs, show the relevant headers or traits. -->

### Internal Changes

<!-- Describe changes to internal components. Reference specific files
and modules by path. -->

## Implementation Strategy

<!-- How will this be landed? Phases, feature flags, migration paths.
Each phase should be a reviewable, testable unit of work. -->

### Phase 1: {{PHASE_1_TITLE}}

<!-- Description, deliverables, estimated scope -->

### Phase 2: {{PHASE_2_TITLE}}

<!-- Description, deliverables, estimated scope -->

## Testing & Validation

<!-- How will correctness be verified?
- Unit tests (which targets?)
- Integration tests (which pipelines/models?)
- Logit verification against PyTorch reference
- Performance benchmarks (latency, throughput, memory)
- GPU configurations to test on (H100, B200, MI355) -->

## Performance Considerations

<!-- Expected impact on:
- Latency (prefill, decode, time-to-first-token)
- Throughput (tokens/sec, requests/sec)
- Memory (peak GPU memory, host memory)
- Scaling behavior (batch size, sequence length, model size)

Include back-of-envelope calculations or Fermi estimates where hard
numbers are not yet available. Express quantities as ratios where possible
(e.g., 2.3x throughput improvement at batch_size=32). -->

## Alternatives Considered

<!-- What other approaches were evaluated and why were they rejected?
Be specific about tradeoffs. -->

### {{ALT_1_TITLE}}

<!-- Description, why rejected -->

## Compatibility & Migration

<!-- Impact on:
- Existing pipeline configurations
- Serving API contracts (OpenAI, KServe, SageMaker)
- Weight formats (GGUF, SafeTensors, PyTorch)
- Public Python/Mojo APIs (breaking changes?)
- BEGIN_PUBLIC/END_PUBLIC considerations for open source components -->

## Open Questions

<!-- Unresolved design decisions. Each should have:
- The question
- Current thinking / leading option
- What data or discussion is needed to resolve it -->

## References

<!-- Links to papers, prior art, related RFCs, external docs. -->
