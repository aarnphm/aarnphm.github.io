---
id: tool calling
tags:
  - ml
  - rfc
  - vllm
date: "2025-05-07"
description: unification of frontend parsers
modified: 2025-05-07 17:43:54 GMT-04:00
title: tool calling
transclude:
  title: false
---

## motivation

https://github.com/vllm-project/vllm/issues/11522 (with draft implementation at https://github.com/vllm-project/vllm/pull/11554)
aims to simplify the logics of the tool parser interface. However, this doesn't cover the cases for reasoning models (where we want to parse
tokens generated within the thinking budgets, etc. Our current solutions involves a reasoning parser, which will soon be running into the same
issue mentioned in #11522 when dealing with very long thinking budget). Additionally, the current implementations of tool calling are relatively
fragile, and not scalable when adding more tool format.

This RFC aims to build on top of some similar ideas from the RFC and unify both tool calling and reasoning parser logic for a more robust
way for us to move forward, especially with v0.10.x.

## proposed change

XGrammar team recently introduced us the next design iterations for structural tags, which aims to support more use case, including function/tool calling, reasoning, custom tags.

The workflow can be seen as follows:

- function/tool calling format for supported models (defined by the LLMEngine)
- Construct structural tags <- said tool/function calling format
- perform constrained decoding with supported backend (xgrammar/guidance)
- structural tag parser to convert string response -> structured objects (xgrammar will include their own parser here)

From vLLM perspective:

```bash
┌───────┐
│Prompt │
└───┬───┘
    │
    ▼
┌────────────────────────────────┐
│ vLLM (OpenAI‑compatible FE)    │
└───┬───────────────────┬────────┘
    │ [tool / func‑call │ reasoning]
    ▼                   │
┌──────────┐            │
│  Parser  │◀───────────┘
└───┬──────┘
    │
    ▼
┌───────────────────────────┐
│ Structural Tags Object    │
└───┬───────────────────────┘
    │
    ▼
┌────────────┐
│ LLM Engine │
└───┬────────┘
    │
    ▼
┌───────────────────────────┐
│ Structural Tags Object    │
└───┬───────────────────────┘
    │
    ▼
┌────────┐
│ Parser │
└───┬────┘
    │
    ▼
┌────────────────────────────┐
│ vLLM (OpenAI‑compatible FE)│
└───┬───────────────┬────────┘
    │               │
    ▼               ▼
┌───────┐      ┌────────┐
│Output │      │ (logs) │   ← optional
└───────┘      └────────┘

```

Aim:

- Structural tags as first class citizen
- Simplified and unified interface called `vllm.Parser`

There are a few compatibility matrix we need to consider:

| features              | function/tool calling | structured outputs | reasoning |
| --------------------- | --------------------- | ------------------ | --------- |
| function/tool calling | -                     |                    |           |
| structured outputs    |                       | -                  |           |
| reasoning             |                       |                    | -         |

_NOTE_: For reasoning logics, there are forced/non-forced mode (which is recently introduced by Qwen3-series of models)

A ad-hoc implementation of the parser would be

```python
class Parser:
  tool: bool = False
  reasoning: bool = False

  def parse_tool_call(self, structural_tag: StructuralTagResult) -> ToolCallResult: ...

  def parse_tool_call_stream(self, structural_tag: StructuralTagResult) -> DeltaToolCallResult: ...

  def parse_reasoning(self, structural_tag: StructuralTagResult) -> ReasoningResult: ...

  def parse_reasoning_stream(self, structural_tag: StructuralTagResult) -> DeltaReasoningResult: ...

class Llama3JSON(Parser, tool=True, name="llama3-json"): ...
class Pythonic(Parser, tool=True, name="pythonic"): ...

class DeepSeek(Parser, tool=True, reasoning=True, name="deepseek_r1"): ...
```

## Feedback period

1-2 weeks. wrt implementations, We will need to wait from the xgrammar team to have this support. Also cc Michal @mmoskal for your inputs on here as well.

## CC List

@mgroin @russelb @robertshaw-nm

## Any Other Thing

- We should probably move all of the tool/chat templates under `vllm/tools`
