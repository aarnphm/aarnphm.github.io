"""Keep literal OCR markup intact in native non-streaming chat responses."""

from __future__ import annotations


def literal_text(text: str, *_: object, **__: object) -> tuple[None, str]:
  return None, text


def no_reasoning_tokens(*_: object, **__: object) -> int:
  return 0


def install() -> None:
  from mlx_vlm.server import openai

  # This dedicated server transcribes documents, including printed <think>
  # tags. The native chat parser otherwise removes that source text even when
  # enable_thinking is false. Generation and batching remain upstream-owned.
  openai._split_thinking = literal_text
  openai._count_thinking_tag_tokens = no_reasoning_tokens
