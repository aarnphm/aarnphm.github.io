from __future__ import annotations

from pathlib import Path


def max_subsequence(line: str, k: int) -> int:
  digits = [int(ch) for ch in line.strip()]
  n = len(digits)

  remove = n - k
  stack: list[int] = []

  for idx, d in enumerate(digits):
    remaining = n - idx - 1
    while remove and stack and stack[-1] < d and (len(stack) + remaining) >= k:
      stack.pop()
      remove -= 1
    stack.append(d)

  if remove:
    stack = stack[:-remove]
  if len(stack) > k:
    stack = stack[:k]

  return int("".join(str(x) for x in stack))


def total_output(lines, k) -> int:
  total = 0
  for line in lines: total += max_subsequence(line.strip(), k)
  return total


def sol() -> tuple[int, int]:
  with Path(__file__).with_suffix(".txt").open() as f:lines = f.readlines()
  return total_output(lines, 2), total_output(lines, 12)

if __name__ == "__main__": print(sol())
