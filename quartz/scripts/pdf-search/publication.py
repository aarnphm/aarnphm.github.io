"""Verify a PDF generation before switching readers and pruning replaced artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import TypedDict

import numpy as np


class VectorShard(TypedDict):
  path: str
  rows: int
  rowOffset: int
  byteLength: int
  byteStride: int
  sha256: str


def atomic_write(path: Path, payload: bytes) -> None:
  temporary = path.with_suffix('.tmp')
  temporary.write_bytes(payload)
  temporary.replace(path)


class PassageShard(TypedDict):
  path: str
  rows: int
  rowOffset: int
  byteLength: int
  sha256: str


def publish(
  output: Path,
  generation: str,
  payload: bytes,
  shards: list[VectorShard],
  passage_shards: list[PassageShard],
  rows: int,
) -> list[str]:
  target = output / generation
  atomic_write(target / 'manifest.json', payload)
  saved = (target / 'manifest.json').read_bytes()
  if saved != payload or json.loads(saved)['generation'] != generation:
    raise ValueError('PDF manifest changed before publication')
  offset = 0
  for shard in shards:
    raw = (target / Path(shard['path']).name).read_bytes()
    if (
      shard['rowOffset'] != offset
      or len(raw) != shard['byteLength']
      or len(raw) != shard['rows'] * 768 * 4
      or hashlib.sha256(raw).hexdigest() != shard['sha256']
    ):
      raise ValueError(f'Invalid saved PDF vector shard: {shard["path"]}')
    matrix = np.frombuffer(raw, dtype=np.float32).reshape(-1, 768)
    if not np.isfinite(matrix).all() or not np.allclose(
      np.linalg.norm(matrix, axis=1), 1, atol=1e-4
    ):
      raise ValueError(f'Invalid saved PDF unit vectors: {shard["path"]}')
    offset += shard['rows']
  if offset != rows:
    raise ValueError('Saved PDF shards do not cover the passage inventory')
  if len(passage_shards) != len(shards):
    raise ValueError('PDF passage shards do not align with the vector shards')
  for vector_shard, shard in zip(shards, passage_shards, strict=True):
    raw = (target / Path(shard['path']).name).read_bytes()
    if (
      shard['rows'] != vector_shard['rows']
      or shard['rowOffset'] != vector_shard['rowOffset']
      or len(raw) != shard['byteLength']
      or hashlib.sha256(raw).hexdigest() != shard['sha256']
      or len(json.loads(raw)) != shard['rows']
    ):
      raise ValueError(f'Invalid saved PDF passage shard: {shard["path"]}')
  atomic_write(output / 'manifest.json', payload)
  return prune_directories(output, generation)


def prune_directories(root: Path, current: str) -> list[str]:
  removed = []
  for directory in root.iterdir():
    if (
      directory.name != current
      and re.fullmatch(r'[a-f0-9]{64}', directory.name)
      and not directory.is_symlink()
      and directory.is_dir()
    ):
      shutil.rmtree(directory)
      removed.append(directory.name)
  return sorted(removed)


def prune_vectors(
  root: Path, fingerprint: str, retained: set[str]
) -> dict[str, object]:
  removed = prune_directories(root, fingerprint)
  files = 0
  for path in (root / fingerprint).iterdir():
    if (
      re.fullmatch(r'[a-f0-9]{64}\.npy', path.name)
      and path.stem not in retained
      and not path.is_symlink()
      and path.is_file()
    ):
      path.unlink()
      files += 1
  return {'removedContracts': removed, 'removedVectors': files}
