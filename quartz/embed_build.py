#!/usr/bin/env python3
import os
import json
import argparse
import hashlib
import math
from pathlib import Path
from collections.abc import Iterable
import random

import numpy as np


DEFAULT_VLLM_URL = os.environ.get("VLLM_URL") or os.environ.get("VLLM_EMBED_URL") or "http://127.0.0.1:8000/v1/embeddings"


def load_jsonl(fp: str) -> Iterable[dict]:
  with open(fp, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      yield json.loads(line)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
  # x: [N, D]
  norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
  norms[norms == 0] = 1.0
  return x / norms


def write_shards(vectors: np.ndarray, shard_size: int, dtype: str, out_dir: Path) -> list[dict]:
  out_dir.mkdir(parents=True, exist_ok=True)
  rows, dims = vectors.shape
  shards_meta: list[dict] = []
  np_dtype = np.float16 if dtype == "fp16" else np.float32
  bytes_per_value = np.dtype(np_dtype).itemsize
  row_offset = 0
  for si, start in enumerate(range(0, rows, shard_size)):
    end = min(start + shard_size, rows)
    shard = vectors[start:end]  # [n, dims]
    bin_path = out_dir / f"vectors-{si:03d}.bin"
    payload = shard.astype(np_dtype, copy=False).tobytes(order="C")
    digest = hashlib.sha256(payload).hexdigest()
    with open(bin_path, "wb") as f:
      f.write(payload)
    shard_rows = int(shard.shape[0])
    shards_meta.append(
      {
        "path": f"/embeddings/{bin_path.name}",
        "rows": shard_rows,
        "rowOffset": row_offset,
        "byteLength": len(payload),
        "sha256": digest,
        "byteStride": dims * bytes_per_value,
      },
    )
    row_offset += shard_rows
  return shards_meta


def write_hnsw_graph(levels: list[list[list[int]]], rows: int, out_path: Path) -> tuple[list[dict], str]:
  out_path.parent.mkdir(parents=True, exist_ok=True)
  offset = 0
  meta: list[dict] = []
  digest = hashlib.sha256()
  with open(out_path, "wb") as f:
    for lvl in levels:
      indptr = np.zeros(rows + 1, dtype=np.uint32)
      edge_accum: list[int] = []
      for idx in range(rows):
        neighbors = lvl[idx] if idx < len(lvl) else []
        indptr[idx + 1] = indptr[idx] + len(neighbors)
        edge_accum.extend(neighbors)
      indptr_bytes = indptr.tobytes(order="C")
      indptr_offset = offset
      f.write(indptr_bytes)
      digest.update(indptr_bytes)
      offset += len(indptr_bytes)

      if edge_accum:
        indices = np.asarray(edge_accum, dtype=np.uint32)
        indices_bytes = indices.tobytes(order="C")
      else:
        indices = np.zeros(0, dtype=np.uint32)
        indices_bytes = indices.tobytes(order="C")
      indices_offset = offset
      f.write(indices_bytes)
      digest.update(indices_bytes)
      offset += len(indices_bytes)

      meta.append(
        {
          "level": len(meta),
          "indptr": {
            "offset": indptr_offset,
            "elements": int(indptr.shape[0]),
            "byteLength": len(indptr_bytes),
          },
          "indices": {
            "offset": indices_offset,
            "elements": int(indices.shape[0]),
            "byteLength": len(indices_bytes),
          },
        },
      )
  return meta, digest.hexdigest()



def embed_vllm(texts: list[str], model_id: str, vllm_url: str) -> np.ndarray:
  import requests
  out: list[np.ndarray] = []
  bs = 64
  for i in range(0, len(texts), bs):
    batch = texts[i : i + bs]
    r = requests.post(vllm_url, json={"model": model_id, "input": batch}, timeout=300)
    r.raise_for_status()
    data = r.json()["data"]
    out.extend([np.asarray(item["embedding"], dtype=np.float32) for item in data])
  return np.stack(out, axis=0)


def embed_hf(texts: list[str], model_id: str, device: str) -> np.ndarray:
  # Prefer sentence-transformers for E5 and similar embed models
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer(model_id, device=device)
  # E5 family benefits from prefixes; treat docs as passages
  prefixed = [f"passage: {t}" for t in texts]
  vecs = model.encode(
    prefixed,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
  )
  return vecs.astype(np.float32, copy=False)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--jsonl", default="public/embeddings-text.jsonl")
  ap.add_argument("--model", default=os.environ.get("SEM_MODEL", "intfloat/multilingual-e5-large"))
  ap.add_argument("--dims", type=int, default=int(os.environ.get("SEM_DIMS", "1024")))
  ap.add_argument("--dtype", choices=["fp16", "fp32"], default=os.environ.get("SEM_DTYPE", "fp32"))
  ap.add_argument("--shard-size", type=int, default=int(os.environ.get("SEM_SHARD", "1024")))
  ap.add_argument("--out", default="public/embeddings")
  ap.add_argument("--use-vllm", action="store_true", default=bool(os.environ.get("USE_VLLM", "")))
  ap.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
  args = ap.parse_args()

  recs = list(load_jsonl(args.jsonl))
  if not recs:
    print("No input found in public/embeddings-text.jsonl; run the site build first to emit JSONL.")
    return

  ids = [r["slug"] for r in recs]
  titles = [r.get("title", r["slug"]) for r in recs]
  texts = [r["text"] for r in recs]

  if args.use_vllm:
    vecs = embed_vllm(texts, args.model, args.vllm_url)
  else:
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    vecs = embed_hf(texts, args.model, device)

  # Coerce dims and re-normalize
  if vecs.shape[1] != args.dims:
    if vecs.shape[1] > args.dims:
      vecs = vecs[:, : args.dims]
    else:
      vecs = np.pad(vecs, ((0, 0), (0, args.dims - vecs.shape[1])))
  vecs = l2_normalize_rows(vecs.astype(np.float32, copy=False))

  out_dir = Path(args.out)
  shards = write_shards(vecs, args.shard_size, args.dtype, out_dir)

  # Build a lightweight HNSW graph and store it in a compact binary layout
  def hnsw_build(data: np.ndarray, M: int = 16, efC: int = 200, seed: int = 0) -> dict:
    rng = random.Random(seed)
    N, D = data.shape
    levels: list[list[list[int]]] = []  # levels[L][i] = neighbors of node i at level L

    # random level assignment using 1/e distribution
    node_levels = []
    for _ in range(N):
      lvl = 0
      while rng.random() < 1 / math.e:
        lvl += 1
      node_levels.append(lvl)
    max_level = max(node_levels) if N > 0 else 0
    for _ in range(max_level + 1):
      levels.append([[] for _ in range(N)])

    def sim(i: int, j: int) -> float:
      return float((data[i] * data[j]).sum())

    entry = 0 if N > 0 else -1

    def search_layer(q: int, ep: int, ef: int, L: int) -> list[int]:
      if ep < 0:
        return []
      visited = set()
      cand: list[tuple[float, int]] = []
      top: list[tuple[float, int]] = []
      def push(node: int):
        if node in visited:
          return
        visited.add(node)
        cand.append((sim(q, node), node))
      push(ep)
      while cand:
        cand.sort(reverse=True)
        s, v = cand.pop(0)
        if len(top) >= ef and s <= top[-1][0]:
          break
        top.append((s, v))
        for u in levels[L][v]:
          push(u)
      top.sort(reverse=True)
      return [n for _, n in top]

    for i in range(N):
      if i == 0:
        continue
      lvl = node_levels[i]
      ep = entry
      for L in range(max_level, lvl, -1):
        c = search_layer(i, ep, 1, L)
        if c:
          ep = c[0]
      for L in range(min(max_level, lvl), -1, -1):
        W = search_layer(i, ep, efC, L)
        # Select top M by similarity
        neigh = sorted(((sim(i, j), j) for j in W if j != i), reverse=True)[:M]
        for _, e in neigh:
          if e not in levels[L][i]:
            levels[L][i].append(e)
          if i not in levels[L][e]:
            levels[L][e].append(i)

    # trim neighbors to M
    for L in range(len(levels)):
      for i in range(N):
        if len(levels[L][i]) > M:
          # keep top M by sim
          nb = levels[L][i]
          nb = sorted(nb, key=lambda j: sim(i, j), reverse=True)[:M]
          levels[L][i] = nb

    return {
      "M": M,
      "efConstruction": efC,
      "entryPoint": entry,
      "maxLevel": max_level,
      "levels": levels,
    }

  hnsw = hnsw_build(vecs, M=16, efC=200)
  hnsw_meta, hnsw_sha = write_hnsw_graph(hnsw["levels"], int(vecs.shape[0]), out_dir / "hnsw.bin")

  manifest = {
    "version": 2,
    "dims": args.dims,
    "dtype": args.dtype,
    "normalized": True,
    "rows": int(vecs.shape[0]),
    "shardSizeRows": args.shard_size,
    "vectors": {
      "dtype": args.dtype,
      "rows": int(vecs.shape[0]),
      "dims": args.dims,
      "shards": shards,
    },
    "ids": ids,
    "titles": titles,
    "hnsw": {
      "M": hnsw["M"],
      "efConstruction": hnsw["efConstruction"],
      "entryPoint": hnsw["entryPoint"],
      "maxLevel": hnsw["maxLevel"],
      "graph": {
        "path": "/embeddings/hnsw.bin",
        "sha256": hnsw_sha,
        "levels": hnsw_meta,
      },
    },
    "bm25": {
      "path": "/embeddings/bm25.json",
    },
  }
  (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
  print(f"Wrote {len(shards)} vector shard(s), HNSW graph, and manifest to {out_dir}")

if __name__ == "__main__":
  main()
