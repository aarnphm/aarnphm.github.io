"""The in-container engine drivers of modal_interp.py: vLLM with the garden plugin, the torch
engine for the harvest, and the volume's rows."""

from __future__ import annotations

import json
from pathlib import Path


VOL = Path('/vol')
BF16 = VOL / 'gemma-bf16'

READOUT = (7, 21, 35)


def garden_cfg(**over) -> dict:
  cfg = dict(
    dict_path=str(VOL / 'plt'), readout=list(READOUT), top=4, steer_slots=8
  )
  cfg.update(over)
  return cfg


def engine(
  cfg: dict,
  *,
  eager: bool = False,
  n_max: int = 8192,
  prefix_cache: bool = True,
):
  """A vLLM engine over the bf16 dequant with the garden model class, in this process."""
  from vllm import EngineArgs
  from vllm.v1.engine.llm_engine import LLMEngine

  args = EngineArgs(
    model=str(BF16),
    dtype='bfloat16',
    max_model_len=4096,
    max_num_seqs=16,
    max_num_batched_tokens=n_max,
    gpu_memory_utilization=0.80,
    enforce_eager=eager,
    async_scheduling=False,  # runner.py reads a step's buffers before the next step writes them
    kv_sharing_fast_prefill=False,  # YOCO skips layers 24..41 on prompt rows
    enable_prefix_caching=prefix_cache,  # cached prompt rows are not recomputed: off for 'all' readouts
    hf_overrides={'architectures': ['GardenGemma4ForCausalLM'], 'garden': cfg},
  )
  return LLMEngine.from_engine_args(args)


def steps(
  eng,
  prompts: list[list[int]],
  specs: list[dict],
  max_tokens: int,
  tags: list[str],
):
  """Drive the engine step by step. Yields (tag, new token ids, {pos: feat} so far, finished)
  after every engine step that moved a request. The planner hands a step's codes to the sink
  inside step(), so a token's features are there when the token is."""
  from garden_vllm import sink_for
  from vllm import SamplingParams
  from vllm.sampling_params import RequestOutputKind

  sink = sink_for('memory')
  qs, feats = {}, {}
  for p, spec, tag in zip(prompts, specs, tags):
    qs[tag], feats[tag] = sink.open(tag), {}
    sp = SamplingParams(
      temperature=0.0,
      max_tokens=max_tokens,
      output_kind=RequestOutputKind.DELTA,
      extra_args={'garden': dict(spec, tag=tag)},
    )
    eng.add_request(tag, {'prompt_token_ids': p}, sp)
  try:
    while eng.has_unfinished_requests():
      for o in eng.step():
        tag, q = o.request_id, qs.get(o.request_id)
        if q is None:
          continue
        while not q.empty():
          for pos, f in q.get().get('readout', []):
            feats[tag][pos] = f
        yield tag, list(o.outputs[0].token_ids), feats[tag], o.finished
  finally:
    for tag in tags:
      sink.close(tag)


def run(
  eng,
  prompts: list[list[int]],
  specs: list[dict],
  max_tokens: int,
  tags: list[str],
) -> dict:
  """To completion: {tag: {'ids', 'prompt', 'readout': {pos: feat}}}."""
  out = {
    t: dict(ids=[], prompt=len(p), readout={}) for p, t in zip(prompts, tags)
  }
  for tag, new, feat, _ in steps(eng, prompts, specs, max_tokens, tags):
    out[tag]['ids'] += new
    out[tag]['readout'] = feat
  return out


def _torch_engine(taps=('mlp.in', 'mlp.out'), n_max=16384):
  import torch
  from garden_vllm import hf
  from garden_vllm.state import GardenConfig
  from transformers import AutoTokenizer, Gemma4ForCausalLM

  model = (
    Gemma4ForCausalLM
    .from_pretrained(
      str(BF16), dtype=torch.bfloat16, attn_implementation='sdpa'
    )
    .cuda()
    .eval()
  )
  model.requires_grad_(False)
  g = hf.install(
    model, GardenConfig(readout=(), steer_slots=0, taps=taps), n_max=n_max
  )
  return model, AutoTokenizer.from_pretrained(str(BF16)), g


def _rows(sidecar: str | None):
  files = sorted((VOL / 'rows').glob(f'{sidecar or ""}*.jsonl'))
  if not files:
    raise SystemExit('no rows in the volume: run ::rows on the Mac first')
  f = files[-1]
  return f.stem, [
    (j['row'], j['text']) for j in map(json.loads, f.read_text().splitlines())
  ]
