"""Analyse a single attention head's ``W_OV`` matrix and low-rank projections.

Usage modes:

1. Local checkpoints: ``--weights path/to/model.safetensors``
2. Hugging Face hub models: ``--hf-repo Qwen/Qwen3-0.6B --auto-infer``

The script extracts the value and output projection matrices for a chosen layer/head,
computes ``W_OV``, prints spectral diagnostics, and reports reconstruction error when
truncating to a specified latent dimension.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from safetensors.torch import load_file as load_safetensors


def get_config_attr(config: Any, name: str) -> Any:
  if config is None:
    return None
  if hasattr(config, name):
    return getattr(config, name)
  if isinstance(config, dict):
    return config.get(name)
  return None


def load_local_config(weights_path: Path) -> Optional[dict[str, Any]]:
  candidates = [
    weights_path.with_suffix(".json"),
    weights_path.parent / "config.json",
  ]
  for candidate in candidates:
    if candidate.exists():
      try:
        return json.loads(candidate.read_text())
      except json.JSONDecodeError:
        continue
  return None


def load_state_from_file(path: Path) -> dict[str, Any]:
  suffix = path.suffix
  if suffix == ".safetensors":
    if load_safetensors is None:
      raise RuntimeError("Install safetensors to load .safetensors checkpoints")
    return load_safetensors(str(path))
  if suffix in {".pt", ".pth", ".bin"}:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
      return state["state_dict"]
    return state
  raise ValueError(f"unsupported checkpoint suffix: {suffix}")


def load_state_from_hub(
  repo: str,
  revision: str | None,
  cache_dir: Path | None,
  keep_model: bool = False,
  attn_impl: str | None = None,
) -> Tuple[dict[str, Any], Any, Optional[Any]]:
  try:
    from transformers import AutoModelForCausalLM
  except ImportError as exc:  # pragma: no cover
    raise ImportError("transformers is required when using --hf-repo") from exc

  kwargs: dict[str, Any] = dict(
    revision=revision,
    cache_dir=str(cache_dir) if cache_dir else None,
    dtype=torch.float32,
    trust_remote_code=True,
  )
  if attn_impl is not None:
    kwargs["attn_implementation"] = attn_impl

  model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
  state = model.state_dict()
  config = getattr(model, "config", None)
  if not keep_model:
    del model
    return state, config, None
  model.eval()
  return state, config, model


def fetch_matrix(state: dict[str, Any], key_template: str, layer: int) -> torch.Tensor:
  key = key_template.format(layer=layer)
  if key not in state:
    raise KeyError(f"missing key {key}")
  return state[key].float()


def split_qkv(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if source.shape[0] % 3 == 0:
    return torch.chunk(source, 3, dim=0)
  if source.shape[1] % 3 == 0:
    return torch.chunk(source, 3, dim=1)
  raise ValueError("cannot split qkv: tensor shape not divisible by three")


def slice_value_head(mat: torch.Tensor, head: int, num_heads: int) -> torch.Tensor:
  rows, cols = mat.shape
  if rows % num_heads == 0:
    span = rows // num_heads
    start = head * span
    return mat[start:start + span, :]
  if cols % num_heads == 0:
    span = cols // num_heads
    start = head * span
    return mat[:, start:start + span].T
  raise ValueError("value projection shape is incompatible with the requested head count")


def slice_output_head(mat: torch.Tensor, head: int, num_heads: int) -> torch.Tensor:
  rows, cols = mat.shape
  if cols % num_heads == 0:
    span = cols // num_heads
    start = head * span
    return mat[:, start:start + span]
  if rows % num_heads == 0:
    span = rows // num_heads
    start = head * span
    return mat[start:start + span, :].T
  raise ValueError("output projection shape is incompatible with the requested head count")


def compute_w_ov(
  v_matrix: torch.Tensor,
  o_matrix: torch.Tensor,
  head: int,
  num_heads: int,
  num_value_heads: int,
) -> torch.Tensor:
  if num_heads <= 0 or num_value_heads <= 0:
    raise ValueError("head counts must be positive")
  if num_heads % num_value_heads == 0:
    ratio = num_heads // num_value_heads
    value_head = head // ratio
  else:
    value_head = head % num_value_heads

  v_head = slice_value_head(v_matrix, value_head, num_value_heads).contiguous()
  o_head = slice_output_head(o_matrix, head, num_heads).contiguous()

  if o_head.shape[1] != v_head.shape[0]:
    if o_head.shape[0] == v_head.shape[0]:
      o_head = o_head.T.contiguous()
    elif o_head.shape[1] == v_head.shape[1]:
      v_head = v_head.T.contiguous()
    else:
      raise RuntimeError(
        "cannot align output/value projections: "
        f"O head {o_head.shape}, V head {v_head.shape}"
      )

  return o_head @ v_head


def latent_error(w_ov: torch.Tensor, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor, float]:
  u, s, vh = torch.linalg.svd(w_ov, full_matrices=False)
  if latent_dim < 1 or latent_dim > s.numel():
    raise ValueError("latent_dim must be within [1, rank]")
  approx = (u[:, :latent_dim] * s[:latent_dim]) @ vh[:latent_dim, :]
  err = torch.linalg.norm(w_ov - approx) / torch.linalg.norm(w_ov)
  return s, approx, err.item()


def matches_layer(key: str, layer: int) -> bool:
  return any(int(chunk) == layer for chunk in re.findall(r"\d+", key))


def generalise_template(key: str, layer: int) -> str:
  token = str(layer)
  substitutions = [
    (f".{token}.", ".{layer}."),
    (f"_{token}_", "_{layer}_"),
    (f"/{token}/", "/{layer}/"),
    (f"-{token}-", "-{layer}-"),
    (f".{token}_", ".{layer}_"),
    (f"_{token}.", "_{layer}."),
  ]
  for old, new in substitutions:
    if old in key:
      return key.replace(old, new)
  suffixes = [f".{token}", f"_{token}", f"/{token}", f"-{token}"]
  for old in suffixes:
    if key.endswith(old):
      return key[: -len(old)] + old.replace(token, "{layer}")
  prefixes = [f"{token}.", f"{token}_", f"{token}/", f"{token}-"]
  for old in prefixes:
    if key.startswith(old):
      return old.replace(token, "{layer}") + key[len(old):]
  return re.sub(token, "{layer}", key, count=1)


def find_key_for_component(state: dict[str, Any], layer: int, patterns: list[str]) -> str | None:
  lowered_items = [(key, key.lower()) for key in state.keys() if matches_layer(key, layer)]
  # prefer exact suffix matches
  for pattern in patterns:
    for key, low in lowered_items:
      if low.endswith(pattern):
        return key
  # fallback to substring matches
  for pattern in patterns:
    for key, low in lowered_items:
      if pattern in low:
        return key
  return None


def infer_templates(state: dict[str, Any], layer: int = 0) -> dict[str, str]:
  templates: dict[str, str] = {}
  qkv_key = find_key_for_component(state, layer, [
    "c_attn.weight",
    "qkv_proj.weight",
    "wqkv.weight",
    "qkv.weight",
    "attn.qkv.weight",
  ])
  if qkv_key:
    templates["qkv"] = generalise_template(qkv_key, layer)
  else:
    q_key = find_key_for_component(state, layer, ["q_proj.weight", "wq.weight", "query.weight"])
    k_key = find_key_for_component(state, layer, ["k_proj.weight", "wk.weight", "key.weight"])
    v_key = find_key_for_component(state, layer, ["v_proj.weight", "wv.weight", "value.weight"])
    if not (q_key and k_key and v_key):
      raise KeyError("Unable to infer Q/K/V key templates; please specify them manually.")
    templates["q"] = generalise_template(q_key, layer)
    templates["k"] = generalise_template(k_key, layer)
    templates["v"] = generalise_template(v_key, layer)
  o_key = find_key_for_component(state, layer, ["o_proj.weight", "out_proj.weight", "wo.weight", "proj.weight"])
  if not o_key:
    raise KeyError("Unable to infer output projection (O) key template.")
  templates["o"] = generalise_template(o_key, layer)
  return templates


def divisors(n: int) -> list[int]:
  result: set[int] = set()
  for i in range(1, int(math.sqrt(n)) + 1):
    if n % i == 0:
      result.add(i)
      result.add(n // i)
  return sorted(result)


def select_head_count(length: int) -> int:
  candidates = [d for d in divisors(length) if d > 1]
  if not candidates:
    return 1
  targets = (64, 128)
  best = None
  best_score = float("inf")
  for d in candidates:
    per_head = length // d
    score = min(abs(per_head - t) for t in targets)
    if score < best_score or (score == best_score and (best is None or d > best)):
      best = d
      best_score = score
  return best or candidates[0]


def compute_effective_rank(svals: torch.Tensor) -> float:
  if svals.numel() == 0:
    return 0.0
  power = svals.pow(2)
  total = power.sum()
  if total <= 0:
    return 0.0
  probs = power / total
  entropy = -(probs * torch.log(probs + 1e-12)).sum()
  return torch.exp(entropy).item()


def resolve_head_counts(
  num_heads: Optional[int],
  num_value_heads: Optional[int],
  config: Any,
  v_matrix: torch.Tensor,
  o_matrix: torch.Tensor,
) -> tuple[int, int]:
  cfg_heads = get_config_attr(config, "num_attention_heads")
  cfg_value_heads = (
    get_config_attr(config, "num_key_value_heads")
    or get_config_attr(config, "num_kv_heads")
    or get_config_attr(config, "num_key_value_groups")
  )

  if num_heads is None and cfg_heads is not None:
    num_heads = int(cfg_heads)
  if num_value_heads is None and cfg_value_heads is not None:
    num_value_heads = int(cfg_value_heads)

  if num_value_heads is None:
    num_value_heads = select_head_count(v_matrix.shape[0])
    warnings.warn(
      f"num_value_heads not provided; guessing {num_value_heads} based on value projection shape",
      RuntimeWarning,
    )

  if num_heads is None:
    num_heads = select_head_count(o_matrix.shape[1])
    warnings.warn(
      f"num_heads not provided; guessing {num_heads} based on output projection shape",
      RuntimeWarning,
    )

  return num_heads, num_value_heads


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", type=Path, help="checkpoint path (.pt/.bin/.safetensors)")
  parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo id, e.g. Qwen/Qwen3-0.6B")
  parser.add_argument("--hf-revision", type=str, default=None)
  parser.add_argument("--hf-cache", type=Path, default=None)
  parser.add_argument("--layer", type=int, default=0)
  parser.add_argument("--head", type=int, default=0)
  parser.add_argument("--num-heads", type=int, default=None)
  parser.add_argument("--num-value-heads", type=int, default=None)
  parser.add_argument("--v-key-template", type=str, default="transformer.h.{layer}.attn.v_proj.weight")
  parser.add_argument("--o-key-template", type=str, default="transformer.h.{layer}.attn.o_proj.weight")
  parser.add_argument("--qkv-key-template", type=str, default=None, help="optional combined qkv key template")
  parser.add_argument("--auto-infer", action="store_true", help="infer key templates from the state dict")
  parser.add_argument("--latent-dim", type=int, default=8)
  parser.add_argument("--tokens", type=int, default=32, help="synthetic tokens for cache reconstruction")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--plot", action="store_true", help="save a spectrum/cumulative energy plot")
  parser.add_argument("--plot-path", type=Path, default=None, help="output path for the saved plot")
  parser.add_argument("--prompt", type=str, default=None, help="text prompt for attention heatmap (HF models only)")
  parser.add_argument("--prompt-file", type=Path, default=None, help="read prompt text from file")
  parser.add_argument("--heatmap", action="store_true", help="save an attention heatmap for the prompt")
  parser.add_argument("--heatmap-path", type=Path, default=None, help="output path for the attention heatmap")
  args = parser.parse_args()

  if args.layer < 0 or args.head < 0:
    raise ValueError("layer and head must be non-negative")

  if args.weights is None and args.hf_repo is None:
    raise ValueError("Either --weights or --hf-repo must be provided")
  if args.weights is not None and args.hf_repo is not None:
    raise ValueError("Specify only one of --weights or --hf-repo")

  prompt_text: Optional[str] = args.prompt
  if args.prompt_file is not None:
    file_text = args.prompt_file.read_text().strip()
    prompt_text = file_text if file_text else prompt_text

  if args.heatmap and prompt_text is None:
    raise ValueError("--heatmap requires --prompt or --prompt-file")
  if prompt_text is not None and args.hf_repo is None:
    raise ValueError("Prompt-based attention requires --hf-repo")

  config = None
  model_obj: Optional[Any] = None
  tokenizer = None
  if args.hf_repo:
    keep_model = args.heatmap or prompt_text is not None
    attn_impl = "eager" if keep_model else None
    state, config, model_obj = load_state_from_hub(
      args.hf_repo,
      args.hf_revision,
      args.hf_cache,
      keep_model=keep_model,
      attn_impl=attn_impl,
    )
  else:
    state = load_state_from_file(args.weights)
    if args.auto_infer:
      local_cfg = load_local_config(args.weights)
      if local_cfg is not None:
        config = local_cfg

  if args.weights is not None:
    stem = Path(args.weights).stem
  elif args.hf_repo is not None:
    stem = args.hf_repo.split("/")[-1]
  else:
    stem = "latent_projection"

  templates: dict[str, str] = {}
  if args.auto_infer:
    templates = infer_templates(state, args.layer)

  qkv_template = args.qkv_key_template or templates.get("qkv")
  v_template = templates.get("v", args.v_key_template)
  o_template = templates.get("o", args.o_key_template)

  if qkv_template:
    qkv = fetch_matrix(state, qkv_template, args.layer)
    _, _, v_mat = split_qkv(qkv)
    v_matrix = v_mat.contiguous()
  else:
    v_matrix = fetch_matrix(state, v_template, args.layer)

  o_matrix = fetch_matrix(state, o_template, args.layer)

  num_heads = args.num_heads
  num_value_heads = args.num_value_heads

  num_heads, num_value_heads = resolve_head_counts(
    num_heads,
    num_value_heads,
    config,
    v_matrix,
    o_matrix,
  )

  if args.head >= num_heads:
    raise ValueError(
      f"requested head index {args.head} but only {num_heads} heads are available"
    )

  w_ov = compute_w_ov(v_matrix, o_matrix, args.head, num_heads, num_value_heads)

  svals, approx, rel_err = latent_error(w_ov, args.latent_dim)

  cond = (svals.max() / svals.min()).item()
  energy = svals.pow(2).cumsum(0) / svals.pow(2).sum()
  r_eff = compute_effective_rank(svals)

  torch.manual_seed(args.seed)
  residual = torch.randn(args.tokens, w_ov.shape[0])
  projected = (w_ov @ residual.T).T
  truncated = (approx @ residual.T).T
  cache_err = torch.linalg.norm(projected - truncated) / torch.linalg.norm(projected)

  print(
    f"layer={args.layer} head={args.head} num_heads={num_heads} "
    f"num_value_heads={num_value_heads}"
  )
  print(f"W_OV shape: {tuple(w_ov.shape)} rank: {torch.linalg.matrix_rank(w_ov)}")
  print(f"condition number: {cond:.3e} effective_rank: {r_eff:.2f}")
  print(
    f"latent_dim={args.latent_dim} rel_error={rel_err:.3e} "
    f"cache_error={cache_err.item():.3e}"
  )
  top = min(10, svals.numel())
  top_vals = " ".join(f"{val:.3e}" for val in svals[:top])
  print(f"singular values (top {top}): {top_vals}")
  energy_vals = " ".join(f"{val:.3f}" for val in energy[:top])
  print(f"cumulative energy (top {top}): {energy_vals}")

  if args.plot:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
    sigmas = svals.detach().cpu().numpy()
    cumulative = energy.detach().cpu().numpy()

    ax1.plot(sigmas)
    ax1.set_title("Singular Values")
    ax1.set_xlabel("index")
    ax1.set_ylabel("sigma")
    ax1.set_yscale("log")

    ax2.plot(cumulative)
    ax2.set_title("Cumulative Energy")
    ax2.set_xlabel("index")
    ax2.set_ylabel("fraction")
    ax2.set_ylim(0.0, 1.0)

    plot_path = args.plot_path or Path("content", "thoughts", "images", f"{stem}_layer{args.layer}_head{args.head}.png")

    fig.suptitle(
      f"layer {args.layer} head {args.head} | num_heads={num_heads} num_value_heads={num_value_heads} r_eff={r_eff:.2f}"
    )
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"saved plot -> {plot_path}")

  if args.heatmap:
    if model_obj is None or prompt_text is None:
      raise ValueError("Heatmap generation requires --hf-repo with an accessible model and prompt text")
    try:
      from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
      raise ImportError("transformers is required for --heatmap") from exc

    tokenizer = AutoTokenizer.from_pretrained(
      args.hf_repo,
      revision=args.hf_revision,
      cache_dir=str(args.hf_cache) if args.hf_cache else None,
      trust_remote_code=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    with torch.no_grad():
      outputs = model_obj(
        **encoded,
        output_attentions=True,
        use_cache=False,
      )

    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
      raise RuntimeError("Model did not return attentions; ensure output_attentions=True is supported")
    if args.layer >= len(attentions):
      raise ValueError(f"layer index {args.layer} exceeds available layers ({len(attentions)})")

    layer_attn = attentions[args.layer]
    if args.head >= layer_attn.shape[1]:
      raise ValueError(f"head index {args.head} exceeds available heads ({layer_attn.shape[1]})")

    heat = layer_attn[0, args.head].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    max_tokens = 64
    if heat.shape[0] > max_tokens:
      heat = heat[:max_tokens, :max_tokens]
      tokens = tokens[:max_tokens]

    fig, ax = plt.subplots(
      figsize=(max(6, len(tokens) * 0.45), max(6, len(tokens) * 0.45)),
      constrained_layout=True,
    )
    im = ax.imshow(heat, cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("source token")
    ax.set_ylabel("target token")
    ax.set_title(f"Attention heatmap | layer {args.layer} head {args.head}")

    heatmap_path = args.heatmap_path or Path("content", "thoughts", "images", f"{stem}_layer{args.layer}_head{args.head}_heatmap.png")
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved heatmap -> {heatmap_path}")


if __name__ == "__main__":
  main()
