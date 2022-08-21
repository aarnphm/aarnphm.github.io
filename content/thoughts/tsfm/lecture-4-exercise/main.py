import argparse
import math
import os
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from pathlib import Path


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str.lower() not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return mapping[dtype_str.lower()]


def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: bool = True,
) -> torch.Tensor:
    """
    Naive scaled dot-product attention.

    Shapes:
      - query, key, value: [batch, heads, seq_len, head_dim]
      - returns: [batch, heads, seq_len, head_dim]
    """
    assert query.dim() == 4 and key.dim() == 4 and value.dim() == 4
    assert query.shape == value.shape
    assert query.shape[:-1] == key.shape[:-1]
    batch_size, num_heads, seq_len, head_dim = query.shape

    # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
    scores = torch.matmul(query, key.transpose(-2, -1))

    if scale:
        scores = scores / math.sqrt(head_dim)

    attn = torch.softmax(scores, dim=-1)

    # [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
    out = torch.matmul(attn, value)

    return out


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    FlashAttention via torch.nn.functional.scaled_dot_product_attention.
    Prefers the Flash backend on CUDA if available; falls back gracefully otherwise.
    """
    # Try to force Flash backend; if not available, fall back to default selection
    try:
        cm = torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        )
    except Exception:
        cm = nullcontext()
    with cm:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
        )


def make_inputs(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    return q, k, v


def profile_with_torch_profiler(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup_steps: int,
    active_steps: int,
    logdir: Optional[str],
    impl: str = "naive",
):
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    is_cuda = q.device.type == "cuda"
    activities = [ProfilerActivity.CPU]
    if is_cuda:
        activities.append(ProfilerActivity.CUDA)

    prof_schedule = schedule(wait=0, warmup=warmup_steps, active=active_steps, repeat=1)

    on_ready = None
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        on_ready = tensorboard_trace_handler(logdir)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=on_ready,
    ) as prof:
        total_steps = warmup_steps + active_steps
        for _ in range(total_steps):
            if impl == "naive":
                _ = naive_attention(q, k, v, scale=True)
            elif impl == "sdpa":
                _ = sdpa_attention(q, k, v)
            else:
                raise ValueError(f"Unknown impl: {impl}")
            prof.step()


def main():
    parser = argparse.ArgumentParser(description="Naive attention with torch.profiler")
    parser.add_argument("--B", type=int, default=4, help="Batch size")
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument("--S", type=int, default=1024, help="Sequence length")
    parser.add_argument("--D", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "float32", "fp16", "float16", "bf16", "bfloat16"],
        help="Tensor dtype",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device"
    )
    parser.add_argument("--iters", type=int, default=20, help="Active profiler steps / timing iters")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps before measuring")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="If set, write PyTorch profiler traces for TensorBoard here",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="naive",
        choices=["naive", "sdpa"],
        help="Attention implementation to run (naive or sdpa/flash)",
    )

    args = parser.parse_args()

    device = get_device(args.device)
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available; falling back to CPU.")

    dtype = parse_dtype(args.dtype)

    q, k, v = make_inputs(args.B, args.H, args.S, args.D, dtype, device, seed=args.seed)

    profile_with_torch_profiler(
        q, k, v,
        warmup_steps=args.warmup,
        active_steps=args.iters,
        logdir=(args.logdir or None),
        impl=args.impl,
    )


if __name__ == "__main__":
    main()


# ------------------------------
# Modal integration (remote GPU)
# ------------------------------
try:
    import modal

    traces = modal.Volume.from_name("naive-attn-traces", create_if_missing=True)
    TRACE_DIR = Path("/traces")

    app = modal.App("naive-attn-profiling")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.5.1"
    )
    config = {"gpu": "h100", "image": image}

    def _make_output_dir(label: Optional[str]) -> Path:
        from uuid import uuid4

        base = TRACE_DIR / ("naive_attention" + (f"_{label}" if label else "")) / str(uuid4())
        base.mkdir(parents=True, exist_ok=True)
        return base

    @app.function(volumes={TRACE_DIR: traces}, **config)
    def profile_naive_attention_remote(
        batch: int = 4,
        heads: int = 8,
        seq: int = 1024,
        dim: int = 64,
        dtype_str: str = "fp16",
        seed: int = 0,
        warmup: int = 10,
        iters: int = 20,
        impl: str = "naive",
        label: Optional[str] = None,
    ):
        # Build inputs on CUDA
        device = torch.device("cuda")
        dtype = parse_dtype(dtype_str)
        q, k, v = make_inputs(batch, heads, seq, dim, dtype, device, seed=seed)

        output_dir = _make_output_dir(label)

        profile_with_torch_profiler(
            q, k, v,
            warmup_steps=warmup,
            active_steps=iters,
            logdir=str(output_dir),
            impl=impl,
        )

        # Return the most recent profiler JSON file path (TensorBoard handler writes many files)
        json_traces = sorted(
            output_dir.glob("**/*.pt.trace.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if json_traces:
            rel = json_traces[0].relative_to(TRACE_DIR)
            print(f"[remote] trace saved to {rel}")
            return json_traces[0].read_text(), str(rel)
        else:
            print(f"[remote] no JSON trace files found in {output_dir}")
            return "", str(output_dir.relative_to(TRACE_DIR))

    @app.local_entrypoint()
    def modal_entrypoint(
        batch: int = 4,
        heads: int = 8,
        seq: int = 1024,
        dim: int = 64,
        dtype: str = "fp16",
        seed: int = 0,
        warmup: int = 10,
        iters: int = 20,
        impl: str = "naive",
        label: Optional[str] = None,
    ):
        results, remote_rel_path = profile_naive_attention_remote.remote(
            batch=batch, heads=heads, seq=seq, dim=dim, dtype_str=dtype, seed=seed, warmup=warmup, iters=iters, impl=impl, label=label
        )

        # Save a copy locally for chrome://tracing
        local_out = Path("/tmp") / (Path(remote_rel_path).name or "trace.pt.trace.json")
        try:
            if results:
                local_out.write_text(results)
                print(f"[local] trace saved at {local_out}")
            else:
                print(f"[local] no trace content returned; remote dir: {remote_rel_path}")
        except Exception as e:
            print(f"[local] failed to write local trace copy: {e}")

except Exception as _modal_exc:
    pass
