"""
Modal remote Triton add kernel

Usage:

  uv run modal run triton_add.py::triton_add_remote \
    --size 98432 --block-size 1024 --seed 0

  # Or use the local entrypoint wrapper:
  uv run modal run triton_add.py
"""

import modal
import torch

app = modal.App("triton-add-kernel")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "triton",
        "numpy"
    )
)

config = {"gpu": "h100", "image": image}


@app.function(**config)
def triton_add_remote(size: int = 98432, block_size: int = 1024, seed: int = 0):
    # Import torch and triton inside the function to avoid local import requirements
    import triton
    import triton.language as tl
    globals()["triton"] = triton
    globals()["tl"] = tl
    @triton.jit
    def add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    def add_host(x_tensor, y_tensor):
        output = torch.empty_like(x_tensor)
        n_elements_local = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements_local, meta["BLOCK_SIZE"]),)
        add_kernel[grid](x_tensor, y_tensor, output, n_elements_local, BLOCK_SIZE=block_size)
        return output

    torch.manual_seed(seed)
    device = torch.device("cuda")
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)

    output_torch = x + y
    output_triton = add_host(x, y)

    max_diff = (output_torch - output_triton).abs().max().item()
    print(f"[remote] Triton add: size={size}, block_size={block_size}, max_diff={max_diff}")
    return {"size": int(size), "block_size": int(block_size), "max_diff": float(max_diff)}


@app.local_entrypoint()
def main(size: int = 98432, block_size: int = 1024, seed: int = 0):
    result = triton_add_remote.remote(size=size, block_size=block_size, seed=seed)
    print("[local] result:", result)


