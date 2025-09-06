"""
Commands for memory profiling:

uv run memray run -o memray.bin computation.py
uv run python -m memray flamegraph memray.bin
uv run memray stats memray.bin
uv run memray table memray.bin

Commands for call stack viz:

uv run python computation.py
speedscope ./trackthis.json
"""

import numpy as np
import fire
import speedscope

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.dot(A, B)

def run_computation(n_layers: int, hidden_dim: int, input_dim: int, output_dim: int):
    weights = []
    biases = []
    
    input_projection = np.random.randn(input_dim, hidden_dim)
    output_projection = np.random.randn(hidden_dim, output_dim)

    for _ in range(n_layers):
        W = np.random.randn(hidden_dim, hidden_dim)
        b = np.random.randn(hidden_dim)
        weights.append(W)
        biases.append(b)
    
    # Create input tensor
    X = np.random.randn(1, input_dim)
    
    hidden_state = matmul(X, input_projection)
    # Forward pass through each layer
    for i in range(n_layers):
        W = weights[i]
        b = biases[i]
        Z = matmul(hidden_state, W) + b
        hidden_state = np.tanh(Z)
    
    output = matmul(hidden_state, output_projection)
    
    return output

def main(
    n_layers: int = 3,
    hidden_dim: int = 100,
    input_dim: int = 100,
    output_dim: int = 100,
):
    with speedscope.track("trackthis.json"):
        run_computation(n_layers, hidden_dim, input_dim, output_dim)


if __name__ == "__main__":
    fire.Fire(main)
