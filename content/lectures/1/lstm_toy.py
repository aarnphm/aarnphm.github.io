# Demonstration: LSTM memory retention vs. token distance
# We model the LSTM cell state's gradient retention from token t to T as the product of forget gates: Π f_k.
# For intuition, we use a constant mean forget gate E[f] = alpha in {0.98, 0.95, 0.90, 0.80}
# and plot retention alpha^D as a function of distance D (number of tokens between dependency ends).
#
# We'll annotate D=4 (distance between "fox" and "jumps" in the short sentence)
# and D=20 (a longer variant with inserted adjectives/clauses).

import numpy as np
import matplotlib.pyplot as plt

distances = np.arange(1, 61)  # up to 60-token gaps
alphas = [0.98, 0.95, 0.90, 0.80]

plt.figure(figsize=(8,5))
for a in alphas:
    plt.plot(distances, a**distances, label=f"E[f]={a:.2f}")


# Annotate two reference gaps
d_fox = 4
d_long = 20
plt.axvline(d_fox, linestyle="--", linewidth=1)
plt.axvline(d_long, linestyle="--", linewidth=1)
plt.text(d_fox+0.5, 0.9, "gap≈4 (fox→jumps)", rotation=90, va="top")
plt.text(d_long+0.5, 0.9, "gap≈20 (with clause)", rotation=90, va="top")

plt.yscale("log")
plt.xlabel("Token distance D")
plt.ylabel("Expected retention  E[∏ f] ≈ (E[f])^D")
plt.title("LSTM memory retention decays exponentially with distance")
plt.legend()
plt.tight_layout()
plt.show()

# Print concrete numbers for the class handout
def retention(alpha, D):
    return alpha**D

print("Retention at D=4 and D=20 for different mean forget gates E[f]=alpha:")
for a in alphas:
    print(f"  alpha={a:.2f}:  D=4 -> {retention(a,4):.4f},  D=20 -> {retention(a,20):.6f}")
