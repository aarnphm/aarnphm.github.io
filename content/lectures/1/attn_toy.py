# keep subject = "fox" as the only singular NOUN before the verb,
# but increase the *distance* by inserting many adverbs (non-nouns) between "fox" and "jumps".
# This shows attention can still directly hop to "fox" even when D is large, because the query
# matches features not path length.
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

DIMS = ["DET","ADJ","NOUN","VERB","PREP","SING","BEFORE"]
d = len(DIMS)

def token_features(tok, pos, verb_pos):
    det = tok.lower() in {"the","a","an"}
    adj = tok.lower() in {"quick","brown","lazy","very"}
    noun = tok.lower() in {"fox","dog"}
    verb = tok.lower() in {"jumps"}
    prep = tok.lower() in {"over"}
    sing = tok.lower() in {"fox","dog"}
    before = pos < verb_pos
    x = np.array([det,adj,noun,verb,prep,sing,before], dtype=float)
    n = np.linalg.norm(x)
    return x / n if n>0 else x

def build_K(tokens):
    verb_pos = tokens.index("jumps")
    K = np.stack([token_features(tok,i,verb_pos) for i,tok in enumerate(tokens)])
    return K

Q = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 1.0, 0.8], dtype=float)
Q = Q / np.linalg.norm(Q)
scale = np.sqrt(len(Q))

def attn_weights(Q, K):
    logits = (K @ Q) / scale
    return softmax(logits)

S1 = "The quick brown fox jumps over the lazy dog".split()

# Insert many adverbs between "fox" and "jumps" to enlarge distance without adding competing nouns
fillers = "really truly extremely absolutely definitely surely".split()
S2 = "The quick brown fox " + " ".join(fillers*4) + " jumps over the very lazy dog"
S2 = S2.split()

K1 = build_K(S1)
K2 = build_K(S2)

w1 = attn_weights(Q, K1)
w2 = attn_weights(Q, K2)

def plot_weights(tokens, w, title):
    plt.figure(figsize=(10,3))
    xs = np.arange(len(tokens))
    plt.bar(xs, w)
    plt.xticks(xs, tokens, rotation=45, ha="right")
    plt.ylabel("Attention weight (from 'jumps')")
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_weights(S1, w1, "Attention from 'jumps' (short sentence)")
plot_weights(S2, w2, "Attention from 'jumps' (long distance, no competing nouns)")

def topk(tokens, w, k=3):
    idx = np.argsort(-w)[:k]
    return [(tokens[i], float(w[i])) for i in idx]

print("Top-3 attention targets from 'jumps' (short):", topk(S1, w1))
print("Top-3 attention targets from 'jumps' (long w/ fillers):", topk(S2, w2))

# Report the attention weight on "fox" in both sentences
fox_idx_1 = S1.index("fox")
fox_idx_2 = S2.index("fox")
print("Weight on 'fox' (short):", float(w1[fox_idx_1]))
print("Weight on 'fox' (long): ", float(w2[fox_idx_2]))
