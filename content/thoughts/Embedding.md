---
date: '2024-02-25'
description: token lookup, position, masking, and image–text representations
id: Embedding
modified: 2026-10-12
seealso:
  - '[[thoughts/Transformers|Transformers]]'
  - '[[thoughts/Attention|Attention]]'
  - '[[thoughts/LLMs|LLMs]]'
tags:
  - ml
  - llm
title: Embedding
---

In [[thoughts/LLMs|LLMs]], a token embedding is a learned table lookup. Let the vocabulary contain $m$ tokens and the model have width $d$. For token ID $u_{b,t}$ at position $t$ in batch item $b$:

$$
E\in\mathbb{R}^{m\times d},\qquad x_{b,t}=E[u_{b,t},:]\in\mathbb{R}^{d}.
$$

The ID selects a row. Two occurrences of the same token start with the same vector; attention later lets their representations depend on their different contexts. The lookup is equivalent to multiplying a one-hot row vector by $E$, without allocating the one-hot vector. See the [embedding operation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Embedding.html).

Position enters according to the architecture. An additive scheme uses $x_{b,t}+p_t$; the original Transformer also scales the token vector by $\sqrt d$ [@vaswani2023attentionneed]. [[thoughts/RoPE|RoPE]] applies position-dependent rotations to attention queries and keys, so adding a position vector to every input is not a universal step. The RoFormer derivation shows how these rotations make the query–key dot product depend on relative position [@su2023roformerenhancedtransformerrotary].

## [[thoughts/Tokenization|tokenization]] and vocab

[[thoughts/byte-pair encoding|BPE]] and WordPiece produce IDs before the embedding lookup. The tokenizer defines which special tokens exist and when to insert them. `BOS`, `EOS`, `PAD`, and `UNK` are possible vocabulary entries; a model need not use all four.

Padding makes a batch rectangular. In PyTorch, a new `Embedding` with `padding_idx` starts with a zero padding row and suppresses its lookup gradient. This option does not install attention or loss masks. A zero vector can still receive attention weight because its score participates in the softmax denominator.

### shapes and flow

see also: @croft2023llm

For a batch of $B=5$ sequences padded to $T=128$ positions, lookup produces $X\in\mathbb{R}^{5\times128\times d}$. Transformer blocks preserve that width. Let $H$ denote their final hidden states, including any final normalization. A separate vocabulary projection produces logits:

$$
W_{\mathrm{out}}\in\mathbb{R}^{d\times m},\qquad
L=HW_{\mathrm{out}}\in\mathbb{R}^{B\times T\times m}.
$$

An output bias may be added. Softmax runs over the vocabulary axis. With tied weights, $W_{\mathrm{out}}=E^\top$. Normalization and embedding scaling belong to the chosen architecture; the shape contract alone specifies neither. The original Transformer separates its decoder states from this linear projection and softmax [@vaswani2023attentionneed].

```mermaid
flowchart LR
  A["Token IDs"] --> B["Embedding lookup"]
  B --> C["Input preparation"]
  C --> D["Position-aware transformer blocks"]
  D --> E["Final hidden states"]
  E --> F["Vocabulary projection"]
  F --> G["Logits"]
  G --> H["Vocabulary probabilities"]
```

### masks

A causal mask prevents a position from reading future tokens. A padding mask prevents it from reading padded keys. For one batch item and attention head, an additive mask can express both:

$$
M_{ij}=\begin{cases}
0, & j\leq i\text{ and position }j\text{ is a valid key},\\
-\infty, & \text{otherwise},
\end{cases}
\qquad
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

Here $d_k$ is the query/key width. Each valid query needs at least one allowed key for this formula to be defined. Masking padded keys does not remove padded query slots or exclude their predictions from the loss. Boolean mask conventions also depend on the API: PyTorch's [scaled dot-product attention](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html) uses `True` for an allowed entry.

For next-token training, the logits at position $t$ predict $u_{b,t+1}$. With $\mathcal{M}$ the set of valid input–next-target pairs, the unweighted mean loss is

$$
\mathcal{L}=-\frac{1}{|\mathcal{M}|}
\sum_{(b,t)\in\mathcal{M}}
\log\operatorname{softmax}(L_{b,t,:})_{u_{b,t+1}},
\qquad |\mathcal{M}|>0.
$$

For `BOS a b EOS PAD`, the supervised pairs are `BOS` predicting `a`, `a` predicting `b`, and `b` predicting `EOS`. The prediction whose target is `PAD` is ignored. This shift is applied once, either by preparing the targets or by the model's loss implementation. PyTorch's [cross-entropy loss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html) accepts logits directly and supports ignored target indices; it does not perform the sequence shift itself.

A literal `[MASK]` token serves another purpose. BERT uses it to hide selected input content during masked-language-model training [@devlin2019bertpretrainingdeepbidirectional]. An attention mask controls which positions can exchange information.

## vision and multimodal embeddings

see also: CLIP, ViT

Image patches already contain numerical values. A patch embedding projects those values to the model width; a token embedding selects a learned row by ID.

### patch embeddings (ViT)

Take an RGB image $I\in\mathbb{R}^{H\times W\times3}$ whose height and width are divisible by patch size $p$. There are

$$
N=\frac{H}{p}\frac{W}{p}
$$

non-overlapping patches. Flatten each patch into a row $r_i\in\mathbb{R}^{3p^2}$. In the original ViT, the input sequence is [@dosovitskiy2021imageworth16x16words]

$$
Z_0=
\begin{bmatrix}
c\\ r_1E_{\mathrm{patch}}\\ \vdots\\ r_NE_{\mathrm{patch}}
\end{bmatrix}+P_{\mathrm{pos}}
\in\mathbb{R}^{(N+1)\times d},
$$

where $E_{\mathrm{patch}}\in\mathbb{R}^{3p^2\times d}$, $c\in\mathbb{R}^{d}$ is a learned class-token row, and $P_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times d}$ contains learned one-dimensional position embeddings. The final normalized class-token state feeds the classification head. Other architectures can pool patch states instead.

For a $224\times224$ image with $16\times16$ patches, there are $196$ patches, each containing $768$ values. Adding the class token gives $197$ input vectors. The model width is a separate choice from the number of values in a patch.

```mermaid
flowchart LR
  A["RGB image"] --> B["Split into patches"]
  B --> C["Flatten and project each patch"]
  C --> D["Prepend class token and add position embeddings"]
  D --> E["ViT encoder"]
  E --> F["Normalized class-token state"]
  F --> G["Classification head"]
```

Padding and reconstruction masks also have different jobs in vision. Padding masks can exclude artificial patches in architectures that support variable-sized inputs. Masked-image pretraining hides actual image content:

- MAE sends only visible patches through the encoder. A smaller decoder receives the encoded patches and inserted mask tokens, then reconstructs the missing pixels [@he2021maskedautoencodersscalablevision].
- BEiT puts mask embeddings into the encoder input and predicts discrete visual-token labels supplied by a separate image tokenizer [@bao2022beitbertpretrainingimage].

### CLIP-style dual encoders (image–text)

CLIP trains on paired images and texts [@radford2021learningtransferablevisualmodels]. Its encoders produce features, and learned linear projections put both in a shared space. For batch pair $i$, write

$$
a_i=f_{\mathrm{img}}(I_i)W_{\mathrm{img}},\qquad
b_i=f_{\mathrm{txt}}(T_i)W_{\mathrm{txt}},\qquad
v_i=\frac{a_i}{\|a_i\|_2},\qquad
t_i=\frac{b_i}{\|b_i\|_2}.
$$

For nonzero projected features, their normalized dot product is cosine similarity. With positive temperature $\tau$, define $s_{ij}=v_i^\top t_j/\tau$. CLIP learns this scale during training. The symmetric contrastive loss is

$$
\mathcal{L}=-\frac{1}{2B}\sum_{i=1}^{B}
\left[
\log\frac{\exp(s_{ii})}{\sum_{j=1}^{B}\exp(s_{ij})}
+\log\frac{\exp(s_{ii})}{\sum_{j=1}^{B}\exp(s_{ji})}
\right].
$$

The first term asks each image to select its paired text from the batch; the second asks each text to select its paired image. This training objective gives cross-modal similarity its use. A shared vector width alone does not align two encoders.

At inference, image–text retrieval ranks candidates by similarity. For zero-shot classification, the text candidates are prompts containing the class names. Their wording and the available classes therefore affect the decision.
