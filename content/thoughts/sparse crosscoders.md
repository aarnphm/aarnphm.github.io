---
date: '2024-11-03'
description: A shared sparse dictionary for activations across layers or models, and the limits of reading model differences from decoder norms.
id: sparse crosscoders
modified: 2026-10-02 09:15:16 GMT-04:00
socials:
  circuits: https://transformer-circuits.pub/2024/crosscoders/index.html
tags:
  - ml
  - interp
title: sparse crosscoders
transclude:
  title: false
---

> [!important] maturity
>
> Anthropic introduced this as a research preview in October 2024. Later work below tests whether decoder norms identify model-specific concepts.

A sparse crosscoder extends a [[thoughts/sparse autoencoder]] to several activation spaces. It computes one vector of latent activations and gives each layer or model its own decoder. The same latent can therefore contribute to several reconstructions. [@lindsey2024sparsecrosscoders]

see also the [Gemma 2 2B reproduction notebook](https://colab.research.google.com/drive/124ODki4dUjfi21nuZPHRySALx9I74YHj?usp=sharing) and its [training code](https://github.com/ckkissane/crosscoder-model-diff-replication).

## motivations

- **Cross-layer features:** fit a dictionary jointly to activity spread across layers.
- **Persistent features:** give a feature that survives through several layers one latent identity.
- **Model diffing:** inspect how the decoder contributions differ between models on the same inputs.

### cross-layer [[thoughts/mechanistic interpretability#superposition hypothesis|superposition]]

![[thoughts/images/additive-residual-stream-llm.webp|given the additive properties of transformers' residual stream, adjacent layers in larger transformers can be thought as "almost parallel"]]

> [!important]- intuition
>
> Under the linear superposition hypothesis, features occupy directions that can overlap. One neuron can participate in several feature directions.
>
> ![[thoughts/images/feature-neurons.webp]]

For a fixed token, write the residual update as

$$
r_{l+1}=r_l+u_l.
$$

The update $u_l$ may depend on earlier layers and other tokens. Addition lets contributions accumulate in a common space; it does not make the blocks independent. The diagrams below illustrate a possible computation distributed across two blocks.

![[thoughts/images/one-step-circuit.webp]]

![[thoughts/images/parallel-joint-branch.webp]]

Joint dictionary learning can capture correlated contributions in these spaces.[^jointlysae]

[^jointlysae]: For a related vision example, see Gorton's SAE study of InceptionV1 and its curve detectors. [@gorton2024missingcurvedetectorsinceptionv1]

### persistent features and complexity

![[thoughts/images/proposal-formed-by-cross-layers-superposition.webp]]

Suppose a signal with strength $z(x)$ persists across three layers. Its contribution at layer $l$ is $z(x)d^l$, where the direction $d^l$ may change. Separate SAEs must discover and match those three occurrences. A crosscoder can assign them one activation $z(x)$ and three decoder vectors.

![[thoughts/images/dedup-features-persistent.webp]]

This can combine repeated occurrences into one graph node.[^risks] The reconstruction alone leaves open whether the original model carried the signal forward or independently recomputed it.

[^risks]: Correlations across layers can support a reconstruction whose causal graph differs from the model's. Interventions are needed to test the proposed mechanism.

## setup.

An SAE reconstructs its input activation space. A [[thoughts/mechanistic interpretability#transcoders|transcoder]] predicts another space, commonly an MLP's output from its input. Both fit within the crosscoder family.

![[thoughts/images/crosscoder-setup.webp]]

> [!math]+ crosscoders
>
> Let $a^l(x)\in\mathbb{R}^{d_l}$ be the activation at a chosen token and layer $l\in L$. With $F$ latents, the basic acausal crosscoder is
>
> $$
> \begin{aligned}
> f(x)&=\operatorname{ReLU}\left(\sum_{l\in L}W_{\mathrm{enc}}^l a^l(x)+b_{\mathrm{enc}}\right),\\
> \hat a^l(x)&=W_{\mathrm{dec}}^l f(x)+b_{\mathrm{dec}}^l.
> \end{aligned}
> $$
>
> The encoder matrix has shape $F\times d_l$; the decoder has shape $d_l\times F$. Each latent has one activation and a separate decoder column $d_i^l=W_{\mathrm{dec},i}^l$ for each layer.

The training objective balances reconstruction and sparsity:

$$
\mathcal{L}=\mathbb{E}_x\left[
\sum_{l\in L}\left\|a^l(x)-\hat a^l(x)\right\|_2^2
+\lambda\sum_{i=1}^{F}f_i(x)\sum_{l\in L}\left\|d_i^l\right\|_2
\right],\qquad \lambda>0.
$$

The decoder norm matters because activation scale is arbitrary. If the sparsity term penalized only $f_i(x)$, replacing $f_i(x)$ by $f_i(x)/c$ and every $d_i^l$ by $c d_i^l$, for $c>1$, would preserve reconstruction while lowering the sparsity penalty. Weighting by $\sum_{l\in L}\|d_i^l\|_2$ makes that scale change leave the penalty unchanged.

The outer sum is an $L_1$ norm of per-layer $L_2$ norms.[^l2weightnorm] For one latent with activation $2$ and decoder norms $3$ and $4$, its penalty is $14\lambda$. Concatenating the decoder vectors and taking one $L_2$ norm would give $10\lambda$. Thus the latter objective gives a discount for spreading a latent across spaces. Which objective is useful depends on what we want to measure; decoder sparsity needs particular care in [[#model diffing]].

[^l2weightnorm]: The two weights are $\sum_l\|d_i^l\|_2$ and $\sqrt{\sum_l\|d_i^l\|_2^2}$. The preview uses the first so its loss can be compared with the sum of per-layer SAE losses at the same sparsity coefficient.

## variants

![[thoughts/images/crosscoders-variants.webp]]

The placement of encoders and decoders determines what information a prediction can use:

- **Acausal:** read from all selected layers and reconstruct them. A reconstruction of an early layer can use later information.
- **Weakly causal:** read at one residual-stream layer and reconstruct that layer and later layers.
- **Strictly causal:** read before a computation and predict its output or downstream outputs. Cross-layer transcoders can read an MLP input and predict that MLP's output plus later MLP outputs.

These masks constrain information flow in the crosscoder. Mechanistic faithfulness still needs testing. Attention also needs separate treatment because it mixes token positions. One approximation fixes the observed attention pattern while analysing a prompt. [@lindsey2024sparsecrosscoders]

## Cross-layer Features

> How can we discover cross-layer structure?

The preview compared a global acausal crosscoder with separate SAEs on all layers of an 18-layer model. Activations were normalized per layer and the sparsity coefficient was fixed. The crosscoder achieved lower evaluation loss at a matched total dictionary size; at large compute budgets, reaching the same loss cost roughly twice the training FLOPs. Dictionary size and training cost answer different questions here. [@lindsey2024sparsecrosscoders]

![[thoughts/images/all-layer-crosscoder-vs-per-layer-sae.webp]]

## model diffing

see also: [[thoughts/model stiching|model stitching]] and [[thoughts/SVCCA]]. These study compatibility or similarity between representations.[^sne]

[^sne]: Laakso and Cottrell compare representations through distances between examples. [@doi:10.1080/09515080050002726] Colah's [representation-visualization post](https://colah.github.io/posts/2015-01-Visualizing-Representations/) discusses a related construction for visualizing networks.

For two models, replace the layer index with $m\in\{\mathrm{base},\mathrm{chat}\}$. Run both on matching inputs and learn a shared $f(x)$ with separate decoder vectors $d_i^{\mathrm{base}}$ and $d_i^{\mathrm{chat}}$. Different hidden widths are allowed by the matrix shapes, though choosing corresponding layers and token positions remains part of the experiment.

A zero base decoder means that this fitted dictionary does not use the latent to reconstruct the base model. Inferring that fine-tuning created the concept requires more evidence.

### challenges with model diffing

Minder et al. identify two failures of decoder-norm attribution: [@minder2025overcomingsparsityartifactscrosscoders]

1. **Complete shrinkage:** the sparsity penalty removes a useful base decoder contribution, leaving it in the reconstruction error.
2. **Latent decoupling:** other latents already reconstruct the concept in the base model, so its nominally chat-only latent receives a zero base decoder.

Their **Latent Scaling** test fits how much a chat latent's contribution explains the base reconstruction error and the base reconstruction. These probe shrinkage and decoupling, respectively. In their Gemma experiment, both models use the same layer-13 residual-stream coordinates, so a chat decoder vector can be applied to base activations.

### BatchTopK loss for improved model diffing

BatchTopK selects the largest $nk$ decoder-norm-weighted latent activations across a batch of $n$ examples. The budget averages $k$ active latents per example; individual examples can use different counts.

On Gemma 2 2B base/chat activations at layer 13, Minder et al. found fewer of the two artifacts and more interpretable chat-specific latents. Patching selected latent contributions into a hybrid forward pass tested their effect on the output distribution. These experiments support the method for that comparison; a new model pair still needs its own checks. [@minder2025overcomingsparsityartifactscrosscoders]

### model diff amplification

[Goodfire's logit diff amplification](https://www.goodfire.com/research/model-diff-amplification) compares output logits directly. It requires no crosscoder. At each generation step, using the same context and aligned token vocabulary, compute

$$
\ell_{\mathrm{amp}}=\ell_{\mathrm{after}}+\alpha(\ell_{\mathrm{after}}-\ell_{\mathrm{before}}),\qquad \alpha>0,
$$

then sample from the amplified distribution and repeat with the new context. This magnifies the logit differences introduced by post-training. Goodfire surfaced rare unwanted behaviours in its case studies. Because amplification changes the sampling distribution, those frequencies cannot estimate how often the original model produces the behaviour.

## open-source implementations

- [Gemma reproduction](https://github.com/ckkissane/crosscoder-model-diff-replication): crosscoder training and analysis for the middle residual stream of Gemma 2 2B base and instruction-tuned models.
- [Crosscoder learning](https://github.com/science-of-finetuning/crosscoder_learning) and [sparsity-artifact experiments](https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders): the training library and analysis code accompanying Minder et al.
- [Crosscode](https://github.com/oclivegriffin/crosscode): training for multi-layer and multi-model crosscoders, including $L_1$ and BatchTopK variants. The earlier [model-diffing repository](https://github.com/model-diffing/model-diffing) points here.
- [Circuit tracer](https://github.com/safety-research/circuit-tracer): attribution graphs and interventions using trained transcoders. Its graph pipeline consumes those weights.
- [Neuronpedia graph interface](https://www.neuronpedia.org/gemma-2-2b/graph): a viewer for attribution graphs; check the selected source set to distinguish per-layer and cross-layer transcoders.

see [[thoughts/circuit tracing]] for the graph workflow.

## questions

> How do features change over model training? When do they form?

> As we make a model wider, do we get more features? or they are largely the same, packed less densely?

> How can we detect when crosscoders are learning spurious exclusive features versus genuine model-specific concepts?
