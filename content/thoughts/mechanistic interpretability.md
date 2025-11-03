---
abstract: The subfield of alignment, or reverse engineering neural network. In a sense, it is the field of learning models' world representation.
aliases:
  - mechinterp
  - reveng neural net
  - interpretability
date: "2024-10-30"
description: and reverse engineering neural networks.
id: mechanistic interpretability
modified: 2025-11-02 23:32:18 GMT-05:00
permalinks:
  - /mechinterp
  - /interpretability
tags:
  - interpretability
  - ml
  - llm
title: mechanistic interpretability
---

[whirlwind tour](https://www.youtube.com/watch?v=veT2VI4vHyU&ab_channel=FAR%E2%80%A4AI), [[thoughts/pdfs/tinymorph exploration.pdf|initial exploration]], [glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)

> The subfield of alignment that delves into reverse engineering of a neural network, especially [[thoughts/LLMs]]

To attack the _curse of dimensionality_, the question remains: _==how do we hope to understand a function over such a large space, without an exponential amount of time?==_ [^lesswrongarc]

[^lesswrongarc]: good read from [Lawrence C](https://www.lesswrong.com/posts/6FkWnktH3mjMAxdRT/what-i-would-do-if-i-wasn-t-at-arc-evals#Ambitious_mechanistic_interpretability) for ambitious mech interp.

![[thoughts/sparse autoencoder#{collapsed: true}]]

![[thoughts/sparse crosscoders#{collapsed: true}]]

![[thoughts/Attribution parameter decomposition#{collapsed: true}]]

## transcoders

Transcoders are variants of SAEs that reconstruct the output of a component given its input, rather than reconstructing activations from themselves [@paulo2025transcodersbeatsparseautoencoders]. Unlike SAEs which encode and decode at a single layer, transcoders bridge layers by predicting downstream activations from upstream ones.

comparing to SAEs:

- Significantly more interpretable features - transcoders find features that better correspond to human-understandable concepts
- Enable analysis of direct feature-feature interactions by bridging over nonlinearities
- Form the basis for replacement models used in {{sidenotes[attribution graphs.]: Skip transcoders add affine skip connections, achieving lower reconstruction loss with no effect on interpretability.}}

> [!note] Cross-layer transcoders
>
> Each feature reads from the residual stream at one layer and contributes to outputs of all subsequent MLP layers. This greatly simplifies resulting circuits by:
>
> - Handling cross-layer superposition directly
> - Allowing features to "jump" across many uninteresting identity circuit connections
> - Matching underlying model outputs in ~50% of cases when substituting for MLPs

Transcoders enable the linear attribution framework used in attribution graphs - by replacing MLP computations, they make feature interactions linear and attribution well-defined.

see also: [[thoughts/circuit tracing]] for open-source tools using transcoders

## open problems

_excerpt from_ @sharkey2025openproblemsmechanisticinterpretability

- differentiate between "reverse engineering" versus "concept-based"
  - reverse engineer:
    - decomposition -> hypotheses -> validation
      - Decomposition via dimensionality [[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/principal component analysis|reduction]]
  - drawbacks with [[thoughts/sparse autoencoder#sparse dictionary learning|SDL]]:
    - SDL reconstruction error are way too high [@rajamanoharan2024improvingdictionarylearninggated{see section 2.3}]
    - SDL assumes linear representation hypothesis against non-linear feature space.
    - SDL leaves feature geometry unexplained ^geometry

## inference

Application in the wild: [Goodfire](https://goodfire.ai/) and [Transluce](https://transluce.org/)

> [!question]- How we would do inference with SAE?
>
> https://x.com/aarnphm_/status/1839016131321016380

idea: treat SAEs as a logit bias, similar to [[thoughts/structured outputs]]

> [!abstract] proposal for [[thoughts/vLLM|vLLM]] plugin architecture to enable SAE-based steering and feature monitoring in production
>
> Design goals: <5% latency overhead, support for CLTs and matryoshka SAEs, feature drift detection, production-grade observability

### motivation

Current SAE deployments (Goodfire, Transluce) run as separate inference services, creating duplication and latency overhead. Integrating SAEs directly into vLLM's execution pipeline enables:

- **Zero-copy activation access** - Direct manipulation of residual stream without serialization
- **Batched processing** - Leverage vLLM's continuous batching for SAE operations
- **Unified deployment** - Single service for base model + interpretability
- **Production monitoring** - Real-time feature drift detection and steering validation

more notes: [[thoughts/vLLM]], [[thoughts/sparse autoencoder]], [[thoughts/sparse crosscoders]]

### architecture overview

The plugin leverages vLLM v1's modular architecture and plugin system via Python entry points:

```python title="setup.py"
setup(
  name='vllm-sae-plugin',
  entry_points={
    'vllm.general_plugins': ['register_sae = vllm_sae_plugin:register']
  },
)
```

**Core components:**

1. **SAERegistry** - Manages multiple SAE/CLT checkpoints per layer
2. **ActivationInterceptor** - Hooks into attention/MLP outputs
3. **FeatureCache** - Persistent batch caching for sparse representations
4. **DriftMonitor** - Tracks feature activation distributions
5. **SteeringController** - Applies feature-level interventions

### design: activation interception

Hook into vLLM's model executor at attention output layer (pre-residual addition):

```python
class ActivationInterceptor:
  def __init__(self, sae_registry, target_modules):
    self.registry = sae_registry
    self.targets = target_modules  # e.g., ["attn_output", "mlp_output"]
    self.intervention_cache = {}

  def register_hooks(self, model):
    """Register forward hooks on target modules"""
    for layer_idx in range(model.config.num_hidden_layers):
      if f'layer.{layer_idx}.attn' in self.targets:
        self._register_attn_hook(model, layer_idx)

  def _register_attn_hook(self, model, layer_idx):
    def hook(module, input, output):
      # output shape: (batch, seq_len, hidden_dim)
      activations = output[0]  # residual stream contribution

      # Get appropriate SAE (could be CLT, matryoshka, standard)
      sae = self.registry.get_sae(layer_idx, 'attn')

      if sae is None:
        return output

      # Encode to sparse features
      sparse_features = sae.encode(activations)

      # Apply steering if configured
      if self.registry.has_intervention(layer_idx):
        sparse_features = self._apply_steering(sparse_features, layer_idx)

      # Monitor for drift
      self._update_drift_metrics(sparse_features, layer_idx)

      # Decode back
      reconstructed = sae.decode(sparse_features)

      return (reconstructed,) + output[1:]

    return model.layers[layer_idx].self_attn.register_forward_hook(hook)
```

### design: cross-layer transcoder support

CLTs read from one layer and write to multiple downstream layers, requiring special handling:

```python
class CrossLayerTranscoderManager:
  def __init__(self, clt_checkpoint):
    self.clt = load_clt(clt_checkpoint)
    self.source_layer = clt_checkpoint.source_layer
    self.target_layers = clt_checkpoint.target_layers
    self.feature_cache = {}

  def encode_source(self, activations, position_ids):
    """Encode at source layer, cache for downstream use"""
    sparse_features = self.clt.encode(activations)

    # Cache by position for reuse
    for pos in position_ids:
      self.feature_cache[pos] = sparse_features[pos]

    return sparse_features

  def apply_to_target(self, layer_idx, position_ids):
    """Apply cached features to target MLP layer"""
    if layer_idx not in self.target_layers:
      return None

    # Retrieve cached features
    cached = torch.stack([self.feature_cache[pos] for pos in position_ids])

    # Project to target layer
    return self.clt.decode_to_layer(cached, layer_idx)
```

### design: matryoshka SAE support

Matryoshka SAEs enable dynamic sparsity levels - use fewer features for simple tokens, more for complex:

```python
class MatryoshkaSAE:
  def __init__(self, checkpoint, sparsity_levels=[32, 64, 128, 256]):
    self.encoder = checkpoint.encoder
    self.decoders = {k: checkpoint.decoders[k] for k in sparsity_levels}
    self.complexity_estimator = ComplexityEstimator()

  def adaptive_encode_decode(self, activations, token_ids):
    """Select sparsity level per token based on complexity"""
    # Estimate complexity (e.g., rare token = complex)
    complexity = self.complexity_estimator(token_ids)

    # Encode once
    all_features = self.encoder(activations)

    # Decode with adaptive sparsity
    reconstructed = []
    for i, complexity_score in enumerate(complexity):
      k = self._select_k(complexity_score)

      # TopK selection
      topk_vals, topk_idx = torch.topk(all_features[i], k)
      sparse = torch.zeros_like(all_features[i])
      sparse.scatter_(0, topk_idx, topk_vals)

      # Decode with appropriate decoder
      reconstructed.append(self.decoders[k](sparse))

    return torch.stack(reconstructed)

  def _select_k(self, complexity_score):
    """Map complexity to sparsity level"""
    if complexity_score < 0.3:
      return 32
    elif complexity_score < 0.6:
      return 64
    elif complexity_score < 0.85:
      return 128
    return 256
```

### design: feature drift monitoring

Track distributional shifts in feature activations over time:

```python
class FeatureDriftMonitor:
  def __init__(self, n_features, window_size=1000):
    self.n_features = n_features
    self.window = window_size

    # Running statistics
    self.activation_counts = torch.zeros(n_features)
    self.activation_means = torch.zeros(n_features)
    self.activation_vars = torch.zeros(n_features)

    # Historical baselines
    self.baseline_counts = None
    self.baseline_means = None

    self.total_samples = 0

  def update(self, sparse_features):
    """Update statistics with new batch"""
    # sparse_features: (batch, n_features) with mostly zeros
    active_mask = sparse_features != 0

    # Count activations
    self.activation_counts += active_mask.sum(dim=0)

    # Update running mean/variance (Welford's algorithm)
    for i in range(self.n_features):
      active_values = sparse_features[:, i][active_mask[:, i]]
      if len(active_values) > 0:
        for val in active_values:
          self.total_samples += 1
          delta = val - self.activation_means[i]
          self.activation_means[i] += delta / self.total_samples
          delta2 = val - self.activation_means[i]
          self.activation_vars[i] += delta * delta2

    # Check for drift every window_size samples
    if self.total_samples % self.window == 0:
      self._check_drift()

  def _check_drift(self):
    """Detect significant distributional shifts"""
    if self.baseline_means is None:
      # First window - establish baseline
      self.baseline_counts = self.activation_counts.clone()
      self.baseline_means = self.activation_means.clone()
      return

    # Compare frequency shifts (which features are firing)
    freq_current = self.activation_counts / self.total_samples
    freq_baseline = self.baseline_counts / self.baseline_counts.sum()

    # KL divergence for frequency drift
    kl_div = torch.sum(
      freq_current * torch.log(freq_current / (freq_baseline + 1e-10) + 1e-10)
    )

    # Mean shift detection (for active features)
    mean_shifts = torch.abs(self.activation_means - self.baseline_means)

    # Alert on significant drift
    if kl_div > 0.1:  # threshold
      self._emit_alert('frequency_drift', kl_div.item())

    significant_shifts = mean_shifts > 2.0  # threshold
    if significant_shifts.any():
      drifted_features = torch.where(significant_shifts)[0]
      self._emit_alert('feature_shift', drifted_features.tolist())

  def _emit_alert(self, drift_type, details):
    """Log or publish drift alert"""
    # Could integrate with monitoring stack
    logger.warning(
      f'Feature drift detected: {drift_type}', extra={'details': details}
    )
```

### optimization: batched sparse operations

Critical for <5% overhead - fuse TopK selection and reconstruction:

```python
class OptimizedSAEKernel:
  """Custom CUDA kernel for batched TopK + reconstruction"""

  @staticmethod
  @torch.compile(mode='max-autotune')
  def fused_topk_decode(encoded, decoder_weights, k):
    """
    Fused operation:
    1. TopK selection on encoded
    2. Sparse matrix multiplication for decode

    Args:
        encoded: (batch, seq_len, n_features)
        decoder_weights: (n_features, hidden_dim)
        k: sparsity level

    Returns:
        reconstructed: (batch, seq_len, hidden_dim)
    """
    batch, seq_len, n_features = encoded.shape

    # TopK selection (batched)
    topk_vals, topk_idx = torch.topk(encoded, k, dim=-1)

    # Efficient sparse decode using gather
    # Only multiply the k active features
    selected_weights = decoder_weights[topk_idx]  # (batch, seq, k, hidden)
    reconstructed = (topk_vals.unsqueeze(-1) * selected_weights).sum(dim=-2)

    return reconstructed

  @staticmethod
  def register_custom_op():
    """Register as vLLM custom operator"""
    from vllm.model_executor.custom_op import CustomOP

    @CustomOP.register_oot(name='sae_topk_decode')
    class SAETopKDecode:
      def forward(self, encoded, decoder, k):
        return OptimizedSAEKernel.fused_topk_decode(encoded, decoder, k)
```

### optimization: persistent caching

Leverage vLLM v1's persistent batch technique for SAE state:

```python
class SAEPersistentCache:
  """Cache sparse features across decode steps"""

  def __init__(self, max_batch_size, max_seq_len, n_features):
    # Pre-allocate buffers
    self.sparse_indices = torch.zeros(
      (max_batch_size, max_seq_len, 128),  # assume k=128
      dtype=torch.int32,
      device='cuda',
    )
    self.sparse_values = torch.zeros(
      (max_batch_size, max_seq_len, 128), dtype=torch.float16, device='cuda'
    )
    self.position_map = {}  # track which positions are cached

  def update_diff(self, request_id, new_positions, sparse_features):
    """Only update new token positions (incremental)"""
    # Get existing cache position for this request
    base_idx = self.position_map.get(request_id, 0)

    # Extract TopK from new features
    topk_vals, topk_idx = torch.topk(sparse_features, 128, dim=-1)

    # Write only new positions
    n_new = len(new_positions)
    self.sparse_indices[base_idx : base_idx + n_new] = topk_idx
    self.sparse_values[base_idx : base_idx + n_new] = topk_vals

    # Update position map
    self.position_map[request_id] = base_idx + n_new

  def reconstruct(self, request_id, decoder_weights):
    """Reconstruct from cached sparse representation"""
    positions = self.position_map[request_id]
    indices = self.sparse_indices[:positions]
    values = self.sparse_values[:positions]

    # Sparse decode
    weights = decoder_weights[indices]
    return (values.unsqueeze(-1) * weights).sum(dim=-2)
```

### API design

Production-first configuration API:

```python
from vllm import LLM
from vllm_sae_plugin import SAEConfig, SteeringVector

# Initialize with SAE config
llm = LLM(
  model='meta-llama/Llama-3.2-3B',
  sae_config=SAEConfig(
    checkpoints={
      'attn_15': 'path/to/clt_layer15.pt',  # CLT
      'attn_20': 'path/to/matryoshka_layer20.pt',  # Matryoshka
      'mlp_23': 'path/to/standard_sae_layer23.pt',  # Standard
    },
    target_modules=['attn_output'],  # Hook attention outputs
    enable_drift_monitoring=True,
    drift_window=1000,  # samples
    cache_config={'max_cached_features': 100_000, 'eviction_policy': 'lru'},
    optimization={
      'use_custom_kernels': True,
      'compile_mode': 'max-autotune',
      'enable_persistent_cache': True,
    },
  ),
)

# Define steering intervention
steering = SteeringVector(
  layer=15,
  features={
    1337: 2.5,  # Amplify feature 1337 by 2.5x
    4242: -1.0,  # Suppress feature 4242
  },
  mode='additive',  # or "multiplicative"
)

# Generate with steering
outputs = llm.generate(
  'The weather in California is', steering_vectors=[steering], temperature=0.7
)

# Access drift metrics
drift_report = llm.sae_plugin.get_drift_report()
# {
#   "layer_15": {
#     "kl_divergence": 0.03,
#     "shifted_features": [128, 1337, 2048],
#     "dead_features": [42, 99],
#   }
# }
```

### monitoring and observability

Integration with production monitoring stacks:

```python
class SAEMetricsExporter:
  """Export SAE metrics to Prometheus/Datadog"""

  def __init__(self, drift_monitor, port=9090):
    self.monitor = drift_monitor
    self.metrics = {
      'sae_feature_activations': Counter(),
      'sae_drift_kl_divergence': Gauge(),
      'sae_reconstruction_error': Histogram(),
      'sae_latency_overhead': Histogram(),
    }

  def export_metrics(self):
    """Called periodically by vLLM metrics collector"""
    return {
      'feature_activation_rate': (
        self.monitor.activation_counts / self.monitor.total_samples
      ).tolist(),
      'drift_kl': self.monitor.compute_kl_divergence(),
      'dead_features': self.monitor.get_dead_features(threshold=0.001),
    }
```

### performance targets

Based on vLLM v1 benchmarks and SAE overhead analysis:

| Metric                  | Target            | Justification                                  |
| ----------------------- | ----------------- | ---------------------------------------------- |
| Latency overhead        | <5%               | Custom kernels + persistent caching            |
| Memory overhead         | <10%              | Sparse representations, shared decoder weights |
| Throughput impact       | <8%               | Batched operations, minimal serialization      |
| Feature refresh rate    | >1000 Hz          | Real-time drift detection                      |
| Max concurrent requests | Same as base vLLM | No batching constraints                        |

**Optimization breakdown:**

- Custom CUDA kernels for TopK: ~2% overhead
- Fused decode operation: ~1.5% overhead
- Persistent cache updates: ~0.5% overhead
- Drift monitoring (async): ~1% overhead
- **Total: ~5% end-to-end latency**

### future extensions

1. **Multi-SAE ensembles** - Run multiple SAEs per layer for robustness
2. **Adaptive sparsity scheduling** - Vary k based on token importance
3. **Feature attribution caching** - Pre-compute common steering vectors
4. **Cross-request feature analysis** - Aggregate statistics across users
5. **Online SAE finetuning** - Adapt to distribution shift automatically

## steering

refers to the process of manually modifying certain activations and hidden state of the neural net to influence its
outputs

For example, the following is a toy example of how a decoder-only transformers (i.e: GPT-2) generate text given the prompt "The weather in California is"

```mermaid
flowchart LR
  A[The weather in California is] --> B[H0] --> D[H1] --> E[H2] --> C[... hot]
```

To steer to model, we modify $H_2$ layers with certain features amplifier with scale 20 (called it $H_{3}$)[^1]

[^1]: An example steering function can be:

    $$
    H_{3} = H_{2} + \text{steering\_strength} * \text{SAE}.W_{\text{dec}}[20] * \text{max\_activation}
    $$

```mermaid
flowchart LR
  A[The weather in California is] --> B[H0] --> D[H1] --> E[H3] --> C[... cold]
```

One usually use techniques such as [[thoughts/mechanistic interpretability#sparse autoencoders]] to decompose model activations into a set of
interpretable features.

For feature [[thoughts/mechanistic interpretability#ablation]], we observe that manipulation of features activation can be strengthened or weakened
to directly influence the model's outputs

example: @panickssery2024steeringllama2contrastive uses [[thoughts/contrastive representation learning|contrastive activation additions]] to [steer](https://github.com/nrimsky/CAA) Llama 2

### [[thoughts/contrastive representation learning|contrastive]] activation additions

intuition: using a contrast pair for steering vector additions at certain activations layers

Uses _mean difference_ which produce difference vector similar to PCA:

Given a dataset $\mathcal{D}$ of prompt $p$ with positive completion $c_p$ and negative completion $c_n$, we calculate mean-difference $v_\text{MD}$ at layer $L$ as follow:

$$
v_\text{MD} = \frac{1}{\mid \mathcal{D} \mid} \sum_{p,c_p,c_n \in \mathcal{D}} a_L(p,c_p) - a_L(p, c_n)
$$

> [!important] implication
>
> by steering existing learned representations of behaviors, CAA results in better out-of-distribution generalization than basic supervised finetuning of the entire model.

## superposition hypothesis

see also: https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb

a phenomena when a neural network represents _more_ than $n$ features in a $n$-dimensional space

> [!abstract]+ tl/dr
>
> Linear representation of neurons can represent more features than dimensions. As sparsity increases, model use
> superposition to represent more [[thoughts/mechanistic interpretability#features]] than dimensions.
>
> neural networks “want to represent more features than they have neurons”.

When features are sparsed, superposition allows compression beyond what linear model can do, at a cost of interference that requires {{sidenotes<dropdown:true>[non-linear]: or "noisy simulation", where small neural networks exploit feature sparsity and properties of high-dimensional spaces to approximately simulate much larger much sparser neural networks}} filtering.

In a sense, superposition is a form of **lossy [[thoughts/Compression|compression]]**

This is plausible because:

- almost _orthogonal vectors_
  - it's only possible to have $n$ orthogonal vectors in an $n$-dimensional space, it's possible to have $\exp (n)$ many "almost orthogonal" ($< \epsilon$ cosine similarity) vectors in {{sidenotes[high-dimensional spaces.]: See the [Johnson–Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma) for the mathematical foundation.}}
- compressed sensing
  - In general, if one projects a vector into a lower-dimensional space, one can't reconstruct the {{sidenotes[original vector.]: However, this changes if one knows that the original vector is sparse - in this case, it is often possible to recover the original vector.}}

### properties

One can think in terms of _four progressively more strict properties_ that [[/tags/ml|neural network]] representations might have:

- **Decomposability**:
  - Neural network activations which are _decomposable_ can be {{sidenotes[decomposed]: This property is ultimately the most important — see the role of decomposition in defeating the curse of dimensionality.}} into features, the meaning of which is not dependent on the value of other features.
- **Linearity**:
  - Features correspond to directions. Each feature $f_i$ has a corresponding representation direction $W_i$.
  - The presence of multiple features $f_1, f_2, \dots$ activating with values $x_{f_1}, x_{f_2}, \dots$ is represented by

  $$
    x_{f_1} W_{f_1} + x_{f_2} W_{f_2} + \dots.
  $$

- **Superposition vs Non-Superposition**:
  - A linear representation exhibits superposition if $W^\top W$ is _not_ invertible.
  - If $W^\top W$ _is_ invertible, it does _not_ exhibit superposition.
- **Basis-Aligned**:
  - A representation is [[thoughts/basis]] aligned if _all_ $W_i$ are one-hot basis vectors.
  - A representation is partially basis aligned if _all_ $W_i$ are sparse. This requires a privileged basis.

The first two (decomposability and linearity) are properties we hypothesize to be widespread, while the latter (non-superposition and basis-aligned) are properties we believe only sometimes occur.

### importance

- sparsity: how _frequently_ is it in the input?

- importance: how useful is it for lowering loss?

### over-complete basis

_reasoning for the set of $n$ directions [^direction]_

[^direction]: Even though features still correspond to directions, the set of interpretable direction is larger than the number of dimensions

## features

> A property of an input to the model

When we talk about features [@elhage2022superposition{see "Empirical Phenomena"}], the theory building around
several observed empirical phenomena:

1. Word Embeddings: have direction which corresponding to semantic properties [@mikolov-etal-2013-linguistic]. For
   example:
   ```prolog
   V(king) - V(man) = V(monarch)
   ```
2. Latent space: similar vector arithmetics and interpretable directions have also been found in generative adversarial
   network.

We can define features as properties of inputs which a sufficiently large neural network will reliably dedicate
a neuron to represent [@elhage2022superposition{see "Features as Direction"}]

## ablation

> refers to the process of removing a subset of a model's parameters to evaluate its predictions outcome.

idea: deletes one activation of the network to see how performance on a task changes.

- zero ablation or _pruning_: Deletion by setting activations to zero
- mean ablation: Deletion by setting activations to the mean of the dataset
- random ablation or _resampling_

## mathematical frameworks to transformers

see also: @elhage2021mathematical

### residual stream

![[thoughts/images/residual-stream-illustration.webp|Residual stream illustration]]

intuition: we can think of residual as highway networks, in a sense portrays linearity of the {{sidenotes[network]: Constructing models with a residual stream traces back to early work by the Schmidhuber group, such as highway networks and LSTMs, which have found significant modern success in the more recent residual network architecture. In transformers, the residual stream vectors are often called the "embedding" - we prefer the residual stream terminology because it emphasizes the residual nature and because the residual stream often dedicates subspaces to tokens other than the present token.}}

residual stream $x_{0}$ has dimension $\mathit{(C,E)}$ where

- $\mathit{C}$: the number of tokens in context windows and
- $\mathit{E}$: embedding dimension.

[[thoughts/Attention]] mechanism $\mathit{H}$ process given residual stream $x_{0}$ as the result is added back to $x_{1}$:

$$
x_{1} = \mathit{H}{(x_{0})} + x_{0}
$$

![[thoughts/induction heads|induction heads]]

## grokking

See also: [writeup](https://www.alignmentforum.org/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking), [code](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20), [circuit threads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

> A phenomena discovered by [@power2022grokkinggeneralizationoverfittingsmall] where small algorithmic tasks like modular addition will initially memorise training data, but after a long time ti will suddenly learn to generalise to unseen data

> [!important] empirical claims
>
> related to phase change

## attribution graph

see also [[thoughts/Attribution parameter decomposition]], [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

Attribution graphs are computational graphs that reveal the mechanisms underlying model behaviors by tracing how features influence each other to produce outputs. They show the intermediate computational steps models use, providing a "path-finder" for activation flows.

```jsx imports={MethodologyStep,MethodologyTree}
<MethodologyTree
  title="methodology"
  description="Distil a transformer run into an interpretable circuit by training a sparse replacement model, freezing the residual context, and pruning the resulting feature graph."
>
  <MethodologyStep
    title="train interpretable transcoders"
    badge="replacement model"
    summary="Swap SAEs for transcoders (ideally cross-layer) so features capture the computation the original MLP stack performs."
    points={[
      "Let features read from a single residual stream position and write into downstream MLP layers to bridge non-linear blocks.",
      "Cross-layer transcoders handle superposition and drastically simplify resulting circuits.",
      "Replacement models should closely reconstruct the base model to keep feature semantics stable.",
    ]}
  >
    <MethodologyStep
      title="prefer cross-layer variants"
      summary="Allow features to jump across residual layers instead of following every identity connection."
      points={[
        "Reduces circuit depth by skipping uninformative residual additions.",
        "Matches the underlying model's outputs in roughly half of substitution trials.",
      ]}
    />
  </MethodologyStep>
  <MethodologyStep
    title="freeze attention patterns"
    badge="context"
    summary="Hold attention weights and normalization denominators fixed so attribution focuses on the induced MLP computation."
    points={[
      "Splits the problem into understanding behaviour with a fixed attention pattern and separately explaining why the model attends there.",
      "Removes the largest non-linearities outside the replacement model.",
    ]}
  />
  <MethodologyStep
    title="make interactions linear"
    badge="linearity"
    summary="Design the setup so feature-to-feature effects are linear for the chosen input."
    points={[
      "Transcoder features replace the non-linear MLP while frozen attention removes the remaining activation-dependent terms.",
      "Enables principled attribution because edges sum to the observed activation.",
    ]}
  />
  <MethodologyStep
    title="construct the feature graph"
    badge="graph build"
    summary="Represent every active feature, prompt token, reconstruction error, and output logit as a node."
    points={[
      "Edges store linear contributions; their weights sum to the downstream activation.",
      "Multi-hop paths capture indirect mediation via other features.",
    ]}
  >
    <MethodologyStep
      title="encode nodes & edges"
      summary="Annotate each node with activation magnitude and each edge with its linear effect."
      points={[
        "Token and embedding nodes anchor the computation back to the prompt.",
        "Include reconstruction error nodes so the graph accounts for approximation mismatch.",
      ]}
    />
  </MethodologyStep>
  <MethodologyStep
    title="prune for interpretability"
    badge="sparsity"
    summary="Select the smallest subgraph that explains the behaviour at the focus token."
    points={[
      "Rank nodes and edges by their marginal contribution to the output logits.",
      "Keep sparse subgraphs so human inspection stays tractable.",
    ]}
  />
  <MethodologyStep
    title="validate with perturbations"
    badge="verification"
    summary="Check that the proposed circuit actually drives the behaviour."
    points={[
      "Ablate or rescale identified features and measure the effect on the model output.",
      "Confirm the replacement model relies on the same mechanisms as the base model.",
    ]}
  />
</MethodologyTree>
```

### parameter decomposition

Attribution graphs and parameter decomposition are complementary views:

- **Attribution graphs** answer "what computational steps happen?" - showing activation/feature-level flow of information through the model
- **Parameter decomposition** answers "which parameters implement those steps?" - identifying which parameter components enable that computational flow

Both address superposition by finding sparse decompositions, but at different abstraction levels: attribution graphs reveal activation flow, parameter decomposition reveals parameter implementation.

### limitations

- Missing {{sidenotes[attention circuits]: Because attention patterns are frozen, attribution graphs miss attention mechanisms - need separate QK attribution methods to understand how attention selects which features interact.}}
- Replacement {{sidenotes[model validity]: Replacement models may use different mechanisms than original models - requires careful validation via perturbation experiments.}}
- Graph pruning subjectivity
  - Pruning introduces subjectivity in determining what's "important" - different pruning criteria may reveal different mechanisms
- Single forward pass focus
  - Current methods focus on single forward passes - understanding mechanism reusability across contexts needs more work

### applications

- Mechanism discovery
  - Finding which computational steps models use for specific behaviors
- Model editing
  - Understanding which features/circuits to modify for targeted behavior changes
- Anomaly detection
  - Identifying when unexpected mechanisms activate (mechanistic anomaly detection)
- Cross-model analysis
  - Comparing how different models implement similar behaviors
- Training analysis
  - Tracking how mechanisms emerge during training

## stochastic parameter decomposition

@bushnaq2025stochasticparameterdecomposition improves upon [[thoughts/Attribution parameter decomposition|APD]] by being more scalable and robust to {{sidenotes[hyperparameters.]: SPD demonstrates decomposition on models slightly larger and more complex than was possible with APD, avoids parameter shrinkage issues, and better identifies ground truth mechanisms in toy models.}}

see also: https://github.com/goodfire-ai/spd

## QK attributions

https://transformer-circuits.pub/2025/attention-qk

> describe attention head scores as a bilinear function of feature activations on the respective query and key positions.

[^ref]
