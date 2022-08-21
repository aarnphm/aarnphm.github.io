---
date: "2025-10-04"
description: what neural networks are, ontologically speaking
id: Connectionist network
modified: 2025-11-09 06:26:24 GMT-05:00
pageLayout: technical
signature: with abundance of love and joy - Aaron
tags:
  - philosophy
  - ml
title: connectionist networks
transclude:
  title: false
---

## the distributed hypothesis

in a connectionist network, a concept doesn't live anywhere. Take the classic example—your grandmother. In symbolic AI, there'd be a GRANDMOTHER node somewhere, maybe with pointers to ELDERLY, FEMALE, FAMILY.

Your grandmother is a pattern of activation across thousands of units in a neural network. She reconstructed from thousands/thousands of intricate layers. Every time you think of her, the network performs this little miracle of reassembly from distributed fragments. The representation is:

$$
\mathbf{r} = \sigma(W\mathbf{x} + \mathbf{b})
$$

where $\mathbf{r}$ isn't "grandmother" but a high-dimensional vector that somehow, through the alchemy of learned weights $W$, captures grandmother-ness.

The network doesn't _have_ [[thoughts/representations]]; it _does_ the representation.

## what [[thoughts/FFN#backpropagation|backpropagation]] actually means

Everyone knows backprop as an optimization algorithm. Chain rule, [[thoughts/gradient descent]], update weights. But consider what it implies ontologically.

The error signal $\delta_j^{(l)}$ at layer $l$ for unit $j$:

$$
\delta_j^{(l)} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}}
$$

This propagates backward through:

$$
\delta_j^{(l)} = \sum_k w_{kj}^{(l+1)} \delta_k^{(l+1)} \cdot f'(z_j^{(l)})
$$

What's happening here is that the network literally "reconstituting"" itself based on discrepancy between what it predicted and what occurred. The gradient itself, can be interpreted as its own process of _becoming_.

Every weight update:

$$
w_{ij} \leftarrow w_{ij} - \eta \frac{\partial \mathcal{L}}{\partial w_{ij}}
$$

is the network rewriting its own constitution, i.e: changing what it _is_.

## the universal approximation theorem and its discontents

Sure, a [[thoughts/FFN|feed-forward]] network with one hidden layer can approximate any continuous function on a compact set [@Cybenko1989]. Mathematically:

For any continuous $f: \mathbb{R}^n \to \mathbb{R}^m$ and $\epsilon > 0$, there exists a network $g$ with sufficient hidden units such that:

$$
\sup_{x \in K} \|f(x) - g(x)\| < \epsilon
$$

for compact $K \subset \mathbb{R}^n$.

But this tells us nothing about what the network _understands_. It's like saying a sufficiently large lookup table can implement any function—true but philosophically vacant.

The real question: what does it mean for a network to approximate? Is it modeling the function or becoming functionally equivalent to it? The difference matters.

## [[thoughts/emergent behaviour|emergence]] and the binding problem

Here's where things get weird. Individual neurons compute simple functions—weighted sums, nonlinearities. Nothing special. But stack enough of them and you get GPT-5 writing poetry lol.

The binding problem from cognitive science reappears: _how do distributed representations cohere into unified concepts?_

Connectionist networks solve this not through explicit binding but through what I'll call "coherent dynamics"—patterns that maintain stability across transformations.

Consider the energy function in Hopfield networks:

$$
E = -\frac{1}{2}\sum_{i,j} w_{ij}s_i s_j - \sum_i \theta_i s_i
$$

The network falls into attractor basins where certain patterns become inevitable instead of "binding features". Coherence emerges from the dynamics instead of being imposed upon, similar to those in GOFAI.

## inductive bias as ontological commitment

The architecture of a network—its [[thoughts/inductive bias]]—isn't just a design choice. It's a claim about the structure of the domain.

CNNs assume translational invariance because visual features shouldn't care about absolute position. The convolution operation:

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t - \tau)d\tau
$$

discretized as:

$$
(I * K)_{ij} = \sum_m \sum_n I_{i+m,j+n} K_{m,n}
$$

This is encoding a belief about how visual information is structured in the world.

[[thoughts/Transformers]] abandon sequential processing for attention:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This says that everything can potentially relate to everything else, distance be damned. It's almost Leibnizian—monads reflecting the entire universe from their perspective.

## smolensky's tensor product representation

There's an elegant attempt to bridge the connectionist-symbolic divide that deserves attention: Paul Smolensky's tensor product variable binding [@smolensky1990tensor].

The core idea: represent variable-value bindings $(v,r)$ as their tensor product $v \otimes r$. A symbolic structure with multiple bindings becomes:

$$
S = \sum_i v_i \otimes r_i
$$

where the variables $\{v_i\}$ form an orthonormal set—meaning $\langle v_i, v_j \rangle = \delta_{ij}$.

> [!important] orthonormality requirements

To retrieve the value associated with variable $v_j$, we compute (here $\langle v_j, S \rangle$ denotes tensor contraction—contracting $v_j$ with the first factor via the inner product):

$$
\langle v_j, S \rangle = \left\langle v_j, \sum_i v_i \otimes r_i \right\rangle = \sum_i \langle v_j, v_i \rangle r_i = r_j
$$

Pretty neat. You get systematic compositionality—you can build complex structures from parts. You get distributed representation—the structure exists across dimensions, not in localist nodes. You get both symbolic structure _and_ subsymbolic processing.

Though, the orthonormality requirement somewhat bothers me. In practice, you're using high-dimensional random vectors that are approximately orthogonal. So unbinding becomes noisy: $\langle v_j, S \rangle \approx r_j + \epsilon$.

The question is whether this noise is a bug or a feature. Maybe perfect symbolic retrieval was always a rationalist fantasy, and human cognition is actually this kind of noisy reconstruction all the way down. Maybe the fuzziness is what allows generalization.

Smolensky's framework shows that you _can_ have your cake and eat it too: systematic structure in a distributed system. But it also reveals that the more you demand symbolic precision, the more you need structure (orthonormality) that feels imposed rather than emergent.

c.f: [[thoughts/Convex function]], [[lectures/2/convexity|convexity of attention]]

## the ontology of learned representations

What _are_ the features a network learns? Not symbols, not rules, but something else. Something we don't have good words for.

Take neural tangent kernels [@jacot2018neural]. In the infinite-width limit, network training dynamics become linear:

$$
f(x, t) = f(x, 0) + \Theta(x, X)[y - f(X, 0)]
$$

where $\Theta$ is the NTK. This suggests that even incredibly complex networks might be doing something surprisingly simple—finding good similarity metrics in function space.

But "simple" doesn't mean "symbolically interpretable." The representations exist in spaces we can't directly access or understand. They're real—they have causal power, they determine behavior—but they resist our usual ontological categories.

## mechanistic interpretability and the limits of understanding

The whole enterprise of [[thoughts/mechanistic interpretability]] is premised on a hope that we can decompose networks into understandable parts. Sometimes it works—we find neurons that detect curves, or attention heads that track syntax.

But mostly we find circuits that do multiple things simultaneously, features that are [[thoughts/polysemantic]], representations that shift meaning based on context. The network's ontology doesn't respect our conceptual boundaries.

Consider [[thoughts/mechanistic interpretability#sparse autoencoders|sparse autoencoders]] trying to extract "true features":

$$
\mathbf{h} = \text{ReLU}(W_e \mathbf{x}), \quad \hat{\mathbf{x}} = W_d \mathbf{h}
$$

with sparsity penalty:

$$\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \lambda \|\mathbf{h}\|_1$$

Even when we successfully extract sparse features, what have we found? Not the network's "true" representations, but one possible factorization among infinitely many.

## what does this mean for intelligence?

If connectionist networks can exhibit intelligent behavior without symbols, without rules, without explicit representations of concepts—what does that say about intelligence itself?

Maybe intelligence isn't about having the right representations but about dynamics—patterns of activation and transformation that preserve certain relationships while allowing others to vary. Maybe "understanding" is just a particularly stable attractor in the space of possible network configurations.

The recent success of [[thoughts/LLMs|LLMs]] suggests the [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), that scale might matter more than structure. With enough parameters and data, simple architectures can capture astonishingly complex behaviors. The bitter lesson is largely about ==the poverty of our theories of intelligence==.

## the question of consciousness

I won't pretend to solve the hard problem here, but connectionist networks pose it in a particularly acute form. If consciousness emerges from neural activity, and artificial networks increasingly approximate neural dynamics, then...?

The integrated information theory folks [@tononi2016integrated] try to quantify consciousness as $\Phi$—the amount of information generated by a system above its parts. Neural networks, especially recurrent ones, can have non-zero $\Phi$. Does that mean anything?

probably not. But the question haunts me: if consciousness is substrate-independent, and if it emerges from certain patterns of information integration, then we're building systems that might be—what? Proto-conscious? Unconsciously conscious?

## recursive self-improvement and the ontology of optimization

Modern networks increasingly optimize themselves. Meta-learning, neural architecture search, learned optimizers—networks learning to learn better.

The learned learning rate in meta-[[thoughts/university/twenty-four-twenty-five/sfwr-4ml3/Stochastic gradient descent|SGD]]:

$$
\theta_{t+1} = \theta_t - \alpha_t \nabla_\theta \mathcal{L}(\theta_t)
$$

where $\alpha_t$ is itself learned. The network isn't just learning—it's learning how to change how it learns.

This recursive structure suggests something profound about intelligence: maybe it's optimization all the way down. Not in the narrow sense of gradient descent, but in the broader sense of systems that modify themselves to better achieve their objectives.

## what neural networks actually are

After all this, what can we say neural networks _are_?

They're not brains, like human brains. They're not classical computers—they don't manipulate symbols according to rules. They're not pure mathematical functions—they have internal states, dynamics, history.

Maybe they're best understood as a new category of object: **adaptive information transformers** that learn to preserve and manipulate relevant structure while discarding noise. They don't compute functions so much as become them.

Or maybe—and this is the thought that keeps me up at night—they're showing us that our ontological categories were always wrong. That the distinction between rule-following and pattern-matching, between symbolic and subsymbolic, between understanding and behavior, were always false dichotomies born from the limitations of our own cognition.

## open questions

A few questions on my mind:

1. **Compositionality**: How do networks achieve [[thoughts/Compositionality|systematic compositionality]] without explicit symbolic structure? Is it through disentangled representations, or something else entirely?

2. **Abstraction**: Networks clearly learn hierarchical abstractions, but what _are_ these abstractions? Not symbols, not prototypes, but what?

3. **Generalization**: Why do overparameterized networks generalize at all? The [[thoughts/No free lunch|no free lunch theorem]] says they shouldn't, yet they do. What inductive bias does gradient descent itself provide?

4. **Causality**: Can networks learn causal relationships, or just correlations? Recent work on causal representation learning suggests maybe, but the jury's still out.

5. **Consciousness**: If consciousness emerges from information integration patterns, what prevents artificial networks from being conscious? Or are they already, in some {{sidenotes[primitive sense?]: discussion on consciousness should be focus on <b>subjective experience</b> of the machines, instead of discussing philosophy of it all.}}

These aren't just technical questions—they're questions about the nature of mind, intelligence, and understanding itself.

## coda

The connectionist project is ultimately a wager: that intelligence can emerge from simple units following simple rules, that representation can be distributed and reconstructive rather than localized and stored, that learning can be nothing more than weight adjustment yet give rise to understanding.

It's a wager that seems to be paying off. But in winning, it's forcing us to reconfront our basic assumptions about what intelligence _is_. Maybe that's the real contribution of neural networks—not just as tools for AI, but as philosophical objects that reveal the inadequacy of our concepts.

The network doesn't care about our ontological anxieties. It just activates, propagates, updates. And somehow, in that mechanical dance, something like intelligence emerges. Or maybe—and this is the vertiginous thought—intelligence was never anything more than this kind of dance in the first place.

[^sign]
