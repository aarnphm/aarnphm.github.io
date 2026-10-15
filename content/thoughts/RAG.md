---
date: '2024-02-07'
description: retrieve passages for each input, then condition a generator on them; the lewis et al. model and what it does and does not fix.
id: RAG
modified: 2026-06-05 15:08:22 GMT-04:00
tags:
  - technical
  - ml
title: RAG
---

A language model stores what it learned in its weights. Updating that knowledge means retraining, and the model cannot show which source supports an answer. Retrieval-augmented generation (RAG) adds an external text collection: for each input, a retriever finds passages, and the generator conditions on them. [@lewis2021retrievalaugmentedgenerationknowledgeintensivenlp]

In Lewis et al., the collection is about 21 million 100-word Wikipedia chunks. The retriever is DPR: two BERT encoders map a query $x$ and a passage $z$ to vectors, and

$$
p_\eta(z \mid x) \propto \exp\left(\mathbf{d}(z)^\top \mathbf{q}(x)\right).
$$

A maximum inner product search over the precomputed passage vectors returns the top $k$ passages. The generator is BART. The **RAG-Sequence** model uses one passage for the whole output and sums over the retrieved passages:

$$
p(y \mid x) \approx \sum_{z \in \operatorname{top-}k} p_\eta(z \mid x) \prod_{i} p_\theta(y_i \mid x, z, y_{<i}).
$$

The **RAG-Token** model can use a different passage for each token:

$$
p(y \mid x) \approx \prod_{i} \sum_{z \in \operatorname{top-}k} p_\eta(z \mid x)\, p_\theta(y_i \mid x, z, y_{<i}).
$$

Because $z$ is summed out, training needs only $(x, y)$ pairs and no labels for which passage is correct. The query encoder and generator are fine-tuned together. The passage encoder stays fixed, so the index does not need rebuilding during training. Replacing the index changes what the model can cite without retraining it.

Most deployed systems keep the retrieve-then-condition structure but drop the joint training. They put retrieved chunks in the prompt of a frozen model. The usual parts are a chunker, an [[thoughts/Embedding|embedding]] model, an approximate nearest-neighbour index (see [[thoughts/Search#HNSW|HNSW]]), often a lexical retriever such as BM25, and sometimes a reranker.

Retrieval changes the source of errors. If the retriever misses the relevant passage, the generator can still answer from its weights, and nothing marks that answer as unsupported. If a retrieved passage is wrong or off-topic, the generator can repeat it. Retrieval recall and the generator's faithfulness to its context need separate measurements.

An agent that calls a search tool uses retrieval as one action among others. Retrieval is one component of such systems, and many agents use no retrieval at all.
