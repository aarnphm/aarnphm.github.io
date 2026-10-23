---
date: '2024-02-07'
description: how a web search engine finds and ranks documents, with hits link analysis, bm25 and hnsw vector search.
id: Search
modified: 2026-06-05 15:08:21 GMT-04:00
seealso:
  - '[[thoughts/PageRank|PageRank]]'
tags:
  - technical
title: Search
---

## Engine

A web search engine answers a query with a ranked list of documents. It is a form of [[thoughts/information retrieval|information retrieval]] and has three main stages:

1. **crawling**: a crawler follows links and revisits known pages on a schedule. The index is therefore always somewhat out of date.
2. **indexing**: the engine builds an inverted index, which maps each term to the documents that contain it, and computes query-independent signals such as [[thoughts/PageRank|PageRank]].
3. **ranking**: for each query, it retrieves candidate documents from the index, scores them, and returns the ordered list as a search engine results page (SERP).

A web directory differs in stage 2: human editors choose and classify the entries.

## BM25

BM25 scores a document $D$ for a query $Q$ by term matches:

$$
\operatorname{score}(D, Q) = \sum_{t \in Q} \operatorname{IDF}(t) \cdot \frac{f(t, D)\,(k_1 + 1)}{f(t, D) + k_1\left(1 - b + b\,\frac{\lvert D\rvert}{\operatorname{avgdl}}\right)}
$$

where $f(t,D)$ is the count of term $t$ in $D$, $\lvert D\rvert$ is the document length and $\operatorname{avgdl}$ is the average length in the collection. A rare term has a high $\operatorname{IDF}$. The fraction saturates: with $k_1$ around $1.2$, the tenth occurrence of a term adds much less than the first. $b$ around $0.75$ reduces the score of long documents, which match more terms by chance ([Robertson and Zaragoza, 2009](https://doi.org/10.1561/1500000019)).

## HITS algorithm

Kleinberg's HITS [(1999)](https://doi.org/10.1145/324133.324140) gives each page two scores. A good **authority** receives links from good hubs. A good **hub** links to good authorities. Let $A$ be the adjacency matrix of the link graph, with $A_{ij}=1$ when page $i$ links to page $j$. Starting from all-ones vectors, repeat

$$
\mathbf{a} \leftarrow A^\top \mathbf{h}, \qquad \mathbf{h} \leftarrow A\,\mathbf{a},
$$

and normalize both vectors after each step. This is power iteration, so $\mathbf{a}$ converges to the principal eigenvector of $A^\top A$ and $\mathbf{h}$ to that of $A A^\top$ (when the top eigenvalue is unique).

HITS depends on the query. It takes the top results of a text search as a root set, adds the pages that link to them or that they link to, and runs on that subgraph. [[thoughts/PageRank|PageRank]] computes one score per page over the whole graph before any query arrives, which is cheaper at query time.

## HNSW

_https://blog.wilsonl.in/graph-vector-search/_

Hierarchical navigable small world graphs [(Malkov and Yashunin, 2018)](https://arxiv.org/abs/1603.09320) find approximate nearest neighbours of a query vector. Each vector is a node, and each node links to about $M$ near neighbours. A node's top layer is drawn as $\lfloor -\ln(u)\, m_L \rfloor$ with $u$ uniform on $(0,1)$ and $m_L = 1/\ln M$, so each layer holds about a $1/M$ fraction of the layer below.

A search starts at the top layer and moves greedily to the neighbour nearest the query until no neighbour is nearer. It then goes down one layer from that node. At layer 0 it keeps a candidate list of size $\mathit{ef}$; a larger $\mathit{ef}$ gives higher recall and slower queries. The upper layers have few nodes and long links, so the search crosses the space in a small number of steps.

Dense retrieval with HNSW matches meaning but can miss exact strings such as names, identifiers and error codes. BM25 matches those strings but misses paraphrases. Hybrid search runs both and merges the two ranked lists, for example with reciprocal rank fusion, which scores a document by $\sum_r 1/(k + \operatorname{rank}_r(d))$ over the lists $r$, with $k=60$ in [Cormack et al., 2009](https://doi.org/10.1145/1571941.1572114).
