---
date: '2024-11-04'
description: test whether one network can use another network's activations through a learned stitching layer.
id: model stiching
modified: 2026-10-02 09:06:45 GMT-04:00
noindex: true
tags:
  - ml
title: model stiching
---

Model stitching asks whether the lower layers of one network can supply the activations needed by the upper layers of another. Lenc and Vedaldi insert a learned transformation between those pieces and evaluate the resulting network on its task. [@lenc2015understandingimagerepresentationsmeasuring]

Write the original networks as $A=A_{>l}\circ A_{\le l}$ and $B=B_{>k}\circ B_{\le k}$. A stitch $S$ gives

$$
B_{>k}\circ S\circ A_{\le l}.
$$

Keep both network pieces fixed and train $S$ on the task loss. A small affine map is a useful starting point; convolutional feature maps can also need spatial resampling. Compare held-out performance with the intact receiving network $B$. The allowed stitch, the cut layers, and the evaluation data are part of the result.

For a simple case, suppose $A_{\le l}(x)=(u,v)$ while $B_{\le k}(x)=(v,u)$. The stitch

$$
S=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

restores exactly what $B_{>k}$ expects. Directly connecting the two pieces would swap the inputs seen by every downstream weight. The learned representations can support the same computation while using different coordinates.

A successful stitch shows that the receiving network can use the donor's representation for the evaluated task. It leaves open how either network computes that representation and what information the receiving layers discard. [[thoughts/SVCCA]] measures correlations between activation subspaces; stitching tests the behaviour of an assembled network.
