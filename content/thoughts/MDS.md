---
date: '2025-11-23'
description: a form of non-linear dimensionality reduction
id: MDS
modified: 2025-11-23 16:38:52 GMT-05:00
tags:
  - ml
title: multidimensional scaling
---

_refers to a set of related ordination techniques in information visualisation_

Takes a some high-dimensional data $x_{1},\dots,x_{n} \in \mathbb{R}^{d}$ and produce a lower-dimensional representation $y_{1},\dots,y_{n} \in \mathbb{R}^{d}$ where pairwise distances (represented by a matrix $D \in \mathbb{R}^{d\times d}$ with $D_{ij}$ corresponds to some measure of distance between points $x_{i}$ and $x_{j}$)

in classical MDS, we use Euclidean distance metrics $d_{ij} =  \lVert x_{i} - x_{j} \rVert_{2}$

- MDS then seek to find points $y_{1},\dots,y_{n}\in \mathbb{R}^{p}$ such that the distances between these new points approximate to the original distance $\lVert y_{i} - y_{j} \rVert{2} \approx d_{ij}$

the sketch below shows a toy Euclidean distance graph (left) and its 2D embedding produced by classical MDS (right); edge labels are the target distances $d_{ij}$ the embedding tries to preserve.

```tikz
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}[>=latex,scale=2]
  \tikzset{
    datapoint/.style={circle, draw=black, fill=cyan!25, minimum size=6pt, inner sep=1.5pt},
    target/.style={circle, draw=black, fill=orange!35, minimum size=6pt, inner sep=1.5pt},
    edge/.style={gray!70},
    dashededge/.style={edge, dashed}
  }

  \begin{scope}
    \node[datapoint] (x1) at (-1.1,  1.6) {$x_1$};
    \node[datapoint] (x2) at ( 1.4,  2.2) {$x_2$};
    \node[datapoint] (x3) at (-1.3, -0.3) {$x_3$};
    \node[datapoint] (x4) at ( 1.6, -0.9) {$x_4$};
    \node[datapoint] (x5) at ( 0.0,  0.5) {$x_5$};

    \draw[dashededge] (x1) -- node[sloped, above]{2.5} (x2);
    \draw[dashededge] (x1) -- node[sloped, left ]{2.1} (x3);
    \draw[dashededge] (x2) -- node[sloped, right]{3.0} (x4);
    \draw[dashededge] (x3) -- node[sloped, below]{2.9} (x4);
    \draw[edge]      (x1) -- node[sloped, above]{1.5} (x5);
    \draw[edge]      (x2) -- node[sloped, above]{1.8} (x5);
    \draw[edge]      (x3) -- node[sloped, left ]{1.2} (x5);
    \draw[edge]      (x4) -- node[sloped, right]{1.6} (x5);

    \node[align=left, anchor=west] at (-2.2, -1.6) {distance graph $d_{ij} \approx ||x_i - x_j||_2$};
  \end{scope}

  \draw[->, thick] (2.2,0.2) -- (4.2,0.2)
    node[midway, below=2pt]{classical MDS};

  \begin{scope}[xshift=6.2cm]
    \node[target] (y1) at (-0.9,  1.1) {$y_1$};
    \node[target] (y2) at ( 1.3,  1.5) {$y_2$};
    \node[target] (y3) at (-0.8, -0.6) {$y_3$};
    \node[target] (y4) at ( 1.5, -0.5) {$y_4$};
    \node[target] (y5) at ( 0.1,  0.2) {$y_5$};

    \draw[dashededge] (y1) -- node[sloped, above]{2.5} (y2);
    \draw[dashededge] (y1) -- node[sloped, left ]{2.0} (y3);
    \draw[dashededge] (y2) -- node[sloped, right]{3.1} (y4);
    \draw[dashededge] (y3) -- node[sloped, below]{2.7} (y4);
    \draw[edge]      (y1) -- node[sloped, above]{1.6} (y5);
    \draw[edge]      (y2) -- node[sloped, above]{1.7} (y5);
    \draw[edge]      (y3) -- node[sloped, left ]{1.3} (y5);
    \draw[edge]      (y4) -- node[sloped, right]{1.5} (y5);

    \node[align=left, anchor=west] at (-1.6, -1.6) {2d embedding $||y_i-y_j||_2 \approx d_{ij}$};
\end{scope}
\end{tikzpicture}
\end{document}
```
