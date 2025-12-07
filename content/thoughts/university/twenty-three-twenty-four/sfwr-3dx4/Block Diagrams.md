---
date: "2024-01-24"
description: control system block diagrams with cascade form, parallel form, feedback loops, summing junctions, and transfer function reduction techniques.
id: Block Diagrams
modified: 2025-10-29 02:16:20 GMT-04:00
tags:
  - sfwr3dx4
title: Block Diagrams
seealso:
  - "[[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/block_diagrams.pdf|sides]]"
---

## Cascade form

```tikz
\usepackage{tikz}
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[auto, node distance=1.8cm, >=latex]

% Part (a) - Series Block Diagram
\node at (2, 1.5) {\textit{(a)}};

% Nodes for first block diagram
\node[] (R) {$R(s)$};
\node[draw, rectangle, right=1.5cm of R, minimum width=1.5cm, minimum height=1cm] (G1) {$G_1(s)$};
\node[draw, rectangle, right=1.5cm of G1, minimum width=1.5cm, minimum height=1cm] (G2) {$G_2(s)$};
\node[draw, rectangle, right=1.5cm of G2, minimum width=1.5cm, minimum height=1cm] (G3) {$G_3(s)$};
\node[right=1.5cm of G3] (C) {$C(s)$};

% Arrows for first block diagram
\draw[->] (R) -- (G1);
\draw[->] (G1) -- (G2);
\draw[->] (G2) -- (G3);
\draw[->] (G3) -- (C);

% Part (b) - Simplified Block Diagram
\node at (2, -1.5) {\textit{(b)}};

% Nodes for simplified block diagram
\node[] (R2) at (0, -3) {$R(s)$};
\node[draw, rectangle, right=3.5cm of R2, minimum width=3.5cm, minimum height=1cm] (G123) {$G_3(s)G_2(s)G_1(s)$};
\node[right=2cm of G123] (C2) {$C(s)$};

% Arrows for simplified block diagram
\draw[->] (R2) -- (G123);
\draw[->] (G123) -- (C2);

\end{tikzpicture}
\end{document}
```

## Parallel form

```tikz
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[auto, node distance=1.5cm, >=Latex]

% -------------------- PART (a) --------------------
\node at (1.5, 1.5) {\textit{(a)}};

% Input node
\node (R) at (1.5, -1.5) {$R(s)$};

% Blocks G1, G2, G3
\node[draw, rectangle, minimum width=1.5cm, minimum height=1cm, right=1.5cm of R, yshift=1.5cm] (G1) {$G_1(s)$};
\node[draw, rectangle, minimum width=1.5cm, minimum height=1cm, right=1.5cm of R] (G2) {$G_2(s)$};
\node[draw, rectangle, minimum width=1.5cm, minimum height=1cm, right=1.5cm of R, yshift=-1.5cm] (G3) {$G_3(s)$};

% Summing point
\node[draw, circle, minimum size=0.5cm, right=4.5cm of R] (sum) {\Large $\Sigma$};

% Output node
\node[right=1.5cm of sum] (C) {$C(s)$};

% Arrows from R to blocks
\draw[->] (R) -- (G1);
\draw[->] (R) -- (G2);
\draw[->] (R) -- (G3);

% Arrows from blocks to summing point
\draw[->] (G1.east) -- ++(2,0) |- (sum.north);
\draw[->] (G2.east) -- ++(0.5,0) -- (sum.west);
\draw[->] (G3.east) -- ++(2,0) |- (sum.south);

% Arrow from summing point to output
\draw[->] (sum.east) -- (C);

% -------------------- PART (b) --------------------
\node at (1.5, -5) {\textit{(b)}};

% Simplified block diagram
\node[draw, rectangle, minimum width=3.5cm, minimum height=1cm, below=4cm of R] (simplified) {$\pm G_1(s) \pm G_2(s) \pm G_3(s)$};

% Input and output for simplified block
\node[left=1.5cm of simplified] (R2) {$R(s)$};
\node[right=1.5cm of simplified] (C2) {$C(s)$};

% Arrows for simplified block
\draw[->] (R2) -- (simplified);
\draw[->] (simplified) -- (C2);

\end{tikzpicture}
\end{document}
```

## Feedback loop

![[thoughts/university/twenty-four-twenty-five/sfwr-4aa4/feedback-loop-transfer-function.webp]]

## Moving through summing junction

![[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/images/summing junction.webp]]

## Reduction via Familiar Forms
