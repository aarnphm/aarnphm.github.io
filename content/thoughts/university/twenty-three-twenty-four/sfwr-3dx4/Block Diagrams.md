---
date: '2024-01-24'
description: control system block diagrams with cascade form, parallel form, feedback loops, summing junctions, and transfer function reduction techniques.
id: Block Diagrams
modified: 2026-06-05 15:08:43 GMT-04:00
seealso:
  - '[[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/block_diagrams.pdf|sides]]'
tags:
  - sfwr3dx4
title: Block Diagrams
---

A block diagram specifies equations between signals. A block multiplies its input by its transfer function, a summing junction adds signed inputs, and a pickoff copies a signal to another branch. Reduction means eliminating internal signals while preserving the chosen input-output relation.

The rules below use linear, time-invariant blocks and zero initial conditions. For physical subsystems, include any loading from their connections in the model before multiplying transfer functions. An unbuffered circuit stage can change the preceding stage's output when connected to it.[^blocks]

## Cascade form

Follow the arrows in order. If the first block produces $G_1(s)R(s)$, the next produces $G_2(s)G_1(s)R(s)$. Thus

$$
\frac{C(s)}{R(s)}=G_3(s)G_2(s)G_1(s).
$$

Scalar transfer functions commute. Transfer matrices generally do not, so retain the order of the signal path when working with multiple inputs and outputs.

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

Each branch receives the same input. Add its output with the sign shown at the summing junction:

$$
C(s)=\bigl[G_1(s)+G_2(s)-G_3(s)\bigr]R(s).
$$

The diagram below leaves the signs as choices in its reduced block. Use the actual junction signs for a particular system; three added branches give $G_1+G_2+G_3$.

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

With forward block $G(s)$ and feedback block $H(s)$, negative feedback gives

$$
\begin{aligned}
E(s)&=R(s)-H(s)C(s),\\
C(s)&=G(s)E(s),\\
\bigl[1+G(s)H(s)\bigr]C(s)&=G(s)R(s),\\
\frac{C(s)}{R(s)}&=\frac{G(s)}{1+G(s)H(s)}.
\end{aligned}
$$

For positive feedback, the junction adds $H(s)C(s)$, giving denominator $1-G(s)H(s)$. Write the signal equation first if the sign is unclear. Negative feedback alone does not establish stability; the resulting closed-loop dynamics still need a stability check.

The figure uses $H_1$ for the forward block and $H_2$ for the feedback block.

```jsx imports={Zoomable,BlockDiagram}
<Zoomable label="Feedback loop reduction">
  <BlockDiagram
    config="feedback"
    caption="Forward block $H_1$ with feedback block $H_2$. Choose feedback sign and step through the reduction: negative gives $H_1/(1+H_1H_2)$, positive gives $H_1/(1-H_1H_2)$."
  />
</Zoomable>
```

## Moving through summing junction

Preserve the gain from each independent input to the output. Moving $G(s)$ from after a junction to the main branch before it requires the same gain on the side branch:

$$
C(s)=G(s)\bigl[R(s)\pm X(s)\bigr]
    =G(s)R(s)\pm G(s)X(s).
$$

Moving $G(s)$ in the other direction requires a compensating inverse on the side branch:

$$
C(s)=G(s)R(s)\pm X(s)
    =G(s)\left[R(s)\pm\frac{X(s)}{G(s)}\right].
$$

The inverse is an algebraic device. For example, the inverse of an integrator is a differentiator, which cannot be implemented as a proper causal transfer function. A reduced diagram may establish an input-output identity while its intermediate blocks are unsuitable for implementation.

```jsx imports={Zoomable,BlockDiagram}
<Zoomable label="Moving a block past a summing junction">
  <BlockDiagram
    config="summing-junction"
    caption="Block move after Figure 5.7 of the course slides. Choose (a) or (b) and step through; the readout expands each output, and matching expansions show equivalence."
  />
</Zoomable>
```

## Reduction via Familiar Forms

Start with the innermost cascade, parallel branch, or feedback loop. Replace it with its equivalent block, then repeat. Keep each summing sign and pickoff signal labelled. When a diagram is hard to reduce, write an equation for each internal signal and eliminate those signals directly.

For example, suppose the signal equations are

$$
\begin{aligned}
E&=G_1R-HC,\\
C&=G_3G_2E.
\end{aligned}
$$

Substitution gives

$$
\frac{C}{R}=\frac{G_1G_2G_3}{1+G_2G_3H}.
$$

The loop contains $G_2$, $G_3$, and $H$. The reference prefilter $G_1$ appears only in the numerator because the feedback returns to the junction after it. With $G_1=2$, $G_2=1/(s+1)$, $G_3=1/(s+3)$, and $H=1$,

$$
\frac{C(s)}{R(s)}
=\frac{2}{(s+1)(s+3)+1}
=\frac{2}{s^2+4s+4}.
$$

Check a reduction against the original signal equations. Also retain the unreduced internal model when assessing stability: cancellation can hide internal modes from the final transfer function, as discussed in [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/State space representation#state space to a transfer function|state space representation]].

[^blocks]: R. J. Leduc and M. Lawford, [[thoughts/university/twenty-three-twenty-four/sfwr-3dx4/block_diagrams.pdf|block-diagram lecture slides]], PDF pages 1–14. The worked signal-equation example here is separate from the slide figures.
