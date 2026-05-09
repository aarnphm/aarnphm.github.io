---
date: "2025-12-12"
description: and tile primitives
id: cutedsl
modified: 2026-05-09 17:51:49 GMT-04:00
socials:
  introduction: https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/
  layout algebra: https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/
  lectures: https://www.youtube.com/watch?v=MVh_guNbWMA
tags:
  - ml
  - dsl
title: CuTeDSL
---

A Tensor is specified by an _iterator_ and a _layout_:

$$
L = S : D
$$

where $S$ denotes the _shape_ of the tensor, and $D$ denotes the _strides_ of the tensor

For example:

row-major versus col-major

$$
\begin{array}{ccc}
\begin{array}{|c|c|c|c|c|c|}
\hline
0 & 1 & 2 & 3 & 4 & 5 \\
\hline
6 & 7 & 8 & 9 & 10 & 11 \\
\hline
12 & 13 & 14 & 15 & 16 & 17 \\
\hline
18 & 19 & 20 & 21 & 22 & 23 \\
\hline
\end{array}
&
\hspace{3em}
&
\begin{array}{|c|c|c|c|c|c|}
\hline
0 & 4 & 8 & 12 & 16 & 20 \\
\hline
1 & 5 & 9 & 13 & 17 & 21 \\
\hline
2 & 6 & 10 & 14 & 18 & 22 \\
\hline
3 & 7 & 11 & 15 & 19 & 23 \\
\hline
\end{array}
\\[1em]
L = (4,6) : (6,1) & & L = (4,6) : (1,4)
\end{array}
$$

broadcast versus strides

$$
\begin{array}{ccc}
\begin{array}{|c|c|c|c|c|c|}
\hline
0 & 0 & 0 & 0 & 0 & 0 \\
\hline
1 & 1 & 1 & 1 & 1 & 1 \\
\hline
2 & 2 & 2 & 2 & 2 & 2 \\
\hline
3 & 3 & 3 & 3 & 3 & 3 \\
\hline
\end{array}
&
\hspace{3em}
&
\begin{array}{|c|c|c|c|c|c|}
\hline
0 & 1 & 2 & 3 & 4 & 5 \\
\hline
12 & 13 & 14 & 15 & 16 & 17 \\
\hline
24 & 25 & 26 & 27 & 28 & 29 \\
\hline
36 & 37 & 38 & 39 & 40 & 41 \\
\hline
\end{array}
\\[1em]
L = (4,6) : (1,0) & & L = (4,6) : (12,1)
\end{array}
$$
