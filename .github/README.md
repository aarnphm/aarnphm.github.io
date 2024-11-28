garden v2, on top of [quartz](https://quartz.jzhao.xyz/) v4.

garden links: https://aarnphm.xyz

> “[One] who works with the door open gets all kinds of interruptions, but [they] also occasionally gets clues as to what the world is and what might be important.” — Richard Hamming

## features

A modified/personal enhancement from bare Quartz

### TikZ support

to use in conjunction with [obsidian-tikzjax](https://github.com/artisticat1/obsidian-tikzjax/)

````
```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\pi^{-1}(U) \arrow[r, "\varphi"] \arrow[d, "\pi"'] & U \times F \arrow[ld, "proj_1"] \\
U &
\end{tikzcd}
\end{document}
```
````

Currently, there is a few pgfplots bug upstream in node port, so to remove the graph from target rendering add `ablate=true`:

````
```tikz ablate=true
```
````

### pseudocode support

````
```pseudo
\begin{algorithm}
\caption{LLM token sampling}
\begin{algorithmic}
\Function{sample}{$L$}
\State $s \gets ()$
\For{$i \gets 1, L$}
\State $\alpha \gets \text{LM}(s, \theta)$
\State Sample $s \sim \text{Categorical}(\alpha)$
\If{$s = \text{EOS}$}
\State \textbf{break}
\EndIf
\State $s \gets \text{append}(s, s)$
\EndFor
\State \Return $s$
\EndFunction
\end{algorithmic}
\end{algorithm}
```
````

The target render should also include a copy button

### collapsible header

inspired by dynalist

### Gaussian-scaling TOC

inspired by press.stripe.com
