garden v2, on top of [quartz](https://quartz.jzhao.xyz/) v4.

> “[One] who works with the door open gets all kinds of interruptions, but [they] also occasionally gets clues as to what the world is and what might be important.” — Richard Hamming

fwiw a lot of plugins are disabled during serve given that I have a lot of math notes (KaTeX is pretty expensive to run.)

## features

A modified/personal enhancement from bare Quartz

### [telescopic-text](https://github.com/jackyzha0/telescopic-text)

Support a small subsets of the features, with wikilinks parsing

````
```telescopic
* reading
  * reading a lot of Nietzsche,
  * hosting functions,
    * go on longs walks,
    * building [[thoughts/work|open-source project]],
    * this [pan](https://example.com)
```
````

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

Currently, there is a few pgfplots bug upstream in node port, so to remove the graph from target rendering add `alt` as the URI svg:

````
```tikz alt="data:image/svg+xml..."
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

### reader view

_press cmd/ctrl+b_
