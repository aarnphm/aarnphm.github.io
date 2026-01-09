---
date: "2024-12-10"
description: nobby designer word for "the design of the current site"
id: colophon
lang: en
modified: 2026-01-09 06:21:52 GMT-05:00
socials:
  twitter: https://x.com/aarnphm/status/1861550609834402129
tags:
  - evergreen
  - design
title: colophon
---

**#technical** -- a [heavily modified fork](https://en.wikipedia.org/wiki/Ship_of_Theseus) of [Quartz](https://quartz.jzhao.xyz/), hosted with Cloudflare Workers, with support for arXiV file pulling, [telescopic](https://github.com/jackyzha0/telescopic-text) [[/index|text]], a [note view](https://notes.aarnphm.xyz/notes?stackedNotes=bm90ZXM), [[/singularity|JSONCanvas supports]], [[cinematheque.base|Obsidian Bases support]] hierarchical grid layout, reader view with custom [[/thoughts|folder]] and [[/tags|tags]] view, nested [[/index.xml|rss]] [[/posts/index.xml|feed]], {{sidenotes[tufte]: This is a tiny sidenote, supporting dynamic view size. Very much along the lines of [Tufte CSS](https://edwardtufte.github.io/tufte-css/)}} sidenotes, sidepanels [^sidepanel], modified [Flexoki](https://stephango.com/flexoki). I also host all of the LFS on Cloudflare R2, and have a middleware as a proxy client.

[^sidepanel]: You can hold <kbd>alt+click</kbd> on any internal links to [popover](https://x.com/aarnphm/status/1884954569341272345) a side panel 😃

**typography** -- [PP Neue Montreal](https://pangrampangram.com/products/neue-montreal), [<span style="font-family: 'Parclo Serif'">Parclo Serif</span>](https://lettermatic.com/fonts/parclo-serif?plan=student), [<span style="font-family: 'ITCGaramondStdLtCond'">ITC Garamond</span>](https://www.typewolf.com/itc-garamond) and [`berkeley mono{:text}`](https://usgraphics.com/products/berkeley-mono)

**accessibility** -- follow ARIA spec. I tried to modify a few value in rose-pine to add a bit more contrast. Press <span style="text-transform: uppercase"><kbd>D</kbd></span> anywhere to toggle between light and dark mode.

**components** -- I added support for rendering [[thoughts/Vector calculus#gradient|tikz graph]], [[thoughts/Transformers#Feynman-Kac|pseudocode]] support, [dynalist](https://dynalist.io)-inspired [[thoughts/mechanistic interpretability#inference|collapsible header]], a few customised [[posts/new#^ending|signature]], tiny [transformers plugins](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/plugins/transformers/aarnphm.ts), micromarks extensions for [wikilinks](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-wikilinks), [sidenotes](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-sidenotes), [[posts/25/n-bday|protected notes]], and additional supports of [`renderPage.tsx`](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/components/renderPage.tsx#L1487)

**usage** -- Some daily drivers that has found [[uses|its way]] into my life. But I use [Obsidian](https://obsidian.md/) for note-taking and Apple Notes while I'm on the go, in conjunction with [neovim](https://neovim.io/) for all my [[thoughts/craft|work-related shenanigans]].

**license** -- All my notes are licensed under CC BY-NC-SA [^license], whilst the code are under [Apache 2.0](http://www.apache.org/licenses/) (not everything here is written by me, but original authors are referenced & linked — they have their own licenses)

[^license]:
    This is not really a huge enforcer per-say, given that [not saying it](https://choosealicense.com/no-permission/) means "oh ALL RIGHTS RESERVED". These are mostly for my own [[notes|consumption]], and if you find it helpful then feel free to use it, just a quick mention would be appreciated 🫶

    I'm pretty lenient towards LLMs, most of the sites are pretty LLM-friendly. For all URL, you can just append `.md` to just get the markdown source.

**plugins** -- Most of additional items I wrote for this website follow [unified](https://unifiedjs.com/) ecosystem and can be exported as a standalone {{sidenotes[plugins.]: We are working on a few integrations separating out logics and improving general [Quartz ecosystem](https://github.com/quartz-community), stay tuned.}}

## a rather non-exhaustive lists

_of plugins that exists on this vault_

Also to run this with

```bash
pnpm exec tsx quartz/scripts/dev.ts --bg

tail -f .dev.log
```

For a more compact highlights, see [[thoughts/craft#^quartz|this list]]

### vim bindings

- `f` to navigate the sites on keyboards.
- `gh` for navigating ToC (supports j/k/Ctrl-n/Ctrl-p scrolling)
- `Cmd/Ctrl+'` for shortcuts helpers

### parser

some remark parsers for wikilinks, callouts, that supports general OFM compatibility

see [ofm-wikilinks](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-wikilinks/) and [ofm-sidenotes](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-sidenotes) for more information.

Supports [[cinematheque|Bases]] [API](https://help.obsidian.md/bases) and [[singularity.canvas|JSON]] [Canvas](https://help.obsidian.md/plugins/canvas)

### [telescopic-text](https://github.com/jackyzha0/telescopic-text)

Support a small subsets of the features, with wikilinks parsing

```telescopic id="reading"
* I'm
* reading
  * reading a lot of
    * Nietzsche,
    * Wittgenstein,
		 * hosting functions,
	 * and, go on longs walks,
	   * building [[thoughts/work|open-source project]],
	    * this [pan](https://example.com)
```

> [!summary]- code
>
> ````
> ```telescopic id="reading"
> * I'm
> * reading
>   * reading a lot of
>     * Nietzsche,
>     * Wittgenstein,
> 		 * hosting functions,
> 	 * and, go on longs walks,
> 	   * building [[thoughts/work|open-source project]],
> 	    * this [pan](https://example.com)
> ```
> ````

### TikZ support

to use in {{sidenotes[conjunction]: Currently, there is a few pgfplots bug upstream in node port, so to remove the graph from target rendering add `alt` as the URI svg (see examples below).}} with [obsidian-tikzjax](https://github.com/artisticat1/obsidian-tikzjax/)

```tikz
\usepackage{tikz-cd}
\begin{document}
\begin{tikzcd}
\pi^{-1}(U) \arrow[r, "\varphi"] \arrow[d, "\pi"'] & U \times F \arrow[ld, "proj_1"] \\
U &
\end{tikzcd}
\end{document}
```

> [!summary]- code
>
> ````
> ```tikz
> \usepackage{tikz-cd}
> \begin{document}
> \begin{tikzcd}
> \pi^{-1}(U) \arrow[r, "\varphi"] \arrow[d, "\pi"'] & U \times F \arrow[ld, "proj_1"] \\
> U &
> \end{tikzcd}
> \end{document}
> ```
>
> ```tikz alt="data:image/svg+xml..."
> ```
> ````

### pseudocode support

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

> [!summary]- code
>
> ````
> ```pseudo
> \begin{algorithm}
> \caption{LLM token sampling}
> \begin{algorithmic}
> \Function{sample}{$L$}
> \State $s \gets ()$
> \For{$i \gets 1, L$}
> \State $\alpha \gets \text{LM}(s, \theta)$
> \State Sample $s \sim \text{Categorical}(\alpha)$
> \If{$s = \text{EOS}$}
> \State \textbf{break}
> \EndIf
> \State $s \gets \text{append}(s, s)$
> \EndFor
> \State \Return $s$
> \EndFunction
> \end{algorithmic}
> \end{algorithm}
> ```
> ````

### collapsible header

inspired by https://dynalist.io/

### Gaussian-scaling TOC

inspired by https://press.stripe.com

> [!NOTE]
>
> Currently, this only works on Chrome and on Desktop.

### reader view

_press cmd/ctrl+b_

### sidepanel view

_press cmd/ctrl+click on any internal links_
