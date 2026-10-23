---
date: '2024-12-10'
description: nobby designer word for "the design of the current site"
id: colophon
lang: en
modified: 2026-10-23 11:51:18 GMT-04:00
socials:
  twitter: https://x.com/aarnphm/status/1861550609834402129
tags:
  - evergreen
  - design
title: colophon
---

**#technical** -- a [heavily modified fork](https://en.wikipedia.org/wiki/Ship_of_Theseus) of [Quartz](https://quartz.jzhao.xyz/), hosted with Cloudflare Workers, with support for arXiv file pulling, [telescopic](https://github.com/jackyzha0/telescopic-text) [[/index|text]], [[/|custom homepage]], a [note view](https://notes.aarnphm.xyz/notes?stackedNotes=bm90ZXM), [[/singularity|JSON Canvas support]], [[cinematheque.base|Obsidian Bases support]] hierarchical grid layout, comment room support, reader view with custom [[/thoughts|folder]] and [[/tags|tags]] view, nested [[/index.xml|rss]] [[/posts/index.xml|feed]], [[queries|Jupyter]] [[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/10 Generalized Parsing/00 Generalized Parsing|Notebooks]] support for client-side interaction, [[thoughts/Sets.fc|flashcards]], [[triathlon|a bunch of triathlon/tracking/analytics/health stuff]], {{sidenotes[tufte]: This is a tiny sidenote, supporting dynamic view size. Very much along the lines of [Tufte CSS](https://edwardtufte.github.io/tufte-css/)}} sidenotes, sidepanels [^sidepanel], custom popover, modified [Flexoki](https://stephango.com/flexoki). I also host all of the LFS on Cloudflare R2, and have a middleware as a proxy client. There are also [[courses/index|course workspaces]], a PDF reader with marks and notes, a saved-link reading queue, browser semantic search, pronunciation tools, and a little mascot wandering around the page edges. The [[/index|homepage]] has local weather and a shared walking trail. Also custom [[/404|404 page]] because why not. Oh, there's also a [[personal canon|Tractatus-specific]] layout, [[thoughts/LLMs|technical docs]] layout, [[fr|french-specific]] layout, and {{sidenotes[more.]: Thanks to Opus and Astra for prototyping a lot of the mocks/ideas I have for these.}}

[^sidepanel]: <kbd>Alt</kbd>, <kbd>Cmd</kbd>, or <kbd>Ctrl</kbd> + click an internal link to [popover](https://x.com/aarnphm/status/1884954569341272345) a side panel 😃

**typography** -- [PP Neue Montreal](https://pangrampangram.com/products/neue-montreal), [<span style="font-family: 'Space Grotesk'">Space Grotesk</span>](https://fonts.google.com/specimen/Space+Grotesk?preview.script=Latn), [<span style="font-family: 'Parclo Serif'">Parclo Serif</span>](https://lettermatic.com/fonts/parclo-serif?plan=student), [<span style="font-family: 'ITCGaramondStdLtCond'">ITC Garamond</span>](https://www.typewolf.com/itc-garamond) and [`berkeley mono{:text}`](https://usgraphics.com/products/berkeley-mono)

**accessibility** -- native buttons, dialogs and popovers, keyboard navigation, visible focus, and ARIA labels for the controls that need them. The mascot and 404 landscape respect the reduced-motion preference. Press <span style="text-transform: uppercase"><kbd>D</kbd></span> outside text fields to cycle through daylight, system, dark, and light themes. <kbd>Cmd/Ctrl+'</kbd> opens the shortcuts helper.

**components** -- I added support for rendering [[thoughts/Vector calculus#gradient|tikz graph]], [[thoughts/Transformers#Feynman-Kac|pseudocode]] support, [dynalist](https://dynalist.io)-inspired [[thoughts/mechanistic interpretability#inference|collapsible header]], a few customised [[posts/new#^ending|signature]], tiny [transformers plugins](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/plugins/transformers/aarnphm.ts), micromarks extensions for [wikilinks](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-wikilinks), [sidenotes](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-sidenotes), [[posts/25/n-bday|protected notes]], [distributed comments](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/components/scripts/collaborative-comments.inline.ts), and additional supports of [`renderPage.tsx`](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/components/renderPage.tsx#L1487)

**usage** -- Some daily drivers that has found [[uses|its way]] into my life. But I use [Obsidian](https://obsidian.md/) for note-taking and Apple Notes while I'm on the go, in conjunction with [neovim](https://neovim.io/) for all my [[thoughts/craft|work-related shenanigans]].

**license** -- All my notes are licensed under CC BY-NC-SA [^license], whilst the code is under [Apache 2.0](http://www.apache.org/licenses/). Original authors are referenced and linked, and their work retains its own licenses. The note footer renders the Creative Commons marks.

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

### courses and lecture workspaces

I want to explore a [spaced repetition](https://notes.andymatuschak.org/zRBUabiBg7c2o9i2vqamMFW) interface for some of the [[courses/index|OCW course]]. This include the lecture notes, alongside problem sets, and decks.

There are some {{sidenotes[video lectures]: Video lecture pages also pair with captions for better accessibility. For folks that enjoy latex, those are also imported to the linked study notes.}} that paired with backlinks that allows better graph navigation. I explicitly did not enable graph given that it creates too much cognitive overload (or at least for me whenever I look at a graph).

### PDF reader

PDFs in the garden open in a reader with a document library, outline, page navigation, zoom, text selection, marks, and Markdown notes with math. It remembers reading position and can link directly to a page or mark. Course resource pages use the same reader, so their marks stay attached when opening the source beside a lecture. The original file remains available through the raw-file link.

### feed

A [reading queue](/arena/feed) combines links from [[are.na|are.na]] and [[curius|curius]], allowing me to go through all of the links backlog I have. There are also some custom parsing courtesy of [defuddle](https://defuddle.md/)

### search

Search expands based on Quartz' default lexical search, with semantic enabled. I used [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) through [Transformers.js](https://github.com/huggingface/transformers.js) and precompute the whole garden's embeddings. Feel free to take a look at the RAG pipeline, pretty standard nowadays.

### training

I use both [Strava](https://www.strava.com/athletes/123158310) and TrainingPeaks, but [[triathlon|triathlon page]] brings together Strava, Garmin, Wahoo, Oura, Apple Health, and CORE records with logs. Of course it has to have our own styling and preferences. There are a bunch of stuff to explore in terms of data analytics and exploration, with also activities that contains wind data, cycling CdA.

There are also a bunch of maths and sports physiology in there for maximize human-performance 😸 (i'm just a sucker for data, but it is all fun and game)

### mascots

Very rudimentary hand-drawn mascot appears in the favicon, social cards, and page artwork. Its animated version perches on headings, code blocks, figures, and other page edges, then rides the scroll rail while the page moves. If you wish to turn the pets off, then press <kbd>Ctrl+p</kbd> (but Blob wouldn't like that)

### kit de presse

Press <kbd>gs</kbd> to open the share-card editor. Think Strava sharing but customization per notes socials. Mostly for me lool but feel free to use it.

### pronunciation and source text

French notes and decks have pronunciation controls with playback-speed settings. <kbd>gv</kbd> opens those settings. Audio uses cached clips when available and an optional browser speech model for synthesis. We use a speech models to support French and English.

The metadata row can copy a note's Markdown source to the clipboard and reports whether the copy succeeded. Appending `.md` to a note URL also serves the source. Notebook cells, mathematical diagrams, and slide views have their own interactive rendering, including numbered slide pages and an overview grid.

### vim bindings

- `f` to navigate the sites on keyboards.
- `gh` for navigating ToC (supports j/k/Ctrl-n/Ctrl-p scrolling)
- `Cmd/Ctrl+'` for shortcuts helpers
- `Cmd/Ctrl+P` for the command palette
- `gs` for the share-card editor, `gv` for pronunciation settings

### parser

some remark parsers for wikilinks, callouts, that supports general OFM compatibility

see [ofm-wikilinks](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-wikilinks/) and [ofm-sidenotes](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-sidenotes) for more information.

Supports [[cinematheque|Bases]] [API](https://help.obsidian.md/bases) and [[singularity.canvas|JSON]] [Canvas](https://help.obsidian.md/plugins/canvas)

Bases render tables, lists, image galleries, maps, and [Kanban columns](https://obsidian.md/help/bases/views/kanban). The [[cinematheque/analysis|film analysis view]] uses the first `studio` entry as the main production company and places each film in one column. This convention is saved as a Bases formula; notes keep their full list of production companies. Obsidian can show those lists as combined group labels. Search filters cards and updates column counts. The `.base` file controls filters, sorting, visible properties, image fit, and column width; note creation and property edits happen in Obsidian.

### [telescopic-text](https://github.com/jackyzha0/telescopic-text)

Support a small subsets of the features, with wikilinks parsing

```telescopic id="reading"
* I'm
* reading
  * reading a lot of
    * Nietzsche,
    * Wittgenstein,
		 * hosting functions,
	 * and, go on long walks,
	   * building [[thoughts/craft|open-source projects]],
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
> 	 * and, go on long walks,
> 	   * building [[thoughts/craft|open-source projects]],
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
\State Sample $w \sim \text{Categorical}(\alpha)$
\If{$w = \text{EOS}$}
\State \textbf{break}
\EndIf
\State $s \gets \operatorname{append}(s, w)$
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
> \State Sample $w \sim \text{Categorical}(\alpha)$
> \If{$w = \text{EOS}$}
> \State \textbf{break}
> \EndIf
> \State $s \gets \operatorname{append}(s, w)$
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

> [!note] browser compatibility
>
> Currently, this only works on Chrome and on Desktop.

### reader view

_press cmd/ctrl+shift+b_

### sidepanel view

_press alt/cmd/ctrl+click on any internal links_

### comments

_inspired by figma multiplayer comments_

I added the ability to leave comments on pages, simply by holding <kbd>Ctrl</kbd> or <kbd>Cmd</kbd> while dragging to add a comment.

### flashcards

credit to [eudoxia0/hashcards](https://github.com/eudoxia0/hashcards). Check out [[thoughts/Sets.fc|an example set of flashcards on sets theory]]. Decks support question/answer and cloze cards, math, and stable card identities. The source note can sit beside the active card, and a course review combines due cards from its decks. Signed-in owner reviews use [FSRS](https://github.com/open-spaced-repetition/ts-fsrs), with saved review history and undo. Grading a cloze card defers its siblings until the next session.
