---
date: '2024-12-10'
description: nobby designer word for "the design of the current site"
id: colophon
lang: en
modified: 2026-10-11 03:03:56 GMT-04:00
socials:
  twitter: https://x.com/aarnphm/status/1861550609834402129
tags:
  - evergreen
  - design
title: colophon
---

**#technical** -- a [heavily modified fork](https://en.wikipedia.org/wiki/Ship_of_Theseus) of [Quartz](https://quartz.jzhao.xyz/), hosted with Cloudflare Workers, with support for arXiv file pulling, [telescopic](https://github.com/jackyzha0/telescopic-text) [[/index|text]], a [note view](https://notes.aarnphm.xyz/notes?stackedNotes=bm90ZXM), [[/singularity|JSON Canvas support]], [[cinematheque.base|Obsidian Bases support]] hierarchical grid layout, comment room support, reader view with custom [[/thoughts|folder]] and [[/tags|tags]] view, nested [[/index.xml|rss]] [[/posts/index.xml|feed]], [[queries|Jupyter]] [[thoughts/university/twenty-five-twenty-six/sfwr-4tb3/10 Generalized Parsing/00 Generalized Parsing|Notebooks]] support for client-side interaction, [[thoughts/Sets.fc|flashcards]], [[triathlon|a bunch of triathlon/tracking/analytics/health stuff]], {{sidenotes[tufte]: This is a tiny sidenote, supporting dynamic view size. Very much along the lines of [Tufte CSS](https://edwardtufte.github.io/tufte-css/)}} sidenotes, sidepanels [^sidepanel], modified [Flexoki](https://stephango.com/flexoki). I also host all of the LFS on Cloudflare R2, and have a middleware as a proxy client.

There are also [[courses/index|course workspaces]], a PDF reader with marks and notes, a saved-link reading queue, browser semantic search, pronunciation tools, and a little mascot wandering around the page edges. The [[/index|homepage]] has local weather and a shared walking trail. The [404 page](/404.html) has acquired a whole landscape. More on those below.

[^sidepanel]: <kbd>Alt</kbd>, <kbd>Cmd</kbd>, or <kbd>Ctrl</kbd> + click an internal link to [popover](https://x.com/aarnphm/status/1884954569341272345) a side panel 😃

**typography** -- [PP Neue Montreal](https://pangrampangram.com/products/neue-montreal), [<span style="font-family: 'Space Grotesk'">Space Grotesk</span>](https://fonts.google.com/specimen/Space+Grotesk?preview.script=Latn), [<span style="font-family: 'Parclo Serif'">Parclo Serif</span>](https://lettermatic.com/fonts/parclo-serif?plan=student), [<span style="font-family: 'ITCGaramondStdLtCond'">ITC Garamond</span>](https://www.typewolf.com/itc-garamond) and [`berkeley mono{:text}`](https://usgraphics.com/products/berkeley-mono)

**accessibility** -- native buttons, dialogs and popovers, keyboard navigation, visible focus, and ARIA labels for the controls that need them. The mascot and 404 landscape respect the reduced-motion preference. Press <span style="text-transform: uppercase"><kbd>D</kbd></span> outside text fields to cycle through light, dark, and system themes. <kbd>Cmd/Ctrl+'</kbd> opens the shortcuts helper.

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

The [[courses/index|course library]] brings together MIT OpenCourseWare materials, my lecture notes, problem sets, and decks. Course pages have resource lists, study progress, and previous/next navigation. Video lecture pages pair the player with captions you can use to jump to a passage. Imported LaTeX sources become linked study notes while keeping their source attribution.

On an authored lecture such as [[courses/18.905-fall-2016/lectures/01|singular simplices and chains]], a workspace sits beside the reading column on wide screens and opens in a dialog on smaller ones. It has the source PDF, a Markdown scratchpad, and notes on selected passages. You can mark, question, disagree, or write a note, then export the scratchpad and notes as Markdown with source citations. These lecture drafts live in the browser. Deleted notes can be restored after a reload.

### PDF reader

PDFs in the garden open in a reader with a document library, outline, page navigation, zoom, text selection, marks, and Markdown notes with math. It remembers reading position and can link directly to a page or mark. Course resource pages use the same reader, so their marks stay attached when opening the source beside a lecture. The original file remains available through the raw-file link.

### saved-link reader

The [reading queue](/arena/feed) combines links from [[are.na|my Arena]] and [[curius|Curius]], merges duplicate URLs, and puts unread Later links first. I can mark an article read, skip to the next one, or save a note on a quoted passage. Article copies and the versions cited by notes live in R2; read marks and saved notes belong to my signed-in account. arXiv links open the PDF reader, and GitHub source files retain their code formatting. The original source stays one click away.

### search

Search has lexical and semantic modes. Lexical search includes titles, text, tags, and path segments. Semantic search uses EmbeddingGemma through [Transformers.js](https://github.com/huggingface/transformers.js) to encode the query in a browser worker and compare it with the garden's precomputed passage vectors. Selecting that mode downloads the model and index, then the browser caches the model. It needs WebGPU; lexical search remains available on other browsers.

### training and activity analysis

[[triathlon|The training workspace]] brings together Strava, Garmin, Wahoo, Oura, Apple Health, and CORE records with my training logs. There are activity and race calendars, route maps, lap and interval views, best efforts, power curves, recovery and sleep views, gear mileage, maintenance records, and equipment notes. A password-protected TrainingPeaks calendar keeps planned and completed workouts together and links matched sessions to their activity pages.

The activity workspace links the route cursor to telemetry and charts. It can compare recorded courses, show shifts and cycling dynamics, and plot wind, weather, and modelled Lake Ontario surface currents. Archived [myWindsock](https://mywindsock.com/) reports add native wind, resistance, Feels Like elevation, and cycling CdA graphs, including test averages and ranges when captured. Empty or invalid graphs stay out of the viewer. Provider values retain their source labels; locally computed physiology, stamina, power, and pace forecasts are labelled as estimates. Commutes remain in the activity log and are excluded from the garden's derived pace and load analytics.

### homepage weather and shared trail

The sun or moon on the [[/index|homepage]] opens weather for the visitor's approximate area, with a [Mapbox](https://www.mapbox.com/) map, [Open-Meteo](https://open-meteo.com/) conditions, wind, and the next sunrise or sunset. The Worker rounds the location supplied by Cloudflare before requesting the forecast.

The trail button opens three paths around the homepage text, through trees to a campsite. [PlayHTML](https://playhtml.fun/) synchronizes a stone for each browser. You can drag yours or move it with the keyboard, and other visitors' stones remain after they leave. The shared state consists of path choices and positions. Weather and location stay separate from it. There is also a campfire and someone roasting a marshmallow.

### mascot and 404 landscape

The hand-drawn mascot appears in the favicon, social cards, and page artwork. Its animated version perches on headings, code blocks, figures, and other page edges, then rides the scroll rail while the page moves. The command palette can turn pets off.

The [404 page](/404.html) has a pixel landscape with mountains, water, a clock gate, a cat, a herd returning to its barn, riders, skiers, an angler, and passing planes. Its light, foliage, snow, and weather follow the visitor's local time, season, and available conditions.

### kit de presse

Press <kbd>gs</kbd> to open the share-card editor. It makes note cards, quotation cards, and links to the page or a heading. Select a passage to use it as a quotation. Cards have story, portrait, square, and landscape formats, four themes, and optional artwork and metadata. You can download a PNG, copy the image, or use native sharing where the browser supports it. Math in selected passages keeps its LaTeX source for rendering on the card.

### pronunciation and source text

French notes and decks have pronunciation controls with playback-speed settings. <kbd>gv</kbd> opens those settings. Audio uses cached clips when available and an optional browser speech model for synthesis. Text stays on the device during synthesis; the model details identify the voice and download size.

The metadata row can copy a note's Markdown source to the clipboard and reports whether the copy succeeded. Appending `.md` to a note URL also serves the source. Notebook cells, mathematical diagrams, and slide views have their own interactive rendering, including numbered slide pages and an overview grid.

### authoring and machine access

The vault has local Obsidian plugins for sidenotes, marked annotations, heading navigation, metadata, and running Quartz. The published page also exposes WebMCP tools to search the garden, read a page, and follow a result through the site router. The Cloudflare Worker serves an MCP endpoint for connected clients, alongside the Markdown sources and generated LLM text feeds.

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
