---
date: "2024-12-10"
description: nobby designer word for "the design of the current site"
id: colophon
lang: en
modified: 2025-11-07 21:32:38 GMT-05:00
socials:
  twitter: https://x.com/aarnphm_/status/1861550609834402129
tags:
  - evergreen
  - design
title: colophon
---

**[[/tags/technical|technical]]** -- a [heavily modified fork](https://en.wikipedia.org/wiki/Ship_of_Theseus) of [Quartz](https://quartz.jzhao.xyz/), hosted with Cloudflare Workers, with support for arXiV file pulling, [telescopic](https://github.com/jackyzha0/telescopic-text) [[/index|text]], a [note view](https://notes.aarnphm.xyz/notes?stackedNotes=bm90ZXM), [[singularity.canvas|JSONCanvas supports]], [[cinematheque.base|Obsidian Bases support]] hierarchical grid layout, reader view with custom [[/thoughts|folder]] and [[/tags|tags]] view, nested [[/index.xml|rss]] [[/posts/index.xml|feed]], sidenotes, sidepanels [^sidepanel], modified [Flexoki](https://stephango.com/flexoki). I also host all of the LFS on Cloudflare R2, and have a middleware as a proxy client.

[^sidepanel]: You can hold <kbd>alt+click</kbd> on any internal links to [popover](https://x.com/aarnphm_/status/1884954569341272345) a side panel 😃

**typography** -- [PP Neue Montreal](https://pangrampangram.com/products/neue-montreal), [<span style="font-family: 'Parclo Serif'">Parclo Serif</span>](https://lettermatic.com/fonts/parclo-serif?plan=student), [<span style="font-family: 'ITCGaramondStdLtCond'">ITC Garamond</span>](https://www.typewolf.com/itc-garamond) and [`berkeley mono{:text}`](https://usgraphics.com/products/berkeley-mono)

**accessibility** -- follow ARIA spec. I tried to modify a few value in rose-pine to add a bit more contrast. Press <kbd>D</kbd> anywhere to toggle between light and dark mode.

**components** -- I added support for rendering [[thoughts/Vector calculus#gradient|tikz graph]], [[thoughts/Transformers#Feynman-Kac|pseudocode]] support, [dynalist](https://dynalist.io)-inspired [[thoughts/mechanistic interpretability#inference|collapsible header]], a few customised [[posts/new#^ending|signature]], tiny [transformers plugins](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/plugins/transformers/aarnphm.ts), micromarks extensions for [wikilinks](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-wikilinks), [sidenotes](https://github.com/aarnphm/aarnphm.github.io/tree/main/quartz/extensions/micromark-extension-ofm-sidenotes), [[posts/25/n-bday|protected notes]], and additional supports of [`renderPage.tsx`](https://github.com/aarnphm/aarnphm.github.io/blob/f2006d75ca76263ffe880b43d7c8bac27aefc6ac/quartz/components/renderPage.tsx#L1487)

**usage** -- Some daily drivers that has found [[uses|its way]] into my life. But I use [Obsidian](https://obsidian.md/) for note-taking and Apple Notes while I'm on the go, in conjunction with [neovim](https://neovim.io/) for all my [[thoughts/craft|work-related shenanigans]].

**plugins** -- Most of additional items I wrote for this website follow [unified](https://unifiedjs.com/) ecosystem and can be exported as a standalone plugin [^plugin].

[^plugin]: We are working on a few integrations separating out logics and improving general [Quartz ecosystem](https://github.com/quartz-community), stay tuned.
