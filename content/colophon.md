---
id: colophon
lang: en
tags:
  - evergreen
  - design
description: nobby designer word for "the design of the current site"
socials:
  twitter: https://x.com/aarnphm_/status/1861550609834402129
date: "2024-12-10"
modified: 2025-08-10 16:08:47 GMT-04:00
title: colophon
---

**[[/tags/technical|technical]]** -- a very heavily modified version of [Quartz](https://quartz.jzhao.xyz/), hosted with Cloudflare Workers, with support for arXiV file pulling, [telescopic](https://github.com/jackyzha0/telescopic-text) [[/index|text]], a [note view](https://notes.aarnphm.xyz/notes?stackedNotes=bm90ZXM), more extensive grid layout support, reader view with custom [[/thoughts|folder]] and [[/tags|tags]] view, some QOL for [[/feed.xml|rss]] feed, sidenotes [^sidepanel], [rose-pine-dawn](https://rosepinetheme.com/). I also host all of the LFS on Cloudflare R2, and have a middleware as a proxy client.

[^sidepanel]: You can hold <kbd>alt+click</kbd> on any internal links to [popover](https://x.com/aarnphm_/status/1884954569341272345) a side panel 😃

**typography** -- [Parclo Serif](https://lettermatic.com/fonts/parclo-serif?plan=student), [ITC Garamond](https://www.typewolf.com/itc-garamond) and [`Berkeley Mono{:prolog}`](https://usgraphics.com/products/berkeley-mono)

**accessibility** -- follow ARIA spec. I tried to modify a few value in rose-pine to add a bit more contrast.

**components** -- I added support for rendering [[thoughts/Vector calculus#gradient|tikz graph]], [[thoughts/Transformers#Feynman-Kac|pseudocode]] support, [dynalist](https://dynalist.io)-inspired [[thoughts/mechanistic interpretability#inference|collapsible header]], a few customised [[posts/new#^ending|signature]], and some tiny [transformers plugins](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/plugins/transformers/aarnphm.ts).

**usage** -- Some daily drivers that has found [[uses|its way]] into my life. But I use [Obsidian](https://obsidian.md/) for note-taking, in conjunction with [neovim](https://neovim.io/) for all my [[thoughts/craft|work-related shenanigans]]

**plugins** -- Most of additional items I wrote for this website follow [unified](https://unifiedjs.com/) ecosystem and can be exported as a standalone plugin [^plugin].

[^plugin]: We are working on a few integrations separating out logics and improving general [Quartz ecosystem](https://github.com/quartz-community), stay tuned.
