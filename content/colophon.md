---
id: colophon
tags:
  - evergreen
  - design
date: "2024-12-10"
description: nobby designer word for 'the design of the current site'
lang: en
modified: 2025-01-04 10:44:19 GMT-05:00
socials:
  twitter: https://x.com/aarnphm_/status/1861550609834402129
title: colophon
---
**[[/tags/technical|technical]]** -- a very heavily modified version of [Quartz](https://quartz.jzhao.xyz/), hosted on GitHub pages, Vercel serverless functions with support for arxiv file pulling, [telescopic](https://github.com/jackyzha0/telescopic-text) [[/index|text]], a [note view](https://notes.aarnphm.xyz/notes?stackedNotes=bm90ZXM), more extensive grid layout support, reader view with custom [[/thoughts|folder]] and [[/tags|tags]] view, some QOL for [[/feed.xml|rss]] feed, sidenotes, [rose-pine-dawn](https://rosepinetheme.com/).

**typography** -- EB Garamond, [ITC Garamond](https://www.typewolf.com/itc-garamond) and `Berkeley Mono` from [US Graphics Company](https://usgraphics.com/products/berkeley-mono)

**accessibility** -- follow ARIA spec. I tried to modify a few value in rose-pine to add a bit more contrast.

**components** -- I added support for rendering [[thoughts/Vector calculus#gradient|tikz graph]], [[thoughts/Transformers#Feynman-Kac|pseudocode]] support, [dynalist](https://dynalist.io)-inspired [[thoughts/mechanistic interpretability#inference|collapsible header]], a few customised [[posts/new|signature]], and some tiny [transformers plugins](https://github.com/aarnphm/aarnphm.github.io/blob/main/quartz/plugins/transformers/aarnphm.ts). Most of it follow [unified](https://unifiedjs.com/) ecosystem and _can potentially_ be exported as a standalone plugin [^plugin].

[^plugin]: We are working on a few integrations separating out logics and improving general [Quartz ecosystem](https://github.com/quartz-community), stay tuned.