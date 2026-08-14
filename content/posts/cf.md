---
date: '2025-08-04'
description: why I moved this Quartz site to Cloudflare Workers.
id: cf
modified: 2026-06-06 00:12:35 GMT-04:00
socials:
  hackernews: https://news.ycombinator.com/item?id=44791222
  twitter: https://x.com/aarnphm/status/1952474293654413636
tags:
  - technical
title: Moving to Cloudflare
---

Over two weekends, I moved this site from a split Vercel and Cloudflare setup to Cloudflare Workers. This is an account of what changed for one custom Quartz site.

The decision came from the shape of this project. Cloudflare already managed the domain, DNS, and the R2 bucket that held large files. Moving the site and its server code to Workers put the parts I operated most often in one place.

## how the site reached that point

This site uses a customized version of [Quartz](https://quartz.jzhao.xyz/). Its deployment changed in four stages.

1. In [812ac42](https://github.com/aarnphm/aarnphm.github.io/commit/812ac42097844bd0470b1b7fbb7ac6ed66e772e6), I first open-sourced this implementation at the request of a few friends. GitHub Pages served the static files. Gen.xyz managed DNS. Vercel Functions handled the [[/curius|Curius feed]], arXiv popovers, and code rendering.
2. In [ce7bcee](https://github.com/aarnphm/aarnphm.github.io/commit/ce7bcee77f7e2e6e4b688c831201fadc9cd2d18b), I started prototyping the site around [Andy Matuschak's notes](https://notes.andymatuschak.org/About_these_notes). I added Vercel Middleware for routing between `notes.aarnphm.xyz` and `aarnphm.xyz`. That left me operating two deployments of the same site.
3. I later exceeded the GitHub LFS Free allowance while hosting PDFs. I moved those assets to Cloudflare R2, then moved DNS and the static deployment to Cloudflare Pages. At that point, Cloudflare already owned most of the path from the domain to the files.
4. In [6aadff3](https://github.com/aarnphm/aarnphm.github.io/commit/6aadff359a5e8ccb7879e6e8a69e79c8ba1542cd), I moved the site from Pages to Workers while upgrading Quartz. Cloudflare's [migration guide](https://developers.cloudflare.com/workers/static-assets/migration-guides/migrate-from-pages/) covered the static asset changes. `wrangler types` generated TypeScript types for bindings, and `wrangler dev` gave me the local Worker runtime I needed.

None of these stages was a controlled provider comparison. Each solved the next problem in a site that had already accumulated parts from both platforms.

## why Workers fit this site

The main gain was operational. DNS, static assets, R2 files, and Worker code now shared one provider and one deployment path. A feature that needed a server route could live beside the site instead of crossing from GitHub Pages to a Vercel Function.

The second gain was configuration I could keep in the repository. Quartz had no Vercel preset for this deployment, so I would have maintained its build and routing configuration myself. Vercel supports many frameworks and also exposes framework-neutral project configuration. The issue was the work required for this Quartz fork, rather than a general limit of the platform.

Cloudflare documents standard unmetered DDoS protection across its plans.[^ddos] That was useful for a public site. It does not establish perfect security or `100%` uptime, and I did not collect enough production data to compare either provider on those measures.

R2 mattered because the large objects were already there. Moving them again would have added work without solving a problem I had. The storage decision made the later hosting decision easier.

## what this decision does not show

This migration does not show that Cloudflare is faster than Vercel, or that one provider is safer. I did not run latency tests, keep a fixed observation window, or compare equivalent paid plans. The old scorecard in this note pretended those measurements existed, so I removed it.

Vercel's framework presets, dashboard, and deployment workflow can be a better fit for a project built around one of those presets. Cloudflare fit this site because the domain, object storage, and edge code were already there. The ratio of migration work to removed operational work was favorable for this repository.

That is the whole recommendation. Map the pieces you already operate, identify the boundaries that cause repeated work, and choose the deployment that removes those boundaries. For this Quartz site in August 2025, that deployment was Cloudflare Workers.

[^ddos]: Cloudflare lists standard unmetered DDoS protection as available across its plans in the [DDoS Protection documentation](https://developers.cloudflare.com/ddos-protection/).
