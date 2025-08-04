---
id: cf
tags:
  - seed
  - technical
description: from Vercel. And why you should too.
date: "2025-08-04"
draft: true
modified: 2025-08-04 13:52:30 GMT-04:00
title: Moving to Cloudflare
---

I finally found some pockets of time over the last two weekends slowly migrating all of [[/|this]] site infrastructure to Cloudflare. I would want to say so far, the experience has been nothing but great.

Previously, this site were being run with a mixed between Vercel (functions) and Cloudflare (Domains and DNS), which in turn has caused me a lot of friction for prototyping/adding new features I want to build.
This post will demonstrates pros and cons of both providers, and will explain the decision behind going with Cloudflare versus other alternatives.

## genealogy of the sites

This website is entirely built with a customised version of [Quartz](https://quartz.jzhao.xyz/) (you can find more details [[thoughts/craft#^quartz|here]]):

- [812ac42](https://github.com/aarnphm/aarnphm.github.io/commit/812ac42097844bd0470b1b7fbb7ac6ed66e772e6): I first open-source this implementations, requested by a few friends
  - Hosted as GitHub Pages, with DNS managed with [gen.xyz](https://gen.xyz/)
  - Vercel was being used for
