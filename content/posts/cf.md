---
id: cf
tags:
  - seed
  - technical
description: And maybe you should, or shouldn't.
date: "2025-08-04"
modified: 2025-08-04 16:44:34 GMT-04:00
title: Moving to Cloudflare
---

> Preface: I just want to say that I absolutely adore the Vercel team and love everything they have been working on so far.
> But Cloudflare is based (just see their recent [blog catalog](https://blog.cloudflare.com/), and the main thing that sold me on Cloudflare was [Quicksilver](https://blog.cloudflare.com/introducing-quicksilver-configuration-distribution-at-internet-scale/), but enough glaze 😅)
>
> Also I'm not a frontend engineer by trade, so I'm not going into too much details about the benefits between these providers for specific frameworks. So I want you,
> the reader to treat this from a perspective of a consumer, who enjoy writing somewhat fruitful static files on his weekend. 🙂

I finally found some pockets of time over the last two weekends slowly migrating all of [[/|this]] site infrastructure to Cloudflare. I would want to say so far, the experience has been nothing but great.

Previously, this site were being run with a mixed between Vercel (functions) and Cloudflare (Domains and DNS), which in turn has caused me a lot of friction for prototyping/adding new features I want to build.
This post will demonstrate what I enjoy about using both and what both providers can improve, and will explain the decision behind going with Cloudflare versus other alternatives.

## "genealogy" of the sites

This website is entirely built with a customised version of [Quartz](https://quartz.jzhao.xyz/) (you can find more details [[thoughts/craft#^quartz|here]]):

- [812ac42](https://github.com/aarnphm/aarnphm.github.io/commit/812ac42097844bd0470b1b7fbb7ac6ed66e772e6): I first open-source this implementations, requested by a few friends
  - Hosted with GitHub Pages, DNS managed by [gen.xyz](https://gen.xyz/)
  - Vercel Functions for [[/curius|curius feed]], arXiV popover, and code rendering
    - This is functional, but rather pretty slow, given that there are a few round-trip needed to be done
      between GitHub Pages and Vercel
- [ce7bcee](https://github.com/aarnphm/aarnphm.github.io/commit/ce7bcee77f7e2e6e4b688c831201fadc9cd2d18b): The first initial version of prototyping [Andy Matuschak's notes](https://notes.andymatuschak.org/About_these_notes) with Quartz
  - https://github.com/jackyzha0/quartz/issues/128 requests for a view with default Quartz
  - Added Vercel Middleware to redirect DNS
    - Given that [notes.aarnphm.xyz](https://notes.aarnphm.xyz) and [aarnphm.xyz](https://aarnphm.xyz) points to the same source,
      I end up having to setup build process on Vercel similar to GitHub Pages.
    - Which ends up having two copies of the exact same sites over two different cloud buckets!
      > Retrospectively, I could have moved all over to Vercel at this point, but I was trying to skim through the free offering from both providers.
- At some point along the way, I had exceed the limits for GitHub LFS Free tier.
  - I was tempted to just add more storages, but it is pretty expensive (ik $10 is 2 cups of coffee, but if you can do it for free, why not)
  - Found https://github.com/milkey-mouse/git-lfs-s3-proxy and https://github.com/milkey-mouse/git-lfs-client-worker contains setup for your own Git LFS with Cloudflare R2
    - Cloudflare is relatively generous with their quotas and storages for [free tier](https://www.cloudflare.com/en-gb/lp/pg-r2-comparison-2)
    - I did a quick comparison with [Vercel Blob](https://vercel.com/docs/vercel-blob/usage-and-pricing). However, the pricing seems to be a bit more on the expensive side.
    - For the tasks that I want to achieve (which is current hosting PDFs), Cloudflare is a better choice here
- Moved all DNS, domain to Cloudflare, and migrated to Cloudflare [Pages](https://pages.cloudflare.com/)
  - I was also considering to use Vercel DNS, but given that I have already been on R2 at this point, it seems prudent to use Cloudflare for the sake of simplicity
    - Lee Rob from Vercel reached out for [feedback](https://x.com/aarnphm_/status/1882982597908955548?s=46&t=K6_tWk-1vuN4JVbmPrSC7A), so I'm indeed very much bullish on Vercel on their care/dedications to their customers.
  - Their free tier encapsulates pretty much all features I would ever needed, for now.
  - The only reasons I haven't yet migrated to Cloudflare fully is because `wrangler` 3 was rather hard to use with TypeScript, and Vercel supports for TypeScript is superior at this point.
- [6aadff3](https://github.com/aarnphm/aarnphm.github.io/commit/6aadff359a5e8ccb7879e6e8a69e79c8ba1542cd): Migrated to Cloudflare Workers
  - Upgrading Quartz to the latest 4.5.x requires a major changes into how it handles incremental builds,
    so I found that this was a good time to switch/migrate fully to Cloudflare Workers.
    - They have now a lot better TS supports, with `wrangler types` and `wrangler dev`, which simplifies the setup a ton with Quartz.
    - Migrating from Pages and Workers are very refreshing, as their [docs](https://developers.cloudflare.com/workers/static-assets/migration-guides/migrate-from-pages/) details all that you need to look out for.
      - I haven't seen a lot of good documentations, other than Cloudflare, Amazon, and Vercel.

## pros and "not-pros-yet".

You can pretty much [Google](https://www.google.com/search?q=cloudflare+vs+vercel&oq=cloudflare+vs+vercel&gs_lcrp=EgZjaHJvbWUqBwgAEAAYgAQyBwgAEAAYgAQyDAgBEAAYFBiHAhiABDIMCAIQABgUGIcCGIAEMgcIAxAAGIAEMgcIBBAAGIAEMggIBRAAGBYYHjIICAYQABgWGB4yCAgHEAAYFhgeMggICBAAGBYYHjIICAkQABgWGB7SAQgyNjcxajBqNKgCALACAQ&sourceid=chrome&ie=UTF-8) comparison
all day long to see whether it is sensible to migrate your whole infrastructure between these two providers.

At the end of the day, the following are what _I care about_ when building a functional website:

- Performance: I want people who cares about optimizing for miliseconds
- Security: reduce crawlers, fast, 100% uptime, so that I don't have to worry about them.
- Intuitive UX: follows zero-config philosophy (I think [Ghostty](https://ghostty.org/docs/config) is a prime example of this). I don't have to configure a tons of stuff and can simply toggle items on the dashboard
- Friendly developer tooling: I want to write code, so tooling must be good.
- Pricing: free offering, otherwise pay-per-usage billing. Lenient on storage is a huge plus.

| Criteria                   | Cloudflare   | Vercel                |
| -------------------------- | ------------ | --------------------- |
| Performance                | ✅           | 🚧 [^vercel-perf]     |
| Security                   | ✅           | 🚧 [^vercel-security] |
| Intuitive UX               | ✅           | ✅                    |
| Friendly developer tooling | 🚧 [^cf-dev] | ✅                    |
| Pricing                    | ✅           | 🚧                    |

[^vercel-perf]:
    There is a case to be made that if I migrated everything on Vercel, it would be a lot smoother. However, this is not the case given that Vercel is designed to integrated well with Next, and supports for
    platform-agnostic framework (such as Quartz) are rather limited. I prefer not to spend that much time on making it work with Vercel. Also the pricing is also a big hurdles for me.

[^vercel-security]:
    There has always been a huge hurdles around Vercel's security on Twitter and the internet in general. The team works very well to circumvent these problems, but as a user it didn't give me a lot of conviction
    when comparing to other alternatives (such as Cloudflare, AWS, GCP)

[^cf-dev]:
    `wrangler` has been historically harder to use comparing to vercel setup. However, with the recent v4, it has become a lot more simpler and easier to use.
    But credits where credits are due, Vercel triumphs in terms of users experience here.

    Their KV as well as D1 databases are a safe bet for me to prototype with a few more features I want to build for the sites. Their AI gateway is also a plus.

From this comparison, Cloudflare did seem to come out on top, for my use-case.
However, I don't think this would be a signal for you to move to Cloudflare completely.

Vercel is good for you if:

- developer UX and building sites with supported frameworks. Next, Svelte, Solid are all amazing to work with when using `vercel` (from my experience on other projects)
- seamless user experience with observability and metrics from their dashboard.
- suite of products and resources for you to build your next 1M ARR business you're willing to 😃

Cloudflare is good for you if:

- If you care about security and don't have to worry too much about the minor details
- static sites
- free plan

> [!NOTE]
>
> They both have AI-related products, but I'm not going to compare it here given that I don't use them extensively.

Finally, I do have a lot of conviction in Cloudflare, through their [open source](https://github.com/cloudflare)and their leadership.
However, it goes without saying that Vercel is still considered as a startup, comparing to Cloudflare, a publicly traded company.
The mere sizes comparison between the two shows how far Vercel has come and essentially brings competition to these giants, and I can't wait for one day, for Vercel to win.
