---
id: Summer fellows
tags:
  - application
date: "2025-03-28"
modified: 2025-04-11 18:49:36 GMT-04:00
noindex: true
title: YC 25 Summer fellows
---

> [!question]
>
> List any competitions/awards you have won, or papers you’ve published.

I won Hack The North 2021 as a [Finalist](https://devpost.com/software/twogether?_gl=1*1fwifmz*_gcl_au*NjM2MzA3NDQ4LjE3NDMyMTE0MTA.*_ga*MzEwMTc5NjEzLjE3NDMyMTE0MTE.*_ga_0YHJK3Y10M*MTc0MzIxMTQxMC4xLjEuMTc0MzIxMTQyMS4wLjAuMA), where I trained and fine-tuned a Gaussian Mixture Model to find the safest path between different locations, trained on past assault data provided by Toronto Police Department from 2015 onwards.
Integrated with Google Maps API to show heat maps of avoided zone.

> [!question]
>
> Please tell us about a time you most successfully hacked some (non-computer) system to your advantage.

> [!question]
>
> What are the most interesting things you've built in your spare time? Include URLs if possible.

I contributed and helped maintain https://github.com/yetone/avante.nvim with my coworkers, where we provide a neovim plugins that emulate Cursor AI IDE.
Implemented bounding UI popover to improve QOL (https://github.com/yetone/avante.nvim/pull/29)
Added Rust crates for .avanterules templates (https://github.com/yetone/avante.nvim/pull/466)
Working on tool-use and function-calling support for multiple LLMs providers.

I maintained Quartz https://github.com/jackyzha0/quartz, A fast, batteries-included static-site generator that transforms Markdown content into fully functional websites. 
Improved performance of graph interaction with HTML5 Canvas (https://github.com/jackyzha0/quartz/pull/1328).
Added support for PDF in popover modal (https://github.com/jackyzha0/quartz/pull/913). Implemented font-fetching before runtime for faster build-time (https://github.com/jackyzha0/quartz/pull/817)
Enhanced search experience with telescope-style layout (https://github.com/jackyzha0/quartz/pull/722, https://github.com/jackyzha0/quartz/pull/774, https://github.com/jackyzha0/quartz/pull/782).
Pretty much then built some more custom features for my website (which is Quartz-based), https://aarnphm.xyz/thoughts/craft#open-source

I authored OpenLLM during my stint on-site in SF with BentoML (2022-2023) https://github.com/bentoml/OpenLLM and currently maintaining a bunch of tooling within BentoML ecosystem (BentoVLLM https://github.com/bentoml/BentoVLLM, BentoLangGraph https://github.com/bentoml/BentoLangGraph/tree/main, etc.)

https://github.com/aarnphm/morph, which is an experimental WYSIWYG file-over-app note editor that utilizes these ML models to generate suggestions based on user essays. Its aim is to help avid writers develop a better sense of writing, rather than relying on a LLM to do it for them.
This is an exploration into mech interp as well as dogfooding structured outputs in vLLM.
I tested out a bunch of stuff, with Three.js, Next.js, vLLM, Goodfire, mech interpretability, building RAG apps, structured outputs generations, etc.

I contributed to vLLM structured outputs features in v0 and v1 https://github.com/vllm-project/vllm
https://github.com/vllm-project/vllm/pull/14625 https://github.com/vllm-project/vllm/pull/12388 https://github.com/vllm-project/vllm/pull/10785 https://github.com/vllm-project/vllm/pull/14868
Currently working on jump-forward decoding, which is a speculative decoding but with 100% correctness based on JSON schema https://github.com/vllm-project/vllm/pull/15490

> [!question]
>
> Link to a video of you demo’ing the most technically impressive thing you've built. Show us the most impressive parts of the code.

https://youtu.be/DLIS8y0tXMg. The composable inference service, self-serve with BentoML and vLLM structured outputs https://github.com/aarnphm/morph/blob/main/python/asteraceae/service.py. Additionally, streaming JSON for generative UI (https://github.com/aarnphm/morph/blob/37a22994157b25555a83c2c1af1b4e98315bd3bc/packages/morph/components/editor.tsx#L1065) where I tried to use minimal amount of library and purely depends on browser API (fetch()). Also most of the frontend in this project has been relying solely on browser API with the exception of Next.js for ISR and Three.js for rendering landing page (otherwise db, interaction, motions are all browser native).

> [!question]
>
> Tell us about the technical project you plan to work on this summer. Why is it interesting to you?

I'm planning to dive deeper onto morph, where I want to experiment some different modalities that allows users to steer generations from text editor. Currently, sticky notes is one modality given that I used sticky notes a lot to quickly jot down ideas/TODO. However, the hardest part of writing is so removed from putting words together that the right way to un-block may not be anything in the typing out of the writing but in something more like planning / conversation partner. I do have some convictions that sticky notes can be one modality, but would want to explore what is also possible.

I also want to build a desktop version of this in Zig, where LLMs/embedding models will also be run locally (so that the app is truly local). Initially, morph was supposed to be distributed as a desktop app because I want to build something in Zig, but I decided to build with TS for faster iterations and reach. It is interesting because I'm a firm believer in local software,
and morph should be throwaway once users' had developed their own taste/feeling for writing.

> [!question]
>
> Do you plan to work on this project full time over the summer?

I also plan to do some research with either how we can improve speculative decoding with KV cache compression in long context generations (in the case of 1M context windows) or how to train interpreter models/SAEs/transcoders to intervene vision LM models (there are some literature wrt mech interp in InceptionV1, but I want to do this exploration but for VisionLM instead)

> [!question]
>
> What have you been thinking about deeply recently?

I think love is a narrative we imposed upon ourselves to categorise and control the uncomfortable reality of our existential isolation. Especially through Ian Mcgilchrist's hemispheric dichotomy where he emphasises that the left hemisphere creates the concepts of romantic partnership as a way to maximise utility while maintaining the illusion of connections.

We also saw this through Jungian's projection of the anima/animus -- we do not love the other but our own unconscious made manifest through them. What we call love are essentially just a sham to create a temporary sensation of wholeness. And through this the ego manifests these labels as "love", while what we truly desire are the yearning to collect fragmentation pieces of ourselves.

As a society we learned that love create stable pair bonds that benefits resource allocation and child-rearing. Therefore, we "invest" in relationship for "returns" on emotional "capital", rather than actually feeling "loved", because there is no such concepts as "loved". Monogamy, in a sense, represents not an authentic expression of connection but rather a very well socially engineered constraint that maximises predictability and minimize emotional transaction costs.
