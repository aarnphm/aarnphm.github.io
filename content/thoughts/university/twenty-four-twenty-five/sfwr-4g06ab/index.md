---
id: w1
tags:
  - university
  - sfwr-4g06ab
date: "2024-09-04"
title: Software Enginering Capstone a la carte.
---
## statement and goals.

Administrative Details
1. natural-language driven terminal
	Possible prof: [Emil Sekerinski](https://www.cas.mcmaster.ca/~emil/) or [Richard Paige](https://www.google.com/search?q=Richard+Paige&sourceid=chrome&ie=UTF-8)
	- [warp](https://www.warp.dev) as an example, but closed source
	- So you can think it like [Alacritty](https://github.com/alacritty/alacritty) but with async command-runner
	- voice-driven assistant: real-time transcribe => generate commands from language to shell commands
		- voice -> natural language
		- natural language -> commands
	- Configuration, maybe in Lua
	- stretch goal: new shell based on rust syntax and borrowing concept of variables.
2. WYSIWYG editor
	Possible prof: [Christopher Anand](https://www.cas.mcmaster.ca/~anand/#research) or [Martin von Mohrenschildt](https://www.cas.mcmaster.ca/~mohrens/)(maybe he is interested in this idk) or [Frantisek (Franya) FRANEK](https://www.cas.mcmaster.ca/~franek/)
	- Markdown renderer
	- train [SAE](https://transformer-circuits.pub/2023/monosemantic-features/index.html) for specific type of writing tonality => manual steering for text generation on creative writing
	- exploration of internals writing features based on text
		- inspired by [Prism](https://x.com/thesephist/status/1747099907016540181)
3. Infrastructure and AI Companion for Engineering Knowledge Management (19)
	- [Quartz](https://quartz.jzhao.xyz/) + similarity search + ANN for reranking