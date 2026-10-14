---
date: '2024-02-08'
description: input errors, units, and the choice of objectives
id: Garbage in Garbage out
modified: 2026-06-05 15:08:28 GMT-04:00
tags:
  - seed
title: Garbage in Garbage out
---

"Garbage in, garbage out" warns that a correct computation can produce a useless answer when its inputs are wrong or misinterpreted. Suppose two thermometers report $20\,^{\circ}\mathrm{C}$ and $68\,^{\circ}\mathrm{F}$. They report the same temperature. Averaging the bare numbers gives $44$, which has no sensible temperature interpretation here. Keep the units with the measurements and convert before combining them.

The slogan leaves the diagnosis to us. A program can reject malformed input; a model may tolerate some noise. Which errors matter for this task, and what check would catch them? For an AI system, [NIST's validity guidance](https://airc.nist.gov/airmf-resources/airmf/3-sec-characteristics/) ties evaluation to its intended use and to test data that represents the conditions where it will operate.

How can we create agenda-free [[thoughts/representations|representations]] of an agenda-filled world? Choosing which [[thoughts/Information Theory|information]] to collect and what to optimize already involves decisions. Accurate click records cannot tell us whether maximizing clicks serves the person using the system. [[thoughts/Alignment|Alignment]] includes deciding whose intentions should guide the system, a question [OpenAI's 2022 research agenda](https://openai.com/index/our-approach-to-alignment-research/) explicitly leaves beyond its technical scope. Cleaning the records leaves that question open.
