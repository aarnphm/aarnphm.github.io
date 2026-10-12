---
aliases:
  - chat
date: '2024-10-02'
description: And its implication on how we assess learning. an overview.
id: chatgpt
modified: 2026-10-12 09:14:55 GMT-04:00
permalinks:
  - /chatgpt
tags:
  - engineer4a03
  - fruit
title: On ChatGPT and its pedagogical consequences
---

> [!question]-
>
> In the context of Gartner’s hype cycle, what has been the trajectory of generative conversational AI?
>
> Should a format including generative conversational AI be introduced to replace traditional essay assignments in educational settings,
> and if so, what are some potential implications for student learning and assessment? [@DWIVEDI2023102642]

## Introduction.

Historically, Alan Turing's seminal work "Computing Machinery and Intelligence" laid the foundation for exploring the possibilities of a thinking machine [@10.1093/mind/LIX.236.433].
Subsequently, the development of [[thoughts/Machine learning|AI]] took a symbolic approach: world representations through systems that manipulate high-level symbols to arrive at a result, commonly referred to as Good Old-Fashioned AI (GOFAI) [@10.7551/mitpress/4626.001.0001].

While GOFAI showed promise through rule-based [[thoughts/reason|reasoning]], its limitations became apparent in the 1980s when the field entered an "AI Winter." This was likely due to the cynicism within the AI researchers' community and a reduction in funding, which halted most research and development [@handler2008avoidanotheraiwinter].

Given Moore's Law and the exponential growth of compute and data, a new approach to [[thoughts/AGI|AI]] arose, focusing on statistical methods and [[thoughts/Connectionist network|connectionist networks]] such as artificial neural networks. [@10.7551/mitpress/4626.001.0001] dubbed this approach as New Fangled AI (NFAI). Fast forward to the $21^{\text{st}}$ century, ML has entered the mainstream through the rise of generative AI (GenAI).

This paper posits that GenAI has passed the "peak of inflated expectations" and is entering the "trough of disillusionment" on Gartner's hype cycle. It will also examine the implications of machine-assisted interfaces beyond conversational UI and their pedagogical consequences for student learning and assessment.

## Gartner's hype cycle.

For context, applications such as ChatGPT are built on top of [[thoughts/Transformers]] architecture and pre-trained on a large corpus of [[thoughts/Language#representations|text]] [@brown2020languagemodelsfewshotlearners]. Given
an input sequence of tokens length $n$, these systems will predict the next tokens at index $n+1$. Most implementations of transformers are autoregressive [@croft2023llm], meaning that the model generates one token at a time, each conditioned on every token before it.
Assuming the Strong Exponential Time Hypothesis, [@keles2022computationalcomplexityselfattention{pp. 4}] establish worst-case lower bounds for exact full self-attention and certain approximations. Under their dimension and error assumptions, these exclude running times of $O(n^{2-\varepsilon})$ for any fixed $\varepsilon>0$. Actual serving costs also depend on the model, workload, and implementation.

Gartner's own 2024 assessment places GenAI past the peak: "GenAI is in the Trough of Disillusionment with the beginning of industry consolidation" [@gartner2024multimodal]. Three factors drove it to the peak: rapid advancement in research, widespread enterprise adoption, and increased public awareness.

Public awareness came first. ChatGPT was released in November 2022 and was estimated to have reached 100 million monthly users two months later, faster than any consumer application before it.[^users]

Ongoing research in GenAI, specifically language models, spans several topics, including mechanistic interpretability [@nanda2023concrete], which explores the inner workings of auto-regressive models, retrieval and verification techniques aimed to improve correctness and reduce hallucinations among LLM systems [@béchard2024reducinghallucinationstructuredoutputs; @dhuliawala2023chainofverificationreduceshallucinationlarge],
as well as vested interests in multimodal applications of transformers [@xu2023multimodallearningtransformerssurvey]. Anthropic's interpretability and alignment work [@elhage2022superposition; @bricken2023monosemanticity; @templeton2024scaling], AI21's Jamba with its hybrid Transformer-Mamba architecture [@jambateam2024jamba15hybridtransformermambamodels], open-weights models from [Meta](https://www.llama.com/), and closed models such as [Google](https://deepmind.google/technologies/gemini/pro/)'s Gemini continue to redefine what these systems are capable of.

Enterprise adoption is evident with Salesforce [@nijkamp2023xgen7btechnicalreport], Oracle's [collaboration with Cohere](https://cohere.com/customer-stories/oracle), and Microsoft's Copilot for its 365 Product Suite. However, widespread implementation doesn't necessarily equate to immediate, measurable productivity gains. Integrating these systems effectively into enterprise workflows to deliver tangible business value takes time and effort.

Despite the field's excitement, the current hype and expectations often exceed its reliable capabilities, especially for complex use cases. Significant challenges persist, including
hallucinations and lack of factual grounding [@huang2023surveyhallucinationlargelanguage{pp. 3}]. We observe such behaviours in ChatGPT, where the knowledge cutoff prevents the system from providing up-to-date information, and it will "hallucinate" inaccurate answers instead. [@DWIVEDI2023102642{pp. 4.4.9.1.2}]

The gap between the first two factors and the third is what puts GenAI in the trough: awareness and adoption ran ahead of what the systems reliably do. As the field moves through the "trough of disillusionment", a more realistic assessment of GenAI's capabilities will likely emerge.

## Implications of machine-assisted interfaces and its pedagogical consequences for student learning and assessment.

The proliferation of conversational user interfaces (CUI) is based upon a simple heuristic of how [[thoughts/Autoregressive models|auto-regressive models]] surface their internal state through generating the next tokens.

CUIs often prove frustrating when dealing with tasks requiring larger information sets. Additionally, for tasks that require frequent information retrieval (research, academic writing), CUIs are suboptimal as they compel users to maintain information in their working memory unnecessarily.
Pozdniakov proposed a framework that incorporates both application and interaction design, emphasizing manual alignment inputs from end users [@pozdniakov2024largelanguagemodelsmeet{pp. 3}].
Should a format like this replace the traditional essay? I argue it should be introduced alongside the essay rather than replace it. Once a student can generate a fluent essay in seconds, the finished text no longer shows what the student did, so an assignment graded only on the final text stops measuring learning. But the skills the essay trains, building an argument and judging whether a claim is supported, are the same skills a student needs to evaluate what a model produces. Introducing the format has two major implications for student learning and assessment: a shift in core competencies and a shift in what gets assessed.

With machine-assisted interfaces, students will need to develop stronger critical thinking skills to evaluate AI-generated content and formulate precise instructions.
The focus will shift towards the process of reaching desired outcomes and improving information retrieval skills. This shift aligns with the potential for machine-assisted proofs to solve novel problems, as discussed by [@tao2024machineassisted].

These new interfaces will require instructors to adapt their evaluation methods. If the product can be generated, the process is what has to be graded: the student submits the record of their work with the model (prompts, drafts, the claims they checked and rejected) along with the final text, and defends it in a short oral discussion with the instructor. Assessment can then consider the student's level of engagement with a given topic, and lets students work at their own pace, since the record shows how they got there.
This approach encourages a more cross-disciplinary understanding, better preparing students for continuous learning.

[^users]: Reuters, "ChatGPT sets record for fastest-growing user base - analyst note", 2 February 2023, citing a UBS estimate from Similarweb data.
