---
id: chatgpt
permalinks:
  - /chatgpt
tags:
  - engineer4a03
  - fruit
date: "2024-10-02"
modified: 2025-08-12 22:40:36 GMT-04:00
aliases:
  - chat
title: On ChatGPT and its pedagogical consequences
socialDescription: And its implication on how we assess learning. an overview.
---

> [!question]-
>
> In the context of Gartner’s hype cycle, what has been the trajectory of generative conversational AI?
>
> Should a format including generative conversational AI be introduced to replace traditional essay assignments in educational settings,
> and if so, what are some potential implications for student learning and assessment? [@DWIVEDI2023102642]

## Introduction.

Historically, Alan Turing's seminal work "Computing Machinery and Intelligence" laid the foundation for exploring the possibilities of a thinking machine [@10.1093/mind/LIX.236.433].
Subsequently, the development of [[thoughts/Machine learning|AI]] had taken a symbolic approach -- world representations through systems that utilise high-level symbols and manipulate tokens to arrive at a result commonly referred to as Good Old-Fashioned AI (GOFAI) [@10.7551/mitpress/4626.001.0001].

While GOFAI showed promise through decision-tree [[thoughts/reason|reasoning]], its limitations became apparent in the 1980s when the field entered "AI Winter." This was likely due to the cynicism within the AI researchers' community and a reduction in funding, which halted most research and development [@handler2008avoidanotheraiwinter].

However, given the rise of Moore's Law and the exponential amount of computing and data available, a new approach to [[thoughts/AGI|AI]] arose, focusing on statistical methods and connectionist networks such as artificial neural networks. [@10.7551/mitpress/4626.001.0001] dubbed this approach as New Fangled AI (NFAI). Fast forward to the $21^{\text{st}}$ century, ML has entered the mainstream through the rise of generative AI (GenAI).

This paper posits that GenAI currently occupies the "peak of inflated expectations", approaching the "trough of disillusionment" on Gartner's hype cycle. It will also examine the implications of machine-assisted interfaces beyond conversational UI and their pedagogical consequences for student learning and assessment.

## Gartner's hype cycle.

For context, applications such as ChatGPT are built on top of [[thoughts/Transformers]] architecture and pre-trained on a large corpus of [[thoughts/Language#representation.|text]] [@brown2020languagemodelsfewshotlearners]. Given
an input sequence of tokens length $n$, these systems will predict the next tokens at index $n+1$. Most implementations of transformers are autoregressive [@croft2023llm], meaning that the model will predict the future values (index $n+1 \to \infty$) based on past values (index $0 \to n$).
However, [@keles2022computationalcomplexityselfattention{pp. 4}] proved that the computation complexity of self-attention is quadratic; therefore, running these systems in production remains a scaling problem [@kaplan2020scalinglawsneurallanguage].

The current positioning of GenAI at the peak of inflated expectations aligns with the [@gartner2024multimodal] prediction. Three key factors support this assessment: rapid advancement in research, widespread enterprise adoption, and increased public awareness.
Ongoing research in GenAI, specifically language models, spans several topics, including mechanistic interpretability [@nanda2023concrete], which explores the inner workings of auto-regressive models, information retrieval techniques aimed to improve correctness and reduce hallucinations among LLM systems [@béchard2024reducinghallucinationstructuredoutputs; @dhuliawala2023chainofverificationreduceshallucinationlarge],
as well as vested interests in multimodal applications of transformers [@xu2023multimodallearningtransformerssurvey]. Leading research labs, from Anthropic on their interpretability and alignment work [@elhage2022superposition; @bricken2023monosemanticity; @templeton2024scaling], AI21's Jamba with its innovative hybrid transformers architecture [@jambateam2024jamba15hybridtransformermambamodels] to open-weights models from [Meta](https://www.llama.com/), [Google](https://deepmind.google/technologies/gemini/pro/) continue lead redefine the boundaries of what these systems are capable of.

Hello

Enterprise adoption is evident with Salesforce [@nijkamp2023xgen7btechnicalreport], Oracle's [collaboration with Cohere](https://cohere.com/customer-stories/oracle), and Microsoft's Copilot for its 365 Product Suite. However, widespread implementation doesn't necessarily equate to immediate, measurable productivity gains. Integrating these systems effectively into enterprise workflows to deliver tangible business value takes time and effort.

Despite the field's excitement, the current hype and expectations often exceed its reliable capabilities, especially for complex use cases. Significant challenges persist, including
hallucinations and lack of factual grounding [@huang2023surveyhallucinationlargelanguage{pp. 3}]. We observe such behaviours in ChatGPT, where the given knowledge cutoff prevents the systems from providing up-to-date information, which will "hallucinate" and provide inaccurate answers. [@DWIVEDI2023102642{pp. 4.4.9.1.2}]

As the field progresses towards the "trough of disillusionment" on Gartner's hype cycle, a more realistic assessment of GenAI's capabilities will likely emerge, paving the way for more effective applications.

## Implications of machine-assisted interfaces and its pedagogical consequences for student learning and assessment.

The proliferation of conversational user interfaces (CUI) is based upon a simple heuristic of how [[thoughts/Autoregressive models|auto-regressive models]] models surface their internal state through generating the next tokens.

CUIs often prove frustrating when dealing with tasks requiring larger information sets. Additionally, for tasks that require frequent information retrieval (research, academic writing), CUIs are suboptimal as they compel users to maintain information in their working memory unnecessarily.
Pozdniakov proposed a framework that incorporate both application and interaction design, emphasizing manual alignment inputs from end users [@pozdniakov2024largelanguagemodelsmeet{pp. 3}].
This approach, when applied replace traditional essay assignments, has two major implications for student learning and assessment: a shift in core competencies and collaborative assessment methods.

With machine-assisted interfaces, students will need to develop stronger critical thinking skills to evaluate AI-generated content and formulate precise instructions.
The focus will shift towards the process of reaching desired outcomes and improving information retrieval skills. This shift aligns with the potential for machine-assisted proofs to solve novel problems, as discussed by [@tao2024machineassisted].

These new interfaces will require instructors to adapt their evaluation methods. Assessment will need to consider students' pace flexibility and their level of engagement with a given topic.
This approach encourages a more holistic, cross-disciplinary understanding, better preparing students for continuous learning in our rapidly evolving technological landscape.

[^ref]
