---
id: structured decoding
tags:
  - technical
  - serving
date: "2024-12-10"
description: and vLLM integration with xgrammar.
draft: true
modified: 2024-12-31 05:51:42 GMT-05:00
title: structured decoding, a guide for the impatient
---

_vLLM is the high-throughput and efficient inference engine for running **large-language model** ([[thoughts/LLMs|LLM]])_

This post will quickly introduce the history of language model, and posit why one should care about [[thoughts/constrained decoding|structured decoding]]. For more information about vLLM, please check out our [documentation](https://docs.vllm.ai/en/latest/).

## language model, a brief historical context

_if you have read about the history of the field before, feel free to skip this part to [[posts/structured decoding#why do we need structured decoding?|reason for guided decoding]]_

The birth of AI started with birth of reasoning (esp plato, but plato seemingly wants to look for semantic rather than syntactic criteria (Republic examples)) => difference in Aristotle for applying in a Platonic project => leads to "The belief that such a total formalization of knowledge must be possible soon came to dominate Western thought." => Leibniz of universal symbolic language => birth of boolean algebra => babbage invention of analytical machine => Turing article which then splits into two tracks of initial GOFAI for symbolic system building expert system (but then funding dried up given that doubts and not being able to scale to generalized items) == Concurrently the PDP group investigated Rossenblatt's perception => leads to explosion of statistical methods using in prediction models (which is black-box by nature) => leads to IBM systems which further proves that statistical modelings works better than symbolic => then the dominance of neural net/connectionist network here which leads to RNN development for address longer context => Attention/Transformers => Scaling law for systems based on language models (but the idea of symbolic is still there and actually become even more prevalent for structured generations and safety)

The inception of [[thoughts/Machine learning|AI]] might well be traced back to the origin of logics, where men put emphasis on reducing reasoning to some specific sets of calculations (a [[thoughts/reductionism|reductionist]] approach).
As such, Plato generalised the belief in total formalisation of knowledge, where knowledge must be universally applicable with explicit definitions[^intuition]. However, according to
Alan Turing's seminal paper "Computing Machinery and Intelligence" where he posited that a high-speed digital computer, programmed with rules, would exhibit [[thoughts/emergent behaviour]] of [[thoughts/intelligence|intelligence]] [@10.1093/mind/LIX.236.433],
In the 1960s, a paradigm quickly emerged among researchers community that focused on symbolic [[thoughts/reason|reasoning]] was born, referred to as Good Old-Fashioned AI (GOFAI) [@10.7551/mitpress/4626.001.0001].
The premises of GOFAI being expert systems designed to replicate the decision-making abilities of human specialists [^expert-system], though it quickly ran into funding problems due to its semantic representation not being able to scaled up to generalized tasks (Also known as the "AI Winter").

[^socrates-belief]: According to [[thoughts/Plato]], Socrates asked Euthyphro, a fellow Athenian who is about to turn in his own father for murder in the name of piety: "I want to know what is characteristic of piety which makes all actions pious. [...] that I may have it to turn to, and to use as a standard whereby to judge your actions and those of other men."

[^intuition]:
    In other words, intuition, feeling would not constitute as the definition of knowing. For Plato, cooks, who proceed by taste and intuition does not involve understanding because they have no knowledge. Intuition is considered as a mere belief.

    Aristotle differed from Plato where he argued intuition was necessary to applying theory into practice [@aristotle_nicomachean_ethics{pp.8, book VI}].

[^expert-system]:
    Allen Newell and Herbert Simon's work at RAND initially showed that computers can simulate important aspects of intelligence.
    Another notable application was found in the medical domain [@10.7551/mitpress/4626.001.0001]. MYCIN, developed at Stanford University in the 1970s, diagnosed and recommended treatments for blood infections [@shortliffe1974mycin].
    MYCIN’s developers recognized the importance of justifying recommendations, implementing what were known as “rule traces” to explain the system’s reasoning in human-understandable terms.

Concurrently, Donald Norman's Parallel Distributed Processing [@10.7551/mitpress/5236.001.0001] group investigated variations of Rosenblatt's perception [@rosenblatt1958perceptron], where they
proposed _hidden layers_ within layers of the network alongside with inputs and outputs to extrapolate appropriate responses based on what it had learned during training process.
These systems, built on top of statistical methods[^5] and thus connectionist networks are often referred as New-Fangled AI (NFAI) [@10.7551/mitpress/4626.001.0001]. Given the abundance
of data and Moore's Law[^moore] resulting in an unprecedented amount of compute available, we see the complete dominance of connectionist networks in both research and production use-cases,
with variants of _decoder-only_ transformers[^lstm] for _text generations_ tasks (ChatGPT, Claude, Copilot, AlphaZero[^gofai-nfai], etc.).

[^gofai-nfai]:
    AlphaZero is a connectionist network based Go playing systems, that uses a deep neural networks to assess new positions and Monte-Carlo Tree Search (a GOFAI algorithm) to determine its next move [@silver2017masteringchessshogiselfplay]. DeepMind then
    applies these techniques to build AlphaFold, a system that predicts a protein’s 3D structure from its amino acid sequence.

> [!note]- on the term "next-token prediction"
>
> The term "next-token prediction" is also used sparsely when describing text-generation tasks, but it rather describes the "autoregressive" objectives of text generations. These models
> are trained to calculate a probability distributions for next tokens based on the causality of existing tokens within a sentence [^token]
>
> You can think of intuitively as "the models try to pick from a set of words what possible options are applicable to be used given a sentence". The model then "chooses" this words (or for more correct term _tokens_) based on a probability distribution and such iteratively repeats this process until it is "being told" to stop (in this case either mechanistically via configurations or out-of-memory (OOM)).
>
> For more information on how these auto-regressive transformers work, I highly recommend checking out [this awesome visualisation by Brendan Bycroft](https://bbycroft.net/llm) or [3Blue1Brown's Deep Learning series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
>
> For now, whenever we mentions LLMs, we assume that they are auto-regressive decoder-only transformers models.

[^token]:
    Machine learning models needs to process arbitrary numbers rather than text, as such we have to convert a given word $X$ to its corresponding encoded tokens (often known as tokenization).

    For example, the model will see the phrase `"The quick brown fox jumps over the lazy dog"` as `['<|begin_of_text|>', 'The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog']`

[^5]:
    In the 1990s, IBM released a sequence of complex statistical models that is trained to perform machine translations [tasks](https://en.wikipedia.org/wiki/IBM_alignment_models) [@IBMModels] (For more information check out this [lecture](https://www.cs.cornell.edu/courses/cs5740/2017sp/lectures/08-alignments.pdf) from Cornell).

    In 2001, Bag of words (BoW)-variants model was trained on 0.3B tokens and was considered SOTA at the time [@mikolov2013efficientestimationwordrepresentations]. These earlier works proved to the research community
    that statistical modeling triumps over symbolic counterpart for language processing given it can capture the general patterns for large corpuses of text.

[^moore]:
    In 2017, The landmark paper "Attention is all You Need" introduced [[thoughts/Transformers]] architecture [@vaswani2023attentionneed] for neural machine translations tasks, which is based on the [[thoughts/Attention|attention]] mechanism first proposed by [@bahdanau2016neuralmachinetranslationjointly].

    OpenAI then introduced the scaling law for neural language models [@kaplan2020scalinglawsneurallanguage], which sets off the race towards building these systems based on foundational language models.

[^lstm]:
    Prior to Attention-based transformers, seq-to-seq models uses RNNs given its ability for longer context length and better memory. However, they are more susceptible to vanishing/exploding gradients comparing to [[thoughts/FFN|feed-forward network]], and thus LSTM [@hochreiter1997long] was proposed to solve this problem.

    The Attention paper proposes a encoder-decoder architecture for translation tasks, however, most of text-generation models nowadays are _decoder-only_, given its superior performance over zero-shot tasks.

In retrospect, GOFAI are [[thoughts/Determinism|deterministic]] in a sense that intentionality is injected within symbolic tokens through explicit programming.
Connectionist networks, on the other hand, are often considered as black-box models, given their hidden nature of intermediate representations of perceptron.
Unlike GOFAI, its internal representation is determined by the state of the entire network states rather than one singular unit. Although these models exihibit [[thoughts/emergent behaviour]] of [[thoughts/intelligence|intelligence]],
one should be aware that this is not [[thoughts/AGI|artificial general intelligence]] _yet_, largely due to researchers' [[thoughts/observer-expectancy effect]].

## why do we need structured decoding?

![[posts/images/shogoth-gpt.png|Shogoth as GPTs]]

LLMs excel at surfacing its internal representation of the world through a simple interface: given
a blob of text, the model will generate a contiguous piece of text that it predicts as the most probable tokens.
For example, if you give it a Wikipedia article, the model should produce text consistent with the remainder of said article.
These models works well given the following assumption: the inputs prompt must be coherent and well-structured
surrounding a given problem the users want to achieve. In other words, generations are not deterministic if one expect
a certain formats of said outputs (most notably JSON[^prompting]). This arises for the need of applying explicit rules
and grammar[^cfg]

[^prompting]:
    One might argue that we can reliably achieve these through few-shot promptings, i.e "Give me a JSON that yields the address of users. Example output can be ...". However, there
    is no guarantee that the generated outputs is a valid JSON.

    One might also argue that one should use specific fine-tuned models for JSON outputs to perform such cases. However, fine-tuning often requires extensive training and a lot more
    labor to curate data, monitor progress, and perform evaluation, which is a huge resources not everyone can afford to do.

[^cfg]: the most

[^ref]

---

- quick history context for autoregressive nature of decoder-only transformers
- why we need guided/structured/constrained decoding
  - generating JSON
  - hypothetically more performance
  - function calling
- current implementation limitation in vLLM, (explain why slow)
- what xgrammar can improve (so quick intro about xgrammar work, then show some graphs)
- quick add on how we implement it in vLLM
- next steps for v1
