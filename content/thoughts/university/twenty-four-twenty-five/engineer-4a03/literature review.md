---
id: literature review
tags:
  - engineer4a03
date: "2024-10-07"
modified: "2024-10-08"
title: literature review
---
See also [[posts/a1|essays on ChatGPT]]

<p class="quotes">
  <i>Mathematicians wish to treat matters of perception mathematically, and make themselves ridiculous [...] the mind [...] does it tacitly, naturally, and without technical rules.</i> -- Pascal, <i>Pensées</i>
</p>

The inception of [[thoughts/Machine learning|artificial intelligence]] (AI) might well begin when the belief of a total formalisation of knowledge must be possible[^1][^2]. From Plato's
separation of the rational soul from the body with its skills and intuition[^2], to Lebniz's invention of the binary systems as a "universal characteristics" [@leibniz_selections_1951{pp. 15, 25, 38}] that
led to Babbage's design of "Analytic Engine" being recognized as the "first digital computer", Alan Turing posited that a high-speed digital computer, programmed
with rules, might exhibit [[thoughts/emergent behaviour]] of [[thoughts/intelligence|intelligence]] [@10.1093/mind/LIX.236.433]. Thus, a paradigm among researchers that focused on symbolic [[thoughts/reason|reasoning]] was born, referred to as Good Old-Fashioned AI (GOFAI) [@10.7551/mitpress/4626.001.0001]. GOFAI was built on a high level symbolic representation of the world, popularized through expert systems [@jackson_introduction_1998]
that tried to mimic human expert on specialized tasks [^3]. Yet, we observed a period of "AI Winter" where most symbolic AI research either reached dead end or funding being dried up [@handler2008avoidanotheraiwinter].
This is largely due to GOFAI's semantic representation which were implausible to scale to generalized tasks.

Concurrently, Donald Norman's Parallel Distributed Processing [@10.7551/mitpress/5236.001.0001] group investigated variations of Rosenblatt's project [@rosenblatt1958perceptron], where they
proposed intermediate processors within the network (often known as "hidden layers") alongside with inputs and outputs would address GOFAI's limitations. These systems, built on top of statistical methods[^4]
and connectionist networks are often referred to by Haugeland as New-Fangled AI (NFAI) [@10.7551/mitpress/4626.001.0001].

In retrospect, GOFAI are [[thoughts/Determinism|deterministic]] in a sense that intentionality is injected within tokens through explicit programming. Whereas connectionist networks are often considered as black-box models, given
their hidden nature of intermediate representations of perceptron. Unlike GOFAI, its internal representation is determined by the state of the entire network rather than any single unit. Given the rise of Moore's Law
and the exponential amount of computing and data available, we are currently witnessing the dominance of connectionist networks, especially with the injection of LLMs into the mainstream [@kaplan2020scalinglawsneurallanguage],
where the majority of research are focused on developing artificial neural networks that optimizes around loss functions [@vaswani2023attentionneed] [@srivastava_dropout_2014]. One notable example that combines both GOFAI and NFAI
systems is AlphaZero, a connectionist network based Go playing systems, that uses a deep neural networks to assess new positions and Monte-Carlo Tree Search (a GOFAI algorithm) to determine its next move [@silver2017masteringchessshogiselfplay].


[^1]: According to Plato, Socrates asked Euthyphro, a fellow Athenian who is about to turn in his own father for murder in the name of piety: "I want to know what is characteristic of piety which makes all actions pious. [...] that I may have it to turn to, and to use as a standard whereby to judge your actions and those of other men."
    This is Socrates' version of [[thoughts/effective procedure]] for modern-day computer scientists.
[^2]: According to Plato, all knowledge must be universally applicable with explicit definitions, in other words, intuition, feeling would not constitute as the definition of knowing
    Aristotle differed from Plato where intuition was necessary to applying theory into practice [@aristotle_nicomachean_ethics{pp.8, book VI}].
    For Plato, cooks, who proceed by taste and intuition does not involve understanding because they have no knowledge. Intuition is considered as a mere belief.
[^3]: Allen Newell and Herbert Simon's work at RAND initially showed that computers can simulate important aspects of intelligence.
[^4]: Notable figures include John Hopfield, Hinton's "A Learning Algorithm for Boltzmann Machines" [@ackley_learning_1985] that introduces the concept of backpropagation and utilisation of Boltzmann's distributions for training neural networks.
