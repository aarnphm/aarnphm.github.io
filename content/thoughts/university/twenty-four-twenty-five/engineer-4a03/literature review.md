---
id: literature review
tags:
  - engineer4a03
description: How we understand machine learning system is how we can move towards a safe futures, yet the road ahead lies many troubles to overcome. A literature review into the inception of the field, as well as where do we go from here.
date: "2024-10-07"
sidenotes: false
modified: 2025-10-04 17:42:08 GMT-04:00
noindex: true
title: machine learning, from the inception of time, a literature review
---

See also [[posts/chatgpt|essays on ChatGPT]], [[thoughts/university/twenty-four-twenty-five/engineer-4a03/case study|case study on Cambridge Analytica]]

## introduction.

```quotes
To understand how AI is fundamentally political, we need to go beyond neural nets and statistical pattern recognition to instead ask _what_ is being optimized, and _for whom_, and _who_ gets to decide. Then we can trace the implications of those choices. -- Kate Crawford
```

1979's "Star-Trek: the Motion Picture" centered around the antagonist, V'Ger, an artificial entity that have outgrown its original programs, sought annihilation upon planet Earth. At the core,
the movie is mostly fictional, yet its prevalence to our current state of affairs is uncanny. Much in Artificial intelligence (AI) has changed since 1960s, including a shift in symbolic
systems to more recent hype about deep connectionist networks. AI has expanded rapidly as a academia field and as a industry[^1]. Yet, the belief of formalising human intelligence and reproduced
by machine has always been the core disputes in the history of AI. There has always been two narratives discussed within academia and industry practitioners on how we should approach such systems:
The likes of Marvin Minsky claiming "machine can think" [@atlasofai{pp. 5-9}]; while Dreyfus [@dreyfus2008why] believed in a Heideggerian AI system would dissolve the framing problem [^framing].
Nowadays, this narrative morphs into two verticals: Entities that seek to build systems capable of outperforming at tasks that a human can do at a greater degree of accuracy and efficiency (OpenAI, Anthropic, SSI, many AI labs, etc.[^ssi]), and
companies that build AI systems to amplify our abilities to create and improve efficiency for our work (Runway, Cohere, etc.).

This literature review aims to provide a comprehensive overview of the current state of AI, through its history and current adoption. It will also include investigations into certain concerns for diversity, equity, and inclusion (DEI) within the field,
as well as the ethical implications of AI systems. It will then conclude and posit questions about where we go from here.

[^1]:
    [@jordan2015machine] described the emerging trends within classical machine learning systems, focusing on recommendation systems. From a recent McKinsey's reports of outlook trend of 2024, they
    reported around 570bn dollars equity investment in the adoption of generative AI, notably the integration of LLMs into enterprises usecase [@mckinsey2024techtrends]

[^framing]:
    An intelligent being learns from its experience, then applies such intuition to predict future events. How does one select appropriate context (frame) for a given situation?<br />
    Dreyfus’ argument is that machines are yet able to represent human’s reliance on many unconscious and subconscious processes [@dreyfus1972what]. A Heideggerian AI would exhibit Dasein (being in the world).

[^ssi]: Their goals are to build “artificial super intelligence” (ASI) systems. This target is largely due to certain observer-expectancy effect we observe in the current AI system.

## growth.

```quotes
Mathematicians wish to treat matters of perception mathematically, and make themselves ridiculous [...] the mind [...] does it tacitly, naturally, and without technical rules. -- Pascal, Pensées
```

The inception of [[thoughts/Machine learning|AI]] might well begin when the belief of a total formalisation of knowledge must be possible[^2]. From Plato's
dichotomy of the rational soul from the body with its skills and intuition[^3], to Leibniz's conception of the binary systems as a "universal characteristics" [@leibniz_selections_1951{pp. 15, 25, 38}] that
led to Babbage's design of "Analytic Engine" being recognized as the "first digital computer", Alan Turing posited that a high-speed digital computer, programmed
with rules, might exhibit [[thoughts/emergent behaviour]] of [[thoughts/intelligence|intelligence]] [@10.1093/mind/LIX.236.433]. Thus, a paradigm among researchers that focused on symbolic [[thoughts/reason|reasoning]] was born, referred to as Good Old-Fashioned AI (GOFAI) [@10.7551/mitpress/4626.001.0001]. GOFAI was built on a high level symbolic representation of the world, popularized through expert systems [@jackson_introduction_1998]
that tried to mimic human expert on specialized tasks [^4]. Yet, we observed a period of "AI Winter" where most symbolic AI research either reached dead end or funding being dried up [@handler2008avoidanotheraiwinter].
This is largely due to GOFAI's semantic representation which were implausible to scale to generalized tasks.

Concurrently, Donald Norman's Parallel Distributed Processing [@10.7551/mitpress/5236.001.0001] group investigated variations of Rosenblatt's project [@rosenblatt1958perceptron], where they
proposed intermediate processors within the network (often known as "hidden layers") alongside with inputs and outputs to extrapolate appropriate responses based on what it had learned during training process.
These systems, built on top of statistical methods[^5] and connectionist networks are often referred to by Haugeland as New-Fangled AI (NFAI) [@10.7551/mitpress/4626.001.0001].

In retrospect, GOFAI are [[thoughts/Determinism|deterministic]] in a sense that intentionality is injected within symbolic tokens through explicit programming.
[[thoughts/Connectionist network]], on the other hand, are often considered as black-box models, given their hidden nature of intermediate representations of perceptron.
Unlike GOFAI, its internal representation is determined by the state of the entire network rather than any single unit.
Given the rise of Moore's Law and the exponential amount of computing and data available, we are currently witnessing the dominance of connectionist networks, especially with the injection of LLMs into the mainstream [@kaplan2020scalinglawsneurallanguage],
where the majority of research are focused on developing artificial neural networks that optimizes around loss functions [@vaswani2023attentionneed; @srivastava_dropout_2014]. One notable example that combines both GOFAI and NFAI
systems is AlphaZero, a connectionist network based Go playing systems, that uses a deep neural networks to assess new positions and Monte-Carlo Tree Search (a GOFAI algorithm) to determine its next move [@silver2017masteringchessshogiselfplay].

[^2]:
    According to [[thoughts/Plato]], Socrates asked Euthyphro, a fellow Athenian who is about to turn in his own father for murder in the name of piety: "I want to know what is characteristic of piety which makes all actions pious. [...] that I may have it to turn to, and to use as a standard whereby to judge your actions and those of other men."
    This is Socrates' version of [[thoughts/effective procedure]] for modern-day computer scientists.

[^3]:
    According to Plato, all knowledge must be universally applicable with explicit definitions, in other words, intuition, feeling would not constitute as the definition of knowing
    Aristotle differed from Plato where intuition was necessary to applying theory into practice [@aristotle_nicomachean_ethics{pp.8, book VI}].
    For Plato, cooks, who proceed by taste and intuition does not involve understanding because they have no knowledge. Intuition is considered as a mere belief.

[^4]: Allen Newell and Herbert Simon's work at RAND initially showed that computers can simulate important aspects of intelligence.

[^5]: Notable figures include John Hopfield, Hinton's "A Learning Algorithm for Boltzmann Machines" [@ackley_learning_1985] that introduces the concept of Boltzmann's distributions in training neural networks, as well as Hinton's later work on backpropagation algorithm.

## adoption.

For context, we produce a lot of data: social media consumption, emails transaction, search, online shopping, mainly due to the rise of the internet and Web 2.0 post 9/11. While
capitialism has always been a fraught system, there are incentives for harvesting our attention and predict our future behaviour -- what Zuboff refers to as "surveillance capitalism" [@carr2019thieves]. In a sense,
surveillance capitalism is built on top of the notion of _extraction imperatives_ where the Google and Facebook of the world have to mine as much information as possible [^6]. Machine learning benefited
of this phenomenon since statistical methods often predict certain pattern from given data and yield certain predictions/decisions. ML can be categorized into two sub fields, supervised learning
(where algorithms are trained on labelled data to provide prediction based on given labels) and unsupervised learning (where algorithms are trained on the basis of "produce _y_ in the form of _x_")[^7].

Supervised learning methods including Naive Bayes, Decision tree, and other Bayesian models have been well integrated into industries to solve forecasting and classification problems [@zhang2020labelingmethod]

[^6]: Some notable quotes:

    - "Unlike financial derivatives, which they in some ways resemble, these new data derivatives draw their value, parasite-like, from human experience.".
    - "[Facebook's algorithm fine-tuning and data wrangling] is aimed at solving one problem: how and when to intervene in the state of play that is your daily life in order to modify your behavior and thus sharply increase the predictability of your actions now, soon, and later."

[^7]: This is a mere simplification of the field. ML researchers also investigate in specific sub-fields

## fairness

See also: MIT Press [@HaoKarBuolamwini2019], Darthmouth investigation in COMPAS system [@doi:10.1126/sciadv.aao5580]

DEI has become a key aspect of technological progress in the $21^{\text{st}}$ century. This applies to AI, where its black-box nature has proven to be difficult for researchers to align certain bias bugs. Two main DEI methods emerge for addressing given problems: improving data
diversity and ensuring fairness during the training procedure.

The primary methods on fighting against bias bugs in contemporary AI system includes increase in data diversity. There is a timeless saying in computer science "[[thoughts/Garbage in Garbage out]]",
which essentially states that bad data will produce outputs that's of equal quality.
This is most prevalent in AI, given the existence of these networks within a black-box model. One case of this is the very first iterations of Google Photos’ image
recognition where it identified people with darker skins as “gorillas” [@BBCGoogleApology2015]. Alliances such as The Data & Trust Alliance, including Meta, Nike, CVS Health, are formed to regulate and
combat algorithmic bias. The Data & Trust Alliance aims to confront dangers of powerful algorithms in the work force before they can cause harm instead of simply reacting after
the damage is done (Lohr, 2021). (Clarke, 2021) proposed that close inspection and regulation of these models should be monitored closely to mitigate misrepresentation of marginalized groups (Khan, 2022).

Truth is, data lacks context. A prime example of this US’ COMPAS used by US courts to assess the likelihood of criminal to reoffend. ProPublica concluded that COMPAS was inherently
biased towards those of African descent, citing that it overestimated the false positives rate for those of African descent by two folds [@AngwinLarsonMattuKirchner2016]. Interestingly, a study done at Darthmouth showed
a surprising accuracy on the rate of recidivism with random volunteers when given the same information as the COMPAS algorithm [@doi:10.1126/sciadv.aao5580].
The question remains, how do we solve fairness and ensure DEI for marginalized groups when there is obviously prejudice and subjectivity that introduce bias at play?
It is not a problem we can’t solve, rather collectively we should define what makes an algorithm **fair**.
