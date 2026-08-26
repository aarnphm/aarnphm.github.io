---
created: '2025-10-01'
date: '2025-10-01'
description: what multilingual model experiments can and cannot say about language and thought
id: linguistic relativity
modified: 2026-06-05 15:08:25 GMT-04:00
tags:
  - pattern
title: linguistic relativity
---

strong linguistic relativity says language determines which thoughts a person can have. The evidence supports the weaker claim: language changes which distinctions are cheap to notice or use.

language models let us test the engineering version of the question. We can hold the model and task fixed while changing the prompt language, the language used for intermediate reasoning, or the required output language. A change in output then shows sensitivity to language under that setup. Claims about private thought or human cognition need evidence from another experiment.

## what the experiments measure

Ray prompted GPT-4o mini with ten culturally salient questions in thirteen languages, then compared the meanings of its answers with multilingual sentence embeddings. The answers shifted with the prompt language [@ray2025linguistic]. That gives evidence for language-conditioned output variation in one model on a small prompt set. Distinct minds or conceptual worlds are outside what the experiment measured.

Reasoning models expose another variable because their visible traces can change language before the final answer. Wang et al. tested fifteen languages, seven levels of difficulty, and eighteen subject areas. The tested models often mixed languages, and script had a large effect on when mixing occurred [@languagemixing2025]. Several models routed part of their reasoning through English or Chinese even when the input used another language.

Li et al. studied Chinese and English reasoning in bilingual models. Reinforcement learning with verifiable rewards increased language mixing, while forcing monolingual decoding reduced accuracy on some mathematical tasks [@bilingualreasoning2024]. This is evidence about a learned computation strategy. The human term "code switching" names a similar visible pattern. The cause still has to be shown inside the model.

## shared and language-specific representations

Multilingual models contain both shared representations and language-specific routing. [Tang et al.](https://aclanthology.org/2024.acl-long.309/) identified neurons whose activation predicts the output language. Changing those neurons can steer which language the model generates. Other work finds increasing overlap among representations of equivalent content during multilingual training.

These results can coexist. Sharing a representation saves model capacity when several languages express the same relation. The model still needs language-specific vocabulary, grammar, and output control. A shared concept therefore does not imply identical processing, and a language-specific neuron does not imply a separate concept.

this is the wrong scoreboard for Whorf against universal grammar. Chomsky's universal grammar is a theory about the human language faculty. It did not predict that neural language models would converge on a specific internal representation. Model compression and transfer results therefore leave that theory and linguistic determinism unresolved.

## culture and performance

Prompt language is entangled with the text used for training. English has more data for many tasks, and the English data comes from particular institutions and populations. A model may therefore change its answer because of data volume, translation quality, cultural associations, or the reasoning route it learned. Calling every change "linguistic relativity" hides these causes.

Tao et al. compared model answers with responses from the World Values Survey and European Values Study. The tested GPT models tended to match English-speaking and Protestant European countries more closely. Country identity prompts often moved answers toward the target country's survey distribution [@culturalbias2024]. The intervention changed an answer distribution. It did not recover a whole cultural worldview, and some prompts made alignment worse.

The same care applies to performance gaps. A lower score in one language can come from less training data, a weak tokenizer, poor benchmark translation, or a model's preference for another reasoning language. Tests should vary these factors separately. A useful report states the model version, prompt language, reasoning constraint, output language, and task. An average gap without that setup explains little.

## what follows

the stable result is narrower: multilingual models share some abstractions and keep some language-specific routes. Prompt language and reasoning language can change accuracy and output distributions. The size and cause of the change depend on the model and task.

This makes language choice an engineering control. Evaluators should test the languages people will use, inspect whether a model silently changes reasoning language, and avoid treating an English result as the default result. If a system serves several cultures, evaluation also needs locally written prompts and local outcome measures. Translation alone preserves words more reliably than it preserves the task.

The philosophical result is narrower. Language can bias a computation without fully determining it. A multilingual model may share a concept across languages and still route the prompt through different features. That is a concrete version of weak linguistic relativity inside an artificial system. Any claim about human thought still needs evidence from humans.
