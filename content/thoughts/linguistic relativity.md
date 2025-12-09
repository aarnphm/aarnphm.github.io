---
created: "2025-10-01"
date: "2025-10-01"
description: Language shapes thought—classical Sapir-Whorf hypothesis finds unexpected validation in language models, where linguistic structure creates measurable emergent properties.
id: linguistic relativity
modified: 2025-12-08 21:55:01 GMT-05:00
tags:
  - pattern
title: linguistic relativity
---

the sapir-whorf hypothesis—that language shapes thought—keeps getting dismissed and then vindicated. 20th century debates focused on whether linguistic structure constrains human cognition. now llms provide unusually clear evidence: different languages, training corpora, and prompt languages produce systematically different reasoning patterns [@ray2025linguistic].

despite multilingual training, large reasoning models predominantly "think" in one or two hub languages—usually english and chinese—regardless of input language. but they still show language-specific behavior. linguistic relativity operates at multiple levels simultaneously: training corpus composition, prompt language, internal representation space, output generation.

> [!important] towards [[thoughts/emergent behaviour|emergent properties]] from [[thoughts/Determinism|determinism]]
>
> the strong form—linguistic determinism—claimed language determines and limits thought.
> This is now largely rejected with modern cognitive science and linguistics study.
>
> However, the weaker form holds: linguistic structures influence perception and reasoning without strictly constraining them.

in [[thoughts/LLMs|language models]], these effects emerge from training dynamics (stochastic prediction) rather than hardcoded constraints. we can now operationalize whorfian effects through embeddings, neuron activation patterns, cross-lingual transfer performance.

## [[thoughts/emergent behaviour|emergent properties]] in language models

### linguistic structure and model behavior

gpt-4o mini generating responses to culturally salient prompts across 13 typologically diverse languages shows measurable linguistic relativity in AI-generated text [@ray2025linguistic]. significant variation in semantic alignment across language pairs.

the models probably don't translate concepts but instead maintain [[lectures/411/notes#Isomorphisms, adjoints, invariant subspaces|subspaces]] of concepts per input language.

### language-specific neurons and shared representations

multilingual models contain both language-agnostic and language-specific regions. [[thoughts/sparse autoencoder|saes]] on haiku 3.5 show prompts with identical meaning across languages activate similar circuits, but models still maintain language-specific neurons for vocabulary, grammar, idioms:

- cross-lingual neuron overlap correlates strongly with zero-shot transfer performance
- compression during pre-training forces shared cross-lingual representations over separate language-specific ones
- implicit alignment degrades in certain pre-training phases
- grammatical features (number, semantic roles) localize in specific embedding regions rather than distributing globally

### language mixing in reasoning

15 languages × 7 difficulty levels × 18 subject areas: all factors influence language mixing in reasoning traces [@languagemixing2025]. script composition aligns with internal representations—models have an internal preference for latin script, which explains why they mix into latin-script reasoning even when prompted in other languages.

forcing models to reason in scripts matching the input language improves accuracy; mismatched scripts degrade performance [@bilingualreasoning2024]. linguistic relativity affects not just outputs but internal reasoning—deepseek-r1 and qwq-32b show human-like language mixing in reasoning chains, patterns emerging from interaction between task difficulty, script types, model architecture.

## domain-specific evidence: humans and models

### color perception

brown and lenneberg's classic work tested whether codability of color terms affects memory and recognition—zuni speakers grouping green/blue struggled distinguishing within that category. gpt-4 replicates cross-linguistic variation in english vs russian color perception, showing human-like color-concept associations.

linguistic relativity effects in color cognition emerge in models through training data rather than perceptual constraints, but mirror human psycholinguistic patterns anyway.

### spatial reasoning

comfort benchmark (consistent multilingual frame of reference test) shows poor robustness across languages in spatial tasks:

- english dominance in resolving spatial ambiguities
- failure to accommodate multiple frames of reference
- zero adherence to language-specific/cultural conventions

languages differ in spatial encoding—some use absolute directions (guugu yimithirr), others relative frames. models struggle with this diversity, defaulting to english-like reasoning.

### temporal reasoning

perfect times benchmark (english, italian, russian, japanese) shows models struggle with human-like temporal causal reasoning. languages encode time differently—english/swedish use distance metaphors ("long meeting"), spanish uses quantity ("big meeting"), mandarin employs vertical metaphors. these correlate with duration judgments in both humans and models.

time representation in embeddings varies by training language distribution.

## practical implications for AI systems

### cross-lingual transfer and alignment

models correctly answer questions in english but fail with identical questions in swahili, igbo, other low-resource languages. performance consistently declines from english → local languages → other foreign languages. english-centric models display 3-4% average performance advantage across languages.

proposed solutions: prealign (multilingual alignment before pre-training), alignx (advance multilingual representation alignment), code-switched fine-tuning to bridge english/low-resource gaps.

### cultural bias and worldview

training corpora bias creates measurable "worldview" in models. english prompts align with western liberal viewpoints; the same model shows different perspectives in different languages. popular llms exhibit western cultural bias, favoring self-expression values from english-speaking/protestant european countries [@culturalbias2024].

models show greater cultural alignment when prompted in a culture's dominant language. misalignment is more pronounced for underrepresented personas and culturally sensitive topics.

anthropological prompting—leveraging anthropological reasoning for cultural alignment—improves alignment for 71-81% of countries/territories [@culturalbias2024]. linguistic relativity in models isn't just semantic differences but cultural worldviews encoded from training data.

### grammatical structure and reasoning

program-of-thought fine-tuning substantially enhances multilingual reasoning, outperforming chain-of-thought. reasoning scaffolding interacts with linguistic structure.

grammatical gender effects appear in models: gender bias is pronounced in gendered languages (spanish, french), with male-to-female ratios in training data ranging 4:1 to 6:1. grammatical categories propagate social biases.

## historical foundations

### ancient philosophy

the idea that [[thoughts/Language|language]] and thought are intertwined is ancient. [[thoughts/Plato|plato]]'s _cratylus_ explores whether conceptions of reality are embedded in language—a question central to [[thoughts/Epistemology|epistemology]]. plato may have held that the world consists of eternal ideas and language should represent these accurately, though his _seventh letter_ claims ultimate truth is inexpressible in words.

st. augustine argued language was merely labels applied to pre-existing concepts. this view persisted through the middle ages. for [[thoughts/Philosophy and Kant|kant]], language was one of several methods humans use to experience the world, not constitutive of thought itself.

### german romantic philosophers

late 18th/early 19th century: ideas of different national characters (_volksgeister_) motivated german romanticism and early ethnic nationalism. johann georg hamann discussed the "genius" of a language: "the lineaments of their language will thus correspond to the direction of their mentality."

wilhelm von humboldt (1820) proposed language as the fabric of thought—thoughts produced through internal dialog using native grammar. linguistic diversity is a diversity of worldviews, with languages creating individual perspectives through lexical/grammatical categories, conceptual organization, syntactic models.

### american anthropology

franz boas challenged the idea that some languages are superior to others. all languages can express the same content by different means, though the form of language is molded by culture.

edward sapir drew on humboldt but explicitly rejected strong [[thoughts/Determinism|determinism]]: "it would be naïve to imagine that any analysis of experience is dependent on pattern expressed in language." language and culture are not intrinsically associated.

benjamin lee whorf, sapir's student, studied native american languages:

> we dissect nature along lines laid down by our native language ... all observers are not led by the same physical evidence to the same picture of the universe, unless their linguistic backgrounds are similar.

whorf's claims about hopi time conceptualization—that hopi treats time as a single process rather than countable instances—remain contested.

### formalization and testing

roger brown and eric lenneberg reformulated linguistic relativity as testable hypothesis through color perception experiments—whether codability of color terms affects memory and recognition. zuni speakers grouping green/blue struggled distinguishing within that category.

brown formulated the well-known weak/strong versions:

- **weak**: structural differences between languages paralleled by nonlinguistic cognitive differences
- **strong**: one's native language strongly influences or fully determines worldview

### universalism and critiques

1960s emphasis on universal grammar—particularly chomsky's work on innate linguistic structures—disfavored linguistic relativity. universalists argued for largely innate structures where differences between languages are surface phenomena. from the 1960s–1980s this view dominated, and relativity was often ridiculed.

ekkehart malotki challenged whorf's hopi time claims with extensive data, though relativists maintain whorf's actual claim concerned different conceptualization rather than absence of time concepts.

steven pinker argues thought is independent of language, proposing "mentalese"—a language of thought distinct from natural language. relativists have accused universalists of misrepresenting whorf through strawman arguments.

from the late 1980s, new research found broad support for non-deterministic versions. current consensus: language influences certain cognitive processes in non-trivial ways, but other processes develop from connectionist factors.

see also: Language-Specific Neurons (ACL 2024, arXiv:2402.16438), Cross-lingual Transfer of Reward Models (arXiv:2410.18027), COMFORT spatial reasoning benchmark, Perfect Times temporal reasoning benchmark.

## dispute

the debate between universalists (chomsky's universal grammar) and relativists finds new expression in language models. universal grammar predicts models should converge on shared representations; linguistic relativity predicts language-specific processing. both occur.

models exhibit:

- **universal aspects**: shared cross-lingual neuron activation for equivalent concepts, transfer learning success
- **relativistic aspects**: language-specific neurons, performance gaps across languages, culture-dependent outputs

this dual nature mirrors human cognition—some aspects universal (perceptual universals in color, basic spatial relations), others shaped by linguistic structure (fine-grained color discrimination, frame-of-reference preferences, temporal reasoning).

### emergent abilities and linguistic structure

the "emergent abilities" debate—whether capabilities appear suddenly at scale or result from metric choices—connects to linguistic relativity. wei et al. (2022) defined emergent abilities as unpredictable capabilities at scale; schaeffer et al. (2023) argued these may be measurement artifacts.

but language-specific behavior in models suggests genuine emergence: compression during pre-training forces multilingual alignment that wasn't explicitly programmed. models develop internal "hub" languages for reasoning without being told to. language mixing patterns emerge from interaction between task difficulty, script types, model architecture.

The are more qualitative changes in how models represent and process linguistic information, analogous to how human language acquisition shapes cognitive development instead of simple scaling effects.

### programming languages and notation

kenneth e. iverson argued notations are tools of thought; more powerful notations aid thinking. paul graham's "blub paradox" observes that thinking in a language can obscure awareness of more expressive ones.

in ai systems, this manifests as models defaulting to english or latin-script reasoning even when prompted in other languages. the training distribution creates a "notational preference" that shapes internal computation—a computational version of whorf's claim that we dissect nature along lines laid down by our languages.
