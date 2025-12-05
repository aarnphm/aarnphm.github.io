---
created: "2025-10-01"
date: "2025-10-01"
description: Language shapes thought—classical Sapir-Whorf hypothesis finds unexpected validation in language models, where linguistic structure creates measurable emergent properties.
id: linguistic relativity
modified: 2025-12-04 17:57:10 GMT-05:00
tags:
  - seed
title: linguistic relativity
---

The hypothesis that language influences thought—classical Sapir-Whorf hypothesis—has found unexpected validation in modern language models. While 20th century debates centered on whether linguistic structure constrains human cognition, recent work demonstrates measurable linguistic relativity effects in LLMs, where different languages, training corpora, and prompt languages lead to systematically different reasoning patterns and outputs [@ray2025linguistic].

Contemporary research reveals that despite multilingual training, large reasoning models predominantly "think" in one or two hub languages—typically English and Chinese—regardless of input language, yet still exhibit language-specific behavior patterns. This suggests linguistic relativity operates at multiple levels: training corpus composition, prompt language, internal representation space, and output generation.

> [!important] towards [[thoughts/emergent behaviour|emergent properties]] from [[thoughts/Determinism|determinism]]
>
> The strong form—linguistic determinism—claimed language determines and limits thought. This has been rejected. Modern evidence supports a weaker form: linguistic structures influence perception and reasoning without strictly constraining them. In [[thoughts/LLMs|language model]], these effects emerge from training dynamics (i.e stochastic prediction) rather than hardcoded constraints, offering new ways to operationalize Whorfian effects through embeddings, neuron activation patterns, and cross-lingual transfer performance.

## [[thoughts/emergent behaviour|emergent properties]] in language models

### linguistic structure and model behavior

A 2025 study provides the first comprehensive quantitative evaluation of linguistic relativity in AI-generated text [@ray2025linguistic]. Using ChatGPT-4o mini to generate responses to culturally salient prompts across 13 typologically diverse languages, the research demonstrates that linguistic structures exert measurable influence on AI outputs—statistical analysis shows significant variation in semantic alignment across language pairs.

I suspect the models doesn't translate concepts, rather having [[lectures/411/notes#Isomorphisms, adjoints, invariant subspaces|subspaces]] of concepts per input language.

### language-specific neurons and shared representations

Multilingual models contain both language-agnostic and language-specific regions.

[[thoughts/sparse autoencoder]] on 3.5 Haiku found that prompts with identical meaning across languages activate similar circuits, yet models maintain language-specific neurons for vocabulary, grammar, and idiomatic expressions:

- High correlation between cross-lingual neuron overlap and zero-shot transfer performance
- Compression during pre-training forces shared cross-lingual representations over separate language-specific ones
- Degradation of implicit alignment detected in certain pre-training phases
- Grammatical features (number, semantic roles) localize in specific embedding regions rather than distributing globally

### language mixing in reasoning

Examining 15 languages across 7 difficulty levels and 18 subject areas reveals that all factors influence language mixing patterns in reasoning traces [@languagemixing2025]. Script composition aligns with internal representations—models show internal preference for Latin script, explaining why they mix into Latin-script reasoning even when prompted in other languages.

Forcing models to reason in scripts matching the input language improves accuracy; mismatched scripts degrade performance [@bilingualreasoning2024]. This suggests linguistic relativity affects not just outputs but internal reasoning processes—DeepSeek-R1 and QwQ-32B show human-like language mixing in reasoning chains, with patterns emerging from interaction between task difficulty, script types, and model architecture.

## domain-specific evidence: humans and models

### color perception

Brown and Lenneberg's classic work tested whether codability of color terms affects memory and recognition—Zuni speakers grouping green/blue had difficulty distinguishing within that category. Recent work shows GPT-4 replicates cross-linguistic variation in English vs. Russian color perception, with models showing human-like color-concept associations.

This suggests linguistic relativity effects in color cognition emerge in models through training data rather than perceptual constraints, yet mirror human psycholinguistic patterns.

### spatial reasoning

The COMFORT benchmark (COnsistent Multilingual Frame Of Reference Test) reveals poor robustness across languages in spatial tasks. Models show:

- English dominance in resolving spatial ambiguities
- Failure to accommodate multiple frames of reference
- Lack of adherence to language-specific/cultural conventions

Languages differ in spatial encoding—some use absolute directions (like Guugu Yimithirr), others relative frames. Models struggle with this diversity, defaulting to English-like reasoning.

### temporal reasoning

The "Perfect Times" quadrilingual benchmark (English, Italian, Russian, Japanese) shows models struggle with human-like temporal causal reasoning. Languages encode time differently—English/Swedish use distance metaphors ("long meeting"), Spanish uses quantity ("big meeting"), Mandarin employs vertical metaphors. These correlate with duration judgments in both humans and models.

Cross-linguistic differences in temporal expression affect model performance, suggesting time representation in embeddings varies by training language distribution.

## practical implications for AI systems

### cross-lingual transfer and alignment

Critical challenges identified in recent work:

- Models correctly answer questions in English but fail with identical questions in Swahili, Igbo, or other low-resource languages
- Performance consistently declines from English to local languages to other foreign languages
- English-centric models display 3-4% average performance advantage across languages

Proposed solutions include PreAlign (establish multilingual alignment before pre-training), AlignX (advance multilingual representation alignment), and code-switched fine-tuning to bridge gaps between English and low-resource languages.

### cultural bias and worldview

Training corpora bias creates measurable "worldview" in models. English prompts align with Western liberal viewpoints; the same model shows different perspectives in different languages. Popular LLMs exhibit Western cultural bias, favoring self-expression values from English-speaking/Protestant European countries [@culturalbias2024].

Models show greater cultural alignment when prompted in a culture's dominant language. Misalignment is more pronounced for underrepresented personas and culturally sensitive topics.

Anthropological prompting—leveraging anthropological reasoning for cultural alignment—improves alignment for 71-81% of countries/territories [@culturalbias2024]. This suggests linguistic relativity in models isn't just about semantic differences but encodes cultural worldviews from training data.

### grammatical structure and reasoning

Program-of-Thought (PoT) fine-tuning substantially enhances multilingual reasoning, outperforming Chain-of-Thought (CoT). This suggests reasoning scaffolding interacts with linguistic structure.

Grammatical gender effects appear in models: gender bias is pronounced in gendered languages (Spanish, French), with male-to-female ratios in training data ranging 4:1 to 6:1. Novel methodology leverages LLMs' contextual understanding to analyze gender representation, revealing how grammatical categories propagate social biases.

## historical foundations

### ancient philosophy

The idea that [[thoughts/Language|language]] and thought are intertwined is ancient. [[thoughts/Plato|Plato]]'s dialogue _Cratylus_ explores whether conceptions of reality are embedded in language—a question central to [[thoughts/Epistemology|epistemology]]. Plato may have held that the world consists of eternal ideas and language should represent these accurately, though his _Seventh Letter_ claims ultimate truth is inexpressible in words.

Following Plato, St. Augustine argued language was merely labels applied to pre-existing concepts. This view persisted through the Middle Ages. For [[thoughts/Philosophy and Kant|Kant]], language was one of several methods humans use to experience the world, but not constitutive of thought itself.

### german romantic philosophers

During the late 18th and early 19th centuries, ideas of different national characters (_Volksgeister_) motivated German romanticism and early ethnic nationalism. This period saw philosophers like [[thoughts/Philosophy and Nietzsche|Nietzsche]] later developing ideas about how language shapes and constrains thought.

Johann Georg Hamann discussed the "genius" of a language, suggesting a people's language affects their worldview: "The lineaments of their language will thus correspond to the direction of their mentality."

Wilhelm von Humboldt (1820) proposed language as the fabric of thought—thoughts produced through internal dialog using native grammar. He held that linguistic diversity is a diversity of worldviews, with languages creating individual perspectives through lexical and grammatical categories, conceptual organization, and syntactic models.

### american anthropology

Franz Boas challenged the idea that some languages are superior to others, stressing equal worth of all cultures and languages. He held that all languages can express the same content by different means, though the form of language is molded by culture.

Edward Sapir drew on Humboldt but explicitly rejected strong [[thoughts/Determinism|determinism]]: "It would be naïve to imagine that any analysis of experience is dependent on pattern expressed in language." He emphasized that language and culture are not intrinsically associated.

Benjamin Lee Whorf, Sapir's student, studied Native American languages and became associated with the "linguistic relativity principle":

> We dissect nature along lines laid down by our native language ... all observers are not led by the same physical evidence to the same picture of the universe, unless their linguistic backgrounds are similar.

Whorf's claims about Hopi time conceptualization—that Hopi treats time as a single process rather than countable instances—became emblematic but remain contested.

### formalization and testing

Roger Brown and Eric Lenneberg reformulated linguistic relativity as testable hypothesis through color perception experiments. They tested whether codability of color terms affects memory and recognition—Zuni speakers grouping green/blue had difficulty distinguishing within that category.

Brown later formulated the well-known weak/strong versions:

- **Weak**: Structural differences between languages are paralleled by nonlinguistic cognitive differences
- **Strong**: One's native language strongly influences or fully determines the worldview acquired

### universalism and critiques

The 1960s emphasis on universal grammar—particularly Chomsky's work on innate linguistic structures—disfavored linguistic relativity. Universalists argued for largely innate structures where differences between languages are surface phenomena. From the 1960s–1980s this view dominated, and relativity was often ridiculed.

Ekkehart Malotki challenged Whorf's Hopi time claims with extensive data, though relativists maintain Whorf's actual claim concerned different conceptualization rather than absence of time concepts.

Steven Pinker argues thought is independent of language, proposing "mentalese"—a language of thought distinct from natural language. Relativists have accused universalists of misrepresenting Whorf through strawman arguments.

From the late 1980s, new research found broad support for non-deterministic versions of linguistic relativity. Current consensus: language influences certain cognitive processes in non-trivial ways, but other processes develop from connectionist factors.

see also: Language-Specific Neurons (ACL 2024, arXiv:2402.16438), Cross-lingual Transfer of Reward Models (arXiv:2410.18027), COMFORT spatial reasoning benchmark, Perfect Times temporal reasoning benchmark.

## dispute

The debate between universalists (Chomsky's universal grammar) and relativists finds new expression in language models. Universal grammar predicts models should converge on shared representations; linguistic relativity predicts language-specific processing. Both occur.

Models exhibit:

- **Universal aspects**: Shared cross-lingual neuron activation for equivalent concepts, transfer learning success
- **Relativistic aspects**: Language-specific neurons, performance gaps across languages, culture-dependent outputs

This dual nature mirrors human cognition—some aspects universal (perceptual universals in color, basic spatial relations), others shaped by linguistic structure (fine-grained color discrimination, frame-of-reference preferences, temporal reasoning).

### emergent abilities and linguistic structure

The "emergent abilities" debate—whether capabilities appear suddenly at scale or result from metric choices—connects to linguistic relativity. Wei et al. (2022) defined emergent abilities as unpredictable capabilities at scale; Schaeffer et al. (2023) argued these may be measurement artifacts.

But language-specific behavior in models suggests genuine emergence: compression during pre-training forces multilingual alignment that wasn't explicitly programmed. Models develop internal "hub" languages for reasoning without being told to. Language mixing patterns emerge from interaction between task difficulty, script types, and model architecture.

These aren't simple scaling effects—they're qualitative changes in how models represent and process linguistic information, analogous to how human language acquisition shapes cognitive development.

### programming languages and notation

Kenneth E. Iverson argued notations are tools of thought; more powerful notations aid thinking. Paul Graham's "Blub paradox" observes that thinking in a language can obscure awareness of more expressive ones.

In AI systems, this manifests as models defaulting to English or Latin-script reasoning even when prompted in other languages. The training distribution creates a "notational preference" that shapes internal computation—a computational version of Whorf's claim that we dissect nature along lines laid down by our languages.
