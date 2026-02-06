---
created: '2025-10-01'
date: '2025-10-01'
description: Language shapes thought—classical Sapir-Whorf hypothesis finds unexpected validation in language models, where linguistic structure creates measurable emergent properties.
id: linguistic relativity
modified: 2025-12-08 21:55:01 GMT-05:00
tags:
  - pattern
title: linguistic relativity
---

the sapir-whorf hypothesis—that language shapes thought—spent most of the 20th century getting beaten up by linguists. chomsky's universal grammar won. language was surface phenomena, thought was universal mentalese underneath.

then we built [[thoughts/LLMs|language models]] and watched them develop the exact cognitive distortions whorf described.

give gpt-4o mini the same culturally salient prompt across 13 typologically diverse languages and you get systematically different reasoning patterns [@ray2025linguistic]. not translation differences—different THOUGHTS. the models don't translate concepts between languages; they maintain separate [[lectures/411/notes#Isomorphisms, adjoints, invariant subspaces|subspaces]] for each input language, like distinct cognitive frames.

here's what's wild: despite multilingual training, large reasoning models predominantly "think" in one or two hub languages—usually english or chinese—regardless of what you prompt them in.[^language] ask deepseek-r1 a question in swahili and its internal monologue {{sidenotes[still runs]: the training distribution creates gravitational wells in representation space}} in english or chinese before translating back.

[^language]: probably bc stochastically these languages dominate the training corpus, creating stronger attractors in the model's latent space

linguistic relativity operates simultaneously across training corpus composition, prompt language, internal representation space, output generation. whorf claimed we dissect nature along lines laid down by our native languages. language models dissect prompts along lines laid down by their training distributions—and those lines are measurable through neuron activation patterns, embedding geometries, cross-lingual transfer performance.

## how linguistic structure creates behavior

crack open a multilingual model with [[thoughts/sparse autoencoder|sparse autoencoders]] and you find something strange: both language-agnostic and language-specific neurons coexisting in the same network. prompts with identical meanings across languages activate similar circuits—but the model also maintains dedicated neurons for vocabulary, grammar, idioms in each language.

this isn't planned. compression during pre-training forces models toward shared cross-lingual representations (cheaper to store one concept than seven), but the training dynamics also carve out language-specific regions. cross-lingual neuron overlap correlates strongly with zero-shot transfer performance; when overlap degrades during certain pre-training phases, so does the model's ability to transfer knowledge between languages.

grammatical features like number and semantic roles don't distribute globally—they localize in specific embedding regions. the model's internal organization mirrors the structure of the languages it learned.

watch a model reason through a hard problem and you'll see it mix languages mid-thought. test across 15 languages × 7 difficulty levels × 18 subject areas and every factor influences language mixing in reasoning traces [@languagemixing2025]. script composition matters most: models have an internal preference for latin script, so they keep slipping into latin-alphabet reasoning even when you prompt them in arabic or mandarin.

force a model to reason in scripts matching the input language and accuracy improves; mismatched scripts degrade performance [@bilingualreasoning2024]. deepseek-r1 and qwq-32b show human-like language mixing in their reasoning chains—switching languages when the problem gets hard, defaulting to their "native" scripts under cognitive load. these patterns emerge from interactions between task difficulty, script types, model architecture. nobody programmed this behavior; it arose from training dynamics, like aphasia patterns in stroke victims.

## where language carves up reality

in the 1950s brown and lenneberg tested whether having words for colors affects how you SEE them. zuni speakers who use one word for green and blue struggled to distinguish shades within that merged category. your language's color vocabulary literally shapes color memory and recognition.

gpt-4 replicates this. prompt it in english vs russian about color perception and you get cross-linguistic variation matching human psycholinguistic patterns. the model has no retina, no perceptual constraints—but training on russian text (which distinguishes синий/blue from голубой/light-blue at the basic color term level) creates the same cognitive distinctions that russian speakers show.

spatial reasoning breaks even harder. some languages encode space through absolute directions—speakers of guugu yimithirr (aboriginal australian language) track cardinal directions constantly, saying things like "there's an ant on your southwest foot." other languages use relative frames ("left/right" depends on perspective). the comfort benchmark {{sidenotes[tests this]: consistent multilingual frame of reference test across languages}} and models fail catastrophically: english dominance in resolving spatial ambiguities, failure to accommodate multiple frames of reference, zero adherence to language-specific conventions.

time works similarly. english and swedish use distance metaphors ("long meeting"), spanish uses quantity ("big meeting"), mandarin employs vertical metaphors (上个月/up-month for "last month"). these aren't just vocabulary differences—they correlate with duration judgments in both humans and models. test temporal causal reasoning across english, italian, russian, japanese and the perfect times benchmark shows models inheriting the temporal reasoning patterns of their training language distributions.

## what this means for building systems

ask a model a question in english and it answers correctly. ask the EXACT same question in swahili or igbo and it fails. performance consistently declines from english → local languages → other foreign languages, with english-centric models showing a 3-4% average performance advantage across all languages.

this isn't just a quality problem—it's an epistemological one. training corpora bias creates measurable "worldview" in models. prompt claude in english and you get western liberal perspectives; prompt the same model in different languages and different {{sidenotes[perspectives emerge]: models show greater cultural alignment when prompted in a culture's dominant language}}. popular llms exhibit western cultural bias, favoring self-expression values from english-speaking and protestant european countries [@culturalbias2024]. misalignment is most pronounced for underrepresented personas and culturally sensitive topics.

anthropological prompting—having the model explicitly reason through cultural context—improves alignment for 71-81% of countries and territories [@culturalbias2024]. but this just papers over the deeper issue: linguistic relativity in models isn't just semantic differences. it's entire cultural worldviews encoded in training distributions, activated by choice of prompt language.

grammatical structure matters too. languages with grammatical gender (spanish, french) show pronounced gender bias in their models, with male-to-female ratios in training data ranging 4:1 to 6:1. the grammatical categories themselves propagate social biases—not through explicit encoding but through the distributions of contexts where gendered words appear.

program-of-thought fine-tuning (using code-like intermediate steps) substantially enhances multilingual reasoning, outperforming chain-of-thought. reasoning scaffolding interacts with linguistic structure in ways nobody fully understands yet.

## how we got here

[[thoughts/Plato|plato]]'s _cratylus_ asked whether reality is embedded in language—whether words capture eternal forms or just conventional labels. his _seventh letter_ claimed ultimate truth is inexpressible in words, suggesting language might constrain rather than convey thought. st. augustine disagreed: language was merely labels applied to pre-existing concepts. this view persisted through the middle ages. for [[thoughts/Philosophy and Kant|kant]], language was one of several tools humans use to experience the world, not constitutive of thought itself.

then came the german romantics. late 18th century, ideas about national characters (_volksgeister_) motivated new thinking about language. johann georg hamann: "the lineaments of their language will thus correspond to the direction of their mentality." wilhelm von humboldt (1820) went further—language as the fabric of thought, with thoughts produced through internal dialog using native grammar. linguistic diversity IS diversity of worldviews; languages create individual perspectives through lexical categories, grammatical structures, syntactic patterns.

across the atlantic, franz boas challenged european assumptions that some languages were superior. all languages can express the same content by different means, though the form of language is molded by culture. edward sapir drew on humboldt but explicitly rejected strong [[thoughts/Determinism|determinism]]: "it would be naïve to imagine that any analysis of experience is dependent on pattern expressed in language."

his student benjamin lee whorf studied native american languages and radicalized the claim:

> we dissect nature along lines laid down by our native language ... all observers are not led by the same physical evidence to the same picture of the universe, unless their linguistic backgrounds are similar.

whorf's claims about hopi time conceptualization—that hopi treats time as a single process rather than countable instances—became the paradigm case. also the most contested. ekkehart malotki later challenged this with extensive data, though relativists maintain whorf's actual claim concerned different conceptualization rather than {{sidenotes[absence]: whether hopi LACKS time concepts vs conceptualizes them differently matters a lot for the strong/weak distinction}} of time concepts.

the 1960s brought chomsky and universal grammar. innate linguistic structures, differences between languages as surface phenomena. relativity got ridiculed for two decades. steven pinker proposed "mentalese"—a language of thought distinct from natural language—and accused relativists of strawmanning whorf.

late 1980s: new research on color perception, spatial reasoning, time conceptualization found broad support for non-deterministic versions. current consensus before llms: language influences certain cognitive processes in non-trivial ways, but doesn't determine them.

## the debate reborn in silicon

chomsky's universal grammar predicted models should converge on shared representations—thought is thought, language is just surface syntax. whorf predicted language-specific processing—different languages create different cognitive patterns.

[[thoughts/LLMs|language models]] do BOTH.

universal aspects: shared cross-lingual neuron activation for equivalent concepts, successful transfer learning between languages, compression forcing convergent representations. relativistic aspects: language-specific neurons for grammar and idioms, systematic performance gaps across languages, culture-dependent outputs activated by prompt language choice.

some things are universal (perceptual basics like color categories, fundamental spatial relations), others shaped by linguistic structure (fine-grained color discrimination, frame-of-reference preferences, temporal reasoning patterns). this mirrors human cognition perfectly—which suggests the old debate missed something. maybe universalism and relativism aren't opposites but different levels of the same system.

here's what's genuinely NEW: compression during pre-training forces multilingual alignment that wasn't explicitly programmed. models develop internal "hub" languages for reasoning without being instructed. language mixing patterns emerge from interactions between task difficulty, script types, model architecture—qualitative changes in how models represent and process linguistic information, analogous to how human language acquisition shapes cognitive development.

these aren't scaling effects. they're [[thoughts/emergent behaviour|emergent properties]] of the training dynamics.

kenneth e. iverson argued notations are tools of thought; more powerful notations aid thinking. paul graham's "blub paradox": thinking IN a language obscures awareness of more expressive ones. in ai systems this manifests as models defaulting to english or latin-script reasoning even when prompted in other languages. the training distribution creates a "notational preference" that shapes internal computation—whorf's claim about dissecting nature along linguistic lines, but now measurable in embedding geometries and neuron activations.
