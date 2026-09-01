---
author: Claire Cheng
date: '2025-09-16'
description: Human annotation in foundation-model training, evaluation, and labor
id: tangents-1
modified: 2026-09-01 09:15:00 GMT-04:00
source: https://tsfm.ca/schedule
tags:
  - ml
  - tangents
  - tsfm
title: Humans of AI
---

Claire Cheng's TSFM tangent was titled [Humans of AI: Human Annotations for LLMs](https://tsfm.ca/schedule). The talk focused on the people who produce demonstrations, preference labels, safety data, and evaluations for language models.

## why

Most pretraining targets come from the data itself. Human annotation enters at more specific boundaries:

- people select and clean training data;
- supervised fine-tuning uses written demonstrations;
- preference training uses comparisons between model outputs;
- evaluations use human judgments when a program cannot score the answer.

InstructGPT used demonstrations and ranked comparisons from hired labelers. Its authors also state the limit directly: the model was aligned to the preferences of a particular group of labelers and researchers. That scope is narrower than a universal account of human values. See [Ouyang et al.](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf) and [[thoughts/Alignment]].

## data labelling

An annotation contract needs a task definition, examples, a review sample, and a rule for disagreements. One useful measure is the held-out acceptance rate

$$
\widehat q=\frac{\text{accepted audited items}}{\text{audited items}}.
$$

That ratio needs its denominator and sampling rule. For tasks with several reasonable labels, report agreement and the adjudication process as well. A single pass rate can hide ambiguous instructions or a reviewer who repeats the same mistake as the annotator.

## labor boundary

Annotators may work directly for a model developer or through an outside contractor. Crowd platforms and delivery centers use other employment models, so a company list cannot describe every worker on a project.

The concrete risks are pay below a local living wage, unpaid qualification work, insecure contracts, workplace surveillance, weak appeal processes, and exposure to traumatic material. Fairwork documented several of these conditions in its [2023 cloudwork ratings](https://fair.work/en/fw/publications/fairwork-cloudwork-ratings-2023-work-in-the-planetary-labour-market/). Those findings are scoped to the workers and firms studied. A global annotation labor model needs separate evidence.

The 2025 talk named Micro1, Mercor, Sama, Appen, and Scale AI as examples. That list records the lecture topic. It should be read as a 2025 snapshot.
