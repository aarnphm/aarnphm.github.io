---
date: '2024-02-12'
description: zero-shot prompting, the role of instruction tuning, and what the original FLAN experiments establish about generalization.
id: zero-shot learning
modified: 2026-10-13 09:04:01 GMT-04:00
tags:
  - llm
title: zero-shot prompting
---

Zero-shot prompting gives [[thoughts/LLMs|LLMs]] a task without demonstrations of how to solve it. For example: "Classify the sentiment of 'The parcel arrived broken' as positive, negative, or neutral." The sentence is the input to classify. A demonstration would pair another sentence with its correct label. [@brown2020languagemodelsfewshotlearners]

The term describes the prompt used at inference time. It says little by itself about which related tasks or examples appeared during training. It also leaves room for [[thoughts/prompt engineering|prompt engineering]]: the instruction can specify labels, context, and an output format while containing no demonstrations.

## what FLAN showed

In _Finetuned Language Models Are Zero-Shot Learners_, Wei et al. instruction-tuned a pretrained, 137-billion-parameter LaMDA-PT model. FLAN names the resulting model. The study used a collection of 62 datasets, expressed through natural-language instructions. [@wei2022finetunedlanguagemodelszeroshot]

To evaluate transfer, the authors held out whole task clusters during instruction tuning. For example, the checkpoint evaluated on entailment had seen no entailment datasets in that tuning stage. Different evaluation clusters used different checkpoints. This separation concerns instruction tuning; pretraining contamination was examined separately.

With the prompt template selected by development-set performance, zero-shot FLAN beat zero-shot GPT-3 on 20 of 25 datasets. GPT-3 had 175 billion parameters; FLAN had about 78% as many. FLAN also improved over its own untuned baseline. These results support instruction tuning for the tested setup. They give no general rule that small models outperform large ones: in the paper's scaling ablation, instruction tuning reduced held-out performance for models with 8 billion parameters or fewer.

## related terms

- **Few-shot prompting** puts solved examples in the context. The model's weights stay fixed during inference.
- **Instruction tuning** trains model weights on tasks expressed as instructions. A model trained this way can receive either zero-shot or few-shot prompts.
- **Prompt tuning**, in Lester et al.'s usage, learns continuous prompt vectors through backpropagation while keeping the underlying model frozen. Those learned vectors differ from the words chosen when editing a text prompt. [@lester2021powerscaleparameterefficientprompt]
