---
date: '2024-02-12'
description: how instructions, examples, and intermediate answers shape language-model outputs, with chain-of-thought and least-to-most prompting.
id: prompt engineering
modified: 2026-10-13 09:04:01 GMT-04:00
seealso:
  - '[[thoughts/linguistic]]'
  - '[[thoughts/NLP]]'
tags:
  - seed
  - ml
title: Prompt engineering
---

Prompt engineering is choosing the instructions, context, and examples given to [[thoughts/LLMs]]. During ordinary prompting, the model's weights stay fixed. The prompt supplies information about the task and the answer we want. [@brown2020languagemodelsfewshotlearners]

What counts as a better prompt depends on what the output must do. For a date extractor, "find the dates" leaves the format and handling of missing information open. A more specific instruction would be: "Return a list of dates explicitly stated in this passage. Use ISO format when the day, month, and year are unambiguous; otherwise, preserve the original wording. Return an empty list if no dates appear." We can then check whether the output follows those rules on dated, ambiguous, and undated passages.

[[thoughts/zero-shot learning|Zero-shot prompting]] describes how many demonstrations appear in the prompt: none. A zero-shot prompt can still be carefully designed. Adding solved examples makes the setup one-shot or few-shot.


## CoT prompting

Chain-of-thought prompting supplies examples with intermediate reasoning before the final answer. In Wei et al.'s experiments, this improved arithmetic, commonsense, and symbolic reasoning on the models and tasks tested. [@wei2023chainofthoughtpromptingelicitsreasoning]

For example, a demonstration might solve: "Three bags contain four marbles each. After giving away two marbles, how many remain?" The answer shows $3 \times 4 = 12$, then $12 - 2 = 10$. A new problem follows that demonstration. The intermediate steps become part of the context from which the model generates its next tokens.

Kojima et al. also studied zero-shot CoT: asking the model to "think step by step" without worked examples. The number of demonstrations and the request for intermediate reasoning are separate choices. [@kojima2023largelanguagemodelszeroshot]

A readable explanation can still conceal what influenced an answer. Turpin et al. found that biased prompts could change model answers while the generated explanations omitted the bias. Check the calculation or evidence independently. [@turpin2023languagemodelsdontsay]

## Least-to-most prompting

Least-to-most prompting first decomposes a problem into subproblems, then solves them in order. In Zhou et al.'s two-stage procedure, each solution is included in the prompt for the next subproblem. Examples demonstrate both decomposition and solving. [@zhou2023leasttomostpromptingenablescomplex]

Consider the task of joining the last letters of "plant, rain, soil." Decompose it into the prefixes "plant," "plant, rain," and "plant, rain, soil." The answers are "t," "tn," and "tnl." At each step, the previous answer is available, so the model only needs to append the next letter. This reuse of earlier answers is the part missing from an instruction that merely asks for a list of subproblems.
