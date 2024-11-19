---
id: OpenLLM 0.7 Problem statement
tags: []
date: "2024-08-06"
title: OpenLLM 0.7 Problem statement
---

## P1 (Aaron: preferred)

### If I want to change the model id, or customize engine config arguments of vllm for a given model locally, how do I do it?

Let say i want to run a [fine-tuned version of llama3](https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored):

1. `openllm model get llama3:8b` yields the following:

```yaml
tag: llama3:8b
repo:
  name: default
  url: https://github.com/bentoml/openllm-models@main
  path: /Users/repo/.openllm/repos/github.com/bentoml/openllm-models/main
path: /Users/repo/.openllm/repos/github.com/bentoml/openllm-models/main/bentoml/bentos/llama3/8b-instruct-fp16-8638
model_card:
  apis:
    /api/chat:
      input:
        messages: array
        model: string
        max_tokens: integer
        stop: array
      output: string
    /api/generate:
      input:
        prompt: string
        model: string
        max_tokens: integer
        stop: array
      output: string
  resources:
    gpu: 1
    gpu_type: nvidia-tesla-l4
  envs:
    - name: HF_TOKEN
  platforms:
    - linux
```

- add `-o json` => `bentopath=$(openllm model get llama3:8b -o json | jq '.engine_config.model')`

2. `cd /Users/repo/.openllm/repos/github.com/bentoml/openllm-models/main/bentoml/bentos/llama3/8b-instruct-fp16-8638` and edit `src/bento_constants.py` and change `CONSTANT_YAML` `engine_config.model` to the desired model id. For example:

```python
CONSTANT_YAML = '''
engine_config:
  dtype: half
  max_model_len: 2048
  model: Orenguteng/Llama-3.1-8B-Lexi-Uncensored
extra_labels:
  model_name: meta-llama/Meta-Llama-3-8B-Instruct
  openllm_alias: 8b,8b-instruct
project: vllm-chat
service_config:
  name: llama3
  resources:
    gpu: 1
    gpu_type: nvidia-tesla-l4
  traffic:
    timeout: 300
'''
```

3. `openllm model run llama3:8b` will use the new model id **LOCALLY**.

Note that this change will be lost the next time you run `openllm repo update`.

### How do I contribute to openllm?

see [contributing](https://github.com/bentoml/openllm-models/blob/main/DEVELOPMENT.md).

Though this means for ppl who are brand new to bentoml, they probably don't know how to create a bento or understand the life cycle of bento.

Is there a easier way to contribute to openllm?

## P1.1 (Aaron: preferred)

`openllm model get-src llama3:8b <target-dir>`

cd `<target-dir>` as a bento project

- learn bento project
- to serve it, do `bentoml serve .`

## P2 (scope creep)

### `openllm run -f model.yaml`

```yaml
"llama3:8b-lexi":
  project: vllm-chat
  service_config:
    name: phi3
    traffic:
      timeout: 300
    resources:
      gpu: 1
      gpu_type: nvidia-rtx-3060
  engine_config:
    model: Orenguteng/Llama-3.1-8B-Lexi-Uncensored
    max_model_len: 4096
    dtype: half
```

- create the bento -> openllm run

- where to save the bento?

  - `openllm-models/custom-models/.gitkeep`

- options to create a PR automatically to `openllm-models`?
  - `openllm repo create-pr <default> <llama3:8b-lexi>`
    - user: repo -> repo/openllm-models
  - tell ppl to add the model.yaml to recipe.yaml
