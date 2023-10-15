---
id: LLM Inferface
tags:
  - seed
  - openllm
title: LLM Interface
---
# Problem

Currently, it seems the API is way too complicated from internal feedback. We want to simplify and provide a simple abstraction on top.

Current UX:

```bash
openllm start llama ... --backend vllm
```

OpenLLM provides both options to either use OpenLLM Client or use the OpenAI compatible API to interact with this LLM.

The following includes all available APIs/SDK that OpenLLM currently offers:

1. Auto classes
> If you aren't sure what LLM to use, then the auto classes API will map the desired backend to the LLM classes itself, via `openllm.[AutoLLM|AutoVLLM|AutoTFLLM|AutoFlaxLLM].for_model`:
```python
pt_llm = openllm.AutoLLM.for_model('llama', model_id='meta-llama/Llama-2-13b-chat-hf')
vllm_llm = openllm.AutoVLLM.for_model('llama', model_id='meta-llama/Llama-2-13b-chat-hf', max_new_tokens=4096)
```

`AutoLLM` contains a `_model_mapping` of `type[openllm.LLMConfig]` to `type[openllm.LLM]` (note that these `openllm.LLM` class are backend specific.) 

`for_model` is responsible for:
- `ensure_available=True|False` will check if the model weights is available under BentoML model store
- Choose the correct `type[openllm.LLM]` backend to load a `openllm.LLM` instance

> [!IMPORTANT]
> OpenLLM's Auto classes therefore is backend-dependent. This is similar to transformers' Auto classes design.

2. `openllm.LLM`
> `openllm.LLM` is a ref class that holds a lot of information that allows access directly to Runner creation, as well as the model and tokenizer.

`openllm.LLM.from_pretrained`: If you know exactly what model to use, then these `openllm.LLM` class can be used directly to load the ref into memory:

```python title="example.py"
pt_llama = openllm.Llama.from_pretrained('meta-llama/Llama-2-13b-chat-hf')

vllm_llama = openllm.VLLMLlama.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
```

Note that this draws similarity with transformers' `from_pretrained` API. In fact, this is pretty much drop-in replacement for it! `openllm.LLM.from_pretrained` supports all attrs from transformers + openllm specific attributes.

One different behaviour with `openllm.LLM.from_pretrained` is that we don't load the model into the memory. Instead, it is then lazily loaded via `openllm.LLM.model`

> The behaviour of loading model, tokenizer from `openllm.LLM` are lazy, meaning it will only be loaded once `openllm.LLM.model` and `openllm.LLM.tokenizer` is called.

> [!NOTE]
> `openllm.Runner` currently relies on this side effect to load model
>
```python title="openllm-python/src/openllm/_llm.py" {11}
def llm_runnable_class(...) -> type[bentoml.Runnable]:
  class _Runnable(bentoml.Runnable):
	  SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'amd.com/gpu', 'cpu')
	  SUPPORTS_CPU_MULTI_THREADING = True
	  backend = self.__llm_backend__
	  
	  def __init__(__self: _Runnable):
	    # NOTE: The side effect of this line is that it will load the
		  # imported model during runner startup. So don't remove it!!
		  if not self.model: raise RuntimeError('Failed to load the model correctly (See traceback above)')
		  if self.adapters_mapping is not None:
		    logger.info('Applying LoRA to %s...', self.runner_name)
		    self.apply_adapter(inference_mode=True, load_adapters='all')
```

3. `openllm.Runner`
> `openllm.Runner` is a light factory that facilitate backend on top of the Auto class API to create BentoML's compatible Runner API.
```python title="service.py"
runner = openllm.Runner('llama', model_id='meta-llama/Llama-2-13b-chat-hf', backend='vllm')
```

The following illustrates the relationship among these APIs:

```mermaid
---
title: Current API state
---
stateDiagram-v2

direction LR

state "openllm.Runner" as r
state "openllm.LLM" as llm
state "__init__" as rin

state "openllm.AutoLLM" as ptllm
state "openllm.AutoVLLM" as vllm
state "openllm.AutoTFLLM" as tfllm
state "openllm.AutoFlaxLLM" as flaxllm

state "type[openllm.Llama]" as ptllama
state "type[openllm.VLLMLlama]" as vllmllama
state "type[openllm.TFLlama]" as tfllama
state "type[openllm.FlaxLlama]" as flaxllama

state ".from_pretrained(model_id, ...)" as fp

state if_state <<choice>>

r --> if_state : ('llama', backend=..., **kwargs)

state rin {
	direction LR
	if_state --> ptllm : backend='pt'
	if_state --> vllm : backend='vllm'
	if_state --> flaxllm : backend='flax'
	if_state --> tfllm : backend='tf'
	state AutoClasses {
		direction LR
	
		ptllm --> ptllama: .for_model('llama', **kwargs)
		vllm --> vllmllama: .for_model('llama', **kwargs)
		flaxllm --> flaxllama: .for_model('llama', **kwargs)
		tfllm --> tfllama: .for_model('llama', **kwargs)
	
		ptllama --> fp
		vllmllama --> fp
		flaxllama --> fp
		tfllama --> fp
	
		fp --> llm
	}
	
	note left of llm: The actual model will be loaded once 'openllm.LLM.model' is accessed
	
	llm --> [*]: ".to_runner()"
}
```
> [!IMPORTANT]
> `openllm.Runner` will provides the most feature-rich APIs to interact with different backends and offers an simple API for people who are familiar with, know of BentoML architecture.

## Issues

While `openllm.Runner` is the most feature-full, people who don't know about BentoML's architecture might not understand what Runner is, as Runner is relatively low-level. Thus, OpenLLM should provide a higher level API that abstract some of these concepts and logics away from the users.

The Auto classes tries to solve this, but `openllm.LLM` doesn't contain all features that Runner has (runner has specific implementation for continuous batching, vllm support, LoRA adapters dynamic mounting, etc.), plus some drawback:

- It is relatively verbose (backend-dependent, therefore users will need to explicit write which Auto classes to use) (this is intended when I first implemented this.)
- The `openllm.LLM` APIs are inherently fragmented (text generation signature are not consistent, which leads to inconsistency between function signature for runners.)
	- PyTorch implementation are sync, whereas vLLM implementation are async
- Currently the most optimized path is implemented in Runner, not `openllm.LLM`

## Target user group


## Goal
- automatic backend detection:
	- if `vllm` is available, default to use vLLM implementation, otherwise fallback to PyTorch implementation
- Async
	- `generate_iterator` should be async?

# Proposal

`openllm.LLM` instead of a bootstrap class, 