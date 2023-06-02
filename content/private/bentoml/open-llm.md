---
id: "open-llm"
tags:
  - "machinelearning"
  - "bentoml"
---


# OpenLLM-Server

OpenLLM-Server is a server for running LLM models in production. It is built on top of [BentoML](https://bentoml.com)

# To install it, run `pip install open-llm-server`

By default, it supports all of the open-source LLM:
- Dolly -> databricks/dolly-v2-7b (figure out revision for transformers)
- Flan-T5
- Bloom
- StableLM
- gpt-neox, and more
- llama.cpp + alpaca.cpp + GPT4All + Vicuna (All having the same cpp binding) (git hash)

Start the server to serve the models locally:
```bash
open-llm-server start flan-t5,llama

open-llm-server start flan-t5
```

Users can also run some of the prebuilt server with any container engine with `open-llm-server run`:
```bash
# 1
docker pull openllm-server/llama:cpu

docker run -p ... openllm-server/llama:cpu

# 2
open-llm-server run-container llama --target-device=cpu

open-llm-server run-container bloom --target-device=cpu --container-engine=podman
```

It will run and serve the model server.


To interact with the server, use cURL, any HTTP Client, or `open-llm-client`:
```python

import open_llm_client

client = open_llm_client.create('http://localhost:5000')
```

To `/chat` with your model, use:

```python
client.chat('Hello, how are you?')
client.chat(prompt='Hello, how are you?', temperature=0.5, top_k=3, top_p=0.15, stop_sequence='--')

client.async_chat
```

It will returns a JSON response:
```json
{
    "message": "Hello, how are you? I am fine, thank you. How are you?",
    "configuration": {
        "prompt": "Hello, how are you?",
        "temperature": 0.5,
        "top_k": 3,
        "top_p": 0.15,
        "stop_sequence": "--"
    }
}
```

To `/complete` a text, use:

```python

client.complete('Hello, how are you? My name is ...')
```
The response will be:
```json
{
    "choices": [ "Hello, how are you? My name is ...", "Hello, how are you? my name is Bento, and I'm a bento box" , "" ]
    "configuration": {
        "prompt": "Hello, how are you? My name is ...",
        "temperature": 0.5,
        "top_k": 3,
        "top_p": 0.15,
        "stop_sequence": "--"
    }
}
```

To create a text `/embed`-dings, one can do 

- Embedding represent the meaning of text as a list of numbers. (can be use to compare for similarity)

```python
embedding = client.embed(['Hello, how are you?', 'soup is my favorite food'])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(embedding[0], embedding[1])) # 0.8
```

This means your service will offer three endpoints:
```bash
/chat
/complete
/embed (create text embedding)
```

> TODO: `/chat-sse`

By default, `open-llm-server` will start a HTTP server on port 5000, and it can start multiple models by specifying its name separated by comma.

`start` will call a BentoML server underneath, and will import model if said weights is not found under the `BENTOML_HOME` directory


The full list of arguments:
-> runners name to match with model name (BENTOML_CONFIGURATION)

- `--temperature 0..1`: specify how random the model output should be
    `--temperature 0` will make model deterministic, and `--temperature 1` will make it completely random

- `--top_k <int>`: specify how the model to pick the next token from top `k` tokens in the list, sorted by prob
    - `--top_k 3` will choose the top 3 tokens

- `--top_p 0..1`: similar to `top_k`, but it based on the sum of their probabilities
    - `--top_p 0.15` will pick the top `p` tokens that adds up to 15% of the total probability.

- `--stop_sequence --`: useful for prompting, to determine a string that tells the model to stop generate more content 
    - Currently only GPT-NeoX supports this
    - prob just move to `--opt`

- `--frequency_penalty 0..1`: penalize tokens that already appreared in preceeding text (including prompt) and scales based on how many times token has appeared

- `--presence_penalty 0..1`: applies penalty regardless of frequency, as long as the token has appeared b4, it will be penalized

- `--prompt /path/to/text | "longtext in here"`: A pre-prompt acts as a warmup phase for said models.

optimization arguments (P1)

- `--lora`: Apply LoRA adapter to supported model
    - https://github.com/microsoft/LoRA
    - Low-Rank Adaptation of LLM
        - reduce # of trainable parameters by learning pairs of rank-decomposition matrices while freezing original weights

- `--perflexity`: Show perflexity (performance) of the language model (P1)
    - mainly for metrics

Other specific model options:

Create each model configuration Pydantic model -> convert to CLI

> pydantic to cli (generate the envvar) -> generate the CLI args

```bash
OPEN_LLM_CONFIGURATION=/path/to/default.yaml open-llm-server start

# generated
OPEN_LLM_STOP_SEQUENCE=0.234 OPEN_LLM_BLOOM_FRQUENCY_PENALTY=0.123 open-llm-server start bloom
```

> An unknown field will be thrown as an ValueError

Server-related args

- `--grpc`: To start with a gRPC server instead of HTTP
    - If multiple models are specified, the following `--deivce 'flan-t5:cpu,llama:gpu:0'` or `--device 'flan-t5:cpu' --device 'llama:gpu:0'` are accepted
- All arguments that is pass through a BentoML server


## Internal implementation

Each LLM model is contained within a `LLMModel` class, which extends `bentoml.Model`:

```python
from open_llm_server.llama import get_runner

llama_runner = get_runner('llama', top_k=3, top_p=0.15)
```

```python
from open_llm_server import Registry

def import_model(model_name: str) -> bentoml.Model:
    # import model here
    # This will be called in start()
    ...

def get_runner(model_name: str):
    # create a bentoml.Runner
    try:
         model = bentoml.models.get(model_name)
    except bentoml.exceptions.NotFound:
         model = import_model()

    runner = model.to_runner()

    if embedded:
        runner.init_local()

    return runner
    ...

class LLMRunnable(bentoml.Runnable, ABC):

    def __init_subclass__(self, model_name: str | None = ...): ...

    def process_cli_args(self, **args) -> dict[str, str]: ...

    @bentoml.Runnable.method()
    def complete(self, data): ...

    @bentoml.Runnable.method()
    def embed(self, data): ...

    @bentoml.Runnable.method()
    def chat(self, data): ...


    # def to_runner(self, embedded: bool = False):
    #     try:
    #        model = bentoml.models.get(self.model_name)
    #     except bentoml.exceptions.NotFound:
    #        self.import_model()
    #     runner = bentoml.Runner(self ,...)
    #
    #     if embedded:
    #         runner.init_local()
    #     return runner



# P1
# @Registry.add_model(model_name='llama.cpp')
# class LlamaCppModel(LLMModel):
#     ...
#
# @Registry.add_model(model_name='alpaca.cpp')
# class AlpacaCppModel(LLMModel):
#     ...
#
# @Registry.add_model(model_name='flan-t5')
# class FlanT5Model(LLMModel):
#     ...
```

<!-- These models will then be managed by a `Registry` class that keeps track of the supported model -->
<!---->
<!-- To add a new model -->
<!---->
<!-- ```python -->
<!---->
<!-- from open_llm_server import Registry, LLModel -->
<!---->
<!-- @Registry.add_model(model_name='new-model') -->
<!-- class NewModel(LLMModel): -->
<!--     ... -->
<!-- ``` -->

-> dynamic add or fork?

To start a server programmatically:

```python

import open_llm_server as olm

olm.start(model="...", **kwargs)
```

The pseudo implementation for `start`:

```python
def start(model, device, ...):

    runners = []
    runner_map = {}
    for model in models:
        runners.append(ref.to_runner())

    for runner in runners:
        runner_server = RunnerServer(...)  # use start-runner-server
        runner_map[runner_name] = runner_server.address
        runner_server.start()

    http_server = HTTPServer(..., runner_map=runner_map)  # use start-http-server

    bentoml.HTTPServer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_llm_server/service.py:svc")), working_dir=os.path.dirname(os.path.abspath(__file__)))

    http_server.start()
```

To quickly create a runner from given sets of models, to use within your BentoML service:
```python

from open_llm_server.llama import get_runner as get_llama_runner
from open_llm_server.bloom import get_runner as get_bloom_runner

llama_runner = get_llama_runner(top_k=3, top_p=0.15)
bloom_runner = get_bloom_runner(freq_penalty=0.123)


svc = bentoml.Service(..., runners=[llama_runner, bloom_runner, my_other_runner])
```

This means it will inherit all features and integration bentoml offers, including adding this into a FastAPI/ASGI app


