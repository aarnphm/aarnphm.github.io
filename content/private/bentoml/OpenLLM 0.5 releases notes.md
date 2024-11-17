
OpenLLM has undergone a significant upgrade in its v0.5 release to enhance compatibility with the BentoML 1.2 architecture. The CLI has also been streamlined to focus on delivering the best experience for deploying open-source LLMs to production. However, version 0.5 introduces breaking changes.
## Breaking changes, and the reason why.

After releasing version 0.4, we realized that while OpenLLM offers a high degree of flexibility and power to users, they encountered numerous issues when attempting to deploy these models. OpenLLM had been trying to accomplish a lot by providing support for different backends (mainly PyTorch for CPU inference and vLLM for GPU inference) and accelerators. Although this provided users with the option to quickly test on their local machines, we discovered that this was actually the crux of the user story. The difference between local and cloud deployment made it difficult for users to understand and control the packaged Bento to behave correctly on the cloud. Additionally, supporting different backends meant there were many different code paths to maintain, further increasing technical debt. Thus, I decided that 0.5 should bring a fresh restart to the project.
## Architecture changes and SDK.

For version 0.5, we have decided to reduce the scope and support the backend that yields the most performance (in this case, vLLM). This means that pip install openllm will also depend on vLLM. In other words, we will currently pause our support for CPU going forward.
This also means that we are going to stop exposing the Python SDK. All interactions with the servers going forward will be done either through clients (i.e., BentoML's Clients, OpenAI, etc.).

> [!NOTE] Users depending on `openllm.LLM`
> `openllm.LLM` is vassly different v0.4.x internally. If your old service still depends on the older BentoML service architecture, make sure to pass in `IMPLEMENTATION=deprecated` to keep your service from breaking. We encourage you to upgrade your service to the BentoML 1.2 architecture at your convenience.

Additionally, the internal service have been migrated to [BentoML's 1.2](https://bentoml.com/blog/introducing-bentoml-1-2) and we have cleaned up a lot of generated boilerplate, which makes the [core](https://github.com/bentoml/OpenLLM/tree/main/openllm-python/src/_openllm_tiny) a lot more compact and DRY.
## CLI

CLI has now been simplified to `openllm start` and `openllm build`

### openllm start

`openllm start` will continue to accept HuggingFace model id for supported model architectures:

```bash
openllm start microsoft/Phi-3-mini-4k-instruct --trust-remote-code
```

> [!NOTE]
> For any models that requires remote code execution, one should pass in `--trust-remote-code`

`openllm start` will also accept serving from local path directly. Make sure to also pass in `--trust-remote-code` if you wish to use with `openllm start`

```bash
openllm start path/to/custom-phi-instruct --trust-remote-code
```

Finally, for private models, we kindly ask users to save the model to [BentoML's model store](https://docs.bentoml.com/en/latest/guides/model-store.html#model-store). You can then use the model directly with `openllm start`

```bash
openllm start my-private-model
```

For quantized model, make sure to pass in `--quantize <quantization-scheme>`:

```bash
openllm start casperhansen/llama-3-70b-instruct-awq --quantize awq
```

See `openllm start --help` for more information

### openllm build

`openllm build` will now be significantly lighter since we have skipped the serialization steps into the generated Bento. In previous versions, OpenLLM would copy the local cache of the models into the generated Bento, resulting in duplications on local machines. From v0.5 going forward, models won't be packaged with the Bento and will be optimized during deployment on BentoCloud.

```bash
openllm build microsoft/Phi-3-mini-4k-instruct --trust-remote-code
```

For your local model, make sure to save it to BentoML model store before `openllm build`

```bash
openllm build casperhansen/llama-3-70b-instruct-awq --quantize awq
```

See `openllm build --help` for more information

## What's next?

Currently, OpenAI's compatibility will only have the `/chat/completions` and `/models` endpoints fully tested and supported. We will continue bringing `/completions` as well as function calling support soon, so stay tuned.

As far as I know, all upstream integrations with `openllm.LLM` are considered broken, so we will submit all upstream integrations for PR fixes. However, all integrations with `openllm.client` will work as normal.

For the OpenLLM "0.6" release, we intend to bring a more concise set of supported models, with prebuilt Bentos and optimization baked in, to ensure the user's path to deployment to the cloud is as smooth as possible.

Thank you for your continued support and trust in us. We would love to hear more of your feedback on the releases.