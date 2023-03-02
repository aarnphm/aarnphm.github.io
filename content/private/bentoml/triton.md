---
id: "triton"
tags:
  - "seed"
  - "machinelearning"
  - "modelserver"
  - "bentoml"
title: "Triton Inference Server with BentoML"
---

<!--toc:start-->

- [BentoML. Triton Inference Server. Choose 2](#bentoml-triton-inference-server-choose-2)
- [What is Triton Inference Server?](#what-is-triton-inference-server)
- [Preparing your model](#preparing-your-model)
- [Create a Triton Runner](#create-a-triton-runner)
- [Packaging BentoService with Triton Inference Server](#packaging-bentoservice-with-triton-inference-server)
- [Conclusion](#conclusion)

<!--toc:end-->

## BentoML. Triton Inference Server. Choose 2

We are seeing a surge in recent months of developments and works on large
language models (LLM) and its applications such as
[ChatGPT](https://chat.openai.com/),
[Stable Diffusion](https://stability.ai/blog/stable-diffusion-v2-release),
[Copilot](https://github.com/features/copilot).

However, deploying and serving LLMs at scale is a challenging task that requires
specific domain expertise and inference infrastructure. A
[rough estimation](https://twitter.com/tomgoldsteincs/status/1600196981955100694?s=20)
of running ChatGPT shows that serving efficiency are critical to make such
models to work at scale. These operations are often known as Large Language
Models Operations (LLMOps). LLMOps, in general, is considered as a subset of
MLOps, which is a set of practices combining software engineering, DevOps, and
data science to automate and scale the end-to-end lifecycle of ML models.

Teams can encounter several problems when running inference on large models,
including:

- _Resource utilisation_: Large models require a significant amount of
  computational power to run, which can be a challenge for teams with limited
  resources. Serving frameworks should utilize all available resources to be
  cost-effective.
- _Model optimization_: Large models often contains a lot of redundant layers
  and parameters that can be pruned to reduce model size and speed up inference.
  A serving framework ideally should be able to provide support for model
  optimization library to aid this process.
- _Serving latency_: Large language models often require complex batching
  strategies to enable real-time inference. A serving framework should be
  equipped with batching strategies to optimize for low-latency serving.

In this blog post, we will be demonstrating the capabilities of BentoML and
Triton Inference Server to help you solve these problems.

## What is Triton Inference Server?

You might wonder, what is Triton Inference Server? Triton Inference Server is a
high performance, open-source inference server for serving deep learning models.
It is designed to handle the variety of deep learning models and frameworks,
such as ONNX, Tensorflow, [TensorRT](https://developer.nvidia.com/tensorrt). It
is also designed with optimization to maximize hardware utilization through
various model execution and efficient batching strategies.

> Triton Inference Server is great for serving large language models, where you
> want a high-performance inference server that can utilize all available
> resources with complex batching strategies.

## What is BentoML?

For those who are not familiar with BentoML, BentoML is an open-source framework
for serving and deploying machine learning models. It provides a high-level API
for defining machine learning models and APIs, and provides a set of

In a nutshell, BentoML provides the capabilities for users to run Triton
Inference Server through BentoML's
[Runner](https://docs.bentoml.org/en/latest/concepts/runner.html) architecture,
via `bentoml.triton.Runner`. This allows users to run Triton Inference Server as
a Runner runtime, with similar APIs as other BentoML's built-in Runners.

```python
triton_runner = bentoml.triton.Runner("triton-runner", model_repository="s3://org/model_repository")
```

We built this integration with Triton Inference Server as a first step in our
progress to improve and optimize BentoML's Runner performance. One of the
reasons why we choose Triton is that the framework is written in C++ and not
Python, therefore it triumphs over its Python counterpart in terms of hardware
utilization and performance.

In order to use the `bentoml.triton` API, users are required to have the Triton
Inference Server
[container image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
available locally.

```bash
docker pull nvcr.io/nvidia/tritonserver:23.01-py3
```

The following PyPI `tritonclient` package is also required:

```bash
pip install "tritonclient[all]"
```

Triton Inference Server evolves around the concepts of model repository, a
filesystem-based persistent volume that contains the models and
[Triton's model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
We will provide a quick walk-through in the section below for you to get
started.

Finally, install BentoML through PyPI:

```bash
pip install bentoml
```

The following section assumes that you have a basic understanding of BentoML
architecture. If you are new to BentoML, we recommend you to read our
[Getting Started](https://docs.bentoml.org/en/latest/tutorial.html) guide first.

You can also find the full example repository
[here](https://github.com/bentoml/BentoML/tree/main/examples/triton_runner).

## Preparing your model

To prepare your model repository under your BentoML project, you will need to
put your model in the following file structure:

```bash
» tree model_repository

model_repository
└── torchscript_yolov5s
    ├── 1
    │   └── model.pt
    └── config.pbtxt
```

Where `1` is the version of the model, and `model.pt` is the TorchScript model.

The `config.pbtxt` file is the model configuration that denotes how Triton can
serve this models.

The example for the `config.pbtxt` for YOLOv5 model is as follows:

```protobuf
platform: "pytorch_libtorch"
input {
name: "INPUT__0"
data_type: TYPE_FP32
dims: -1
dims: 3
dims: 640
dims: 640
}
output {
name: "OUTPUT__0"
data_type: TYPE_FP32
dims: -1
dims: 25200
dims: 85
}
```

> Note that for PyTorch models, you will need to export your model to
> TorchScript first. Refer to
> [PyTorch's guide](https://pytorch.org/docs/stable/jit.html) to learn more
> about how to convert your model to TorchScript.

## Create a Triton Runner

Now that we have our model repository ready, we can create a Triton Runner to
interact with others BentoML Runners.

```python
triton_runner = bentoml.triton.Runner("triton-runner", model_repository="./model_repository")
```

> Note: You can also use S3 or GCS as your model repository, by passing the path
> to your S3/GCS bucket to the `model_repository` argument.

```python
triton_runner = bentoml.triton.Runner("triton-runner", model_repository="gcs://org/model_repository")
```

Each model in the model repository can be accessed via the signature of this
`triton_runner` object. For example, the model `torchscript_yolov5s` can be
accessed via `triton_runner.torchscript_yolov5s`, and you can invoke the
inference of such model with `run` or `async_run` method. This is similar to how
other BentoML's built-in Runners work.

```python
@svc.api(
    input=bentoml.io.Image.from_sample("./data/0.png"), output=bentoml.io.NumpyNdarray()
)
async def infer(im: Image) -> NDArray[t.Any]:
    inputs = preprocess(im)
    InferResult = await triton_runner.torchscript_yolov5s.async_run(inputs)
    return InferResult.as_numpy("OUTPUT__0")
```

Let's unpack this code snippet. First we define an async API that takes in an
image and returns a Numpy array. We then do some pre-processing to the input
images and pass it into the model `torchscript_yolov5s` via
`triton_runner.torchscript_yolov5s.async_run`.

The signature of `async_run` or `run` method is as follows:

- `async_run` and `run` can only take either all positional arguments or all
  keyword arguments. The arguments must match the input signature of the model
  specified in the `config.pbtxt` file.

  From the aboved `config.pbtxt`, we can see that the input signature of the
  model is `INPUT__0`, which is a 3-dimensional tensor of type `TYPE_FP32` with
  a batch dimension. This means `async_run`/`run` method can only take in eithe
  ra single positional argument or a single keyword argument with the name
  `INPUT__0`.

  ```python
  # valid
  triton_runner.torchscript_yolov5s.async_run(inputs)
  triton_runner.torchscript_yolov5s.async_run(INPUT__0=inputs)
  ```

  If the models has multiple inputs, the following are deemed as invalid:

  ```python
  # invalid
  triton_runner.torchscript_yolov5s.async_run(inputs, INPUT__1=inputs)
  ```

- `run`/`async_run` returns a `InferResult` object, which is a
  [wrapper](https://github.com/triton-inference-server/client/blob/403ebafda3f174eddc5b5a130a74b8d5c07607dd/src/python/library/tritonclient/grpc/__init__.py#L1997)
  around the response from Triton Inference Server. Refer to the internal
  docstring for more details.

Additionally, the Triton runner also exposes all `tritonclient` model management
APIs so that users can fully utilize all features provided by Triton Inference
Server.

## Packaging BentoService with Triton Inference Server

To package your BentoService with Triton Inference Server, you can add the
following to your existing `bentofile.yaml`:

```yaml
include:
  - /model_repository
docker:
  base_image: nvcr.io/nvidia/tritonserver:22.12-py3
```

Note that the `base_image` is the Triton Inference Server docker image from
[NVIDIA's container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).

If the model repository is stored in S3 or GCS, there is no need to add the
`include` section.

That's it! Build the BentoService and containerize with `bentoml build` and
`bentoml containerize` respectively:

```bash
bentoml build

bentoml containerize triton-integration:latest
```

## Conclusion

Congratulations! You can now fully utilize the power of Triton Inference Server
with BentoML through `bentoml.triton`. Our internal benchmark reveals upward of
40% performance improvement when using Triton Runner in comparison to its Python
counterpart. While Triton brings a lot of benefits to the table, it is also
important to understand its design philosophy. Triton follows a more
engineering-driven design, which means that it requires a lot of specific domain
knowledge to fully utilize its features, whereas BentoML's focuses on the ease
of use and developer agility.

This integration brings the best of both worlds, enabling users to easily deploy
their models with BentoML and enjoy all of the performance benefits from Triton
Inference Server.

You can read more about this integration from our
[documentation](https://docs.bentoml.org/en/latest/integrations/triton.html).

If you enjoyed this article, feel free to support us by starring our
[GitHub](https://github.com/bentoml/BentoML), and join our community
[Slack](https://l.linklyhq.com/l/ktOX) channel!
