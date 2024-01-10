---
id: BentoML and Triton Inference Server
tags:
  - technical
  - ml
  - fruit
date: "2023-03-22"
title: BentoML. Triton Inference Server. Choose 2
---

We are seeing a surge in recent months of developments and works on large
language models (LLM) and its applications such as ChatGPT,
[Stable Diffusion](https://modelserving.com/blog/creating-stable-diffusion-20-service-with-bentoml-and-diffusers),
Copilot.

However, deploying and serving LLMs at scale is a challenging task that requires
specific domain expertise and inference infrastructure. A
[rough estimation](https://twitter.com/tomgoldsteincs/status/1600196981955100694?s=20)
of running ChatGPT shows that serving efficiency is critical to making such
models work at scale. These operations are often known as Large Language Models
Operations (LLMOps). LLMOps, in general, is considered as a subset of MLOps,
which is a set of practices combining software engineering, DevOps, and data
science to automate and scale the end-to-end lifecycle of ML models.

Teams can encounter several problems when running inference on large models,
including:

- _Resource utilisation_: Large models require a significant amount of
  computational power to run, which can be challenging for teams with limited
  resources. Serving frameworks should utilise all available resources to be
  cost-effective.
- _Model optimisation_: Large models often contains a lot of redundant layers
  and parameters that can be pruned to reduce model size and speed up inference.
  A serving framework ideally should be able to provide support for model
  optimisation library to aid this process.
- _Serving latency_: Large language models often require complex batching
  strategies to enable real-time inference. A serving framework should be
  equipped with batching strategies to optimise for low-latency serving.

In this blog post, we will be demonstrating the capabilities of BentoML and
Triton Inference Server to help you solve these problems.

## What is Triton Inference Server?

Triton Inference Server is a high performance, open-source inference server for
serving deep learning models. It is designed to serve a variety of deep learning
models and frameworks, such as ONNX, TensorFlow, TensorRT. It is also designed
with optimisations to maximise hardware utilisation through various model
execution and efficient batching strategies.

> Triton Inference Server is great for serving large language models, where you
> want a high-performance inference server that can utilise all available
> resources with complex batching strategies.

## What is BentoML?

[[dump/projects#bentoml--build-production-grade-ai-application|BentoML]] is an open-source platform designed to facilitate the development,
shipping, and scaling of AI applications. It empowers teams to rapidly develop
AI applications that involve multiple models and custom logic using Python. Once
developed, BentoML allows these applications to be seamlessly shipped to
production on any cloud platform with engineering best practices already
integrated. Additionally, BentoML makes it easy to scale these applications
efficiently based on usage, ensuring that they can handle any level of demand.

## What Motivated the Triton Integration?

Starting BentoML v1.0.16, Triton Inference Servers can now be seamlessly used as
a [Runner](https://docs.bentoml.org/en/latest/concepts/runner.html). Runners are
abstractions of logic that can execute on either CPU or GPU and scale
independently. Prior to the Triton integration, one of the drawbacks of using
Python runners is the Global Interpreter Lock (GIL), where it only allows one
thread to be executed at a time. While the model inference can still run on GPU
or multi-threaded CPU, the IO logic is still subjective to the limitations of
GIL, which limits the underlying hardware utilisation (CPU and GPU). Triton’s
C++ runtime is optimised for high throughput model serving. By using Triton as a
runner, users can take full advantages of Triton’s high-performance inference,
while continue enjoy all features that BentoML offers.

## Too much talking? Ok lets dive in!

In a nutshell, BentoML provides the capabilities for users to run Triton
Inference Server via `bentoml.triton.Runner`:

```python
triton_runner = bentoml.triton.Runner("triton-runner", model_repository="s3://org/model_repository")
```

In order to use the `bentoml.triton` API, users are required to have the Triton
Inference Server
[container image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)
available locally.

```bash
docker pull nvcr.io/nvidia/tritonserver:23.01-py3
```

Install the extension for BentoML with Triton support:

```bash
pip install -U "bentoml[triton]"
```

Triton Inference Server evolves around the concepts of model repository, a
filesystem-based persistent volume that contains the models and
[Triton's model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).
We will provide a quick walk-through in the section below for you to get
started.

The following section assumes that you have a basic understanding of BentoML
architecture. If you are new to BentoML, we recommend you to read our
[Getting Started](https://docs.bentoml.org/en/latest/tutorial.html) guide first.

You can also find the full example repository
[here](https://github.com/bentoml/BentoML/tree/main/examples/triton).

### Preparing your model

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

> Note that the model weight file name must prefix with `model.<extensions>` for
> all Triton model. Refer to their
> [documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html?highlight=model%20configuration)
> for more details.

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

### Create a Triton Runner

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
@svc.api(input=bentoml.io.Image(), output=bentoml.io.NumpyNdarray())
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

### Packaging BentoService with Triton Inference Server

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
counterpart.

You can read more about this integration from our
[documentation](https://docs.bentoml.org/en/latest/integrations/triton.html).
