---
title: "Runner: A universal interface to run distributed ML application"
tags:
  - technical
  - seed
---

# Runner Interface

Currently, our Runner are loosely defined. We have a `Runnable` interface that allows user to implement the runner server.

Current drawbacks:
- No payload type is specified. Runner should specify the type of the payload on wire:
	- Tensor (bytes)
	- Array-like (`np.array`, `jnp.array`)
	- Tabular-like (`pd.DataFrame`, `cudf.DataFrame`)
	- Python objects? `{str: Any}`
- Runner doesn't have a client signature
	- `runner.run` should support `*args` and `**kwargs`

###### Data format
- Arrow


---

## Requirements

> Each runner inference should be a RPC call.

- Makes the local/remote behaviour more consistent.
	- Local:
		- IPC vs. RPC
		- low latency &arr; minimal serialization on the wire
		- Arrow format vs. Flatbuffer <- single element types
		- Composite type?
			`{str: Tensor}`
	- Remote:
		- gRPC-flatbuffer vs. [gRPC-Flight](https://arrow.apache.org/blog/2019/10/13/introducing-arrow-flight/) (*I need to do more testing around this*)
- Returns value
	- How do we handle tuple or composite type?
		- ONNX and TF support dict inputs
	- ask users to define signatures?
		- ONNX define composite type?
		- `tf.function`
		
- Implementation:
	- [Triton Inference Server][#triton-inference-server]
	- [TFServing](https://github.com/tensorflow/serving)
	- [TorchServe](https://pytorch.org/serve/)
- KServe [Predict V2](https://kserve.github.io/website/modelserving/inference_api/#inference) protocol
	- Only support tensor format
	- Currently use protobuf, which is unnecessary overhead when serving in local use-case 
	- Triton Inference Server, ONNXRuntime Server, and TFServing is adopting this design.
	- Has a `ServerMetadata` and `ModelMetadata`
	

- polyglot environment
- benchmarks: C++ and Go

### LOOK INTO
- KServe and Triton handles dictionary inputs?
- arrow supports dictionary? with different shape?
- Arrow format?
	- Tensor?
		- High dimension not supported? (reshape)
		- Tensorflow

- flatbuffer vs. protobuf
	- flatbuffer + protobuf?
	- bytes field: flatbuffer

grpc+flatbuffer (flatbuffer bytes into bytes protobuf bytes field)

serialization framework?
- protobuf
- flatbuffer
- cap'n proto


---
## Q?

We can have multiple interfaces for this design.

> How one determine payload format?
	- Predict V2 protocol are currently only designed for Tensor format.
	- Uses protobuf, which probably too much overhead for local cases
	- bytes layout

> How do we schedule runners?
	- add in runner metadata?
		- triton: [instance group](https://github.com/triton-inference-server/server/blob/ff131f3b9d7c896c0a91614d6deb3d405d317ecd/docs/user_guide/model_configuration.md)
	- strategy: transformers <- configuration (instance)
	- Yatai:
		- triton: runner -> bentoml -> yatai
		- translation: 
Config:
- relay to triton <- users
- BentoML configuration (+ scheduling strategy on hardware profile)
- multiple instance of triton server?
	- strategy -> deploy n instances (triton allows multiple server on different process)
	- 2GPUS: 2 instance of triton server

> How does this work in Yatai?
> 	- triton config as code?
```python
bentoml.triton.Runner(config={}, ...)
```
- Mixed hardware for single bento?
- bentoml & triton configuration interop

?: Generalise strategy on the runner interface?

Supervisor -> runner

> Batch configuration?

flatbuffer -> gRPC transform payload

---
#Nov-1

Dictionary format?
- inputs/outputs is heterogenous?
```python
inputs = {"key": 1, "key2": np.ndarray([[1,2,3,4]])}
```




---
###### Appendix
[#triton-inference-server]: https://github.com/triton-inference-server/server
[#predict-v2]: https://kserve.github.io/website/modelserving/inference_api/