---
date: '2024-03-04'
description: device API for compiling and running ML computations across hardware backends
id: PJRT
modified: 2026-10-09 09:03:58 GMT-04:00
tags:
  - ml
title: PJRT
---

PJRT gives ML frameworks a common interface to a hardware backend's compiler and runtime. A framework uses it to discover devices, prepare input buffers, compile a computation, and execute it. Hardware providers implement the interface for targets such as [[thoughts/TPU|TPUs]] and [[thoughts/GPU programming|GPUs]]. The framework can call those implementations without knowing their internals. [Google's introduction](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) describes this plugin boundary; the [source](https://github.com/openxla/xla/tree/main/xla/pjrt) contains the interfaces and implementations.

## where compilation happens

StableHLO represents a computation as tensor operations. [[thoughts/XLA|XLA]] optimizes that representation and generates code for a target. [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#introduction) is NVIDIA's virtual instruction set, which a GPU compiler can emit on the way to native instructions. [XLA architecture](https://openxla.org/xla/architecture) describes that compilation path.

PJRT defines how the framework requests compilation and uses the resulting executable. The plugin supplies the compiler and runtime behind those calls, so it can use XLA or another toolchain.

## objects the framework handles

The [C++ API overview](https://openxla.org/xla/pjrt/cpp_api_overview) names the main objects:

- A **client** holds the state for communication with a backend and owns its devices and memory spaces.
- A **device** identifies an execution target. Its memory spaces describe where data can live and which devices can access it.
- A **buffer** holds input or output data in a memory space. Its ownership and readiness rules determine when data can be read, reused, or released.
- A **loaded executable** holds a compiled computation ready to accept input buffers and produce output buffers.

## following one computation

For a computation such as $f(x) = x + 1$, the framework first lowers the function to a program the backend accepts, such as a StableHLO module. Through `PJRT_Client_Compile`, it passes the program and compilation options to the backend and receives a loaded executable.

The framework prepares the input with `PJRT_Client_BufferFromHostBuffer`, then supplies the resulting buffer to `PJRT_LoadedExecutable_Execute`. The output buffer may be returned while the device is still working. `PJRT_Buffer_ReadyEvent` reports when the data is ready or an error has occurred; `PJRT_Buffer_ToHostBuffer` requests a transfer back to host memory. These operations and their ownership rules are specified in the [C API header](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h).

A hardware plugin exposes those C functions through a `PJRT_Api` function-pointer table in a shared library. Its implementation can use the C API directly or wrap PJRT's C++ classes. That lets a framework load a backend without linking against its private C++ types. The [integration guide](https://github.com/openxla/xla/blob/main/docs/pjrt/pjrt_integration.md) covers both routes.
