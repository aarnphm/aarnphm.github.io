---
date: '2025-03-25'
description: multi-level intermediate representation for compiler infrastructure
id: MLIR
modified: 2026-09-16 13:30:23 GMT-04:00
seealso:
  - '[[thoughts/Compiler]]'
  - '[[thoughts/XLA]]'
  - '[[thoughts/PyTorch]]'
  - '[[thoughts/GPU programming]]'
tags:
  - ml
  - compilers
  - infrastructure
title: MLIR
---

Good blogpost write up:

- https://www.stephendiehl.com/posts/mlir_introduction/
- https://mlir.llvm.org/docs/Tutorials/
- https://www.jeremykun.com/2023/08/10/mlir-getting-started/

> a compiler infrastructure project under the LLVM umbrella that provides a flexible framework for building reusable and extensible compiler infrastructure.

MLIR lets a compiler keep a matrix multiplication recognizable while choosing how to execute it. A `linalg.matmul` still exposes its reduction and indexing structure. Once lowered to scalar loads and branches, that information takes more analysis to recover. The point of multiple levels is to make each decision while the relevant structure is available.

## core architecture

### the dialect system

A dialect groups operations, types, and attributes under a namespace. A function can contain `tensor`, `linalg`, and `arith` operations together. They share MLIR's representation; their definitions supply the semantics. A compiler must still provide the conversions between them. [MLIR language reference](https://mlir.llvm.org/docs/LangRef/#dialects).

For example, this function computes $AB$:

```mlir
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x512xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                          outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
```

The fill matters. Matmul accumulates into its destination:

$$
C^{\mathrm{out}}_{ij}=C^{\mathrm{init}}_{ij}+\sum_{k=0}^{K-1} A_{ik}B_{kj}.
$$

`tensor.empty` supplies a shape with unspecified contents. Passing it directly to this reduction leaves the initial sum unspecified. An elementwise operation that overwrites every output without reading the destination can use an empty tensor directly. Check what the operation reads. [Tensor `empty`](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-tensoremptyop), [Linalg structured operations](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/).

### operation definition specification (ODS)

ODS uses LLVM's TableGen language to describe an operation's operands, results, attributes, traits, and assembly format. `mlir-tblgen` generates C++ classes, accessors, builders, and verification code from those declarations. Custom constraints can require a handwritten verifier.

Setting `hasFolder` or `hasCanonicalizer` declares hooks for implementations. It does not derive the arithmetic, prove the rewrite, or generate a backend. Keep the operation's semantics next to its definition so a pass author can tell which transformations are legal. [ODS reference](https://mlir.llvm.org/docs/DefiningDialects/Operations/).

### region, block, SSA hierarchy

An operation can contain regions; regions contain blocks; blocks contain operations. `builtin.module` and `func.func` are operations too. Values come from operation results or block arguments.

The enclosing operation determines a region's meaning. Function bodies use control-flow regions; graph regions need no sequential execution order. In a control-flow region, branches pass values to block arguments, serving the role of LLVM's phi nodes. [Regions and blocks](https://mlir.llvm.org/docs/LangRef/#regions).

A loop also needs an explicit value for each iteration's state:

```mlir
func.func @count(%lower: index, %upper: index) -> index {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %result = scf.for %i = %lower to %upper step %one iter_args(%acc = %zero) -> (index) {
    %next = arith.addi %acc, %one : index
    scf.yield %next : index
  }
  return %result : index
}
```

`%acc` starts at zero. Each `scf.yield` supplies the next iteration's argument, and the final yield becomes `%result`. With no iterations, `%result` is the initial value. [SCF `for`](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop).

### traits and interfaces

Traits attach shared constraints or behavior to operations. `SameOperandsAndResultType`, for example, checks the types on an IR operation. That verification happens when checking the IR; calling it a C++ compile-time property confuses two different programs. [Traits](https://mlir.llvm.org/docs/Traits/).

Interfaces let transformations ask an operation about its behavior without switching on every operation name. `MemoryEffectOpInterface` describes memory effects; `LoopLikeOpInterface` describes loop structure. A dialect has to implement the relevant contract before a generic pass can use it.

`NoMemoryEffect` covers memory effects. Speculation also has to account for undefined behavior and nontermination. Hoisting a computation above a conditional needs those conditions checked too. [Side effects and speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/).

### pattern rewriting infrastructure

A rewrite has a match, legality conditions, and a replacement. C++ patterns use `PatternRewriter` for mutations so the driver can track changed operations. DRR expresses patterns in TableGen; PDLL provides a language for matching and rewriting operation graphs. [Pattern rewriting](https://mlir.llvm.org/docs/PatternRewriter/), [PDLL](https://mlir.llvm.org/docs/PDLL/).

The driver determines how patterns run. A greedy driver repeats applicable rewrites until convergence or a configured limit. Dialect conversion instead checks a specified legality target. Neither driver proves that a replacement preserves the original computation.

For fusion, matching two matmuls is only the start. The rewrite must build legal IR while preserving accumulator values, shapes, and every use of the intermediate result.

### pass manager and composition

Pass pipelines name the operations they run on. This pipeline canonicalizes each function, then runs common-subexpression elimination on the module:

```sh
mlir-opt input.mlir --pass-pipeline='builtin.module(func.func(canonicalize),cse)'
```

Nesting controls scope. It also matters for analysis ownership and parallel pass execution. A pipeline must register its passes and respect the IR each pass accepts; putting pass names in a list does not establish those preconditions. [Pass management](https://mlir.llvm.org/docs/PassManagement/).

## key dialects

### high-level frontend dialects

| Dialect     | What it carries                                                                     |
| ----------- | ----------------------------------------------------------------------------------- |
| `tf`        | TensorFlow operations in TensorFlow's compiler infrastructure                       |
| `torch`     | PyTorch semantics imported by Torch-MLIR                                            |
| `tosa`      | A specified tensor operator set for neural-network computation                      |
| `stablehlo` | A versioned tensor computation format for exchange between frameworks and compilers |

These are entry points for particular compiler stacks. Availability of a lowering depends on the supported operations, types, and shapes. TOSA defines computation, including quantized arithmetic, rather than a schedule for a particular accelerator. [TOSA dialect](https://mlir.llvm.org/docs/Dialects/TOSA/), [Torch-MLIR](https://github.com/llvm/torch-mlir).

StableHLO's compatibility policy applies to artifacts produced through its portable serialization APIs. It specifies five years of backward compatibility and two years of forward compatibility, with exceptions for features absent from the older consumer. Arbitrary pretty-printed MLIR is outside that promise. [StableHLO compatibility](https://openxla.org/stablehlo/compatibility).

### mid-level structured operations

Linalg keeps the iteration structure explicit: indexing maps relate loop indices to operands, iterator types identify parallel and reduction dimensions, and a region supplies the scalar computation. For matmul, the maps are

$$
(i,j,k)\mapsto(i,k),\qquad
(i,j,k)\mapsto(k,j),\qquad
(i,j,k)\mapsto(i,j).
$$

The first two select elements of $A$ and $B$; the third selects the accumulator. The $k$ dimension is a reduction. That structure lets a transformation find which input slices are needed for an output tile. [Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/).

Tensor-form Linalg operations use destination-passing style: an `outs` operand is tied to each tensor result. The operation produces a new SSA value. Bufferization may later reuse the destination's storage if its old contents are no longer needed. [Destination-passing style](https://mlir.llvm.org/docs/Bufferization/#destination-passing-style).

### affine dialect

Affine operations restrict loop bounds and accesses to expressions that dependence analysis can reason about. Multiplication by a constant is affine; multiplication of two loop dimensions is not. MLIR also admits floor division, ceiling division, and modulo by positive integer constants in affine expressions.

The following buffer-form example accumulates $AB$ into an existing $C$. The caller must initialize $C$ and supply non-overlapping input and output storage for this intended matmul computation.

```mlir
func.func @matrix_multiply(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %a = affine.load %A[%i, %k] : memref<1024x1024xf32>
        %b = affine.load %B[%k, %j] : memref<1024x1024xf32>
        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}
```

Affine syntax makes access relationships available to analysis. Whether interchange, fusion, or parallelization is legal still depends on the accesses and their dependencies. [Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/).

### SCF (structured control flow)

SCF provides `for`, `while`, `if`, and parallel loop operations with nested regions. It can express bounds and conditions outside the affine restrictions. Loop-carried tensors follow the same `iter_args` and `scf.yield` discipline as the counter above: producing an updated tensor inside the body has no effect on the next iteration unless it is yielded.

Lowering SCF to the Control Flow dialect replaces this nesting with blocks and branches. Passes that need a visible loop nest should run before that structure is discarded. [SCF dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/).

### tensor vs memref

A tensor is a value. `tensor.insert` returns an updated value; it does not mutate the input tensor. `tensor.extract_slice` likewise returns a tensor value, leaving storage sharing to later decisions. `tensor.empty` introduces unspecified contents and a shape, with no requirement to allocate a buffer at that point. [Tensor dialect](https://mlir.llvm.org/docs/Dialects/TensorOps/).

A memref describes addressable storage. Loads read it, stores mutate it, and views can alias it. Bufferization chooses how to represent tensor values using such buffers while preserving the values that later readers observe.

One-Shot Bufferize analyzes SSA uses and considers reusing a destination buffer or allocating a new one. Function boundaries need suitable configuration and supported operations. Allocation hoisting and buffer deallocation are separate work; the pass does not promise a globally optimal memory plan. [Bufferization](https://mlir.llvm.org/docs/Bufferization/).

### vector dialect

Vector operations describe groups of scalar computations before choosing target instructions. A multidimensional vector is useful for expressing a tile even when the hardware's registers have a different shape. `vector.contract` describes a contraction, while transfer operations describe movement between shaped storage and vectors.

A transfer read supplies a padding value for out-of-bounds elements. Lowering may split the vector, introduce masks, or select target instructions. The vector's type alone does not guarantee one instruction. [Vector dialect](https://mlir.llvm.org/docs/Dialects/Vector/).

### GPU/NVVM/ROCDL dialects

The GPU dialect describes kernels, launch dimensions, and synchronization. Target-specific lowering can produce NVVM operations for NVIDIA or ROCDL operations for AMD, followed by LLVM IR and the relevant backend. A SPIR-V path is another option.

Kernel code generation is only part of this pipeline. The host must load the device binary, arrange memory, and issue the launch through a runtime. A `gpu.launch` operation does not settle those choices by itself. [GPU compilation](https://mlir.llvm.org/docs/Dialects/GPU/).

### LLVM dialect

The LLVM dialect represents LLVM-level operations inside MLIR. It can then be translated to LLVM IR:

```mlir
llvm.func @add(%a: i32, %b: i32) -> i32 {
  %result = llvm.add %a, %b : i32
  llvm.return %result : i32
}
```

This is a common exit from MLIR for native code generation. Other compilers emit SPIR-V, runtime calls, or hardware descriptions. An MLIR-based compiler chooses its output representation. [LLVM dialect](https://mlir.llvm.org/docs/Dialects/LLVM/), [CIRCT](https://circt.llvm.org/docs/).

## progressive lowering examples

The arrows below describe possible routes. They omit supporting dialects and cleanup passes. Treat them as a map of representation changes, then inspect the pipeline implemented by the compiler and revision in use.

### PyTorch → Torch-MLIR → Linalg → LLVM

```text
imported PyTorch graph
  → Torch dialect
  → Linalg on tensors
  → bufferization and loop/vector lowering
  → LLVM dialect
  → LLVM IR
```

Torch-MLIR supplies imports and conversions for downstream compilers. Its current project documentation lists FX and ONNX entry points. The supported subset and backend determine the remaining route. [Torch-MLIR project](https://github.com/llvm/torch-mlir).

Ordinary `torch.compile` uses TorchDynamo for graph capture and TorchInductor as its default backend. Torch-MLIR is a separate integration path. [PyTorch compiler documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html).

### TensorFlow → StableHLO → Linalg → Affine → LLVM

StableHLO can be an exchange boundary between a framework and a compiler. A consumer that provides StableHLO-to-Linalg conversion may then bufferize and lower to affine loops or vectors.

XLA has its own HLO optimization and backend pipelines. It does not require the full route in this heading. Whether a TensorFlow program uses XLA also depends on how compilation was requested. [XLA architecture](https://openxla.org/xla/architecture).

### TOSA → Linalg → Vector → LLVM

TOSA-to-Linalg lowering translates supported tensor operations into structured computations. Tiling and vectorization can then prepare those computations for LLVM lowering. Layout changes, quantization, and operation coverage determine what extra work is needed. There is no single convolution rewrite that covers every type and target. [TOSA dialect](https://mlir.llvm.org/docs/Dialects/TOSA/).

## key optimization passes

### tiling

Tiling splits an iteration space into bounded pieces so reused data may remain close to the computation. For an $f32$ matmul tile of sizes $(M_t,N_t,K_t)$, the three dense operand tiles occupy

$$
4\bigl(M_tK_t+K_tN_t+M_tN_t\bigr)\text{ bytes}.
$$

With $(256,256,128)$, that is $512\,\mathrm{KiB}$ before packing or other live data. Cache capacity, layout, concurrency, and the target's instructions determine whether that tile fits and how efficiently it runs.

Tensor tiling also has a correctness obligation: each reduction tile must receive the preceding tile's accumulator. The full result must be carried through the surrounding loops. [Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/).

### fusion

Producer-consumer fusion computes the part of a producer needed by a consumer tile. For a matmul followed by an elementwise operation, this can avoid writing and rereading the entire intermediate matrix. Other uses of that matrix may still require materialization or recomputation.

A bias or nonlinear activation must be applied at the correct point relative to the reduction. Adding the bias once per reduction tile changes the answer. Fusion needs both indexing information and the scalar computation's semantics. [Structured transformations](https://mlir.llvm.org/docs/Dialects/Linalg/#set-of-key-transformations).

### vectorization

Vectorization groups scalar work into vector operations. For a simple contiguous loop, a width of eight processes eight elements per main-loop iteration. A length that is not divisible by eight needs masking, padding, or a remainder path.

For a contraction, the lowering also chooses how vectors map to registers and matrix instructions. Wider IR vectors can increase register pressure. Inspect the generated code and measure the actual kernel before treating vector width as a speedup. [Vector dialect](https://mlir.llvm.org/docs/Dialects/Vector/).

### buffer allocation

The useful question is which old values remain observable after a proposed in-place update. If a later operation reads the original tensor, overwriting its only buffer would change that read. Bufferization must keep the old value available.

A destination passed through loop arguments can make storage reuse visible to analysis. An unrelated `tensor.empty` inside every iteration does not establish that one allocation will be hoisted and reused. Check the resulting memref IR, then account for ownership and deallocation. [Bufferization analysis](https://mlir.llvm.org/docs/Bufferization/).

### canonicalization and folding

Folding evaluates or simplifies an operation using its defined semantics. Canonicalization applies local cleanup patterns repeatedly, with limits. It is best-effort; a correct pipeline must not depend on every possible cleanup firing. CSE removes equivalent computations when the effect and dominance conditions permit it. [Canonicalization](https://mlir.llvm.org/docs/Canonicalization/).

Arithmetic identities need types and assumptions. Integer addition by zero is straightforward. Floating-point rewrites must respect signed zero, NaNs, rounding, and any fast-math permissions. A familiar algebraic identity is insufficient justification for changing an `arith` operation's behavior.

### loop transformations

Interchange changes loop order. For row-major storage, making the last index vary in the innermost loop gives contiguous accesses. Dependencies can forbid that order, so checking layout is only the performance half of the decision.

Skewing changes the coordinates of the iteration space; unroll-and-jam duplicates outer iterations and combines their inner loops. Each transformation must preserve dependencies, and floating-point reductions may impose additional ordering constraints. [Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/).

## MLIR vs traditional compilers

### vs LLVM

LLVM IR provides a defined instruction set for optimization and code generation. MLIR lets a compiler define operations that retain language or domain structure, then lower them when needed. The projects share infrastructure, and an MLIR compiler can use LLVM as its backend. [MLIR's representation](https://mlir.llvm.org/docs/LangRef/).

### vs XLA HLO

XLA is a compiler; HLO is one of its representations. StableHLO is an MLIR dialect used for exchange, with a compatibility contract. These names refer to different layers of a compiler stack. Asking whether a system uses “MLIR or StableHLO” misses that StableHLO is represented using MLIR. [XLA architecture](https://openxla.org/xla/architecture), [StableHLO compatibility](https://openxla.org/stablehlo/compatibility).

### vs TVM

TVM provides a tensor compilation stack with graph-level and tensor-program representations, scheduling, and runtime components. Its architecture documentation describes Relax and TensorIR, so the older shorthand “Relay plus TIR” needs a version attached.

MLIR supplies reusable IR and transformation infrastructure. A project using it still has to choose its scheduling policy, runtime, and supported models. A count of IR levels tells us little about those decisions. [TVM architecture](https://tvm.apache.org/docs/arch/).

### reusability story

A transformation can work across operations that implement the contracts it needs. This is why Linalg exposes structured indexing and why dialects implement interfaces. Reuse stops where semantics or representations stop matching. Sharing the parser and pass manager alone cannot make an optimization valid for a new dialect. [Linalg transformations](https://mlir.llvm.org/docs/Dialects/Linalg/#set-of-key-transformations).

## compiler and runtime projects

These projects use MLIR at different points in their compiler and runtime stacks.

### TensorFlow ecosystem

TFRT uses MLIR to represent host programs and translates them to its Binary Executable Format for execution. That is a specific runtime design. Its repository documents the compiler-to-BEF path; it does not establish that TFRT replaced every TensorFlow execution path. [TFRT](https://github.com/tensorflow/runtime).

### IREE (Intermediate Representation Execution Environment)

IREE includes both an MLIR-based compiler and a runtime. Its compiler has input conversions, kernel code generation, and dialects for dispatch and execution, including Flow, Stream, HAL, and VM. The runtime supplies device interfaces and executes the compiled module.

Keeping host scheduling separate from kernel lowering matters: compiling an individual matmul does not determine when its buffers are available or when another device may use the result. [IREE developer overview](https://iree.dev/developers/general/developer-overview/).

### Torch-MLIR

Torch-MLIR bridges FX and ONNX entry points to MLIR-based compilers. Its import paths and conversions are useful to downstream projects such as IREE. Selecting Torch-MLIR means selecting an integration with a supported input subset and backend. The default `torch.compile` backend remains TorchInductor. [Torch-MLIR](https://github.com/llvm/torch-mlir), [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

### Flang (Fortran compiler)

Flang parses and checks Fortran before lowering into MLIR-based representations. HLFIR retains higher-level Fortran expression and assignment semantics; FIR represents lower-level Fortran operations and eventually lowers toward LLVM. Describing the entire frontend as MLIR would skip its parser and semantic analysis. [Flang phases](https://flang.llvm.org/docs/Overview.html), [HLFIR](https://flang.llvm.org/docs/HighLevelFIR.html).

### CIRCT (Circuit IR Compilers and Tools)

CIRCT uses MLIR for hardware compilation. Its dialects include HW for hardware structure, Comb for combinational logic, Seq for sequential logic, and SV for SystemVerilog constructs. Hardware descriptions are a useful counterexample to the claim that every MLIR pipeline ends in LLVM machine code. [CIRCT documentation](https://circt.llvm.org/docs/).

## practical code examples

### simple dialect definition

Start with the [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/) for a buildable custom dialect. A constant operation needs an attribute whose type matches the result; an elementwise addition needs shape and element-type rules. A custom assembly format needs its parser and printer definitions too.

ODS can generate the structural code, while the dialect supplies semantics and lowerings. A bare declaration of `toy.add` does not make its arithmetic executable. For experiments with matmul, the existing `arith`, `tensor`, and `linalg` operations above already provide those definitions.

### pattern rewriting example

Use a small integer fold to inspect the existing canonicalizer. Save this as `fold.mlir`:

```mlir
func.func @fold() -> i32 {
  %a = arith.constant 7 : i32
  %b = arith.constant 9 : i32
  %sum = arith.addi %a, %b : i32
  return %sum : i32
}
```

```sh
mlir-opt fold.mlir --canonicalize
```

The result should return an `i32` constant of $16$; SSA names and printer formatting may change. This tests an existing folder without introducing a custom floating-point constant evaluator. [Canonicalization and folding](https://mlir.llvm.org/docs/Canonicalization/).

### tiling transformation

This example tiles only the reduction dimension of a matmul. Keeping the output dimensions intact makes the loop-carried accumulator visible. The tile size divides the static reduction extent exactly.

```mlir
func.func @matmul_k_tiles(%A: tensor<8x16xf32>, %B: tensor<16x8xf32>, %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %result = scf.for %k = %c0 to %c16 step %c4 iter_args(%acc = %C) -> (tensor<8x8xf32>) {
    %a = tensor.extract_slice %A[0, %k] [8, 4] [1, 1] : tensor<8x16xf32> to tensor<8x4xf32>
    %b = tensor.extract_slice %B[%k, 0] [4, 8] [1, 1] : tensor<16x8xf32> to tensor<4x8xf32>
    %next = linalg.matmul ins(%a, %b : tensor<8x4xf32>, tensor<4x8xf32>)
                          outs(%acc : tensor<8x8xf32>) -> tensor<8x8xf32>
    scf.yield %next : tensor<8x8xf32>
  }
  return %result : tensor<8x8xf32>
}
```

Each iteration starts from the previous `%acc`, so all four reduction tiles contribute. Passing `%C` directly as every tile's destination would discard earlier partial sums. To compute $AB$, supply a zero-filled `%C`; otherwise this computes $C+AB$.

General tiling also handles partial boundary tiles and carries updated output slices through outer loops. MLIR's Transform dialect exposes tiling operations for eligible payload operations; its tutorial describes the required interfaces and handles. [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/), [Transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/).

### conversion between dialects

Start with a buffer-form matmul to isolate Linalg-to-loop conversion. Save this as `matmul-buffer.mlir`:

```mlir
func.func @matmul_buffer(%A: memref<8x16xf32>, %B: memref<16x8xf32>, %C: memref<8x8xf32>) {
  linalg.matmul ins(%A, %B : memref<8x16xf32>, memref<16x8xf32>)
                outs(%C : memref<8x8xf32>)
  return
}
```

```sh
mlir-opt matmul-buffer.mlir --convert-linalg-to-loops
```

The caller initializes `%C` and keeps its storage separate from `%A` and `%B`. The pass expresses the computation with SCF loops, loads, arithmetic, and stores. Tensor-form input needs bufferization or an appropriate tensor transformation first. [Linalg loop lowering](https://mlir.llvm.org/docs/Dialects/Linalg/#property-1-input-and-output-operands-define-the-iteration-space).

For a custom dialect conversion, declare which operations are legal in the result, provide rewrite patterns, and use a type converter where representations change. A full conversion fails if illegal operations remain. That failure is useful evidence of a missing lowering. [Dialect conversion](https://mlir.llvm.org/docs/DialectConversion/).

## references

- [MLIR documentation](https://mlir.llvm.org/docs/) and [tutorials](https://mlir.llvm.org/docs/Tutorials/)
- [ODS specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [Affine dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [Pattern rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
- [IREE](https://iree.dev/), [Torch-MLIR](https://github.com/llvm/torch-mlir), [StableHLO](https://openxla.org/stablehlo), [CIRCT](https://circt.llvm.org/)
- [Stephen Diehl's introduction](https://www.stephendiehl.com/posts/mlir_introduction/)
- [Jeremy Kun's getting-started article](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) and [MLIR series](https://www.jeremykun.com/tags/mlir/)
