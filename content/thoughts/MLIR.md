---
id: MLIR
tags:
  - ml
  - compilers
  - infrastructure
description: multi-level intermediate representation for compiler infrastructure
date: "2025-03-25"
modified: 2025-10-05 02:25:13 GMT-04:00
title: MLIR
---

See also: [[thoughts/Compiler]], [[thoughts/XLA]], [[thoughts/PyTorch]], [[thoughts/GPU programming]]

> a compiler infrastructure project under the LLVM umbrella that provides a flexible framework for building reusable and extensible compiler infrastructure.

Unlike traditional single-level IRs like LLVM IR, MLIR allows dialects at multiple abstraction levels to coexist and transform progressively.

## core architecture

### the dialect system

> a way to define domain-specific operations, types, and attributes that can coexist in the same IR.
> Each dialect is ==self-contained== with its own semantics, verification, and transformations.

dialects are composable. you can mix operations from `tensor`, `linalg`, `arith`, and `func` dialects in the same function.

> This [[thoughts/Compositionality|composability]] enables progressive lowering where high-level abstractions incrementally transform to lower-level representations.

```mlir
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %empty = tensor.empty() : tensor<128x512xf32>
  %result = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
                          outs(%empty : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
```

### operation definition specification (ODS)

operations are defined using TableGen[^note] to generate C++ code. ODS provides declarative specifications that generate boilerplate:

[^note]: LLVM's own DSL

```tablegen
def MatmulOp : LinalgStructuredOp<"matmul"> {
  let arguments = (ins AnyTensor:$A, AnyTensor:$B, AnyTensor:$C);
  let results = (outs AnyTensor:$result);
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}
```

this generates:

- operation class with accessors
- verification methods
- builders and printers
- trait implementations

### region, block, SSA hierarchy

MLIR uses a nested structure:

- **Module**: top-level container
- **Operation**: fundamental unit (everything is an op)
- **Region**: contains control flow graphs
- **Block**: basic block with arguments (phi nodes as arguments)
- **Value**: SSA values (operation results or block arguments)

blocks take arguments instead of phi nodes, simplifying control flow representation:

```mlir
func.func @loop(%arg0: index, %arg1: index) -> index {
  %c1 = arith.constant 1 : index
  %result = scf.for %i = %arg0 to %arg1 step %c1 iter_args(%acc = %arg0) -> index {
    %next = arith.addi %acc, %c1 : index
    scf.yield %next : index
  }
  return %result : index
}
```

### traits and interfaces

**traits** define properties that can be checked at compile time:

- `SameOperandsAndResultType`: all operands and results have same type
- `Commutative`: operation is commutative
- `NoMemoryEffect`: operation has no side effects

**interfaces** define methods operations can implement:

- `MemoryEffectOpInterface`: describe memory read/write effects
- `InferTypeOpInterface`: infer result types from operands
- `LoopLikeOpInterface`: standard loop operations

```cpp
def MatmulOp : Op<LinalgDialect, "matmul", [
  NoMemoryEffect,
  DeclareOpInterfaceMethods<DestinationStyleOpInterface>
]> {
  // ...
}
```

### pattern rewriting infrastructure

MLIR provides two pattern rewriting systems:

**declarative rewrite rules (DRR)** - TableGen-based pattern matching:

```tablegen
// Fold consecutive reshapes
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp(ReshapeOp $arg)),
  (ReshapeOp $arg)
>;

// Constant folding
def FoldConstantAdd : Pat<
  (AddIOp (ConstantOp $a), (ConstantOp $b)),
  (ConstantOp (AddInts $a, $b))
>;
```

**PDLL (PDL Language)** - newer, more expressive pattern language:

```pdll
Pattern FuseMatmuls {
  let matmul1 = op<linalg.matmul>(a: Value, b: Value, c0: Value) -> (r1: Type);
  let matmul2 = op<linalg.matmul>(r1, d: Value, c1: Value) -> (r2: Type);

  rewrite matmul2 with {
    let fused = op<linalg.matmul_matmul>(a, b, d, c1) -> (r2);
    replace matmul2 with fused;
    erase matmul1;
  };
}
```

pattern application uses a greedy rewrite driver. patterns fire until fixpoint. debug with `-debug-only=greedy-rewriter`.

### pass manager and composition

passes organize into pipelines:

```cpp
mlir::PassManager pm(&context);
pm.addPass(mlir::createCanonicalizerPass());
pm.addPass(mlir::createCSEPass());
pm.addNestedPass<func::FuncOp>(createLinalgTilingPass());
pm.addPass(createConvertLinalgToLoopsPass());
```

passes can be function-scoped, module-scoped, or op-scoped. nested pass managers handle different IR levels.

## key dialects

### high-level frontend dialects

**`tf` (TensorFlow)**: represents TensorFlow operations

- preserves TensorFlow semantics during ingestion
- lowered through StableHLO or directly to TOSA

**`torch`**: PyTorch operations via Torch-MLIR

- two variants: Torch dialect (Python semantics) and TMTensor
- progressive lowering: Torch → Linalg → loops

**`tosa` (Tensor Operator Set Architecture)**:

- portable operator set for NN workloads
- ~60 operations covering conv, matmul, pooling, elementwise
- designed for hardware compliance testing

**StableHLO**: successor to MHLO, standardizes XLA's HLO

- 5-year backward compatibility guarantee
- 2-year forward compatibility
- canonical format for framework interop
- ~100 operations with full specifications

### mid-level structured operations

**`linalg` dialect**: structured operations on tensors/memrefs

linalg operations describe computation through:

- indexing maps (affine expressions relating iteration space to data space)
- iterator types (parallel, reduction, window)
- region describing scalar computation

named operations:

```mlir
linalg.matmul ins(%A, %B) outs(%C)
linalg.conv_2d_nhwc_hwcf ins(%input, %filter) outs(%output)
linalg.pooling_nhwc_sum ins(%input, %window) outs(%output)
```

generic operations with explicit indexing:

```mlir
linalg.generic {
  indexing_maps = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,  // A
    affine_map<(d0, d1, d2) -> (d2, d1)>,  // B
    affine_map<(d0, d1, d2) -> (d0, d1)>   // C
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}
ins(%A, %B : tensor<128x256xf32>, tensor<256x512xf32>)
outs(%C : tensor<128x512xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b : f32
    %add = arith.addf %c, %mul : f32
    linalg.yield %add : f32
}
```

linalg-on-tensors uses destination-passing style (DPS): output tensor passed as operand. this enables sophisticated fusion without intermediate allocations.

### affine dialect

affine dialect models loop nests and array accesses using polyhedral compilation techniques. affine expressions: `d0 * 4 + d1 + 16` where `d0, d1` are dimensions.

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

affine dialect enables:

- **dependence analysis**: precise data dependencies using polyhedral methods
- **loop transformations**: interchange, tiling, skewing, unroll-and-jam
- **automatic parallelization**: detect parallel loops
- **vectorization**: multi-dimensional vectorization

### SCF (structured control flow)

SCF provides imperative control flow constructs:

```mlir
// for loop
scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<f32>) {
  %next = some_op %acc
  scf.yield %next : tensor<f32>
}

// while loop
scf.while (%arg = %init) : (tensor<f32>) -> tensor<f32> {
  %cond = some_condition %arg
  scf.condition(%cond) %arg : tensor<f32>
} do {
^bb0(%arg: tensor<f32>):
  %next = some_op %arg
  scf.yield %next : tensor<f32>
}

// conditional
scf.if %cond -> tensor<f32> {
  scf.yield %true_val : tensor<f32>
} else {
  scf.yield %false_val : tensor<f32>
}

// parallel
scf.parallel (%i, %j) = (%c0, %c0) to (%N, %M) step (%c1, %c1) {
  // parallel work
  scf.reduce(%val) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      scf.reduce.return %sum : f32
  }
}
```

### tensor vs memref

**tensor dialect**: value semantics, immutable

- `tensor.empty`: allocate uninitialized tensor
- `tensor.extract`: extract scalar element
- `tensor.insert`: insert element (returns new tensor)
- `tensor.extract_slice`: get subtensor view

**memref dialect**: buffer semantics, mutable

- `memref.alloc`: allocate buffer
- `memref.load`/`memref.store`: read/write
- `memref.view`: reinterpret cast
- `memref.subview`: create view into memref

conversion: **bufferization** transforms tensor IR to memref IR

```mlir
// before bufferization
func.func @add(%a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
  %empty = tensor.empty() : tensor<1024xf32>
  %result = linalg.add ins(%a, %b) outs(%empty)
  return %result : tensor<1024xf32>
}

// after bufferization
func.func @add(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) {
  linalg.add ins(%a, %b) outs(%c)
  return
}
```

one-shot bufferization analyzes entire program, avoids unnecessary allocations by reusing buffers when safe.

### vector dialect

hardware-agnostic SIMD operations. targets can be multi-dimensional:

```mlir
// 2D vector load
%vec = vector.transfer_read %memref[%i, %j], %pad
  : memref<1024x1024xf32>, vector<8x16xf32>

// vector contraction (generalized dot product)
%result = vector.contract {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>,
    affine_map<(i,j,k) -> (k,j)>,
    affine_map<(i,j,k) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}
%a, %b, %acc : vector<8x16xf32>, vector<16x8xf32> into vector<8x8xf32>

// multi-dim transpose
%transposed = vector.transpose %vec, [1, 0] : vector<8x16xf32> to vector<16x8xf32>
```

vector operations lower to target-specific intrinsics (AVX, NEON, SVE).

### GPU/NVVM/ROCDL dialects

**GPU dialect**: target-agnostic parallel execution model

```mlir
gpu.launch blocks(%bx, %by, %bz) in (%grid_x, %grid_y, %grid_z)
           threads(%tx, %ty, %tz) in (%block_x, %block_y, %block_z) {
  %thread_id = gpu.thread_id x
  %block_id = gpu.block_id x
  // kernel code
  gpu.terminator
}
```

**NVVM dialect**: NVIDIA CUDA operations

- maps to NVVM IR (NVIDIA's LLVM variant)
- intrinsics: `nvvm.shfl.sync.bfly`, `nvvm.wmma.load`, `nvvm.barrier0`

**ROCDL dialect**: AMD ROCm operations

- maps to AMD GCN/CDNA ISA
- intrinsics: `rocdl.workitem.id.x`, `rocdl.barrier`

lowering path: `gpu` → `nvvm`/`rocdl` → LLVM → PTX/GCN

### LLVM dialect

1:1 correspondence with LLVM IR. final lowering target before native code:

```mlir
llvm.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = llvm.add %arg0, %arg1 : i32
  llvm.return %0 : i32
}

llvm.func @malloc(%size: i64) -> !llvm.ptr {
  %ptr = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  llvm.return %ptr : !llvm.ptr
}
```

all MLIR eventually lowers to LLVM dialect, then to LLVM IR, then to machine code.

## progressive lowering examples

### PyTorch → Torch-MLIR → Linalg → LLVM

```
PyTorch nn.Linear
    ↓ (torch.jit.script)
Torch dialect: torch.aten.linear
    ↓ (decompose-complex-ops)
Torch dialect: torch.aten.mm
    ↓ (convert-torch-to-linalg)
Linalg: linalg.matmul on tensors
    ↓ (linalg-bufferize)
Linalg: linalg.matmul on memrefs
    ↓ (convert-linalg-to-loops)
SCF: nested scf.for loops
    ↓ (convert-scf-to-cf)
CF: br, cond_br
    ↓ (convert-to-llvm)
LLVM dialect
    ↓ (mlir-translate)
LLVM IR
```

### TensorFlow → StableHLO → Linalg → Affine → LLVM

```
TensorFlow MatMul
    ↓ (tf-to-stablehlo)
StableHLO: stablehlo.dot_general
    ↓ (stablehlo-to-linalg)
Linalg: linalg.matmul
    ↓ (linalg-tile)
Linalg: tiled linalg.matmul in loops
    ↓ (linalg-bufferize)
Linalg on memrefs
    ↓ (convert-linalg-to-affine)
Affine: affine.for with affine.load/store
    ↓ (affine-loop-fusion, affine-vectorize)
Optimized affine + vector ops
    ↓ (lower-affine, convert-vector-to-llvm)
LLVM dialect
```

### TOSA → Linalg → Vector → LLVM

```
TOSA: tosa.conv2d
    ↓ (tosa-to-linalg)
Linalg: linalg.conv_2d_nhwc_hwcf
    ↓ (linalg-tile-and-fuse)
Tiled linalg ops
    ↓ (linalg-vectorize)
Vector dialect ops
    ↓ (vector-lower-to-llvm)
LLVM vector intrinsics
```

## key optimization passes

### tiling

tiling divides iteration space into smaller tiles for better cache locality:

```mlir
// before tiling
linalg.matmul ins(%A, %B) outs(%C)
  : tensor<1024x1024xf32>, tensor<1024x1024xf32> into tensor<1024x1024xf32>

// after tiling with tile sizes [256, 256, 128]
scf.for %i = %c0 to %c1024 step %c256 {
  scf.for %j = %c0 to %c1024 step %c256 {
    scf.for %k = %c0 to %c1024 step %c128 {
      %A_tile = tensor.extract_slice %A[%i, %k][256, 128]
      %B_tile = tensor.extract_slice %B[%k, %j][128, 256]
      %C_tile = tensor.extract_slice %C[%i, %j][256, 256]
      %result = linalg.matmul ins(%A_tile, %B_tile) outs(%C_tile)
      %C_updated = tensor.insert_slice %result into %C[%i, %j]
    }
  }
}
```

multi-level tiling for cache hierarchy:

- L1 tiling: 32x32x32
- L2 tiling: 256x256x128
- L3 tiling: 1024x1024x512

### fusion

**producer-consumer fusion**: fuse ops where output of one feeds another

```mlir
// before fusion
%1 = linalg.matmul ins(%A, %B) outs(%C_init)
%2 = linalg.add ins(%1, %bias) outs(%D_init)

// after fusion
scf.for %i, %j {
  %tile_mm = linalg.matmul ins(%A_tile, %B_tile) outs(%C_tile)
  %tile_add = linalg.add ins(%tile_mm, %bias_tile) outs(%D_tile)
  // no intermediate materialization
}
```

**sibling fusion**: fuse independent ops accessing same data

```mlir
// before
%1 = linalg.reduce ins(%X) outs(%sum)
%2 = linalg.reduce ins(%X) outs(%max)

// after
%sum, %max = scf.for %i iter_args(%s, %m) {
  // compute both reductions in single loop
}
```

linalg-on-tensors enables fusion without premature buffer allocation.

### vectorization

affine vectorization finds vectorizable loops:

```mlir
// scalar loop
affine.for %i = 0 to 1024 {
  %a = affine.load %A[%i] : memref<1024xf32>
  %b = affine.load %B[%i] : memref<1024xf32>
  %c = arith.addf %a, %b : f32
  affine.store %c, %C[%i] : memref<1024xf32>
}

// vectorized (factor 8)
affine.for %i = 0 to 1024 step 8 {
  %a_vec = vector.transfer_read %A[%i] : memref<1024xf32>, vector<8xf32>
  %b_vec = vector.transfer_read %B[%i] : memref<1024xf32>, vector<8xf32>
  %c_vec = arith.addf %a_vec, %b_vec : vector<8xf32>
  vector.transfer_write %c_vec, %C[%i] : vector<8xf32>, memref<1024xf32>
}
```

multi-dimensional vectorization for matrix operations:

```mlir
// 2D vectorization of matmul inner loops
%result_vec = vector.contract %A_vec, %B_vec, %C_vec
```

### buffer allocation

one-shot bufferization analyzes tensor dataflow, places allocations optimally:

strategies:

- **in-place updates**: reuse input buffer when safe
- **buffer hoisting**: allocate outside loops
- **aliasing analysis**: avoid copies when possible

```mlir
// analysis determines %empty can be allocated once, reused
func.func @chain(%x: tensor<1024xf32>) -> tensor<1024xf32> {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index

  %result = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %x) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %next = linalg.add ins(%acc, %acc) outs(%empty)
    scf.yield %next : tensor<1024xf32>
  }
  return %result : tensor<1024xf32>
}

// bufferized: single allocation hoisted out of loop
```

### canonicalization and folding

canonicalization simplifies IR using local rewrite patterns:

- constant folding: `add(constant(1), constant(2))` → `constant(3)`
- dead code elimination: remove unused operations
- algebraic simplification: `x + 0` → `x`, `x * 1` → `x`
- operation strength reduction: `x << 2` instead of `x * 4`

CSE (common subexpression elimination) removes duplicate computations.

### loop transformations

**loop interchange**: reorder loops for better cache access

```mlir
// before: column-major access pattern (bad for row-major arrays)
affine.for %i {
  affine.for %j {
    %a = affine.load %A[%j, %i]
  }
}

// after interchange
affine.for %j {
  affine.for %i {
    %a = affine.load %A[%j, %i]
  }
}
```

**loop skewing**: handle loop-carried dependencies

**unroll-and-jam**: unroll outer loop, jam inner bodies

```mlir
// unroll outer by 2, jam
affine.for %i = 0 to 1024 step 2 {
  affine.for %j {
    // iteration i
    work(%i, %j)
    // iteration i+1
    work(%i+1, %j)
  }
}
```

## MLIR vs traditional compilers

### vs LLVM

| aspect        | LLVM IR                     | MLIR                              |
| ------------- | --------------------------- | --------------------------------- |
| abstraction   | single-level, low-level     | multi-level, extensible           |
| types         | primitive types, pointers   | dialects define arbitrary types   |
| operations    | fixed instruction set       | extensible via dialects           |
| optimization  | SSA-based, dataflow         | multiple levels, dialect-specific |
| target domain | general purpose compilation | specialized for ML/HPC/DSLs       |
| control flow  | CFG with phi nodes          | blocks with arguments             |

LLVM IR is final lowering target. MLIR complements it by providing higher-level abstractions that preserve domain semantics during optimization.

### vs XLA HLO

XLA (Accelerated Linear Algebra) compiles TensorFlow/JAX to hardware.

HLO (High-Level Optimizer) is XLA's IR. StableHLO standardizes it.

| aspect        | HLO/StableHLO           | MLIR                                  |
| ------------- | ----------------------- | ------------------------------------- |
| scope         | ML-specific operations  | general compiler framework            |
| extensibility | fixed op set (~100 ops) | unlimited via dialects                |
| lowering      | HLO → LLVM directly     | progressive through multiple dialects |
| reusability   | XLA-specific            | infrastructure for many compilers     |
| abstractions  | single-level tensor ops | multi-level from high to low          |

StableHLO is now an MLIR dialect. XLA increasingly uses MLIR infrastructure.

### vs TVM

TVM provides ML compilation stack: Relay (graph IR) → TIR (tensor IR) → target code.

| aspect      | TVM                            | MLIR                                    |
| ----------- | ------------------------------ | --------------------------------------- |
| approach    | Python-driven with C++ runtime | C++ infrastructure with Python bindings |
| IR levels   | 2 levels (Relay, TIR)          | unlimited dialects                      |
| scheduling  | TVM schedule primitives        | transformation dialect + passes         |
| auto-tuning | AutoTVM, Ansor                 | typically external (like IREE)          |
| community   | focused on ML                  | broader (ML, HPC, languages)            |

both use progressive lowering. TVM emphasizes auto-tuning. MLIR emphasizes infrastructure reuse.

### reusability story

MLIR enables code sharing across compilers:

- **IREE**: uses StableHLO, Linalg, Vector, GPU dialects
- **Torch-MLIR**: reuses Linalg, Arith, Func
- **Flang**: uses MLIR for Fortran, shares LLVM lowering
- **CIRCT**: hardware design, reuses core MLIR patterns

dialects are mix-and-match. write optimization once, apply everywhere.

## production deployments

### TensorFlow ecosystem

**TFRT (TensorFlow Runtime)**: uses MLIR for graph optimization

- BEF (Binary Executable Format) generated from MLIR
- replaces legacy TensorFlow runtime

**TF-to-StableHLO-to-XLA**: canonical TensorFlow compilation path

- TensorFlow → StableHLO → XLA → hardware
- MLIR-based passes at each level

### IREE (Intermediate Representation Execution Environment)

MLIR-based ML compiler for edge/mobile/datacenter.

ingestion: TensorFlow (via IREE), PyTorch (via Torch-MLIR), JAX (via StableHLO)

compilation flow:

```
StableHLO/Torch → Linalg → Flow (data flow) → Stream (scheduling) → HAL (hardware abstraction)
```

targets: CPU (LLVM), GPU (Vulkan/CUDA/ROCm), mobile (Metal), edge (embedded).

focus: low-latency inference, small binary size, cross-platform portability.

notable: AMD submitted IREE-based SDXL to MLPerf (2025).

### Torch-MLIR

PyTorch integration into MLIR ecosystem.

two frontends:

- **TorchScript**: via `torch.jit.script`
- **TorchDynamo**: via `torch.compile` (newer, more coverage)

lowering paths:

- Torch dialect → Linalg → backend
- Torch dialect → TOSA → backend
- Torch dialect → StableHLO → XLA

enables: portable PyTorch models, hardware vendor integration, custom backends.

### Flang (Fortran compiler)

LLVM's Fortran frontend built entirely on MLIR.

FIR (Fortran IR) dialect:

- represents Fortran semantics (array slicing, complex numbers, etc.)
- progressive lowering: FIR → LLVM dialect → LLVM IR

demonstrates MLIR for traditional language compilation, not just ML.

### CIRCT (Circuit IR Compilers and Tools)

hardware design using MLIR.

dialects:

- **HW**: hardware structure (modules, instances)
- **Comb**: combinational logic
- **Seq**: sequential logic (registers, clocks)
- **SV**: SystemVerilog constructs

integrates with Chisel (Scala-based HDL). lowering: high-level hardware → Verilog.

### other production users

- **Google**: XLA, StableHLO, internal tooling
- **Meta**: PyTorch 2.0 compilation via Torch-MLIR
- **AMD**: IREE for GPUs, MLIR-based compiler stack
- **Intel**: oneAPI uses MLIR dialects
- **Modular**: Mojo language uses MLIR

## practical code examples

### simple dialect definition

define a toy dialect with `constant` and `add` operations:

```tablegen
// ToyDialect.td
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

def ConstantOp : Toy_Op<"constant"> {
  let summary = "constant operation";
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value)>
  ];
}

def AddOp : Toy_Op<"add", [NoMemoryEffect, Commutative]> {
  let summary = "element-wise addition";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

usage:

```mlir
func.func @example() -> tensor<2x3xf64> {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]> : tensor<2x3xf64>
  %2 = toy.add %0, %1 : tensor<2x3xf64>
  return %2 : tensor<2x3xf64>
}
```

### pattern rewriting example

constant folding for `add`:

```cpp
// ToyPatterns.cpp
struct ConstantFoldAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs().getDefiningOp<ConstantOp>();
    auto rhs = op.getRhs().getDefiningOp<ConstantOp>();

    if (!lhs || !rhs)
      return failure();

    // Fold: add(constant(a), constant(b)) -> constant(a + b)
    DenseElementsAttr lhsAttr = lhs.getValue();
    DenseElementsAttr rhsAttr = rhs.getValue();

    SmallVector<APFloat> results;
    for (auto [lhsVal, rhsVal] : llvm::zip(
           lhsAttr.getValues<APFloat>(),
           rhsAttr.getValues<APFloat>())) {
      results.push_back(lhsVal + rhsVal);
    }

    auto resultAttr = DenseElementsAttr::get(
      op.getType(), ArrayRef<APFloat>(results));

    rewriter.replaceOpWithNewOp<ConstantOp>(op, resultAttr);
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<ConstantFoldAdd>(context);
}
```

### tiling transformation

tile matmul with specific tile sizes:

```cpp
// TilingPass.cpp
void tileMatmul(linalg::MatmulOp matmul) {
  OpBuilder builder(matmul);

  SmallVector<int64_t> tileSizes = {256, 256, 128}; // M, N, K

  auto tilingOptions = linalg::LinalgTilingOptions()
    .setTileSizes(tileSizes)
    .setLoopType(linalg::LinalgTilingLoopType::Loops);

  FailureOr<linalg::TiledLinalgOp> tiled =
    linalg::tileLinalgOp(builder, matmul, tilingOptions);

  if (failed(tiled))
    return;

  matmul.replaceAllUsesWith(tiled->tensorResults);
  matmul.erase();
}
```

resulting IR:

```mlir
scf.for %i = %c0 to %c1024 step %c256 {
  scf.for %j = %c0 to %c1024 step %c256 {
    scf.for %k = %c0 to %c1024 step %c128 {
      %A_tile = tensor.extract_slice %A[%i, %k][256, 128][1, 1]
      %B_tile = tensor.extract_slice %B[%k, %j][128, 256][1, 1]
      %C_tile = tensor.extract_slice %C[%i, %j][256, 256][1, 1]

      %result_tile = linalg.matmul
        ins(%A_tile, %B_tile : tensor<256x128xf32>, tensor<128x256xf32>)
        outs(%C_tile : tensor<256x256xf32>) -> tensor<256x256xf32>

      %C_new = tensor.insert_slice %result_tile into %C[%i, %j][256, 256][1, 1]
      // update %C for next iteration
    }
  }
}
```

### conversion between dialects

convert linalg to loops:

```cpp
// ConvertLinalgToLoops.cpp
class MatmulToLoopsPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul, PatternRewriter &rewriter) const override {
    Location loc = matmul.getLoc();

    Value A = matmul.getInputs()[0];
    Value B = matmul.getInputs()[1];
    Value C = matmul.getOutputs()[0];

    auto ATy = A.getType().cast<MemRefType>();
    int64_t M = ATy.getShape()[0];
    int64_t K = ATy.getShape()[1];

    auto BTy = B.getType().cast<MemRefType>();
    int64_t N = BTy.getShape()[1];

    // Generate triple nested loop
    auto buildLoop = [&](int64_t ub) {
      Value ubVal = rewriter.create<arith::ConstantIndexOp>(loc, ub);
      Value lbVal = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value stepVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      return rewriter.create<scf::ForOp>(loc, lbVal, ubVal, stepVal);
    };

    auto iLoop = buildLoop(M);
    rewriter.setInsertionPointToStart(iLoop.getBody());

    auto jLoop = buildLoop(N);
    rewriter.setInsertionPointToStart(jLoop.getBody());

    auto kLoop = buildLoop(K);
    rewriter.setInsertionPointToStart(kLoop.getBody());

    Value i = iLoop.getInductionVar();
    Value j = jLoop.getInductionVar();
    Value k = kLoop.getInductionVar();

    // C[i,j] += A[i,k] * B[k,j]
    Value aVal = rewriter.create<memref::LoadOp>(loc, A, ValueRange{i, k});
    Value bVal = rewriter.create<memref::LoadOp>(loc, B, ValueRange{k, j});
    Value cVal = rewriter.create<memref::LoadOp>(loc, C, ValueRange{i, j});

    Value prod = rewriter.create<arith::MulFOp>(loc, aVal, bVal);
    Value sum = rewriter.create<arith::AddFOp>(loc, cVal, prod);

    rewriter.create<memref::StoreOp>(loc, sum, C, ValueRange{i, j});

    rewriter.eraseOp(matmul);
    return success();
  }
};
```

## references

- MLIR documentation: https://mlir.llvm.org/docs/
- ODS specification: https://mlir.llvm.org/docs/DefiningDialects/Operations/
- Linalg dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
- Affine dialect: https://mlir.llvm.org/docs/Dialects/Affine/
- Pattern rewriting: https://mlir.llvm.org/docs/PatternRewriter/
- Transform dialect: https://mlir.llvm.org/docs/Dialects/Transform/
- IREE: https://iree.dev/
- Torch-MLIR: https://github.com/llvm/torch-mlir
- StableHLO: https://openxla.org/stablehlo
- CIRCT: https://circt.llvm.org/
- Jeremy Kun's MLIR tutorials: https://www.jeremykun.com/tags/mlir/
