---
date: '2025-10-05'
description: Tenstorrent hardware as explicit tiled dataflow, with current Wormhole and Blackhole specs
id: Tenstorrent
modified: 2026-08-26 09:31:16 GMT-04:00
seealso:
  - '[[thoughts/GPU programming]]'
  - '[[thoughts/MLIR]]'
  - '[[thoughts/Compiler]]'
  - "[[thoughts/pdfs/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf|Jasmina's talk at HotChip]]"
  - '[[thoughts/XLA]]'
  - '[[thoughts/PyTorch]]'
tags:
  - ml
  - asic
  - hardware
title: Tenstorrent
---

Tenstorrent is useful when the model already looks like tiled dataflow. The machine gives you many Tensix cores, local SRAM, and Ethernet links. The cost is that the software has to place tiles and schedule movement.

The architectural bet is simple:

1. Move tensors as tiles.
2. Keep tile queues in explicit local SRAM.
3. Split compute from data movement.
4. Scale the same fabric across a card, a box, and a rack.

That is the part worth tracking. The rest is SKU fog.

## current cards

The product line changed enough that stale numbers become actively misleading. Current public docs give this shape. [@tenstorrent2024wormhole; @tenstorrent2024blackhole; @tenstorrent2026cards]

| card  |               chip | active Tensix cores |      memory | memory bandwidth | board power | fabric                                        |
| ----- | -----------------: | ------------------: | ----------: | ---------------: | ----------: | --------------------------------------------- |
| n150  |           Wormhole |                  72 | 12 GB GDDR6 |         288 GB/s |       160 W | two QSFP-DD 200G ports                        |
| n300  | two Wormhole chips |                 128 | 24 GB GDDR6 |         576 GB/s |       300 W | two QSFP-DD 200G ports plus 200G chip-to-chip |
| p100a |          Blackhole |                 120 | 28 GB GDDR6 |         448 GB/s |       300 W | no external QSFP-DD fabric                    |
| p150a |          Blackhole |                 120 | 32 GB GDDR6 |         512 GB/s |       300 W | four QSFP-DD 800G ports                       |
| p150b |          Blackhole |                 120 | 32 GB GDDR6 |         512 GB/s |       300 W | four QSFP-DD 800G ports                       |

Blackhole is a 120 Tensix core part in current docs. It also has 16 SiFive x280 big RISC-V cores and 180 MB of on-chip SRAM. Wormhole has 80 Tensix cores at chip level, with fewer active cores exposed by n150 and n300 board products.

The easiest arithmetic trap is the Blackhole p150 fabric number:

$$
4 \cdot 800\ \mathrm{Gb/s}
  = 3200\ \mathrm{Gb/s}
  = 400\ \mathrm{GB/s}.
$$

The raw line rate is 400 GB/s before protocol overhead, topology, collective algorithms, and congestion. Writing this as 3.2 TB/s is a bits-to-bytes bug.

The NVIDIA H100 SXM row is useful as a scale marker because NVIDIA publishes both HBM bandwidth and NVLink bandwidth in one table. That table lists 3.35 TB/s HBM bandwidth and 900 GB/s NVLink bandwidth for the 80 GB SXM part. [@nvidia2026h100]

$$
\frac{900\ \mathrm{GB/s}}{400\ \mathrm{GB/s}} = 2.25.
$$

So a p150 card's four-port external fabric has about $1/2.25$ of an H100 SXM NVLink aggregate budget at raw line rate. The two fabrics use different topologies. The ratio establishes one narrow point: fabric math becomes a first-order constraint once collectives dominate.

## the useful atom

The useful atom is one Tensix core running compute kernels and data-movement kernels that talk through circular buffers.

The local picture is:

1. A reader kernel pulls tiles from DRAM or another core.
2. It pushes those tiles into a circular buffer in local SRAM.
3. A compute kernel pops tiles, runs the tile operation, and pushes output tiles.
4. A writer kernel drains output tiles back to DRAM or onward through the fabric.

This is why Tenstorrent feels closer to a dataflow machine than to a CUDA block with a large invisible runtime behind it. CUDA lets a lot of people defer locality decisions to libraries. TT-Metalium makes locality part of the program text.

A circular buffer in TT-Metalium is a bounded queue with one producer and one consumer. The important API fact is that `cb_reserve_back` blocks until the requested pages are free. [@tenstorrent2026circularbuffers]

That changes the failure model. A bad program can starve, deadlock, or feed tiles in the wrong order. Treat the buffer as a synchronized queue rather than a C ring buffer that silently overwrites unread tiles.

The performance question for a kernel is mechanical:

$$
T_{\mathrm{step}}
  =
  \max\left(
    T_{\mathrm{read}},
    T_{\mathrm{compute}},
    T_{\mathrm{write}}
  \right)
  + T_{\mathrm{sync}}.
$$

If the pipeline is healthy, reads, compute, and writes overlap. If one stage stalls, all stages inherit the slowest stage. This is the whole game. Keep the queues fed, keep the tile order correct, keep synchronization out of the inner loop.

## what the software stack is now

The stack has four public layers worth naming.

1. TT-Metalium is the low-level C++ programming model for kernels, buffers, command queues, and devices. [@tenstorrent2024metalium; @ttmetal2024]
2. TT-NN is the Python and C++ neural-network op layer on top of TT-Metalium. Current docs describe more than 200 operations, mesh devices, custom ops, graph tracing, parameter caching, and comparison mode. [@tenstorrent2026ttnn]
3. TT-MLIR is the compiler infrastructure. Public docs name TTIR as the frontend dialect and TTNN or TTMetal as backend paths, with flatbuffer serialization and runtime support. [@ttmlir2024; @tenstorrent2026ttmlir]
4. TT-XLA packages a PJRT plugin with thin JAX and PyTorch/XLA wrappers. It feeds StableHLO into the Tenstorrent compiler path. [@tenstorrent2026ttxla]

The current question is coverage and maturity. A model is easy when it already has a TT-NN lowering, a TT-Metal demo, or a TT-XLA path with the needed ops. A model gets expensive when it falls through to custom kernels, strange layouts, or collectives that the compiler cannot yet make boring.

## observed model demos

The official `tt-metal` README is the public smoke-test table I trust most because it names hardware, release, and commit context. The table warns that performance can move under other runtimes. These demo numbers belong to the listed releases and configurations. [@ttmetal2024]

| model                   |           hardware | tensor parallelism | TT-Metalium release | time to first token | tokens/s/user | aggregate tokens/s |
| ----------------------- | -----------------: | -----------------: | ------------------: | ------------------: | ------------: | -----------------: |
| Llama 3.3 70B           |   Galaxy, Wormhole |                 32 |         v0.65.0-rc7 |               53 ms |          72.5 |             2268.8 |
| Qwen 2.5 7B             |     n300, Wormhole |                  2 |        v0.62.0-rc35 |              109 ms |          22.1 |              707.2 |
| Qwen 2.5 72B            | QuietBox, Wormhole |                  8 |        v0.62.0-rc25 |              223 ms |          15.4 |              492.8 |
| Whisper distil-large-v3 |     n150, Wormhole |                  1 | v0.65.0-dev20251208 |              163 ms |         105.0 |              105.0 |
| Whisper distil-large-v3 |    p150, Blackhole |                  1 | v0.65.0-dev20251208 |               63 ms |         263.4 |              263.4 |

The useful ratio in that table is the same Whisper model on n150 and p150:

$$
\frac{263.4}{105.0} \approx 2.51.
$$

On that demo, Blackhole p150 produces about $2.5{:}1$ tokens per second against Wormhole n150. That ratio should be read as a software-stack measurement because it includes model implementation, runtime release, compiler behavior, clock, memory bandwidth, and kernel coverage.

## where the architecture wins

Tenstorrent has a clean shot when the workload has these properties:

1. Dense tile math.
2. Predictable memory movement.
3. Enough reuse to pay for explicit local SRAM scheduling.
4. Coarse communication that can ride Ethernet fabric.
5. A compiler or hand-written kernel path that already covers the op set.

That describes a lot of inference. It also describes training kernels after enough compiler work. The bet gets worse as soon as the workload needs fine-grained synchronization, irregular indexing, dynamic shapes, or frequent all-reduce across a large topology.

The hardware wants a program shaped like this:

$$
\text{tiles in}
  \rightarrow
\text{local SRAM queues}
  \rightarrow
\text{Tensix compute}
  \rightarrow
\text{tiles out}.
$$

The compiler has to keep that shape intact across the whole graph. Once the graph falls back to the host, or inserts extra layout conversions, the machine starts paying for its own explicitness.

## the real comparison to GPUs

The honest comparison is ecosystem depth per watt-dollar, with kernel coverage as the hidden multiplier.

For a mature CUDA path, the working set is:

$$
\text{model}
  \rightarrow
\text{PyTorch}
  \rightarrow
\text{cuDNN or CUTLASS or custom CUDA}
  \rightarrow
\text{GPU}.
$$

For Tenstorrent, the working set is:

$$
\text{model}
  \rightarrow
\text{TT-NN, TT-XLA, or TT-Forge}
  \rightarrow
\text{TT-MLIR}
  \rightarrow
\text{TT-Metalium}
  \rightarrow
\text{Tensix}.
$$

That extra visibility is powerful when the path is optimized. It is painful when a needed op has no mature lowering. Useful due diligence starts with the exact model, sequence length, batch shape, precision, collective pattern, and runtime release. Peak FLOPS alone cannot answer that query.

## scratch rules

When checking a Tenstorrent claim, use this order:

1. Read the current hardware docs for the exact card.
2. Convert all fabric numbers from bits to bytes.
3. Check whether the claim is chip-level, card-level, box-level, or rack-level.
4. Look for an official TT-Metal demo or a source commit for the model.
5. Treat marketing peak numbers as hardware ceilings.
6. Treat model demo numbers as release-bound software evidence.

Most stale Tenstorrent takes fail at step 2 or step 3. The chip/card boundary is a chaos goblin. Kill it with units.
