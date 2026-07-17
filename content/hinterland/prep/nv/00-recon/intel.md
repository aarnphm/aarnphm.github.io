---
date: '2026-07-16'
description: source-backed role and interview evidence for NVIDIA JR2015076
id: intel
modified: 2026-07-16 13:15:50 GMT-04:00
tags:
  - cs
title: NVIDIA AI inference systems intel
---

## role snapshot

The target is [Software Engineer, AI Inference Systems, New College Graduate 2026](https://jobs.nvidia.com/careers/job/893394392298?ncid=prsy-896405), NVIDIA requisition JR2015076. The listing names the CentML team, Santa Clara, full-time work, and an onsite arrangement.

As of July 16, 2026, NVIDIA's normal careers search no longer returns this job. The [NVIDIA detail endpoint](https://jobs.nvidia.com/api/pcsx/position_details?position_id=893394392298&domain=nvidia.com&hl=en) still returns the original record after a careers-page session is established. It marks the record as fallback data and sets the public posting time to zero. The role is currently delisted or inactive, while the job description remains recoverable. The [canonical Workday URL](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Software-Engineer--AI-Inference-Systems---New-College-Graduate-2026_JR2015076) provides the same title and requisition.

## what the job owns

NVIDIA describes five main responsibilities:

- Add vLLM features for new models and NVIDIA GPU hardware. Optimize speculative decoding, parallel execution, and prefill-decode disaggregation.
- Develop and benchmark hand-tuned and compiler-generated GPU kernels. Work on fusion, autotuning, memory layout, DSLs, and compiler infrastructure.
- Build inference benchmarks and contribute to MLPerf Inference submissions.
- Schedule and orchestrate containerized inference deployments across GPU clusters and clouds.
- Publish ML systems research and move useful prototypes into NVIDIA products.

The required foundation includes Python, C or C++, algorithms, data structures, operating systems, computer architecture, parallel programming, distributed systems, deep learning, CUDA, GPU memory hierarchy, streams, NCCL, Docker, Kubernetes, Slurm, Linux namespaces, and cgroups.

The preferred list names vLLM, SGLang, Triton, TorchDynamo, Inductor, MLIR, LLVM, XLA, CUTLASS, CUDA Graphs, Tensor Cores, containerd, CRI-O, CRIU, cloud platforms, infrastructure as code, CI/CD, observability, open-source work, and publications.

## evidence rules

This kit uses these labels:

| label   | meaning                                                                                                             |
| ------- | ------------------------------------------------------------------------------------------------------------------- |
| high    | A first-person candidate report names the exact question or gives enough detail to identify it.                     |
| medium  | An anonymous report or retelling gives a concrete prompt, with some missing context.                                |
| low     | A secondary source attributes a prompt to NVIDIA and the original report could not be recovered.                    |
| tagged  | A public company-tag dataset associates the LeetCode problem with NVIDIA. It is not an individual interview report. |
| derived | The prompt was written for this kit from the job description. NVIDIA did not publish or confirm it.                 |

## strongest adjacent evidence

The closest public completed interview report is for an NVIDIA Deep Learning Software Engineer, Inference role. The first screen asked [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/), then moved into CUDA kernel optimization, vLLM, and model serving. See the [candidate report](https://leetcode.com/discuss/post/6809037/).

A candidate for Senior Software Engineer, Deep Learning Inference reported [LRU Cache](https://leetcode.com/problems/lru-cache/) in HackerRank. See the [candidate follow-up](https://www.reddit.com/r/womenintech/comments/1nme839/how_to_prep_for_nvidia_senior_software_engineer/).

A recent anonymous inference-optimization retelling reports tensor quantization, cache-aware matrix transpose, operation-tree pruning, graph validation, and C++ debugging with memory and race bugs. This is useful and less verifiable than the first-person reports. See the [inference optimization report](https://www.reddit.com/r/OfferEngineering/comments/1tsd80v/nvidia_senior_sde_interview_experience/).

## what this predicts for the first rounds

The role and reports point to ordinary coding problems with systems follow-ups:

- A tree question can become computation-graph serialization or pruning.
- A graph question can become dependency validation or execution scheduling.
- A cache question can become KV-cache policy, ownership, and concurrency.
- A matrix question can become strides, layout, locality, and tiling.
- A heap or interval question can become batching and GPU assignment.
- A queue question can become a bounded producer-consumer buffer.
- A bit question can become alignment, packed state, or fixed-width arithmetic.

Expect the interviewer to care about constant factors, copies, allocation, locality, ownership, iterator invalidation, contention, and failure behavior after the algorithm is correct.

## source map

- [NVIDIA careers listing](https://jobs.nvidia.com/careers/job/893394392298?ncid=prsy-896405)
- [NVIDIA Workday listing](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Software-Engineer--AI-Inference-Systems---New-College-Graduate-2026_JR2015076)
- [NVIDIA careers detail endpoint](https://jobs.nvidia.com/api/pcsx/position_details?position_id=893394392298&domain=nvidia.com&hl=en)
- [Public NVIDIA company-tag page](https://codejeet.com/company/nvidia)
- [Underlying company-tag dataset](https://github.com/liquidslr/interview-company-wise-problems/tree/main/Nvidia)
- [Closest inference interview report](https://leetcode.com/discuss/post/6809037/)
- [NVIDIA software engineer report from December 2021](https://leetcode.com/discuss/post/1632862/nvidia-software-engineer-10-dec-2021/)
- [NVIDIA system software report from 2023](https://leetcode.com/discuss/post/4032849/nvidia-interview-or-off-campus-or-2023/)
- [NVIDIA compiler engineer report](https://www.geeksforgeeks.org/interview-experiences/nvidia-interview-experience-for-compiler-engineer/)
