# paged attention profiling

The `paged_attention_cute` kernel shipped in `content/lectures/420/examples/10_cute_paged_attention_optimized.cu` is structured so each cooperative thread array handles a single `(sequence, head)` pair with warp-level online softmax. Profiling that kernel requires launching the standalone harness that file provides.

> [!info] build
> ```bash
> nvcc -std=c++20 -O3 -I/usr/local/cuda/include \
>   content/lectures/420/examples/10_cute_paged_attention_optimized.cu \
>   -o paged_attention_kernel
> ```

## ncu

> [!example] kernel-level metrics
> ```bash
> ncu --set full --target-processes all \
>   --kernel-name ::paged_attention_cute \
>   --launch-skip 0 --launch-count 1 \
>   ./paged_attention_kernel
> ```
>
> Key sections: `sm__pipe_fma_cycles_active.avg`, `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, and the dram throughput block. Expect single-warp occupancy with ~80% FP32 pipe utilization when `HEAD_DIM=128`.

## nsys

> [!example] timeline capture
> ```bash
> nsys profile --sample=cpu --trace=cuda,osrt,nvtx \
>   --cuda-graph-trace=graph-node \
>   --output=profiles/paged_attention ./paged_attention_kernel
> ```
>
> Inspect the `.qdrep` in NVIDIA Nsight Systems to confirm the launch stream performs a single short-lived kernel per `(sequence, head)` pair and that there are no serialized host synchronizations between pages.

The sandbox lacks an attached NVIDIA GPU, so `.ncu-rep` and `.qdrep` artifacts are not generated here.
