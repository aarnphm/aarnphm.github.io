# clustered

```python
clustered(*, size, rdma=False)
```

Run a Function or Server on a cluster of colocated, networked containers.

Apply below `@app.function()`, `@app.cls()`, or `@app.server()`. Each container
must request all GPUs on its host (for example, `gpu="H100:8"`); CPU-only
clusters are not supported. A clustered Cls can expose only one method.
Use a Server for HTTP serving; clustered Web Functions are not supported.

Function inputs are broadcast to every container, and only rank 0's output
is returned. Server requests are routed only to rank 0 and are not broadcast
to the other containers. Use `modal.Cluster.from_context()` inside a container
to discover its rank and the cluster's container IP addresses:

```python notest
cluster = modal.Cluster.from_context()
rank = cluster.container_rank()
container_ips = cluster.container_ips()
```

`min_containers`, `max_containers`, and `buffer_containers` count individual
containers and must be multiples of `size`. For example, `size=4` with
`min_containers=8` keeps two clusters warm.

See the [multi-node clusters guide](https://modal.com/docs/guide/multi-node-clusters)
for hardware requirements and networking details.

Parameters:
size: int
Number of containers in each cluster.
rdma: bool = False
Request RDMA networking for fast communication between nodes, such as
GPU collectives during distributed training. With False, containers
can still communicate over the private IP network without requiring
RDMA-capable placement.
