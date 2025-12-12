import cutlass.cute as cute, numpy as np
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import T
from cutlass.cute.runtime import from_dlpack
from collections import defaultdict

THREADS = 256


def parse(filename: str):
  graph, names = defaultdict(list), {}

  def get_id(name):
    if name not in names: names[name] = len(names)
    return names[name]

  with open(filename) as f:
    for line in f:
      line = line.strip()
      if not line or ':' not in line: continue
      src, rest = line.split(':', 1)
      get_id(src := src.strip())
      for dst in rest.split():
        get_id(dst)
        graph[src].append(dst)
  return dict(graph), names


def build_csr(graph: dict, names: dict):
  n = len(names)
  edge_counts = np.zeros(n + 1, dtype=np.int32)
  for name, neighbors in graph.items():
    edge_counts[names[name] + 1] = len(neighbors)

  adj_offset = np.cumsum(edge_counts).astype(np.int32)
  adj_list = np.zeros(adj_offset[-1], dtype=np.int32)
  temp = adj_offset.copy()
  for name, neighbors in graph.items():
    src_id = names[name]
    for dst in neighbors:
      adj_list[temp[src_id]] = names[dst]
      temp[src_id] += 1
  return adj_offset, adj_list


def topo_depth(adj_offset: np.ndarray, adj_list: np.ndarray, n: int):
  in_deg = np.zeros(n, dtype=np.int32)
  for v in adj_list:
    in_deg[v] += 1

  depth = np.full(n, -1, dtype=np.int32)
  queue = [i for i in range(n) if in_deg[i] == 0]
  for i in queue:
    depth[i] = 0

  head, max_d = 0, 0
  while head < len(queue):
    u = queue[head]
    head += 1
    max_d = max(max_d, depth[u])
    for i in range(adj_offset[u], adj_offset[u + 1]):
      v = adj_list[i]
      in_deg[v] -= 1
      if in_deg[v] == 0:
        depth[v] = depth[u] + 1
        queue.append(v)
  return depth, np.array(queue, dtype=np.int32), max_d


def group_levels(depth: np.ndarray, max_d: int):
  levels = [[] for _ in range(max_d + 1)]
  for i, d in enumerate(depth):
    if d >= 0:
      levels[d].append(i)
  return [np.array(l, dtype=np.int32) for l in levels]


@cute.kernel
def propagate_kernel(
  adj_offset: cute.Tensor, adj_list: cute.Tensor, paths: cute.Tensor, level_nodes: cute.Tensor, level_size: int
):
  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
  idx = bidx * THREADS + tidx
  if idx >= level_size:
    return

  node = level_nodes[idx]
  node_paths = paths[node]
  if node_paths == 0:
    return

  # https://github.com/NVIDIA/cutlass/issues/2346
  for i in range(adj_offset[node], adj_offset[node + 1]):
    neighbor = adj_list[i]
    ptr = paths.iterator + neighbor
    nvvm.atomicrmw(res=T.i64(), op=nvvm.AtomicOpKind.ADD, ptr=ptr.llvm_ptr, a=node_paths.ir_value())


@cute.jit
def compute_paths(
  adj_offset: cute.Tensor,
  adj_list: cute.Tensor,
  paths: cute.Tensor,
  levels: list,
  src_depth: int,
  max_depth: int,
  stream,
):
  for d in range(src_depth, max_depth):
    level = levels[d]
    sz = len(level)
    if sz == 0: continue

    level_t = from_dlpack(level.__dlpack__())
    blocks = (sz + THREADS - 1) // THREADS
    propagate_kernel(adj_offset, adj_list, paths, level_t, sz).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1], stream=stream)


def paths_from(adj_offset, adj_list, src, n, topo_order):
  paths = np.zeros(n, dtype=np.int64)
  paths[src] = 1
  for u in topo_order:
    if paths[u] == 0: continue
    for i in range(adj_offset[u], adj_offset[u + 1]): paths[adj_list[i]] += paths[u]
  return paths


def main():
  graph, names = parse('d11.txt')
  n = len(names)
  adj_offset, adj_list = build_csr(graph, names)
  depth, topo_order, max_d = topo_depth(adj_offset, adj_list, n)

  get = lambda k: names.get(k, -1)
  you, out, svr, dac, fft = get('you'), get('out'), get('svr'), get('dac'), get('fft')

  p1 = paths_from(adj_offset, adj_list, you, n, topo_order)[out]
  print(f'p1: {p1}')

  from_svr = paths_from(adj_offset, adj_list, svr, n, topo_order)
  from_dac = paths_from(adj_offset, adj_list, dac, n, topo_order)
  from_fft = paths_from(adj_offset, adj_list, fft, n, topo_order)

  p2 = from_svr[dac] * from_dac[fft] * from_fft[out] + from_svr[fft] * from_fft[dac] * from_dac[out]
  print(f'p2: {p2}')


if __name__ == '__main__': main()
