#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NODES 1024
#define MAX_EDGES 8192
#define MAX_NAME_LEN 16
#define MAX_LINE_LEN 512
#define THREADS_PER_BLOCK 256

typedef struct {
  int* adj_offset;
  int* adj_list;
  int num_nodes;
  int num_edges;
  int start_node;
  int end_node;
} Graph;

// name -> id mapping
typedef struct {
  char names[MAX_NODES][MAX_NAME_LEN];
  int count;
} NameTable;

// precomputed structure with persistent buffers
typedef struct {
  int* depth;
  int** levels;
  int* level_sizes;
  int max_depth;

  // persistent buffers
  int* d_adj_offset;
  int* d_adj_list;
  int* d_level;
  long long* d_paths;
} GPUContext;

int get_or_create_id(NameTable* table, const char* name) {
  for (int i = 0; i < table->count; i++) {
    if (strcmp(table->names[i], name) == 0) return i;
  }
  strcpy(table->names[table->count], name);
  return table->count++;
}

int get_id(NameTable* table, const char* name) {
  for (int i = 0; i < table->count; i++) {
    if (strcmp(table->names[i], name) == 0) return i;
  }
  return -1;
}

Graph* parse(const char* filename, NameTable* names) {
  FILE* f = fopen(filename, "r");

  int* from = (int*)malloc(MAX_EDGES * sizeof(int));
  int* to = (int*)malloc(MAX_EDGES * sizeof(int));
  int edge_count = 0;

  char line[MAX_LINE_LEN];
  while (fgets(line, MAX_LINE_LEN, f)) {
    int len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
    if (len == 0) continue;

    char* colon = strchr(line, ':');
    if (!colon) continue;

    *colon = '\0';
    int src_id = get_or_create_id(names, line);

    char* rest = colon + 1;
    char* token = strtok(rest, " ");
    while (token) {
      int dst_id = get_or_create_id(names, token);
      from[edge_count] = src_id;
      to[edge_count] = dst_id;
      edge_count++;
      token = strtok(NULL, " ");
    }
  }
  fclose(f);

  Graph* g = (Graph*)malloc(sizeof(Graph));
  g->num_nodes = names->count;
  g->num_edges = edge_count;
  g->start_node = get_id(names, "you");
  g->end_node = get_id(names, "out");

  g->adj_offset = (int*)calloc(g->num_nodes + 1, sizeof(int));
  g->adj_list = (int*)malloc(edge_count * sizeof(int));

  for (int i = 0; i < edge_count; i++) {
    g->adj_offset[from[i] + 1]++;
  }
  for (int i = 1; i <= g->num_nodes; i++) {
    g->adj_offset[i] += g->adj_offset[i - 1];
  }

  int* tmp_offset = (int*)malloc((g->num_nodes + 1) * sizeof(int));
  memcpy(tmp_offset, g->adj_offset, (g->num_nodes + 1) * sizeof(int));
  for (int i = 0; i < edge_count; i++) {
    g->adj_list[tmp_offset[from[i]]++] = to[i];
  }

  free(tmp_offset);
  free(from);
  free(to);
  return g;
}

__global__ void count_paths(int* adj_offset, int* adj_list, long long* paths, int* level_nodes, int level_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= level_size) return;

  int node = level_nodes[idx];
  long long node_paths = paths[node];
  if (node_paths == 0) return;

  for (int i = adj_offset[node]; i < adj_offset[node + 1]; i++) {
    atomicAdd((unsigned long long*)&paths[adj_list[i]], node_paths);
  }
}

void compute_depths(Graph* g, int* depth, int* max_depth) {
  int* in_degree = (int*)calloc(g->num_nodes, sizeof(int));
  for (int i = 0; i < g->num_edges; i++) {
    in_degree[g->adj_list[i]]++;
  }

  int* queue = (int*)malloc(g->num_nodes * sizeof(int));
  int head = 0, tail = 0;

  for (int i = 0; i < g->num_nodes; i++) {
    depth[i] = -1;
    if (in_degree[i] == 0) {
      queue[tail++] = i;
      depth[i] = 0;
    }
  }

  // BFS
  *max_depth = 0;
  while (head < tail) {
    int u = queue[head++];
    if (depth[u] > *max_depth) *max_depth = depth[u];

    for (int i = g->adj_offset[u]; i < g->adj_offset[u + 1]; i++) {
      int v = g->adj_list[i];
      if (--in_degree[v] == 0) {
        depth[v] = depth[u] + 1;
        queue[tail++] = v;
      }
    }
  }

  free(queue);
  free(in_degree);
}

// group nodes by depth level
void build_levels(Graph* g, int* depth, int max_depth, int** levels, int* level_sizes) {
  memset(level_sizes, 0, (max_depth + 1) * sizeof(int));
  for (int i = 0; i < g->num_nodes; i++) {
    if (depth[i] >= 0) level_sizes[depth[i]]++;
  }

  for (int d = 0; d <= max_depth; d++) {
    levels[d] = (int*)malloc(level_sizes[d] * sizeof(int));
  }

  int* idx = (int*)calloc(max_depth + 1, sizeof(int));
  for (int i = 0; i < g->num_nodes; i++) {
    if (depth[i] >= 0) {
      levels[depth[i]][idx[depth[i]]++] = i;
    }
  }
  free(idx);
}

GPUContext* init_ctx(Graph* g) {
  GPUContext* ctx = (GPUContext*)malloc(sizeof(GPUContext));

  ctx->depth = (int*)malloc(g->num_nodes * sizeof(int));
  compute_depths(g, ctx->depth, &ctx->max_depth);

  ctx->levels = (int**)malloc((ctx->max_depth + 1) * sizeof(int*));
  ctx->level_sizes = (int*)calloc(ctx->max_depth + 1, sizeof(int));
  build_levels(g, ctx->depth, ctx->max_depth, ctx->levels, ctx->level_sizes);

  // allocate persistent buffer
  // NOTE: Make sure to only call this function once.
  cudaMalloc(&ctx->d_adj_offset, (g->num_nodes + 1) * sizeof(int));
  cudaMalloc(&ctx->d_adj_list, g->num_edges * sizeof(int));
  cudaMalloc(&ctx->d_level, g->num_nodes * sizeof(int));
  cudaMalloc(&ctx->d_paths, g->num_nodes * sizeof(long long));

  cudaMemcpy(ctx->d_adj_offset, g->adj_offset,
             (g->num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ctx->d_adj_list, g->adj_list,
             g->num_edges * sizeof(int), cudaMemcpyHostToDevice);

  return ctx;
}

void compute_paths(GPUContext* ctx, Graph* g, int src, long long* h_paths) {
  int src_depth = ctx->depth[src];

  // reset paths array
  cudaMemset(ctx->d_paths, 0, g->num_nodes * sizeof(long long));
  long long one = 1;
  cudaMemcpy(&ctx->d_paths[src], &one, sizeof(long long), cudaMemcpyHostToDevice);

  // propagate through levels
  for (int d = src_depth; d < ctx->max_depth; d++) {
    int level_size = ctx->level_sizes[d];
    if (level_size == 0) continue;

    cudaMemcpy(ctx->d_level, ctx->levels[d],
               level_size * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (level_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    count_paths<<<blocks, THREADS_PER_BLOCK>>>(ctx->d_adj_offset, ctx->d_adj_list, ctx->d_paths, ctx->d_level, level_size);
  }

  cudaDeviceSynchronize();
  cudaMemcpy(h_paths, ctx->d_paths, g->num_nodes * sizeof(long long), cudaMemcpyDeviceToHost);
}

long long p1(GPUContext* ctx, Graph* g) {
  if (g->start_node < 0 || g->end_node < 0) return 0;

  long long* paths = (long long*)calloc(g->num_nodes, sizeof(long long));
  compute_paths(ctx, g, g->start_node, paths);
  long long result = paths[g->end_node];
  free(paths);
  return result;
}

long long p2(GPUContext* ctx, Graph* g, int svr, int out, int dac, int fft) {
  long long* from_svr = (long long*)calloc(g->num_nodes, sizeof(long long));
  long long* from_dac = (long long*)calloc(g->num_nodes, sizeof(long long));
  long long* from_fft = (long long*)calloc(g->num_nodes, sizeof(long long));

  compute_paths(ctx, g, svr, from_svr);
  compute_paths(ctx, g, dac, from_dac);
  compute_paths(ctx, g, fft, from_fft);

  // two orderings: svr->dac->fft->out or svr->fft->dac->out
  long long route1 = from_svr[dac] * from_dac[fft] * from_fft[out];
  long long route2 = from_svr[fft] * from_fft[dac] * from_dac[out];

  free(from_svr);
  free(from_dac);
  free(from_fft);

  return route1 + route2;
}

int main() {
  NameTable names = {0};
  Graph* g = parse("d11.txt", &names);

  GPUContext* ctx = init_ctx(g);

  long long p1_result = p1(ctx, g);
  printf("p1: %lld\n", p1_result);

  int svr = get_id(&names, "svr");
  int out = get_id(&names, "out");
  int dac = get_id(&names, "dac");
  int fft = get_id(&names, "fft");

  long long p2_result = p2(ctx, g, svr, out, dac, fft);
  printf("p2: %lld\n", p2_result);

  cudaFree(ctx->d_adj_offset);
  cudaFree(ctx->d_adj_list);
  cudaFree(ctx->d_level);
  cudaFree(ctx->d_paths);

  for (int d = 0; d <= ctx->max_depth; d++) {
    free(ctx->levels[d]);
  }
  free(ctx->levels);
  free(ctx->level_sizes);
  free(ctx->depth);
  free(ctx);
  free(g->adj_offset);
  free(g->adj_list);
  free(g);
  return 0;
}
