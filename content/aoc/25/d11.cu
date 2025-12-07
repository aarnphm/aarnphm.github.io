#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINES 1024
#define MAX_LINE_LEN 256

typedef struct {
  char** lines;
  int num_lines;
  int max_width;
} Input;

Input read_input(const char* filename) {
  Input input = {NULL, 0, 0};
  FILE* f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "could not open %s\n", filename);
    exit(1);
  }

  input.lines = (char**)malloc(MAX_LINES * sizeof(char*));
  char buffer[MAX_LINE_LEN];

  while (fgets(buffer, MAX_LINE_LEN, f) && input.num_lines < MAX_LINES) {
    int len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') buffer[--len] = '\0';
    if (len == 0) continue;

    input.lines[input.num_lines] = strdup(buffer);
    if (len > input.max_width) input.max_width = len;
    input.num_lines++;
  }

  fclose(f);
  return input;
}

void free_input(Input* input) {
  for (int i = 0; i < input->num_lines; i++) {
    free(input->lines[i]);
  }
  free(input->lines);
}

// device kernel stub
__global__ void solve_kernel(char* d_grid, int rows, int cols, int* d_result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    *d_result = 0;
  }
}

int p1(Input* input) {
  if (input->num_lines == 0) return 0;

  int rows = input->num_lines;
  int cols = input->max_width;
  size_t grid_size = rows * cols;

  // flatten grid to 1D array
  char* h_grid = (char*)calloc(grid_size, sizeof(char));
  for (int r = 0; r < rows; r++) {
    int len = strlen(input->lines[r]);
    memcpy(h_grid + r * cols, input->lines[r], len);
  }

  char* d_grid;
  int* d_result;
  int h_result = 0;

  cudaMalloc(&d_grid, grid_size);
  cudaMalloc(&d_result, sizeof(int));
  cudaMemcpy(d_grid, h_grid, grid_size, cudaMemcpyHostToDevice);

  solve_kernel<<<1, 1>>>(d_grid, rows, cols, d_result);

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_grid);
  cudaFree(d_result);
  free(h_grid);

  return h_result;
}

int p2(Input* input) {
  return 0;
}

int main() {
  Input input = read_input("d11.txt");

  printf("p1: %d\n", p1(&input));
  printf("p2: %d\n", p2(&input));

  free_input(&input);
  return 0;
}
