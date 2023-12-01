#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SHARED 1
#define NUM_THREADS 2

struct Args {
  int *a;
  int l;
  int u;
  int n;
  double avg;
};

void *worker(struct Args *arg) {
  // arg.a has arg.n elements && 0 <= arg.l <= arg.u <= arg.n
  struct Args *args = (struct Args *)arg;
  int *a = args->a;
  int l = args->l;
  int u = args->u;
  double partial_sum = 0.0;

  for (int i = l; i < u; i++) {
    partial_sum += a[i];
  }

  args->avg = partial_sum;

  pthread_exit(NULL);
}

double sequentialaverage(int a[], int n) {
  // a has n elements
  double s = 0;
  for (int i = 0; i < n; i++)
    s += a[i];
  return s / n;
}

static double parallelaverage(int a[], int n) {
  // a has n elements
  pthread_t threads[NUM_THREADS];
  struct Args args[NUM_THREADS];

  for (int i = 0; i < NUM_THREADS; i++) {
    args[i].a = a;
    args[i].l = i * n / NUM_THREADS;
    args[i].u = (i == NUM_THREADS - 1) ? n : (i + 1) * (n / NUM_THREADS);
    args[i].n = n;

    pthread_create(&threads[i], NULL, (void *)worker, (void *)&args[i]);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  double total_sum = 0.0;

  for (int i = 0; i < NUM_THREADS; i++) {
    total_sum += args[i].avg;
  }

  return total_sum / n;
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  int a[n];
  srand(time(NULL));
  for (int i = 0; i < n; i++)
    a[i] = rand() % 10000;

  struct timeval start, end;
  gettimeofday(&start, 0);
  double avg = sequentialaverage(a, n);
  gettimeofday(&end, 0);
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  long elapsed = seconds * 1e6 + microseconds;
  printf("Sequential: %f Time: %i microseconds\n", avg, elapsed);

  gettimeofday(&start, 0);
  avg = parallelaverage(a, n);
  gettimeofday(&end, 0);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds * 1e6 + microseconds;
  printf("Parallel:   %f Time: %i microseconds\n", avg, elapsed);
}
