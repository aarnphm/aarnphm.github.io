#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define SHARED 1

sem_t empty, full; /* the global semaphores */
volatile int data; /* shared buffer         */
int numIters;

/* deposit 1, ..., numIters into the data buffer */
void *Producer(void *arg) {
  int id = *((int *)arg);
  printf("Producer %d created\n", id);
  for (int produced = 0; produced < numIters; produced++) {
    sem_wait(&empty);
    data = produced;
    sem_post(&full);
  }
  return NULL;
}

/* fetch numIters items from the buffer and sum them */
void *Consumer(void *arg) {
  int id = *((int *)arg);
  printf("Consumer %d created\n", id);
  int sum = 0;
  for (int consumed = 0; consumed < numIters; consumed++) {
    sem_wait(&full);
    sum += data;
    sem_post(&empty);
  }
  printf("For %d iterations, the sum of consumer %d is %d\n", numIters, id,
         sum);
  return (void *)(intptr_t)sum; // Return the sum as a pointer
}

/* main program: read command line and create threads */
int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <numIters> <numProducersAndConsumers>\n", argv[0]);
    exit(1);
  }

  numIters = atoi(argv[1]);
  int numProducersAndConsumers = atoi(argv[2]);
  pthread_t pids[numProducersAndConsumers], cids[numProducersAndConsumers];
  int ids[numProducersAndConsumers];
  int computedTotalSum = 0;

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
  sem_init(&empty, SHARED, 1); /* sem empty = 1 */
  sem_init(&full, SHARED, 0);  /* sem full = 0  */

  for (int i = 0; i < numProducersAndConsumers; i++) {
    ids[i] = i;
    pthread_create(&pids[i], &attr, Producer, &ids[i]);
    pthread_create(&cids[i], &attr, Consumer, &ids[i]);
  }

  for (int i = 0; i < numProducersAndConsumers; i++) {
    pthread_join(pids[i], NULL);
    void *sum;
    pthread_join(cids[i], &sum);
    computedTotalSum += (intptr_t)sum; // Collect the sum from each consumer
  }

  int expectedTotalSum =
      numProducersAndConsumers * numIters * (numIters - 1) / 2;
  printf("The expected total sum is %d\n", expectedTotalSum);
  printf("The computed total sum is %d\n", computedTotalSum);

  return 0;
}
