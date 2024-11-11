/*
 ============================================================================
 Name        : lab6-0.c ---Robert, 2024 version
 Author      : Dr Asghar Bokhari
 Version     : 2024
 Copyright   : Your copyright notice
 Description : Create three tasks with 0 priority and SCHED_OTHER, to observer
 the need for synchronization
 ============================================================================
 */

// #define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define MY_PRIORITY1                                                           \
  (0) /* we use max value of 49 as the PRREMPT_RT use 50                       \
         as the priority of kernel tasklets                                    \
         and interrupt handler by default */
#define MY_PRIORITY2 (0)
#define MY_PRIORITY3 (0)
#define NUM_TASKS (3)
const int TASK_PRIORITIES[NUM_TASKS] = {MY_PRIORITY1, MY_PRIORITY2,
                                        MY_PRIORITY3};

#define MAX_SAFE_STACK                                                         \
  (8 * 1024) /* The maximum stack size which is                                \
                guaranteed safe to access without                              \
                faulting */

#define NSEC_PER_SEC (1000000000) /* The number of nsecs per sec. */
#define INTERVAL (200000)

void stack_prefault(void) {

  unsigned char dummy[MAX_SAFE_STACK];

  memset(dummy, 0, MAX_SAFE_STACK);
  return;
}

char *PolicyString(int policy) {
  switch (policy) {
  case SCHED_OTHER:
    return "SCHED_OTHER";
  // case SCHED_IDLE:
  //	return "SCHED_IDLE";
  // case SCHED_BATCH:
  //			return "SCHED_BATCH";
  case SCHED_FIFO:
    return "SCHED_FIFO";
  case SCHED_RR:
    return "SCHED_RR";
  }
  return "undefined policy";
}

// Function that will be used by Task1
void *tfun1(void *n) {
  pthread_attr_t getattr;
  struct sched_param param;
  int SchedPolicy;
  struct timespec t;
  int interval = INTERVAL;
  int num_runs =
      20; // Instead of infinite loop, use finite number of iterations

  /* Assign priority and scheduling policy to the task */
  // however, calling the sched_setscheduler() will put the FIFO or RR thread at
  // the start of the queue list to run. which will make the last created thread
  // of equal priority to run first.
  /*
          param.sched_priority = MY_PRIORITY2;
          if(sched_setscheduler(0, SCHED_OTHER, &param) == -1) {
                  perror("sched_setscheduler failed");
                  exit(-1);
          }
  */

  /* Lock memory */
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
    perror("mlockall failed");
    exit(-2);
  }

  /* Pre-fault our stack */
  stack_prefault();

  pthread_getattr_np(pthread_self(), &getattr);
  pthread_attr_getschedparam(&getattr, &param);
  pthread_attr_getschedpolicy(&getattr, &SchedPolicy);

  clock_gettime(CLOCK_MONOTONIC, &t);

  while (num_runs) {

    printf("Hello! This is task1, %s, Priority:%d\n", PolicyString(SchedPolicy),
           param.sched_priority);

    // sleep for the first interval
    /* calculate next shot */
    t.tv_nsec += interval;

    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }

    /* wait until next shot */
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    // sleep for one more interval
    t.tv_nsec += interval;
    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    num_runs = num_runs - 1;
  }
  return NULL;
}

void *tfun2(void *n) {
  pthread_attr_t getattr;
  struct sched_param param;
  int SchedPolicy;
  struct timespec t;
  int interval = INTERVAL;
  int num_runs = 20; // replaced infinite loop

  /* Assign priority and scheduling policy to the task */
  /*
          param.sched_priority = MY_PRIORITY2;
          if(sched_setscheduler(0, SCHED_OTHER, &param) == -1) {
                  perror("sched_setscheduler failed");
                  exit(-1);
          }
  */
  /* Lock memory */
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
    perror("mlockall failed");
    exit(-2);
  }

  /* Pre-fault our stack */
  stack_prefault();

  pthread_getattr_np(pthread_self(), &getattr);
  pthread_attr_getschedparam(&getattr, &param);
  pthread_attr_getschedpolicy(&getattr, &SchedPolicy);

  clock_gettime(CLOCK_MONOTONIC, &t);

  while (num_runs) {

    printf("Hello! This is task2, %s, Priority:%d\n", PolicyString(SchedPolicy),
           param.sched_priority);

    // sleep for the first interval
    /* calculate next shot */
    t.tv_nsec += interval;

    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }

    /* wait until next shot */
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    // sleep for one more interval
    t.tv_nsec += interval;
    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    num_runs = num_runs - 1;
  }
  return NULL;
}

void *tfun3(void *n) {
  pthread_attr_t getattr;
  struct sched_param param;
  int SchedPolicy;
  struct timespec t;
  int interval = INTERVAL;
  int num_runs = 20; // replaced infinite loop

  /* Declare ourself as a real time task */
  /*
          param.sched_priority = MY_PRIORITY3;
          if(sched_setscheduler(0, SCHED_OTHER, &param) == -1) {
                  perror("sched_setscheduler failed");
                  exit(-1);
          }
  */
  /* Lock memory */
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
    perror("mlockall failed");
    exit(-2);
  }

  /* Pre-fault our stack */
  stack_prefault();

  pthread_getattr_np(pthread_self(), &getattr);
  pthread_attr_getschedparam(&getattr, &param);
  pthread_attr_getschedpolicy(&getattr, &SchedPolicy);

  clock_gettime(CLOCK_MONOTONIC, &t);

  while (num_runs) {

    printf("Hello! This is task3, %s, Priority:%d\n", PolicyString(SchedPolicy),
           param.sched_priority);

    // sleep for the first interval
    /* calculate next shot */
    t.tv_nsec += interval;

    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }

    /* wait until next shot */
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    // sleep for one more interval
    t.tv_nsec += interval;
    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    num_runs = num_runs - 1;
  }
  return NULL;
}

int main(int argc, char *argv[]) {
  int i;
  cpu_set_t cpus;
  // Force the program to run on one cpu,
  CPU_ZERO(&cpus);   // Initialize cpus to nothing clear previous info if any
  CPU_SET(0, &cpus); // Set cpus to a cpu number zero here

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpus) < 0)
    perror("set affinity");

  pthread_t pthreadIDS[NUM_TASKS];
  void *TASKNAMES[NUM_TASKS] = {tfun1, tfun2, tfun3};
  pthread_attr_t attr;

  // Create three threads
  for (i = 0; i < NUM_TASKS; i++) {
    struct sched_param sp;
    sp.sched_priority = TASK_PRIORITIES[i];

    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_OTHER);
    pthread_attr_setschedparam(&attr, &sp);
    if (pthread_create(&pthreadIDS[i], &attr, TASKNAMES[i], NULL))
      perror("error in creating pthread ");
    pthread_attr_destroy(&attr);
  }

  // Wait for the threads to terminate
  for (i = 0; i < NUM_TASKS; i++) {
    pthread_join(pthreadIDS[i], NULL);
  }

  return 0;
}
