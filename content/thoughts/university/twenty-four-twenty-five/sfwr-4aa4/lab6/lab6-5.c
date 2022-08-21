/*
 ============================================================================
 Name                                                : lab6-5.c ---Joe
 Author                  : Dr Asghar Bokhari
 Version                 : Summer 2016
 Copyright   : Your copyright notice
 Description : Create three tasks with 0 priority and SCHED_OTHER, to observer
 the need for synchronization
 ============================================================================
 */
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

// Convenient utility functions and typedef's
typedef struct sched_param sched_param_t;
// The maximum stack size which is guaranteed safe to access without faulting
#define MAX_SAFE_STACK (8 * 1024)
void stack_prefault(void);
char *policy_to_string(int policy);

// Task template
void *task(void *args);
// args to customize the behaviour of each task
typedef struct {
  const char *name;
  void (*work)(void);
} task_args_t;
// Example "work to do" inside a task
void spins(void);
void sleeps(void);
void do_noting(void);
// ... Declare more if you wish, remember to provide the function definitions

#define NUMB_TASKS (4)
// We use max value of 49 as the PREEMPT_RT use 50 as the priority of kernel
// tasklets and interrupt handler by default
const int MAIN_PRIORITY = 35;
const int TASK_PRIORITIES[NUMB_TASKS] = {48, 45, 42, 40};
const task_args_t TASK_ARGS[NUMB_TASKS] = {
    // {.name=<task name>, .work=<work to do>}
    {"task1", spins},
    {"task2", do_noting},
    {"task3", spins},
    {"task4", sleeps},
};

// Global synchronization mechanisms
pthread_mutex_t mutex;
pthread_cond_t cond;

int main(int argc, char *argv[]) {
  // Initialize the mutex and condition variable used to synchronize threads
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&cond, NULL);

  // Force the program to run on one cpu,
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);   // Initialize cpus to nothing clear previous info if any
  CPU_SET(0, &cpu_set); // Set cpus to a cpu number zeor here
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set) < 0)
    perror("sched set affinity");

  // Set scheduling policy and priority for main
  sched_param_t main_param;
  main_param.sched_priority = MAIN_PRIORITY;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &main_param) < 0)
    perror("pthread set schedparam for main");

  // Create the tasks as pthreads
  pthread_t pthreads[NUMB_TASKS];
  for (int i = 0; i < NUMB_TASKS; i++) {
    sched_param_t sched_param;
    sched_param.sched_priority = TASK_PRIORITIES[i];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    pthread_attr_setschedparam(&attr, &sched_param);
    if (pthread_create(&pthreads[i], &attr, task, (void *)&TASK_ARGS[i]))
      perror("pthread create");
    pthread_attr_destroy(&attr);
  }

  printf("main:\ttasks will start running in 1 second\n");
  sleep(1);
  // Release the condition variable. All tasks will be woken up at the same time
  pthread_mutex_lock(&mutex);
  pthread_cond_broadcast(&cond);
  pthread_mutex_unlock(&mutex);

  // Wait for tasks to finish
  for (int i = 0; i < NUMB_TASKS; i++) {
    pthread_join(pthreads[i], NULL);
  }

  return 0;
}

// Boilerplate to:
// 1. Prefault the stack for RT threads
// 2. Surround the work function of a task with some debugging printf statements
void *task(void *args) {
  // Retrieve the arguments passed into this task
  task_args_t *task_args = (task_args_t *)args;

  // Lock memory
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
    perror("mlockall failed");
    exit(-2);
  }
  // Pre-fault our stack
  stack_prefault();

  // Retrieve the scheduling policy and priority currently applied to this
  // thread
  int policy;
  sched_param_t sched_param;
  pthread_getschedparam(pthread_self(), &policy, &sched_param);

  // Listen on the condition variable which will be released by main()
  pthread_mutex_lock(&mutex);
  pthread_cond_wait(&cond, &mutex);
  pthread_mutex_unlock(&mutex);

  printf("%s:\tstarts working\tPolicy: %s\tPriority:%d\n", task_args->name,
         policy_to_string(policy), sched_param.sched_priority);
  task_args->work(); // ** doing the work here **
  printf("%s:\tfinishes\tPolicy: %s\tPriority:%d\n", task_args->name,
         policy_to_string(policy), sched_param.sched_priority);

  pthread_exit(NULL);
}

// Definitons of example "work to do" inside a task
void spins(void) {
  u_int64_t i = (u_int64_t)1 << 26;
  while (i--)
    ;
}

void sleeps(void) { sleep(1); }

void do_noting(void) { return; }

// Utility functions
void stack_prefault(void) {
  unsigned char dummy[MAX_SAFE_STACK];
  memset(dummy, 0, MAX_SAFE_STACK);
}

char *policy_to_string(int policy) {
  switch (policy) {
  case SCHED_OTHER:
    return "SCHED_OTHER";
  case SCHED_FIFO:
    return "SCHED_FIFO";
  case SCHED_RR:
    return "SCHED_RR";
  }
  return "Unknown policy";
}
