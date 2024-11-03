#include "MyRio.h"
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Global mutex and mutex attribute declarations
pthread_mutex_t mutex;
pthread_mutexattr_t mutex_attr;

// Global semaphore declarations
sem_t semHi_rdy, semMid_rdy;       // Semaphores to signal tasks are ready
sem_t semLow_r, semMid_r, semHi_r; // Semaphores to control task execution

void workfor(long n, int task) {
  long counter = 0;
  while (counter < n) {
    counter++;
  }
  printf("Task %d finished spinning\n", task);
}

// Structure to hold thread information
typedef struct {
  int task_id;
  int priority;
  long workload;
} TaskInfo;

void stack_prefault(void) {
  unsigned char dummy[8192];
  memset(dummy, 0, 8192);
}

void *task1_function(void *arg) {
  TaskInfo *task_info = (TaskInfo *)arg;
  struct timespec t;
  long sec;

  // Set priority
  struct sched_param param;
  param.sched_priority = task_info->priority;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  // Get start time
  clock_gettime(CLOCK_MONOTONIC, &t);
  sec = t.tv_sec;
  printf("Task1 created: sec %ld, nsec %ld\n", t.tv_sec, t.tv_nsec);

  // Signal ready and wait for permission to run
  stack_prefault();
  sem_post(&semHi_rdy);
  sem_wait(&semHi_r);

  // Critical section
  pthread_mutex_lock(&mutex);
  printf("Hello! This is task1, Priority:%d\n", param.sched_priority);
  workfor(task_info->workload, task_info->task_id);
  pthread_mutex_unlock(&mutex);

  // Get end time and print duration
  clock_gettime(CLOCK_MONOTONIC, &t);
  printf("Task1 took %ld seconds, %ld nsec\n", (t.tv_sec - sec), t.tv_nsec);

  return NULL;
}

void *task2_function(void *arg) {
  TaskInfo *task_info = (TaskInfo *)arg;
  struct timespec t;
  long sec;

  // Set priority
  struct sched_param param;
  param.sched_priority = task_info->priority;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  // Get start time
  clock_gettime(CLOCK_MONOTONIC, &t);
  sec = t.tv_sec;
  printf("Task2 created: sec %ld, nsec %ld\n", t.tv_sec, t.tv_nsec);

  // Signal ready and wait for permission to run
  stack_prefault();
  sem_post(&semMid_rdy);
  sem_wait(&semMid_r);

  // Task 2 doesn't use mutex
  printf("Hello! This is task2, Priority:%d\n", param.sched_priority);
  workfor(task_info->workload, task_info->task_id);

  // Get end time and print duration
  clock_gettime(CLOCK_MONOTONIC, &t);
  printf("Task2 took %ld seconds, %ld nsec\n", (t.tv_sec - sec), t.tv_nsec);

  return NULL;
}

void *task3_function(void *arg) {
  TaskInfo *task_info = (TaskInfo *)arg;
  struct timespec t;
  long sec;

  // Set priority
  struct sched_param param;
  param.sched_priority = task_info->priority;
  pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

  // Get start time
  clock_gettime(CLOCK_MONOTONIC, &t);
  sec = t.tv_sec;
  printf("Task3 created: sec %ld, nsec %ld\n", t.tv_sec, t.tv_nsec);

  // Wait for permission to run
  sem_wait(&semLow_r);

  // Critical section with proper ordering
  pthread_mutex_lock(&mutex);
  printf("Task3 acquired mutex, Priority:%d\n", param.sched_priority);

  // Wait for both higher priority tasks to be ready before releasing them
  sem_wait(&semMid_rdy);
  sem_wait(&semHi_rdy);

  // Now release them in order of decreasing priority
  sem_post(&semMid_r); // Release medium priority task first

  // Do some work while still holding the mutex
  printf("Hello! This is task3, Priority:%d\n", param.sched_priority);
  workfor(task_info->workload, task_info->task_id);

  // Release high priority task last and then release mutex
  sem_post(&semHi_r);
  pthread_mutex_unlock(&mutex);

  // Get end time and print duration
  clock_gettime(CLOCK_MONOTONIC, &t);
  printf("Task3 took %ld seconds, %ld nsec\n", (t.tv_sec - sec), t.tv_nsec);

  return NULL;
}

int main(int argc, char **argv) {
  NiFpga_Status status;

  // Initialize semaphores to 0
  sem_init(&semHi_rdy, 0, 0);
  sem_init(&semMid_rdy, 0, 0);
  sem_init(&semLow_r, 0, 0);
  sem_init(&semMid_r, 0, 0);
  sem_init(&semHi_r, 0, 0);

  // Initialize mutex with PTHREAD_PRIO_INHERIT protocol
  pthread_mutexattr_init(&mutex_attr);
  if (pthread_mutexattr_setprotocol(&mutex_attr, PTHREAD_PRIO_INHERIT)) {
    perror("mutex protocol init");
    exit(1);
  }
  if (pthread_mutex_init(&mutex, &mutex_attr)) {
    perror("mutex init");
    exit(1);
  }

  // Open the myRIO NiFpga Session
  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    return status;
  }

  // Create threads with specific priorities
  pthread_t thread1, thread2, thread3;
  TaskInfo tasks[3] = {
      {1, 45, 1000000},   // Task 1: High priority
      {2, 42, 100000000}, // Task 2: Medium priority
      {3, 40, 200000000}  // Task 3: Low priority
  };

  // Create threads in order of increasing priority
  pthread_create(&thread3, NULL, task3_function,
                 &tasks[2]); // Low priority first
  pthread_create(&thread2, NULL, task2_function,
                 &tasks[1]); // Medium priority second
  pthread_create(&thread1, NULL, task1_function,
                 &tasks[0]); // High priority last

  // Sleep for 1 second before allowing low priority task to run
  sleep(1);
  sem_post(&semLow_r);

  // Wait for all threads to complete
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
  pthread_join(thread3, NULL);

  // Cleanup
  pthread_mutex_destroy(&mutex);
  pthread_mutexattr_destroy(&mutex_attr);
  sem_destroy(&semHi_rdy);
  sem_destroy(&semMid_rdy);
  sem_destroy(&semLow_r);
  sem_destroy(&semMid_r);
  sem_destroy(&semHi_r);

  // Close the myRIO NiFpga Session
  status = MyRio_Close();

  return status;
}
