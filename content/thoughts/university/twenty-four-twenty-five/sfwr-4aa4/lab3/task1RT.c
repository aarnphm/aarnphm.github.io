/* ============================================================================
 Name        : task1RT.c
 Author      : Dr Asghar Bokhari
 Version     : Summer 2016
 Copyright   : Your copyright notice
 Description : This code creates a simple real time task of known priority and
 FIFO scheduling. Generally tasks run in an infinite loop however we use a
 finite loop for convenience in labs.
 ============================================================================*/

// Required for making loop finite
#if !defined(LoopDuration)
#define LoopDuration 2 /* How long to output the signal, in seconds */
#endif

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define MY_PRIORITY                                                            \
  (49) /* we use 49 as the PRREMPT_RT use 50                                   \
          as the priority of kernel tasklets                                   \
          and interrupt handler by default */

#define MAX_SAFE_STACK                                                         \
  (8 * 1024) /* The maximum stack size which is                                \
                guaranteed safe to access without                              \
                faulting */

#define NSEC_PER_SEC (1000000000) /* The number of nsecs per sec. */

void stack_prefault(void) {

  unsigned char dummy[MAX_SAFE_STACK];

  memset(dummy, 0, MAX_SAFE_STACK);
  return;
}

int main(int argc, char *argv[]) {
  struct timespec t;
  struct sched_param param;
  int interval = 50000000; // Determines the time period of the task

  // required for looping for a set time
  time_t currentTime;
  time_t finalTime;

  /* Declare ourself as a real time task */
  param.sched_priority = MY_PRIORITY;
  if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
    perror("sched_setscheduler failed");
    exit(-1);
  }

  /* Lock memory */
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
    perror("mlockall failed");
    exit(-2);
  }

  /* Pre-fault our stack */
  stack_prefault();

  clock_gettime(CLOCK_MONOTONIC, &t);

  /* start after one second */
  t.tv_sec++;

  // Normally, the main function runs a long running or infinite loop.
  // A finite loop is used for convenience in the labs.

  time(&currentTime);
  finalTime = currentTime + LoopDuration;

  while (currentTime < finalTime) {
    /* wait until next shot */
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    /* do the stuff */
    printf("Hello\n");

    /* calculate next shot */
    t.tv_nsec += interval;

    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }
    time(&currentTime);
  } // end of while (currentTime)
  return 0;
}
