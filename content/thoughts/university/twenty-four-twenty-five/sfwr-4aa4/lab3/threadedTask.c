/*============================================================================
 Name        : threadedTask.c
 Author      : Yangdi Lu
 Version     : Fall 2017
 Copyright   : Your copyright notice
 Description : This code creates a simple real time task of known priority and FIFO scheduling.
               Generally tasks run in an infinite loop
               however we use loop a finite loop for convenience in labs.
 ============================================================================*/

//#define _GNU_SOURCE //if you define the _GNU_SOURCE here, you do not need to, cnanot, add a symbol for Cross Compiler in Eclispse
 // if you add the symbol in Eclipse, you do not need to, cannot define the symbol here.

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <string.h>


#define MY_PRIORITY1 (49) /* we use 49 as the PRREMPT_RT use 50
                            as the priority of kernel tasklets
                            and interrupt handler by default */

#define MAX_SAFE_STACK (8*1024) /* The maximum stack size which is
                                   guaranteed safe to access without
                                   faulting */

#define NSEC_PER_SEC    (1000000000) /* The number of nsecs per sec. */

void stack_prefault(void) {

        unsigned char dummy[MAX_SAFE_STACK];

        memset(dummy, 0, MAX_SAFE_STACK);
        return;
}

void* tfun1(void*n){
  struct timespec t;
  struct sched_param param;
  int interval = 50000; /* 50us*/
  int num_runs = 5; // replaced infinite loop

  /* Declare ourself as a real time task */

  param.sched_priority = MY_PRIORITY1;
  if(sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
          perror("sched_setscheduler failed");
          exit(-1);
  }

  /* Lock memory */

  if(mlockall(MCL_CURRENT|MCL_FUTURE) == -1) {
          perror("mlockall failed");
          exit(-2);
  }

  /* Pre-fault our stack */

  stack_prefault();

  clock_gettime(CLOCK_MONOTONIC ,&t);
  /* start after one second */
  t.tv_sec++;

  while(num_runs) {
    /* wait until next shot */
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    /* do the stuff */
    printf("Hello! This is thread: %u Priority:%d\n", (int)pthread_self(), param.sched_priority);
   // pthread_t pthread_self()

   // this return current pthread_t, which is thread id, you can convert it to type "unsigned int",

    /* calculate next shot */
    t.tv_nsec += interval;

    while (t.tv_nsec >= NSEC_PER_SEC) {
           t.tv_nsec -= NSEC_PER_SEC;
            t.tv_sec++;
    }
    num_runs = num_runs -1;
  }
  return NULL;
}


int main(int argc, char* argv[])
{
   pthread_t tid1, tid2, tid3;
   cpu_set_t cpus;
   // Force the program to run on one cpu,
   CPU_ZERO(&cpus); //Initialize cpus to nothing clear previous info if any
   CPU_SET(0, &cpus); // Set cpus to a cpu number zeor here

   if (sched_setaffinity(0, sizeof(cpu_set_t), &cpus)< 0)
      perror("set affinity");

   pthread_create(&tid1, NULL, tfun1, NULL);
   pthread_create(&tid2, NULL, tfun1, NULL);
   pthread_create(&tid3, NULL, tfun1, NULL);

   pthread_join(tid1, NULL);
   pthread_join(tid2, NULL);
   pthread_join(tid3, NULL);

   return 0;

}
