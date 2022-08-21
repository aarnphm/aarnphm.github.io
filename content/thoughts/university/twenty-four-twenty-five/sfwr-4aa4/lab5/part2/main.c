#include "DIO.h"
#include "MyRio.h"

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define _GNU_SOURCE

#define MY_PRIORITY (49)
#define MAX_SAFE_STACK (8 * 1024)
#define NSEC_PER_SEC (1000000000)

void stack_prefault(void) {
  unsigned char dummy[MAX_SAFE_STACK];
  memset(dummy, 0, MAX_SAFE_STACK);
  return;
}

void *led_blink_task(void *arg) {
  struct timespec t;
  struct sched_param param;
  int interval = 500000000; // 500ms
  int num_runs = 20;        // 10 seconds of blinking
  uint8_t led_value = 0x01;

  param.sched_priority = MY_PRIORITY;
  if (sched_setscheduler(0, SCHED_FIFO, &param) == -1) {
    perror("sched_setscheduler failed");
    exit(-1);
  }

  if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
    perror("mlockall failed");
    exit(-2);
  }

  stack_prefault();

  clock_gettime(CLOCK_MONOTONIC, &t);
  t.tv_sec++;

  while (num_runs > 0) {
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

    NiFpga_Status status = NiFpga_WriteU8(myrio_session, DOLED30, led_value);
    if (MyRio_IsNotSuccess(status)) {
      printf("Error writing to LED register\n");
      break;
    }
    led_value = (led_value << 1);
    if (led_value == 0x08) {
      led_value = 0x01;
    }

    printf("LED state: %s\n", led_value ? "ON" : "OFF");

    t.tv_nsec += interval;
    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }
    num_runs--;
  }

  // Turn off all LEDs before exiting
  NiFpga_WriteU8(myrio_session, DOLED30, 0);
  return NULL;
}

int main(int argc, char *argv[]) {
  NiFpga_Status status;
  pthread_t led_thread;
  cpu_set_t cpus;

  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    return status;
  }

  CPU_ZERO(&cpus);
  CPU_SET(0, &cpus);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpus) < 0)
    perror("set affinity");

  pthread_create(&led_thread, NULL, led_blink_task, NULL);
  pthread_join(led_thread, NULL);

  status = MyRio_Close();
  return status;
}
