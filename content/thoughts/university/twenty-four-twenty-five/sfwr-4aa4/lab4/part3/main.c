#include "DIO.h"
#include "TimerIRQ.h"
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define LoopDuration 60 // How long to run the program, in seconds
#define LoopSteps 3     // How long to step between printing, in seconds
#define LED_BLINK_INTERVAL 500000 // 500ms in microseconds

typedef struct {
  NiFpga_IrqContext irqContext;
  NiFpga_Bool irqThreadRdy;
  MyRio_IrqTimer *irqTimer;
} ThreadResource;

void *Timer_Irq_Thread(void *resource) {
  ThreadResource *threadResource = (ThreadResource *)resource;
  uint8_t currentLed = 0;
  uint8_t ledValue = 0;

  while (1) {
    uint32_t irqAssert = 0;

    Irq_Wait(threadResource->irqContext, TIMERIRQNO, &irqAssert,
             (NiFpga_Bool *)&(threadResource->irqThreadRdy));

    if (irqAssert & (1 << TIMERIRQNO)) {
      // Turn off all LEDs
      NiFpga_WriteU8(myrio_session, DOLED30, 0);

      // Turn on the current LED
      ledValue = 1 << currentLed;
      NiFpga_WriteU8(myrio_session, DOLED30, ledValue);

      // Move to the next LED
      currentLed = (currentLed + 1) % 4;

      // Acknowledge the IRQ
      Irq_Acknowledge(irqAssert);

      // Unregister and re-register the Timer IRQ to make it repeating
      Irq_UnregisterTimerIrq(threadResource->irqTimer,
                             threadResource->irqContext);
      Irq_RegisterTimerIrq(threadResource->irqTimer,
                           &threadResource->irqContext, LED_BLINK_INTERVAL);
    }

    if (!(threadResource->irqThreadRdy)) {
      printf("The IRQ thread ends.\n");
      break;
    }
  }

  pthread_exit(NULL);
  return NULL;
}

int main(int argc, char **argv) {
  int32_t status;
  MyRio_IrqTimer irqTimer0;
  ThreadResource irqThread0;
  pthread_t thread;
  time_t currentTime, finalTime, printTime;

  printf("Timer IRQ LED Blink:\n");

  irqTimer0.timerWrite = IRQTIMERWRITE;
  irqTimer0.timerSet = IRQTIMERSETTIME;
  irqThread0.irqTimer = &irqTimer0;

  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    return status;
  }

  status = Irq_RegisterTimerIrq(&irqTimer0, &irqThread0.irqContext,
                                LED_BLINK_INTERVAL);
  if (status != NiMyrio_Status_Success) {
    printf("CONFIGURE ERROR: %d, Configuration of Timer IRQ failed.\n", status);
    return status;
  }

  irqThread0.irqThreadRdy = NiFpga_True;

  status = pthread_create(&thread, NULL, Timer_Irq_Thread, &irqThread0);
  if (status != NiMyrio_Status_Success) {
    printf("CONFIGURE ERROR: %d, Failed to create a new thread!\n", status);
    return status;
  }

  time(&currentTime);
  finalTime = currentTime + LoopDuration;
  printTime = currentTime;
  while (currentTime < finalTime) {
    static uint32_t loopCount = 0;
    time(&currentTime);

    if (currentTime > printTime) {
      printf("main loop,%d\n", ++loopCount);
      printTime += LoopSteps;
    }
  }

  irqThread0.irqThreadRdy = NiFpga_False;
  pthread_join(thread, NULL);

  status = Irq_UnregisterTimerIrq(&irqTimer0, irqThread0.irqContext);
  if (status != NiMyrio_Status_Success) {
    printf("CONFIGURE ERROR: %d\n", status);
    printf("Clear configuration of Timer IRQ failed.\n");
    return status;
  }

  // Turn off all LEDs before exiting
  NiFpga_WriteU8(myrio_session, DOLED30, 0);

  status = MyRio_Close();
  return status;
}
