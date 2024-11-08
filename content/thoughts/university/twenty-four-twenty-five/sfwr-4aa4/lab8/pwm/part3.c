/*
 * Motor control application for lab 8 part 3
 * Uses shared memory to receive servo angles
 */

#include "MyRio.h"
#include "PWM.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

extern NiFpga_Session myrio_session;

#define NOT_READY -1
#define READY 0
#define TAKEN 1

// Shared memory structure
struct ServoData {
  int status; // Communication status
  int angle;  // Desired servo angle
};

// Convert angle in degrees to PWM compare value
uint16_t angleToPWM(int angle) {
  // Map 0-180 degrees to 500-5000 PWM compare values
  if (angle < 0)
    angle = 0;
  if (angle > 180)
    angle = 180;
  return 500 + (int)((4500.0 * angle) / 180.0);
}

int main(int argc, char **argv) {
  NiFpga_Status status;
  MyRio_Pwm pwmA0;
  uint8_t selectReg;
  int prev_angle = -1; // Track previous angle to avoid unnecessary updates

  // Shared memory variables
  key_t shmKey;
  int shmID;
  struct ServoData *shmPTR;

  printf("Motor Control Application (Shared Memory)\n");

  // Create shared memory key
  shmKey = ftok("./", 'h');
  if (shmKey == -1) {
    printf("Error: Failed to create shared memory key\n");
    return 1;
  }

  // Create shared memory segment
  shmID = shmget(shmKey, sizeof(struct ServoData), IPC_CREAT | 0666);
  if (shmID == -1) {
    printf("Error: Failed to create shared memory segment\n");
    return 1;
  }

  // Attach shared memory segment
  shmPTR = (struct ServoData *)shmat(shmID, NULL, 0);
  if ((int)shmPTR == -1) {
    printf("Error: Failed to attach shared memory segment\n");
    return 1;
  }

  // Initialize shared memory
  shmPTR->status = NOT_READY;
  shmPTR->angle = 0;

  // Initialize PWM0 on MXP connector A
  pwmA0.cnfg = PWMA_0CNFG;
  pwmA0.cs = PWMA_0CS;
  pwmA0.max = PWMA_0MAX;
  pwmA0.cmp = PWMA_0CMP;
  pwmA0.cntr = PWMA_0CNTR;

  // Open the myRIO NiFpga Session
  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    shmdt((void *)shmPTR);
    shmctl(shmID, IPC_RMID, NULL);
    return status;
  }

  // Configure PWM output
  Pwm_Configure(&pwmA0, Pwm_Invert | Pwm_Mode, Pwm_NotInverted | Pwm_Enabled);
  Pwm_ClockSelect(&pwmA0, Pwm_16x);
  Pwm_CounterMaximum(&pwmA0, 49999);

  // Enable PWM0 output on connector A
  status = NiFpga_ReadU8(myrio_session, SYSSELECTA, &selectReg);
  if (MyRio_IsNotSuccess(status)) {
    shmdt((void *)shmPTR);
    shmctl(shmID, IPC_RMID, NULL);
    return status;
  }

  selectReg = selectReg | (1 << 2); // Set bit 2 to enable PWM0

  status = NiFpga_WriteU8(myrio_session, SYSSELECTA, selectReg);
  if (MyRio_IsNotSuccess(status)) {
    shmdt((void *)shmPTR);
    shmctl(shmID, IPC_RMID, NULL);
    return status;
  }

  printf("Ready to receive angle inputs...\n");
  shmPTR->status = READY; // Signal ready for communication

  // Main control loop
  while (1) {
    // Check if there's a new angle to process
    if (shmPTR->status == READY && shmPTR->angle != prev_angle) {
      int angle = shmPTR->angle;

      // Validate angle range
      if (angle < 0)
        angle = 0;
      if (angle > 180)
        angle = 180;

      // Update servo position
      Pwm_CounterCompare(&pwmA0, angleToPWM(angle));
      printf("Moving servo to %d degrees\n", angle);

      prev_angle = angle;
      shmPTR->status = TAKEN; // Signal that we've processed the angle
    }

    // Small delay to prevent CPU hogging
    usleep(10000); // 10ms delay
  }

  // Cleanup (this code won't be reached in this version)
  status = MyRio_Close();
  shmdt((void *)shmPTR);
  shmctl(shmID, IPC_RMID, NULL);

  return status;
}
