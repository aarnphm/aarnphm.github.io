#include "MyRio.h"
#include "PWM.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

extern NiFpga_Session myrio_session;

// Shared memory structure
struct SharedData {
  int angle;  // Desired angle
  int status; // Status flag: -1=not ready, 0=ready for new angle, 1=new angle
              // available
};

// Function to map angle (0-180) to PWM compare value (499-4999)
uint16_t angleToPWM(int angle) {
  // Constrain angle to 0-180 range
  if (angle < 0)
    angle = 0;
  if (angle > 180)
    angle = 180;

  // Map angle to PWM value
  // 0° = 0.2ms = 500 ticks
  // 180° = 2.0ms = 5000 ticks
  return 499 + (angle * 4500) / 180;
}

int main(int argc, char **argv) {
  NiFpga_Status status;
  MyRio_Pwm pwmA0;
  uint8_t selectReg;

  // Shared memory variables
  key_t key;
  int shmid;
  struct SharedData *shared_data;

  printf("Motor Control Server Starting...\n");

  // Create key for shared memory
  key = ftok(".", 'S');
  if (key == -1) {
    perror("ftok failed");
    exit(1);
  }

  // Create shared memory segment
  shmid = shmget(key, sizeof(struct SharedData), IPC_CREAT | 0666);
  if (shmid == -1) {
    perror("shmget failed");
    exit(1);
  }

  // Attach shared memory segment
  shared_data = (struct SharedData *)shmat(shmid, NULL, 0);
  if ((int)shared_data == -1) {
    perror("shmat failed");
    exit(1);
  }

  // Initialize shared memory
  shared_data->status = -1; // Not ready
  shared_data->angle = 0;

  // Initialize PWM struct with registers
  pwmA0.cnfg = PWMA_0CNFG;
  pwmA0.cs = PWMA_0CS;
  pwmA0.max = PWMA_0MAX;
  pwmA0.cmp = PWMA_0CMP;
  pwmA0.cntr = PWMA_0CNTR;

  // Open the myRIO NiFpga Session
  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    printf("Failed to open myRIO session\n");
    return status;
  }

  // Configure PWM
  Pwm_Configure(&pwmA0, Pwm_Invert | Pwm_Mode, Pwm_NotInverted | Pwm_Enabled);

  // Set clock divider to 16x to get slower clock
  // 40MHz / 16 = 2.5MHz
  Pwm_ClockSelect(&pwmA0, Pwm_16x);

  // Set maximum count for 50Hz (20ms) period
  // 2.5MHz / 50Hz = 50,000 counts
  Pwm_CounterMaximum(&pwmA0, 49999);

  // Enable PWM0 on connector A by setting bit 2
  status = NiFpga_ReadU8(myrio_session, SYSSELECTA, &selectReg);
  selectReg |= (1 << 2);
  status = NiFpga_WriteU8(myrio_session, SYSSELECTA, selectReg);

  // Signal that we're ready to receive commands
  shared_data->status = 0;
  printf("Motor Control Server Ready - Waiting for commands...\n");

  while (1) {
    // Check if new angle is available
    if (shared_data->status == 1) {
      int angle = shared_data->angle;

      // Constrain angle and convert to PWM value
      uint16_t pwmValue = angleToPWM(angle);

      // Set PWM compare value
      Pwm_CounterCompare(&pwmA0, pwmValue);

      printf("Received command: Set angle to %d degrees (PWM value: %d)\n",
             angle < 0 ? 0 : (angle > 180 ? 180 : angle), pwmValue);

      // Signal that we're ready for next command
      shared_data->status = 0;
    }
    usleep(10000); // Sleep for 10ms to prevent busy waiting
  }

  // Cleanup (this won't be reached due to infinite loop)
  shmdt(shared_data);
  shmctl(shmid, IPC_RMID, NULL);
  status = MyRio_Close();
  return status;
}
