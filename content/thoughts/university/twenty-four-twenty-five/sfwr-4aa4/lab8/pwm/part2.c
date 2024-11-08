/*
 * Motor control application for lab 8 part 2
 * Creates FIFO and controls servo based on received angles
 */

#include "MyRio.h"
#include "PWM.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

extern NiFpga_Session myrio_session;

#define FIFO_NAME "/tmp/servo_fifo"
#define BUFFER_SIZE 80

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
  int fd;
  char buffer[BUFFER_SIZE];
  int angle;

  printf("Motor Control Application\n");

  // Create the FIFO if it doesn't exist
  umask(0);
  mknod(FIFO_NAME, S_IFIFO | 0666, 0);

  // Initialize PWM0 on MXP connector A
  pwmA0.cnfg = PWMA_0CNFG;
  pwmA0.cs = PWMA_0CS;
  pwmA0.max = PWMA_0MAX;
  pwmA0.cmp = PWMA_0CMP;
  pwmA0.cntr = PWMA_0CNTR;

  // Open the myRIO NiFpga Session
  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    return status;
  }

  // Configure PWM output
  Pwm_Configure(&pwmA0, Pwm_Invert | Pwm_Mode, Pwm_NotInverted | Pwm_Enabled);
  Pwm_ClockSelect(&pwmA0, Pwm_16x);
  Pwm_CounterMaximum(&pwmA0, 49999);

  // Enable PWM0 output on connector A
  status = NiFpga_ReadU8(myrio_session, SYSSELECTA, &selectReg);
  MyRio_ReturnValueIfNotSuccess(status, status,
                                "Could not read from SYSSELECTA register!");

  selectReg = selectReg | (1 << 2); // Set bit 2 to enable PWM0

  status = NiFpga_WriteU8(myrio_session, SYSSELECTA, selectReg);
  MyRio_ReturnValueIfNotSuccess(status, status,
                                "Could not write to SYSSELECTA register!");

  printf("Waiting for angle inputs from user application...\n");

  // Open FIFO for reading
  fd = open(FIFO_NAME, O_RDONLY);

  // Main control loop - read angles from FIFO and update servo
  while (1) {
    // Read angle from FIFO
    int bytes_read = read(fd, buffer, BUFFER_SIZE);
    if (bytes_read > 0) {
      buffer[bytes_read] = '\0';

      // Convert string to integer
      angle = atoi(buffer);

      // Validate angle range
      if (angle < 0)
        angle = 0;
      if (angle > 180)
        angle = 180;

      // Update servo position
      Pwm_CounterCompare(&pwmA0, angleToPWM(angle));
      printf("Moving servo to %d degrees\n", angle);
    }
  }

  // Cleanup (note: this code won't be reached in this version)
  close(fd);
  unlink(FIFO_NAME);
  status = MyRio_Close();
  return status;
}
