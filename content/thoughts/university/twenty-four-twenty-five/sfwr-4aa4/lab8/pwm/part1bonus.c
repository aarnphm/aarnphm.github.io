/*
 * Servo motor sweep program implementing lab 8 bonus
 * Makes servo sweep continuously between 0-180 degrees
 */

#include "MyRio.h"
#include "PWM.h"
#include <stdio.h>
#include <unistd.h>

extern NiFpga_Session myrio_session;

// Sweep control parameters
#define STEP_SIZE 1        // Degrees per step
#define STEP_DELAY 15000   // Microseconds between steps (15ms)
#define SWEEP_DELAY 500000 // Microseconds pause at endpoints (500ms)

// Convert angle in degrees to PWM compare value
uint16_t angleToPWM(int angle) {
  // Map 0-180 degrees to 500-5000 PWM compare values
  // 500 = 0.2ms pulse (0 degrees)
  // 2750 = 1.1ms pulse (90 degrees)
  // 5000 = 2.0ms pulse (180 degrees)
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
  int angle = 0;
  int direction = 1; // 1 = increasing angle, -1 = decreasing angle

  printf("Servo Motor Sweep Program\n");
  printf("Press Ctrl+C to exit\n");

  /*
   * Initialize PWM0 on MXP connector A
   */
  pwmA0.cnfg = PWMA_0CNFG;
  pwmA0.cs = PWMA_0CS;
  pwmA0.max = PWMA_0MAX;
  pwmA0.cmp = PWMA_0CMP;
  pwmA0.cntr = PWMA_0CNTR;

  /*
   * Open the myRIO NiFpga Session
   */
  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    return status;
  }

  /*
   * Configure PWM output:
   * - Not inverted
   * - PWM generation enabled
   */
  Pwm_Configure(&pwmA0, Pwm_Invert | Pwm_Mode, Pwm_NotInverted | Pwm_Enabled);

  /*
   * Set clock divider to 16x
   * Base clock = 40MHz
   * PWM clock = 40MHz/16 = 2.5MHz
   */
  Pwm_ClockSelect(&pwmA0, Pwm_16x);

  /*
   * Set counter maximum to 49,999
   * PWM frequency = 2.5MHz/50000 = 50Hz (20ms period)
   */
  Pwm_CounterMaximum(&pwmA0, 49999);

  /*
   * Enable PWM0 output on connector A
   */
  status = NiFpga_ReadU8(myrio_session, SYSSELECTA, &selectReg);
  MyRio_ReturnValueIfNotSuccess(status, status,
                                "Could not read from SYSSELECTA register!");

  selectReg = selectReg | (1 << 2); // Set bit 2 to enable PWM0

  status = NiFpga_WriteU8(myrio_session, SYSSELECTA, selectReg);
  MyRio_ReturnValueIfNotSuccess(status, status,
                                "Could not write to SYSSELECTA register!");

  printf("Starting sweep motion...\n");

  /*
   * Main sweep loop
   */
  while (1) {
    // Update servo position
    Pwm_CounterCompare(&pwmA0, angleToPWM(angle));

    // Print every 10 degrees for smoother output
    if (angle % 10 == 0) {
      printf("\rAngle: %d degrees   ", angle);
      fflush(stdout); // Ensure output is displayed immediately
    }

    // Small delay for smooth motion
    usleep(STEP_DELAY);

    // Update angle
    angle += (direction * STEP_SIZE);

    // Check endpoints
    if (angle >= 180) {
      angle = 180;
      direction = -1;
      usleep(SWEEP_DELAY); // Pause at max angle
      printf("\nReversing direction - sweeping to 0 degrees\n");
    } else if (angle <= 0) {
      angle = 0;
      direction = 1;
      usleep(SWEEP_DELAY); // Pause at min angle
      printf("\nReversing direction - sweeping to 180 degrees\n");
    }
  }

  // This code won't be reached due to infinite loop,
  // but included for completeness
  status = MyRio_Close();
  return status;
}
