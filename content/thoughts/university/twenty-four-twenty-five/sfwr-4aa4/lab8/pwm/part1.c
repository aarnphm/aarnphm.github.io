/*
 * Servo motor control program implementing lab 8 part 1
 * Controls servo position from 0-180 degrees using PWM
 */

#include "MyRio.h"
#include "PWM.h"
#include <stdio.h>

extern NiFpga_Session myrio_session;

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
  int angle;

  printf("Servo Motor Control\n");
  printf("Enter angle (0-180 degrees): ");
  scanf("%d", &angle);

  // Coerce input to valid range
  if (angle < 0)
    angle = 0;
  if (angle > 180)
    angle = 180;

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
   * Set compare value based on desired angle
   */
  Pwm_CounterCompare(&pwmA0, angleToPWM(angle));

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

  printf("Moving servo to %d degrees\n", angle);
  printf("Press Enter to exit...\n");
  getchar(); // Clear previous newline
  getchar(); // Wait for enter

  /*
   * Close the myRIO NiFpga Session
   */
  status = MyRio_Close();

  return status;
}
