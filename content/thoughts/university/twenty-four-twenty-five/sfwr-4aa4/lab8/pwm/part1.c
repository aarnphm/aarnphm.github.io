#include "MyRio.h"
#include "PWM.h"
#include <stdio.h>

extern NiFpga_Session myrio_session;

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
  int angle;

  printf("Servo Motor Position Control\n");

  // Initialize PWM struct with registers
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

  while (1) {
    printf("\nEnter desired angle (0-180 degrees): ");
    scanf("%d", &angle);

    // Constrain angle and convert to PWM value
    uint16_t pwmValue = angleToPWM(angle);

    // Set PWM compare value
    Pwm_CounterCompare(&pwmA0, pwmValue);

    printf("Set angle to %d degrees (PWM value: %d)\n",
           angle < 0 ? 0 : (angle > 180 ? 180 : angle), pwmValue);
  }

  // Cleanup (this won't be reached due to infinite loop)
  status = MyRio_Close();
  return status;
}
