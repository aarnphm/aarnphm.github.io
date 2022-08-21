/*
 * Example for performing PWM
 *
 * Copyright (c) 2015,
 * National Instruments Corporation.
 * All rights reserved.
 */

#include <stdio.h>

/*
 * Include the myRIO header file.
 * The target type must be defined in your project, as a stand-alone #define,
 * or when calling the compiler from the command-line.
 */
#include "MyRio.h"
#include "PWM.h"

/*
 * Declare the myRIO NiFpga_Session so that it can be used by any function in
 * this file. The variable is actually defined in myRIO.c.
 *
 * This removes the need to pass the myrio_session around to every function and
 * only has to be declared when it is being used.
 */
extern NiFpga_Session myrio_session;

/**
 * Sets options for the PWM configuration register.
 *
 * Pwm_Inverted inverts output high and low. Equivalent to 100% - <duty cycle>.
 *
 * Pwm_Disabled disables PWM output
 * Pwm_Enabled enabled PWM output
 *
 * @param[in]  channel  A struct containing the registers on the PWM channel to
 *                      modify.
 * @param[in]  mask     Array of flags that indicate which of the configure
 *                      settings are valid. This can be one or more of the
 *                      following values: Pwm_Invert, Pwm_Mode.
 * @param[in]  settings Array of flags that indicate the configuration settings.
 *                      If mask sets Pwm_Invert, set either Pwm_NotInverted or
 *                      Pwm_Inverted. If mask sets Pwm_Mode, set one of
 *                      Pwm_Disabled, Pwm_Enabled.
 */
void Pwm_Configure(MyRio_Pwm *channel, Pwm_ConfigureMask mask,
                   Pwm_ConfigureSettings settings) {
  NiFpga_Status status;
  uint8_t cnfgValue;

  /*
   * Get the current value of the configure register.
   *
   * The returned NiFpga_Status value is stored for error checking.
   */
  status = NiFpga_ReadU8(myrio_session, channel->cnfg, &cnfgValue);

  /*
   * Check if there was an error reading from the PWM registers.
   *
   * If there was an error then the rest of the function cannot complete
   * correctly so print an error message to stdout and return from the
   * function early.
   */
  MyRio_ReturnIfNotSuccess(status,
                           "Could not read from the PWM configure registers!")

      /*
       * Clear the value of the masked bits in the configure register. This is
       * done so that the correct value can be set later on.
       */
      cnfgValue = cnfgValue & ~mask;

  /*
   * Set the value of the settings bits in the configure register. If the
   * value to set is 0 this operation would not work unless the bit was
   * previously cleared.
   */
  cnfgValue = cnfgValue | settings;

  /*
   * Write the new value of the configure register to the device.
   */
  NiFpga_MergeStatus(&status,
                     NiFpga_WriteU8(myrio_session, channel->cnfg, cnfgValue));

  /*
   * Check if there was an error writing to PWM configure registers.
   *
   * If there was an error then print an error message to stdout.
   */
  MyRio_ReturnIfNotSuccess(status,
                           "Could not write to the PWM configure registers!")
}

/**
 * Sets the PWM clock divider. The PWM clock increments/decrements at this
 * frequency. This, along with the value set by Pwm_CounterMaximum sets the
 * frequency of the PWM waveform.
 *
 * The frequency is equal to the myRIO clock frequency (f_clk) divided by this
 * clock divider. The default myRIO FPGA clock frequency is 40 MHz.
 *
 * | divider  | clock          | default clock |
 * ---------------------------------------------
 * | Pwm_OffF  | Off (no clock) | Off           |
 * | Pwm_1X   | f_clk / 1      | 40 MHz        |
 * | Pwm_2X   | f_clk / 2      | 20 MHz        |
 * | Pwm_4X   | f_clk / 4      | 10 MHz        |
 * | Pwm_8X   | f_clk / 8      | 5 MHz         |
 * | Pwm_16X  | f_clk / 16     | 2.5 MHz       |
 * | Pwm_32X  | f_clk / 32     | 1.25 MHz      |
 * | Pwm_64X  | f_clk / 64     | 625 kHz       |
 *
 * @param[in]  channel  A struct containing the registers of the PWM channel to
 *                      modify.
 * @param[in]  divider  The PWM waveform clock divider.
 */
void Pwm_ClockSelect(MyRio_Pwm *channel, Pwm_ClockDivider divider) {
  NiFpga_Status status;

  /*
   * Write the new value of the clock select register to the device.
   */
  status = NiFpga_WriteU8(myrio_session, channel->cs, divider);

  /*
   * Check if there was an error writing to PWM configure registers.
   *
   * If there was an error then print an error message to stdout.
   */
  MyRio_ReturnIfNotSuccess(status,
                           "Could not write to the PWM clock select register!")
}

/**
 * Sets the maximum counter value for the PWM. The behavior of the counter
 * depends on waveform set by Pwm_Waveform.
 *
 * Pwm_Disabled:
 * The value is ignored
 *
 * Pwm_Enabled:
 * The PWM counter counts to MAX, then resets to 0
 *
 * @param[in]  channel  A struct containing the registers of the PWM channel to
 *                      modify.
 * @param[in]  counterMax  The maximum counter value.
 */
void Pwm_CounterMaximum(MyRio_Pwm *channel, uint16_t counterMax) {
  NiFpga_Status status;

  /*
   * Write the new value of the clock select register to the device.
   */
  status = NiFpga_WriteU16(myrio_session, channel->max, counterMax);

  /*
   * Check if there was an error writing to PWM configure registers.
   *
   * If there was an error then print an error message to stdout.
   */
  MyRio_ReturnIfNotSuccess(status,
                           "Could not write to the PWM maximum register!")
}

/**
 * Sets the comparison counter value for the PWM. The behavior of the output
 * depends on waveform set by Pwm_Configure. This, along with the value set by
 * Pwm_CounterMaximum, sets the duty cycle.
 *
 * Pwm_Disabled:
 * The value is ignored
 *
 * Pwm_Enabled and Pwm_NotInverged:
 * Clears the output when CNTR = CMP
 *
 * Pwm_Enabled and Pwm_Inverted:
 * Sets the output when CNTR = CMP
 *
 * @param[in]  channel  A struct containing the registers of the PWM channel to
 *                      modify.
 * @param[in]  counterCompare  The comparison counter value.
 */
void Pwm_CounterCompare(MyRio_Pwm *channel, uint16_t counterCompare) {
  NiFpga_Status status;

  /*
   * Write the new value of the clock select register to the device.
   */
  status = NiFpga_WriteU16(myrio_session, channel->cmp, counterCompare);

  /*
   * Check if there was an error writing to PWM configure registers.
   *
   * If there was an error then print an error message to stdout.
   */
  MyRio_ReturnIfNotSuccess(status,
                           "Could not write to the PWM maximum register!")
}

/**
 * Gets the current value of the PWM counter. The behavior of the counter
 * depends on the waveform set by Pwm_Configure and the maximum counter value
 * set by Pwm_CounterMaximum.
 */
uint16_t Pwm_Counter(MyRio_Pwm *channel) {
  NiFpga_Status status;
  uint16_t cntrValue;

  /*
   * Get the value of the counter register.
   *
   * The returned NiFpga_Status value is stored for error checking.
   */
  status = NiFpga_ReadU16(myrio_session, channel->cntr, &cntrValue);

  /*
   * Check if there was an error reading from the counter register.
   *
   * If there was an error then print an error message to stdout.
   */
  MyRio_ReturnValueIfNotSuccess(status, 0,
                                "Could not read from the PWM counter register!")

      return cntrValue;
}
