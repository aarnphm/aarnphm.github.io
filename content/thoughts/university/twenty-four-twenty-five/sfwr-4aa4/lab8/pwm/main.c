/*
 * Copyright (c) 2015,
 * National Instruments Corporation.
 * All rights reserved.
 */

#include <stdio.h>
#include <time.h>
#include "PWM.h"
#include "MyRio.h"

extern NiFpga_Session myrio_session;

#if !defined(LoopDuration)
#define LoopDuration    60  /* How long to output the signal, in seconds */
#endif


/**
 * Overview:
 * Demonstrates using the PWM. Generates a PWM signal from PWM 0 on
 * connector A.
 *
 * Instructions:
 * 1. Connect an oscilloscope to the PWM 0 pin on connector A.
 * 2. Run this program.
 *
 * Output:
 * The program generates a 25% duty cycle signal at 10 kHz for 60 s.
 *
 * Note:
 * The Eclipse project defines the preprocessor symbol for the myRIO-1900.
 * Change the preprocessor symbol to use this example with the myRIO-1950.
 */
int main(int argc, char **argv)
{
    NiFpga_Status status;

    MyRio_Pwm pwmA0;

    uint8_t selectReg;

    time_t currentTime;
    time_t finalTime;

    printf("PWM\n");

    /*
     * Initialize the PWM struct with registers from the FPGA personality.
     */
    pwmA0.cnfg = PWMA_0CNFG;
    pwmA0.cs = PWMA_0CS;
    pwmA0.max = PWMA_0MAX;
    pwmA0.cmp = PWMA_0CMP;
    pwmA0.cntr = PWMA_0CNTR;

    /*
     * Open the myRIO NiFpga Session.
     * This function MUST be called before all other functions. After this call
     * is complete the myRIO target will be ready to be used.
     */
    status = MyRio_Open();
    if (MyRio_IsNotSuccess(status))
    {
        return status;
    }

    /*
     * Set the waveform, enabling the PWM onboard device.
     */
    Pwm_Configure(&pwmA0, Pwm_Invert | Pwm_Mode,
            Pwm_NotInverted | Pwm_Enabled);

    /*
     * Set the clock divider. The internal PWM counter will increments at
     * f_clk / 4
     *
     * where:
     *  f_clk = the frequency of the myRIO FPGA clock (40 MHz default)
     */
    Pwm_ClockSelect(&pwmA0, Pwm_4x);

    /*
     * Set the maximum counter value. The counter counts from 0 to 1000.
     *
     * The counter increments at 40 MHz / 4 = 10 MHz and the counter counts
     * from 0 to 1000. The frequency of the PWM waveform is 10 MHz / 1000
     * = 10 kHz.
     */
    Pwm_CounterMaximum(&pwmA0, 1000);

    /*
     * Set the comparison counter value. The PWM counter counts from 0 to 1000
     * and outputs from 0 to the comparison value (250).
     *
     * The duty cycle is 250 / 1000 = 25%.
     */
    Pwm_CounterCompare(&pwmA0, 250);

    /*
     * PWM outputs are on pins shared with other onboard devices. To output on
     * a physical pin, select the PWM on the appropriate SELECT register. See
     * the MUX example for simplified code to enable-disable onboard devices.
     *
     * Read the value of the SYSSELECTA register.
     */
    status = NiFpga_ReadU8(myrio_session, SYSSELECTA, &selectReg);
    MyRio_ReturnValueIfNotSuccess(status, status,
        "Could not read from the SYSSELECTA register!")

    /*
     * Set bit2 of the SYSSELECTA register to enable PWMA_0 functionality.
     * The functionality of the bit is specified in the documentation.
     */

    selectReg = selectReg | (1 << 2);

    /*
     * Write the updated value of the SYSSELECTA register.
     */
    status = NiFpga_WriteU8(myrio_session, SYSSELECTA, selectReg);
    MyRio_ReturnValueIfNotSuccess(status, status,
        "Could not write to the SYSSELECTA register!")

    /*
     * Normally, the main function runs a long running or infinite loop.
     * Keep the program running for 60 seconds so that the PWM output can be
     * observed using an external instrument.
     */
    time(&currentTime);
    finalTime = currentTime + LoopDuration;
    while (currentTime < finalTime)
    {
        time(&currentTime);
    }

    /*
     * Close the myRIO NiFpga Session.
     * This function MUST be called after all other functions.
     */
    status = MyRio_Close();

    /*
     * Returns 0 if successful.
     */
    return status;
}
