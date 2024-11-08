/*
 * DigitalInputOutput.h
 *
 * Copyright (c) 2015,
 * National Instruments Corporation.
 * All rights reserved.
 */

#ifndef PWM_h_
#define PWM_h_

#include "MyRio.h"

#if NiFpga_Cpp
extern "C" {
#endif


/**
 * Flags that indicate which of the PWM configure functions are set.
 */
typedef enum
{
    Pwm_Invert = 0x01,  /**< The waveform inversion option is set */
    Pwm_Mode = 0x04     /**< The waveform mode option is set */
} Pwm_ConfigureMask;


/**
 * PWM configure register (waveform generation) options.
 */
typedef enum
{
    Pwm_NotInverted = 0x00,     /**< PWM signal is not inverted */
    Pwm_Inverted = 0x01,        /**< PWM signal is inverted */
    Pwm_Disabled = 0x00,        /**< No PWM generation */
    Pwm_Enabled = 0x04,         /**< Asymmetric PWM generation */
} Pwm_ConfigureSettings;


/**
 * Clock divider options.
 */
typedef enum
{
    Pwm_Off = 0x00,     /**< No clock */
    Pwm_1x = 0x01,      /**< 1x (f_clk) */
    Pwm_2x = 0x02,      /**< 2x (f_clk / 2) */
    Pwm_4x = 0x03,      /**< 4x (f_clk / 4) */
    Pwm_8x = 0x04,      /**< 8x (f_clk / 8) */
    Pwm_16x = 0x05,     /**< 16x (f_clk / 16) */
    Pwm_32x = 0x06,     /**< 32x (f_clk / 32) */
    Pwm_64x = 0x07      /**< 64x (f_clk / 64) */
} Pwm_ClockDivider;


/**
 * Registers for a particular PWM.
 */
typedef struct
{
    uint32_t cnfg;  /**< Counter configuration register */
    uint32_t cs;    /**< Selects speed of the clock register */
    uint32_t max;   /**< Maximum value of the counter register */
    uint32_t cmp;   /**< Output compare value register */
    uint32_t cntr;  /**< Current value of the counter register */
} MyRio_Pwm;


/**
 * Sets the PWM waveform and and invert configuration options.
 */
void Pwm_Configure(MyRio_Pwm* channel, Pwm_ConfigureMask mask,
                   Pwm_ConfigureSettings settings);


/**
 * Sets the PWM clock select divider, that is, the frequency of the PWM internal
 * clock.
 */
void Pwm_ClockSelect(MyRio_Pwm* channel, Pwm_ClockDivider divider);


/**
 * Sets the maximum value of the PWM counter.
 */
void Pwm_CounterMaximum(MyRio_Pwm* channel, uint16_t counterMax);


/**
 * Sets the period of the PWM counter with output.
 */
void Pwm_CounterCompare(MyRio_Pwm* channel, uint16_t counterCompare);


/**
 * Gets the value of the PWM counter.
 */
uint16_t Pwm_Counter(MyRio_Pwm* channel);

#if NiFpga_Cpp
}
#endif

#endif /* PWM_h_ */
