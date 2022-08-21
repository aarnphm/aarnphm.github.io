/*
 * TimerIRQ.h
 *
 * Copyright (c) 2015,
 * National Instruments.
 * All rights reserved.
 */

#ifndef TimerIRQ_h_
#define TimerIRQ_h_

#include "IRQConfigure.h"

#if !defined(TIMERIRQNO)
#define TIMERIRQNO 0
#endif

#if NiFpga_Cpp
extern "C" {
#endif

/**
 * Registers and settings for timer IRQ I/O.
 */
typedef struct {
  uint32_t timerWrite;      /**< Timer IRQ interval register */
  uint32_t timerSet;        /**< Timer IRQ setting register */
  Irq_Channel timerChannel; /**< Timer IRQ supported I/O */
} MyRio_IrqTimer;

/**
 * Configure the timer IRQ.
 */
int32_t Irq_RegisterTimerIrq(MyRio_IrqTimer *irqChannel,
                             NiFpga_IrqContext *irqContext, uint32_t timeout);

/**
 * Clear the timer IRQ setting.
 */
int32_t Irq_UnregisterTimerIrq(MyRio_IrqTimer *irqChannel,
                               NiFpga_IrqContext irqContext);

#if NiFpga_Cpp
}
#endif

#endif /* TimerIRQ_h_ */
