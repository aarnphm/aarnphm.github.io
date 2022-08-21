/*
 * IRQConfigure.h
 *
 * Copyright (c) 2014,
 * National Instruments Corporation.
 * All rights reserved.
 */

#ifndef IRQConfigure_h_
#define IRQConfigure_h_

#include "MyRio.h"
#include <malloc.h>

/**
 * No errors or warnings.
 */
static const int32_t NiMyrio_Status_Success = 0;

/**
 * The specified channel is already registered with another interrupt. 
 * Specify a different channel or unregister the interrupt with the same channel name.
 */
static const int32_t NiMyrio_Status_IrqChannelNotUsable = -363024;

/**
 * The specified IRQ Number is out of range or 
 * already registered an interrupt with the same interrupt number.
 */
static const int32_t NiMyrio_Status_IrqNumberNotUsable = -363025;


#if NiFpga_Cpp
extern "C" {
#endif


/*
 * Flags that indicate which of the IRQ IOs are supported.
 */
typedef enum
{
    Irq_Ai_A0,          /* AI Interrupt on PIN3 from Connector A */
    Irq_Ai_A1,          /* AI Interrupt on PIN5 from Connector A */
    Irq_Dio_A0,         /* DI Interrupt on PIN11 from Connector A */
    Irq_Dio_A1,         /* DI Interrupt on PIN13 from Connector A */
    Irq_Dio_A2,         /* DI Interrupt on PIN15 from Connector A */
    Irq_Dio_A3,         /* DI Interrupt on PIN17 from Connector A */
    Irq_Button_0,       /* Button Interrupt */
    Irq_Timer_0         /* Timer Interrupt*/
} Irq_Channel;


/**
 * Linker structure to store the reserved IRQ resources.
 */
typedef struct ReservedIrq
{
    Irq_Channel channel;            /* Channel supported by IRQ */
    uint8_t number;                 /* IRQ number */
    struct ReservedIrq* pNext;      /* Pointer to next node */
} ReservedIrq, *ReservedIrqPointer;


/**
 * Check whether the channel and IRQ number are already reserved.
 */
int32_t Irq_CheckReserved(Irq_Channel channel, uint8_t irqNumber);


/**
  * Add a new node into the reserved IRQ resource list.
 */
int32_t Irq_AddReserved(Irq_Channel channel, uint8_t irqNumber);


/**
 * Remove the node in the resource list.
 */
int32_t Irq_RemoveReserved(uint8_t irqNumber);


/**
 * Wait until the specified IRQ number occurred or ready signal arrived.
 */
void Irq_Wait(NiFpga_IrqContext irqContext, NiFpga_Irq irqNumber,
              uint32_t* irqAssert, NiFpga_Bool* continueWaiting);


/**
 * Acknowledge the IRQ(s).
 */
void Irq_Acknowledge(uint32_t irqAssert);

#if NiFpga_Cpp
}
#endif

#endif /* IRQConfigure_h_ */
