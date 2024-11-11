/*
 * myRIO FPGA IRQ configuration wrapper source file
 * Copyright (c) 2014,
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
#include "IRQConfigure.h"

/*
 * Declare the myRIO NiFpga_Session so that it can be used by any function in
 * this file. The variable is actually defined in myRIO.c.
 *
 * This removes the need to pass the myrio_session around to every function and
 * only has to be declared when it is being used.
 */
extern NiFpga_Session myrio_session;


/*
 * Declare the header point which points to the reserved IRQ resource list.
 * If you are programming in a multi-thread pattern, make sure this is accessed
 * in a thread-safe way.
 */
static ReservedIrqPointer reserved_irq_head = NULL;


/**
 * Check whether the channel and IRQ number are already reserved.
 *
 * This function prevents setting duplicated IRQ number or channel.
 *
 * @param[in]  channel      IRQ supported channel name.
 * @param[in]  irqNumber    IRQ number.
 * @return the configure status.
 */
int32_t Irq_CheckReserved(Irq_Channel channel, uint8_t irqNumber)
{
    ReservedIrqPointer pCurr = reserved_irq_head;
    if (pCurr != NULL)
    {
        do
        {
            if (irqNumber == pCurr->number)
            {
                return NiMyrio_Status_IrqNumberNotUsable;
            }
            else if (channel == pCurr->channel)
            {
                return NiMyrio_Status_IrqChannelNotUsable;
            }
            pCurr = pCurr->pNext;
        } while (pCurr != NULL);
    }
    return NiMyrio_Status_Success;
}


/**
 * Create a new node in the resource list.
 *
 * @param[in]  channel       IRQ supported channel name.
 * @param[in]  irqNumber     IRQ number.
 * @return the configure status.
 */
int32_t Irq_AddReserved(Irq_Channel channel, uint8_t irqNumber)
{
    ReservedIrqPointer pNew = (ReservedIrqPointer)malloc(sizeof(ReservedIrq));
    pNew->channel = channel;
    pNew->number = irqNumber;
    if (reserved_irq_head == NULL)
    {
        pNew->pNext = NULL;
        reserved_irq_head = pNew;
    }
    else
    {
        pNew->pNext = reserved_irq_head->pNext;
        reserved_irq_head->pNext = pNew;
    }

    return NiMyrio_Status_Success;
}


/**
 * Delete the pointed registered resource in the resource list.
 *
 * @param[in]  irqNumber     IRQ number.
 * @return the configure status. 
 */
int32_t Irq_RemoveReserved(uint8_t irqNumber)
{
    ReservedIrqPointer pCurr = reserved_irq_head;
    ReservedIrqPointer pPrev = reserved_irq_head;
    if (pCurr != NULL)
    {
        do
        {
            if (irqNumber == pCurr->number)
            {
                if (pCurr == reserved_irq_head)
                {
                    reserved_irq_head = reserved_irq_head->pNext;
                }
                else
                {
                    pPrev->pNext = pCurr->pNext;
                }
                free(pCurr);
                return NiMyrio_Status_Success;
            }
            else
            {
                pPrev = pCurr;
                pCurr = pCurr->pNext;
            }
        } while (pCurr != NULL);
    }

    return NiMyrio_Status_Success;
}


/**
 * Wait until the specified IRQ number occurred or ready signal arrived.
 *
 * @param[in]  irqContext           Context of current IRQ
 * @param[in]  irqNumber            IRQ number
 * @param[out] irqAssert            Asserted IRQ number
 * @param[in]  continueWaiting      Signal which aborts the waiting thread.
 */
void Irq_Wait(NiFpga_IrqContext irqContext, NiFpga_Irq irqNumber,
              uint32_t* irqAssert, NiFpga_Bool* continueWaiting)
{
    NiFpga_Status status;

    /*
     * Break the infinite timeout up into an finite loop calling NiFpga_WaitOnIrqs() with
     * a finite timeout, and check if the thread should stop every loop.
     */
    do
    {
        /*
         * This is a blocking function that stops the calling thread until the FPGA
         * asserts any IRQ in the number parameter, or until the function call times out.
         * Before calling this function, use NiFpga_ReserveIrqContext to reserve an
         * IRQ context. No other threads can use the same context when this function is
         * called. You can use the irqsAsserted parameter to determine which IRQs were
         * asserted for each function call.
         *
         */
        status = NiFpga_WaitOnIrqs(myrio_session, irqContext, 1 << irqNumber,
                                   100,  /* break into infinite 100ms timeouts */
                                   irqAssert, NULL);
    } while ((status == -NiFpga_Status_IrqTimeout) && *continueWaiting);

    /*
     * Check if the NiFpga_WaitOnIrqs() meet any error/warning except IrqTimeout.
     */
    if (*continueWaiting == NiFpga_True)
    {
        MyRio_ReturnIfNotSuccess(status,
                "Could not enter the NiFpga_WaitOnIrqs().\n")
    }
}


/**
 * Acknowledges an IRQ or set of IRQs.
 *
 * @param[in]  irqAssert  Bitwise OR of IRQs to acknowledge.
 */
void Irq_Acknowledge(uint32_t irqAssert)
{
    NiFpga_Status status;

    /*
     * The returned NiFpga_Status value is stored for error checking.
     */
    status = NiFpga_AcknowledgeIrqs(myrio_session, irqAssert);

    /*
     * Check if there was an error when acknowledge an IRQ.
     *
     * If there was an error then print an error message to stdout and return.
     */
    MyRio_ReturnIfNotSuccess(status,
            "Could not acknowledge Irqs!")
}
