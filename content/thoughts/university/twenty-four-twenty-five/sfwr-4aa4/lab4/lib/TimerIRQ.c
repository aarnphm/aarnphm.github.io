/*
 * Configuration for Timer Interrupt Request (IRQ)
 *
 * Copyright (c) 2015,
 * National Instruments.
 * All rights reserved.
 */
#include <stdio.h>

/*
 * Include the myRIO header file.
 * The target type must be defined in your project, as a stand-alone #define,
 * or when calling the compiler from the command-line.
 */
#include "MyRio.h"
#include "TimerIRQ.h"

/*
 * Declare the myRIO NiFpga_Session so that it can be used by any function in
 * this file. The variable is actually defined in myRIO.c.
 *
 * This removes the need to pass the myrio_session around to every function and
 * only has to be declared when it is being used.
 */
extern NiFpga_Session myrio_session;

/**
 * Reserve the interrupt from FPGA and configure Timer IRQ.
 *
 * @param[in]  irqChannel   A structure containing the registers and settings
                                for timer IRQ IO to modify.
 * @param[in]  irqContext   IRQ context under which you reserve IRQ.
 * @param[in]  timeout      The time, in microseconds, after which the IRQ
 occurs.
 * @return the configuration status.
 */
int32_t Irq_RegisterTimerIrq(MyRio_IrqTimer *irqChannel,
                             NiFpga_IrqContext *irqContext, uint32_t timeout) {
  int32_t status;

  /*
   * Reserve an IRQ context. IRQ contexts are single-threaded; only one thread
   * can wait with a particular context at any given time. To minimize jitter
   * when first waiting on IRQs, reserve as many contexts as the application
   * requires. If a context is successfully reserved, you must unreserve it
   * later. Otherwise a memory leak will occur.
   */
  status = NiFpga_ReserveIrqContext(myrio_session, irqContext);

  /*
   * Check if there was an error when you reserved the IRQ.
   * If there was an error, print an error message to stdout and return the
   * configuration status.
   */
  MyRio_ReturnStatusIfNotSuccess(
      status, "A required NiFpga_IrqContext was not reserved.")

      /*
       * Check if the IRQ number or channel value already exists in the resource
       * list, return the configuration status and print an error message.
       */
      status = Irq_CheckReserved(irqChannel->timerChannel, TIMERIRQNO);
  if (status == NiMyrio_Status_IrqNumberNotUsable ||
      status == NiMyrio_Status_IrqChannelNotUsable) {
    printf("You have already registered the only timer interrupt.\n");
    return status;
  }

  /*
   * Write the value to the TIMERWRITE register.
   *
   * The returned NiFpga_Status value is stored for error checking.
   */
  status = NiFpga_WriteU32(myrio_session, irqChannel->timerWrite, timeout);

  /*
   * Check if there was an error when you reserved the IRQ.
   *
   * If there was an error, print an error message to stdout and return the
   * configuration status.s.
   */
  MyRio_ReturnStatusIfNotSuccess(status,
                                 "Could not write to TIMERWRITE register!")

      /*
       * Write the value to the TIMERSETTIME register.
       */
      status =
          NiFpga_WriteBool(myrio_session, irqChannel->timerSet, NiFpga_True);

  /*
   * Check if there was an error when you reserved the IRQ.
   *
   * If there was an error, print an error message to stdout and return the
   * configuration status.
   */
  MyRio_ReturnStatusIfNotSuccess(status,
                                 "Could not write to TIMERSETTIME register!")

      /*
       * Add the channel value and IRQ number to the list.
       */
      Irq_AddReserved(irqChannel->timerChannel, TIMERIRQNO);

  return NiMyrio_Status_Success;
}

/**
 * Unreserve the interrupt from FPGA and disable timer IRQ IO.
 *
 * @param[in]  irqChannel  A structure containing the registers and settings
 *                             for timer IRQ IO to modify.
 * @param[in]  irqContext  IRQ context under which you unreserve the IRQ.
 * @return the configuration status.
 */
int32_t Irq_UnregisterTimerIrq(MyRio_IrqTimer *irqChannel,
                               NiFpga_IrqContext irqContext) {
  int32_t status;

  /*
   * Check if the specified IRQ resource is registered.
   */
  status = Irq_CheckReserved(irqChannel->timerChannel, TIMERIRQNO);
  if (status == NiMyrio_Status_Success) {
    /*
     * Did not find the resource in the list.
     */
    printf("You didn't register an interrupt with this IRQ number.\n");
    return NiMyrio_Status_Success;
  }

  /*
   * Write the value to the IRQTIMERSETTIME register.
   */
  status = NiFpga_WriteU32(myrio_session, irqChannel->timerWrite, 0);
  status = NiFpga_WriteBool(myrio_session, irqChannel->timerSet, NiFpga_True);

  /*
   * Check if there was an error writing to the IRQTIMERSETTIME register.
   *
   * If there was an error then print an error message to stdout and return
   * configuration status.
   */
  MyRio_ReturnStatusIfNotSuccess(status,
                                 "Could not write to IRQTIMERSETTIME register!")

      /*
       * Delete the reserved resource in the list.
       */
      status = Irq_RemoveReserved(TIMERIRQNO);
  /*
   * Check if there was an error releasing the resource from list.
   *
   * If there was an error then print an error message to stdout.
   */
  MyRio_ReturnStatusIfNotSuccess(status, "Could not release the irq resource!")

      /*
       * Unreserve an IRQ context obtained from Irq_ReserveIrqContext.
       * The returned NiFpga_Status value is stored for error checking.
       */
      status = NiFpga_UnreserveIrqContext(myrio_session, irqContext);

  /*
   * Check if there was an error when unreserve an IRQ.
   *
   * If there was an error then print an error message to stdout and return
   * configuration status.
   */
  MyRio_ReturnStatusIfNotSuccess(
      status, "A required NiFpga_IrqContext was not unreserved.")

      return NiMyrio_Status_Success;
}
