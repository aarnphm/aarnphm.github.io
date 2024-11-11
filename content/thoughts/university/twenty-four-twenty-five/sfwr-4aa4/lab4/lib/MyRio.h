/*
 * myRIO NI FPGA Interface C API Wrapper header file.
 *
 * Copyright (c) 2013,
 * National Instruments Corporation.
 * All rights reserved.
 */

#ifndef MyRio_h_
#define MyRio_h_

#if defined(MyRio_1900)

#include "MyRio1900.h"

#elif defined(MyRio_1950)

#include "MyRio1950.h"

#else
#   error "Target Not Defined"
#endif

#if NiFpga_Cpp
extern "C" {
#endif


/*
 * Simple error handling for the cases where the function returns on error.
 */
#define MyRio_ReturnIfNotSuccess(status, message)\
if (MyRio_IsNotSuccess((status))){\
    MyRio_PrintStatus((status));\
    printf((message));\
    return;\
}


/*
 * Simple error handling for the cases where the function returns on error.
 */
#define MyRio_ReturnValueIfNotSuccess(status, value, message)\
if (MyRio_IsNotSuccess((status))){\
    MyRio_PrintStatus((status));\
    printf((message));\
    return (value);\
}


/*
 * Simple error handling for the cases where the function returns on error.
 */
#define MyRio_ReturnStatusIfNotSuccess(status, message)\
if (MyRio_IsNotSuccess((status))){\
    MyRio_PrintStatus((status));\
    printf((message));\
    return (status);\
}


/**
 * Tests whether a status is not equal to NiFpga_Status_Success.
 *
 * Both errors and warnings will return NiFpga_True. If you only care about
 * checking errors and not warnings then use NiFpga_IsNotError.
 *
 * @param[in]  status  The NiFpga_Status to check for an error
 * @return  NiFpga_Bool which indicates if the status is an error or warning
 * @see NiFpga_IsNotError
 */
static NiFpga_Inline NiFpga_Bool MyRio_IsNotSuccess(NiFpga_Status status)
{
    return (status != NiFpga_Status_Success);
}


/**
 * Print the value of status if there is an error or warning.
 */
void MyRio_PrintStatus(NiFpga_Status status);


/**
 * Opens a session to the myRIO FPGA Personality.
 */
NiFpga_Status MyRio_Open();


/**
 * Closes the session to the myRIO FPGA Personality.
 */
NiFpga_Status MyRio_Close();

#if NiFpga_Cpp
}
#endif

#endif /* MyRio_h_ */
