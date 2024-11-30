#include "AIO.h"
#include "MyRio.h"
#include <stdio.h>
#include <stdlib.h> //for atof()
#include <time.h>
#include <unistd.h>

#define LoopDuration 20

#define TIMESTEP 0.001            // calculation time step, in second
#define NSEC_PER_SEC (1000000000) /* The number of nsecs per sec. */

int main(int argc, char **argv) {
  NiFpga_Status status;
  /* Variable Initialization and AIO setup */
  time_t currentTime;
  time_t finalTime;
  struct timespec t;

  uint8_t count = 1;

  // the following four will be passed by command line argument
  double Kp = 1.0;
  double Ki = 0;
  double Kd = 0;
  double setPointDeg; // passed in in unit of Degree

  double errorC = 0,
         errorP = 0;      // the error for current step and for previous step
  double tolerance = 0.5; // error tolerance, than 0.5 Degree,

  double eInt = 0;
  double eDiff; // for PID calculation, the error integral, the error
                // differential b

  double PIDout; // this program calculated PID in Degree

  double positionV, positionDeg;
  double outputV; // measured position in V, the output control signal in V,

  double T;
  T = TIMESTEP;

  MyRio_Aio CI0, CO0;

  if (argc < 5) {
    printf("Please provide a setpoint and the three PID parameters, Kp, Ki and "
           "Kd\n");
    return (-1);
  } else {
    setPointDeg = atof(argv[1]);
    Kp = atof(argv[2]);
    Ki = atof(argv[3]);
    Kd = atof(argv[4]);
  }

  printf(" The parameters: setpoint: %.2lf deg,Kp=%lf, Ki=%lf, Kd=%lf\n",
         setPointDeg, Kp, Ki, Kd);

  CI0.val = AIC_0VAL;
  CI0.wght = AIC_0WGHT;
  CI0.ofst = AIC_0OFST;
  CI0.is_signed = NiFpga_True;

  CO0.val = AOC_0VAL;
  CO0.wght = AOC_0WGHT;
  CO0.ofst = AOC_0OFST;
  CO0.set = AOSYSGO;
  CO0.is_signed = NiFpga_True;

  /*
   * Open the myRIO NiFpga Session.
   * This function MUST be called before all other functions. After this call
   * is complete the myRIO target will be ready to be used.
   */
  status = MyRio_Open();
  if (MyRio_IsNotSuccess(status)) {
    return status;
  }

  /* Setup and Aquire Data */
  Aio_Scaling(&CO0);
  Aio_Scaling(&CI0);

  /*
          FILE *f = fopen("Labs/logFile11-1.txt","w");
          if (f == NULL){
                  puts("no file made");
                  exit(1);
          }
          fprintf(f, "%f\n", (ai_C0 * (176/5.0)));
          fclose(f);
  */

  time(&currentTime);
  finalTime = currentTime + LoopDuration;
  while (currentTime < finalTime) {
    clock_gettime(CLOCK_MONOTONIC, &t);

    // calculate the time for nanosleep
    t.tv_nsec += T * NSEC_PER_SEC;

    while (t.tv_nsec >= NSEC_PER_SEC) {
      t.tv_nsec -= NSEC_PER_SEC;
      t.tv_sec++;
    }

    // do the job------------
    positionV = Aio_Read(&CI0);
    positionDeg = positionV * 176 / 5.0;

    errorC = setPointDeg - positionDeg;

    //   printf(" errorc: %.2f\n", errorC);

    if (abs(errorC) < tolerance)
      errorC = 0;

    //	 printf(" tolerance, errorc: %.2f, %.2f\n", tolerance, errorC);

    eInt += errorC * TIMESTEP; // accumation error
    eDiff = (errorC - errorP) / TIMESTEP;

    PIDout = Kp * errorC + Ki * eInt + Kd * eDiff;

    errorP = errorC;

    // convert the PIDout (in degree) to Volt
    outputV = PIDout / 36;

    // limit the output
    if (outputV > 6.0)
      outputV = 6.0;
    if (outputV < -6.0)
      outputV = -6.0;

    Aio_Write(&CO0, outputV);

    if (((count % 5) == 0) & (count < 500)) {
      printf("the real position: %.2f, errorC: %.2f", positionDeg, errorC);
      printf("   the write out V: %.2f\n", outputV);
    }

    count++;

    time(&currentTime);

    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);
  }

  /*
   * Close the myRIO NiFpga Session.
   * This function MUST be called after all other functions.
   */
  status = MyRio_Close();

  return status;
}
