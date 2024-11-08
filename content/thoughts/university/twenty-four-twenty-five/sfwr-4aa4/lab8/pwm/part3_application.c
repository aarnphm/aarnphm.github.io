/*
 * User application for lab 8 part 3
 * Uses shared memory to send angles to motor control
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

#define NOT_READY -1
#define READY 0
#define TAKEN 1

// Shared memory structure
struct ServoData {
  int status; // Communication status
  int angle;  // Desired servo angle
};

int main(int argc, char **argv) {
  // Shared memory variables
  key_t shmKey;
  int shmID;
  struct ServoData *shmPTR;
  int angle;

  printf("User Application (Shared Memory)\n");

  // Get shared memory key
  shmKey = ftok("./", 'h');
  if (shmKey == -1) {
    printf("Error: Failed to get shared memory key\n");
    printf("Make sure motor control application is running first!\n");
    return 1;
  }

  // Connect to shared memory segment
  shmID = shmget(shmKey, sizeof(struct ServoData), 0666);
  if (shmID == -1) {
    printf("Error: Failed to connect to shared memory segment\n");
    printf("Make sure motor control application is running first!\n");
    return 1;
  }

  // Attach shared memory segment
  shmPTR = (struct ServoData *)shmat(shmID, NULL, 0);
  if ((int)shmPTR == -1) {
    printf("Error: Failed to attach shared memory segment\n");
    return 1;
  }

  // Wait for motor control to be ready
  while (shmPTR->status == NOT_READY) {
    printf("Waiting for motor control to be ready...\n");
    sleep(1);
  }

  printf("Connected to motor control application!\n");

  // Main input loop
  while (1) {
    // Get angle from user
    printf("\nEnter angle (0-180 degrees) or -1 to exit: ");
    scanf("%d", &angle);

    if (angle == -1) {
      break;
    }

    // Validate input
    if (angle < 0)
      angle = 0;
    if (angle > 180)
      angle = 180;

    // Wait for previous command to be processed
    while (shmPTR->status != READY) {
      usleep(1000); // 1ms delay
    }

    // Send new angle
    shmPTR->angle = angle;
    shmPTR->status = READY;

    // Wait for acknowledgment
    while (shmPTR->status != TAKEN) {
      usleep(1000); // 1ms delay
    }
  }

  // Cleanup
  shmPTR->status = NOT_READY; // Signal that we're done
  shmdt((void *)shmPTR);

  return 0;
}
