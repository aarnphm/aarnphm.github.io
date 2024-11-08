#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

// Shared memory structure - must match server
struct SharedData {
  int angle;  // Desired angle
  int status; // Status flag: -1=not ready, 0=ready for new angle, 1=new angle
              // available
};

int main(int argc, char **argv) {
  // Shared memory variables
  key_t key;
  int shmid;
  struct SharedData *shared_data;

  printf("Servo Control Client\n");

  // Get key for shared memory
  key = ftok(".", 'S');
  if (key == -1) {
    perror("ftok failed");
    exit(1);
  }

  // Get shared memory segment
  shmid = shmget(key, sizeof(struct SharedData), 0666);
  if (shmid == -1) {
    perror("shmget failed - Make sure motor control server is running");
    exit(1);
  }

  // Attach shared memory segment
  shared_data = (struct SharedData *)shmat(shmid, NULL, 0);
  if ((int)shared_data == -1) {
    perror("shmat failed");
    exit(1);
  }

  printf("Connected to motor control server!\n");

  // Wait for server to be ready
  while (shared_data->status == -1) {
    printf("Waiting for server to be ready...\n");
    sleep(1);
  }

  while (1) {
    int angle;

    printf("\nEnter desired angle (0-180 degrees, or -1 to quit): ");
    if (scanf("%d", &angle) != 1) {
      // Clear input buffer if invalid input
      while (getchar() != '\n')
        ;
      continue;
    }

    if (angle == -1) {
      break;
    }

    // Wait for server to be ready for next command
    while (shared_data->status != 0) {
      usleep(10000); // Sleep for 10ms
    }

    // Send new angle
    shared_data->angle = angle;
    shared_data->status = 1; // Signal new angle available

    printf("Sent command to set angle to %d degrees\n", angle);
  }

  // Detach from shared memory
  shmdt(shared_data);
  return 0;
}
