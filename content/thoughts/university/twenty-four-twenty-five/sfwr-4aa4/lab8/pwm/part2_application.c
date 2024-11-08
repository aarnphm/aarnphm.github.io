/*
 * User application for lab 8 part 2
 * Gets angle input from user and sends to motor control via FIFO
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define FIFO_NAME "/tmp/servo_fifo"
#define BUFFER_SIZE 80

int main(int argc, char **argv) {
  int fd;
  char buffer[BUFFER_SIZE];
  int angle;

  printf("User Application\n");

  // Open FIFO for writing
  fd = open(FIFO_NAME, O_WRONLY);
  if (fd == -1) {
    printf("Error: Motor control application must be running first!\n");
    return 1;
  }

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

    // Convert angle to string and write to FIFO
    snprintf(buffer, BUFFER_SIZE, "%d", angle);
    write(fd, buffer, strlen(buffer));
  }

  // Cleanup
  close(fd);
  return 0;
}
