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
  char buf[BUFFER_SIZE];
  int angle;

  printf("Servo Control Client\n");
  printf("Opening FIFO for writing...\n");

  fd = open(FIFO_NAME, O_WRONLY);
  if (fd == -1) {
    printf("Failed to open FIFO - Make sure motor control server is running\n");
    return 1;
  }

  printf("Connected to motor control server!\n");

  while (1) {
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

    // Convert angle to string and send through FIFO
    snprintf(buf, BUFFER_SIZE, "%d", angle);
    write(fd, buf, strlen(buf));
    printf("Sent command to set angle to %d degrees\n", angle);
  }

  close(fd);
  return 0;
}
