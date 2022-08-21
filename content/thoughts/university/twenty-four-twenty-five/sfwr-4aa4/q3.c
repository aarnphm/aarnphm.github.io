#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  fork();
  fork();
  fork();
  printf("hello\n");
  return 0;
}
