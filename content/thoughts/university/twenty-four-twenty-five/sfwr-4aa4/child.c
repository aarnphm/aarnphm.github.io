#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  int child = fork();
  int c = 0;
  if (child)
    c += 5;
  else {
    child = fork();
    c += 5;
    if (child)
      c += 5;
  }
  printf("%d ", c);
}
