#include <stdio.h>

#define N 3 + 5

void swap(int a, int b);

int main() {

  int a = N * 2;

  int b = N * 4;

  swap(a, b);

  printf("a=%d,b=%d\n", a, b);
}

void swap(int a, int b) {

  int c = a;

  a = b;

  b = c;
}
