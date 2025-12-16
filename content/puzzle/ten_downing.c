// gcc -std=c99 -o td ten_downing.c && ./td
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  long long m = 12;
  long long n = 12;
  long long total = m * n;
  long long smaller_color = total / 2;

  printf("%lld\n", smaller_color & ~1ll);
  return 0;
}
