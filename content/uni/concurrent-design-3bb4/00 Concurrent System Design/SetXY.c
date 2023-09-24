
#include <pthread.h>
#include <stdio.h>

int x,y;

void *SetX(void *args) {
    x = 1; return NULL;
}
void *SetY(void *args) {
    y = 1; return NULL;
}

int main(int argc, char *argv[]) {
    pthread_t setX, setY;
    pthread_create(&setX, NULL, SetX, NULL);
    pthread_create(&setY, NULL, SetY, NULL);
    pthread_join(setX, NULL);
    pthread_join(setY, NULL);
    printf("%d %d\n", x,y);
}
