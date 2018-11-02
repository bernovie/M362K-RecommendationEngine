#include <stdio.h>

int main() {
    int x = 1024;
    x = x << 31;
    x = x << 1;
    printf("X is: %d\n", x);
    return 0;
}
