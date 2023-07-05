#include    <stdio.h>
#include    <stdlib.h>
#include    <time.h>

#define MAX_SIZE    4000000

int main(void)
{
    int i;
    int *dst,*src1,*src2;

    clock_t start = clock();

    dst = (int*)malloc(sizeof(float)*MAX_SIZE);
    src1 = (int*)malloc(sizeof(float)*MAX_SIZE);
    src2 = (int*)malloc(sizeof(float)*MAX_SIZE);

    srand(time(NULL));

    for(i=0;i<MAX_SIZE;i++)
    {
        src1[i] = rand();
        src2[i] = rand();
    }

    printf("time : %d ms\n",((int)clock() - start) / (CLOCKS_PER_SEC / 1000));
    return  0;
}