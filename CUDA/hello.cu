#include <stdio.h>

__global__ void helloFromGPU(void)
{
    int tid=threadIdx.x;

    printf("[Thread: %d] Hello World From GPU!\n",tid);
}

int main(void)
{
    printf("Hello World from CPU!\n");
    helloFromGPU<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}