#include    <stdio.h>
#include    <stdlib.h>
#include    <time.h>

#define MAX_SIZE    4000000

#define NUM_OF_BLOCKS               5
#define NUM_OF_THREADS_PER_BLOCK    (MAX_SIZE/NUM_OF_BLOCKS)

#define DST     0
#define SRC1    1
#define SRC2    2
#define MAX_MEM 3

#define min(a,b)    ((a)<(b)?(a):(b))

__global__ void vector_add(int *dst,int *src1,int* src2)
{
    int thread_id = blockDim.x*blockIdx.x+threadIdx.x;
    dst[thread_id] = src1[thread_id]+src2[thread_id];
}

int main(void)
{
    int i;
    int *cpu_mem[MAX_MEM],*cuda_mem[MAX_MEM];

    clock_t start = clock();

    for(i=0;i<MAX_MEM;i++)
    {
        cpu_mem[i] = (int*)malloc(sizeof(int)*MAX_SIZE);
        cudaMalloc((void**)&cuda_mem[i],sizeof(int)*MAX_SIZE);
    }

    srand(time(NULL));

    for(i=0;i<MAX_SIZE;i++)
    {
        cpu_mem[SRC1][i] = rand();
        cpu_mem[SRC2][i] = rand();
    }

    cudaMemcpy(cuda_mem[SRC1],cpu_mem[SRC1],sizeof(int)*MAX_SIZE,cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_mem[SRC2],cpu_mem[SRC2],sizeof(int)*MAX_SIZE,cudaMemcpyHostToDevice);

    vector_add<<<NUM_OF_BLOCKS,NUM_OF_THREADS_PER_BLOCK>>>(cuda_mem[DST],cuda_mem[SRC1],cuda_mem[SRC2]);
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_mem[DST],cuda_mem[DST],sizeof(int)*MAX_SIZE,cudaMemcpyDeviceToHost);

    for(i=0;i<MAX_MEM;i++)
    {
        free(cpu_mem[i]);
        cudaFree(cuda_mem[i]);
    }

    printf("time : %d ms\n",((int)clock() - start) / (CLOCKS_PER_SEC / 1000));
    return  0;
}