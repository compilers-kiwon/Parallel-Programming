#include    <stdio.h>
#include    <stdlib.h>
#include    <time.h>

#define MAX_SIZE    4000000

#define NUM_OF_BLOCKS               400
#define NUM_OF_THREADS_PER_BLOCK    1000
#define VECTOR_SIZE_PER_THREAD      (MAX_SIZE/(NUM_OF_BLOCKS*NUM_OF_THREADS_PER_BLOCK))

#define DST     0
#define SRC1    1
#define SRC2    2
#define MAX_MEM 3

#define mem
__global__ void vector_add(int *dst,int *src1,int* src2)
{
    int begin = (blockDim.x*blockIdx.x+threadIdx.x)*VECTOR_SIZE_PER_THREAD;
    int end = begin+VECTOR_SIZE_PER_THREAD;

    //printf("(%d,%d) thread takes vector[%d~%d]\n",
    //                    blockIdx.x,threadIdx.x,begin,end);

    for(;begin<end;begin++)
    {
        dst[begin] = src1[begin]+src2[begin];
    }
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
/*
    for(i=0;i<MAX_SIZE;i++)
    {
        if( cpu_mem[SRC1][i]+cpu_mem[SRC2][i] != cpu_mem[DST][i] )
        {
            printf("Unexpected!!\n");
            printf("[%07d] %d = %d + %d\n",i,cpu_mem[DST][i],cpu_mem[SRC1][i],cpu_mem[SRC2][i]);
            break;
        }
        else
        {
            if( i%1000000 == 0 )
            {
                printf("[%07d] %d = %d + %d\n",i,cpu_mem[DST][i],cpu_mem[SRC1][i],cpu_mem[SRC2][i]);
            }
        }
    }
*/
    for(i=0;i<MAX_MEM;i++)
    {
        free(cpu_mem[i]);
        cudaFree(cuda_mem[i]);
    }

    printf("time : %d ms\n",((int)clock() - start) / (CLOCKS_PER_SEC / 1000));
    return  0;
}