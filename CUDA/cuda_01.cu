#include  <stdio.h>

/*
__device__ void hiDeviceFunction(void)
{
      printf("Hello! This is in hiDeviceFunction\n");
}
*/

#define get_unique_id(num_of_threads_per_block,block_idx,thread_idx) \
                        ((num_of_threads_per_block)*(block_idx)+(thread_idx))
                        
__global__ void helloCUDA(void)
{
    printf("Hello! There is a thread [%d] in block[%d]\n",threadIdx.x,blockIdx.x);
    printf("An unique id of this thread is %d\n",get_unique_id(blockDim.x,blockIdx.x,threadIdx.x));
    //printf("There are %d threads in %d block\n",blockDim.x,blockIdx.x);
    //hiDeviceFunction();
}

int main(void)
{
    helloCUDA<<<3,4>>>();
    cudaDeviceSynchronize();

    return  0;
}