#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

__global__ void add(int *a,int *b, int *c, int size) {
	int tid = blockIdx.x *  blockDim.x + threadIdx.x;
        if(tid < size){
          c[tid] = a[tid]+b[tid];
        }
}

int main(int argc, char *argv[])  {
	int N = 10, T = 10, B = 1;            // threads per block and blocks per grid
	int *a, *b, *c, *d;
	int *dev_a, *dev_b, *dev_c;

	do {
        printf("\nEnter size of vector, (currently %d): ",N);
        scanf("%d",&N);
        printf("\nLimitation: there is only 1 grid.\n");
		printf("\nEnter number of threads per block: ");
		scanf("%d",&T);
		printf("\nEnter number of blocks per grid: ");
		scanf("%d",&B);
		if (T * B != N) printf("Error T x B != N, try again\n");
	} while (T * B != N);

	cudaEvent_t start, stop;     // using cuda events to measure time
	float elapsed_time_ms;       // which is applicable for asynchronous code also

    a = (int*) malloc(N*sizeof(int));		//this time use dynamically allocated memory for arrays on host
    b = (int*) malloc(N*sizeof(int));
    c = (int*) malloc(N*sizeof(int));
    d = (int*) malloc(N*sizeof(int));

    for(int i=0;i<N;i++) {    // load arrays with some numbers
		a[i] = i;
		b[i] = i;
	}

    cudaEventCreate( &start ); 
	cudaEventCreate( &stop );

    /* ------------- COMPUTATION ON DEVICE GPU ----------------------------*/
	cudaEventRecord( start, 0 ); // instrument code to measure start time

	cudaMalloc((void**)&dev_a,N * sizeof(int));
	cudaMalloc((void**)&dev_b,N * sizeof(int));
	cudaMalloc((void**)&dev_c,N * sizeof(int));

	cudaMemcpy(dev_a, a , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c , N*sizeof(int),cudaMemcpyHostToDevice);
    
	add<<<B,T>>>(dev_a,dev_b,dev_c,N);

	cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);

	cudaEventRecord( stop, 0 );     // instrument code to measure end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // print out execution time

    /* ------------- COMPUTATION ON HOST CPU ----------------------------*/
    cudaEventRecord(start, 0);		// use same timing

    for(int i=0;i<N;i++) {
        d[i] = a[i]+b[i];
    }
    
    cudaEventRecord(stop, 0);     	// instrument code to measure end time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop );

    printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);  // print out execution time

    /* ------------------- check device creates correct results -----------------*/
    for(int i=0;i<N;i++) {
        if (c[i] != d[i]) 
            printf("*********** ERROR in results, CPU and GPU create different answers ********\n");
        break;
    }

	// clean up
    free(a);
    free(b);
    free(c);
    free(d);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}