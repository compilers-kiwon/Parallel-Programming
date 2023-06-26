#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define NUM_OF_THREADS      10
#define NUM_OF_ITERATIONS   5

void*   thread_func(void* arg)
{
    int thread_no = *(int*)arg;
    
    for(int i=0;i<NUM_OF_ITERATIONS;i++)
        printf("Hello, World (thread %d)\n",thread_no);
    
    pthread_exit((void*)NULL);
}

int     main(void)
{
    pthread_t   thread_ID[NUM_OF_THREADS];
    void*       thread_result;
    int         args[NUM_OF_THREADS] = {0,1,2,3,4,5,6,7,8,9};

    for(int i=0;i<NUM_OF_THREADS;i++)
    {
        pthread_create(&thread_ID[i],NULL,thread_func,(void*)&args[i]);
    }

    for(int i=0;i<NUM_OF_THREADS;i++)
    {
        pthread_join(thread_ID[i],&thread_result);
    }

    return  0;
}