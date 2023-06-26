#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

#define MAX_NUM 99

void*   get_square_root(void* arg)
{
    int     n = *(int*)arg;
    double* ret = (double*)malloc(sizeof(double));

    *ret = sqrt((double)n);
    return  (void*)ret;  
}

int     main(void)
{
    pthread_t   thread_ID[MAX_NUM+1];
    int         args[MAX_NUM+1]; 

    for(int i=0;i<=MAX_NUM;i++)
    {
        args[i] = i;
        pthread_create(&thread_ID[i],NULL,get_square_root,(void*)&args[i]);
    }

    for(int i=0;i<=MAX_NUM;i++)
    {
        void*   thread_result;

        pthread_join(thread_ID[i],&thread_result);
        printf("square root of %d = %f\n",i,*(double*)thread_result);
        free(thread_result);
    }

    return  0;    
}