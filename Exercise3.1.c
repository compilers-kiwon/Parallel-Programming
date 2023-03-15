#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

pthread_mutex_t lock;
int shared_data;

void*   thread_function(void* arg)
{
    int i;

    for(i=0;i<1024*1024;++i)
    {
        pthread_mutex_lock(&lock);
        shared_data++;
        pthread_mutex_unlock(&lock);
    }

    return  NULL;
}

int main(void)
{
    pthread_t   thread_ID;
    void*       thread_result;
    int         i;

    pthread_mutex_init(&lock,NULL);
    pthread_create(&thread_ID,NULL,thread_function,NULL);

    for(i=0;i<10;++i)
    {
        // sleep(1);
        pthread_mutex_lock(&lock);
        printf("Shared integer's value = %d\n",shared_data);
        pthread_mutex_unlock(&lock);
    }

    pthread_join(thread_ID,&thread_result);
    pthread_mutex_destroy(&lock);

    return  0;
}