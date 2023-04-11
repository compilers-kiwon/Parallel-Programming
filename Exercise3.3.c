#include    <stdio.h>
#include    <pthread.h>
#include    <unistd.h>

pthread_cond_t  is_zero;
pthread_mutex_t mutex;

int shared_data = 0x1000;

void*   thread_function(void* arg)
{
    while(shared_data>0)
    {
        pthread_mutex_lock(&mutex);
        --shared_data;
        pthread_mutex_unlock(&mutex);
    }

    printf("shared_data gets 0 and it sends a signal to another threads.\n");
    pthread_cond_signal(&is_zero);
    return  NULL;
}

int main(void)
{
    pthread_t   thread_ID;
    void*       exit_status;
    int         i;

    pthread_cond_init(&is_zero,NULL);
    pthread_mutex_init(&mutex,NULL);

    pthread_create(&thread_ID,NULL,thread_function,NULL);

    pthread_mutex_lock(&mutex);
    
    while(shared_data!=0)
    {
        printf("shared data is not 0 and wait for a signal to indicate that it gets 0.\n");
        pthread_cond_wait(&is_zero,&mutex);
        printf("receive a signal.\n");
    }

    pthread_mutex_unlock(&mutex);
    pthread_join(thread_ID,&exit_status);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&is_zero);

    return  0;
}