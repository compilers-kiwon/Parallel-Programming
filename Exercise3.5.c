#include    <pthread.h>
#include    <stdio.h>

int                 shared;
pthread_rwlock_t    lock;

void*   thread_function(void* arg)
{
    for(;shared!=0x10;)
    {
        pthread_rwlock_rdlock(&lock);
        printf("read value = 0x%08X\n",shared);
        pthread_rwlock_unlock(&lock);
    }

    pthread_exit((void*)NULL);
}

int     main(void)
{
    pthread_t   thread_ID;
    void*       thread_result;
    int         i;

    pthread_rwlock_init(&lock,NULL);
    pthread_create(&thread_ID,NULL,thread_function,NULL);


    for(i=0;i<0x10;i++)
    {
        pthread_rwlock_wrlock(&lock);
        printf("write value = 0x%08X\n",++shared);
        pthread_rwlock_unlock(&lock);
    }

    pthread_join(thread_ID,&thread_result);
    pthread_rwlock_destroy(&lock);

    return  0;
}