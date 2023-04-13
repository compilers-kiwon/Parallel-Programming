#include    <pthread.h>
#include    <semaphore.h>
#include    <stdio.h>

#define PCBUFFER_SIZE   8

typedef struct{
    void*   buffer[PCBUFFER_SIZE];
    pthread_mutex_t lock;
    sem_t   used;
    sem_t   free;
    int     next_in;
    int     next_out;
} pcbuffer_t;

void    pcbuffer_init(pcbuffer_t*);
void    pcbuffer_destroy(pcbuffer_t*);
void    pcbuffer_push(pcbuffer_t*,void*);
void*   pcbuffer_pop(pcbuffer_t*);

void    pcbuffer_init(pcbuffer_t* p)
{
    pthread_mutex_init(&p->lock,NULL);
    sem_init(&p->used,0,0);
    sem_init(&p->free,0,PCBUFFER_SIZE);
    p->next_in = 0;
    p->next_out =0;
}

void    pcbuffer_destroy(pcbuffer_t* p)
{
    pthread_mutex_destroy(&p->lock);
    sem_destroy(&p->used);
    sem_destroy(&p->free);
}

void    pcbuffer_push(pcbuffer_t* p,void* value)
{
    sem_wait(&p->free);
    pthread_mutex_lock(&p->lock);
    p->buffer[p->next_in++] = value;
    if( p->next_in == PCBUFFER_SIZE ) p->next_in = 0;
    pthread_mutex_unlock(&p->lock);
    sem_post(&p->used);
}

void*   pcbuffer_pop(pcbuffer_t* p)
{
    void*   return_value;

    sem_wait(&p->used);
    pthread_mutex_lock(&p->lock);
    return_value = p->buffer[p->next_out++];
    if( p->next_out == PCBUFFER_SIZE ) p->next_out = 0;
    pthread_mutex_unlock(&p->lock);
    sem_post(&p->free);
    return  return_value;
}

#define NUM_OF_THREADS      10
#define NUM_OF_ITERATIONS   5

pcbuffer_t  buf;

void*   thread_func(void* arg)
{
    int thread_no = *(int*)arg;
    int data = *(int*)arg;

    for(int i=0;i<NUM_OF_ITERATIONS;i++)
    {
        pcbuffer_push(&buf,&data);
        printf("push %d (thread %d)\n",data,thread_no);
        
        data = *(int*)pcbuffer_pop(&buf);
        printf("pop %d (thread %d)\n",data,thread_no);
    }

    pthread_exit((void*)NULL);
}

int     main(void)
{
    pthread_t   thread_ID[NUM_OF_THREADS];
    void*       thread_result;
    int         args[NUM_OF_THREADS] = {0,1,2,3,4,5,6,7,8,9};

    pcbuffer_init(&buf);

    for(int i=0;i<NUM_OF_THREADS;i++)
    {
        pthread_create(&thread_ID[i],NULL,thread_func,(void*)&args[i]);
    }

    for(int i=0;i<NUM_OF_THREADS;i++)
    {
        pthread_join(thread_ID[i],&thread_result);
    }

    pcbuffer_destroy(&buf);

    return  0;
}