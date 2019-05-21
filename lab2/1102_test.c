/******************************************************
Author: Su Ming Yi
Date: 11/01/2018
Goal: 
	To learn how to use condition variable in pthread.
How to compile it:
	icc -o test2 1102_main.c -pthread
How to run it:
	./test2
********************************************************/	

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

pthread_mutex_t array_mutex;
pthread_cond_t cond_C, cond_P;

typedef struct
{

    int full;
    int empty;
} work_entry;

void *producer(void *ptr);
void *consumer(void *ptr);



int main(int argc, char *argv[])
{
    printf("Start to modify my pthreads program.\n");
    
    work_entry *ptr = malloc(sizeof(work_entry));
    
    pthread_t pro, con;
    
    // Initialize the mutex and condition variables
    pthread_mutex_init(&array_mutex, NULL);	
    pthread_cond_init(&cond_C, NULL);		
    pthread_cond_init(&cond_P, NULL);	
    
    printf("Initialize the mutex and condition vairables Successfully.\n");
    
    ptr->empty = 1;
    ptr->full = 0;
    printf("Initialize empty: %d, full: %d \n.", ptr->empty, ptr->full);
    
    
    
    
    pthread_create(&pro, NULL, producer, (void *)ptr);
	pthread_create(&con, NULL, consumer, (void *)ptr);
    
    
    pthread_join(pro, NULL);
    pthread_join(con, NULL);
    
    /*
	int i, rc;
	long t1=1, t2=2, t3=3;
	pthread_t threads[3];
	pthread_attr_t attr;
	
	// Initialize mutex and condition variable objects
	pthread_mutex_init(&count_mutex, NULL);
	pthread_cond_init(&count_threshold_cv, NULL);
	
	// For portability, explicitly create threads in a joinable state
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_create(&threads[0], &attr, watch_count, (void *)t1);
	pthread_create(&threads[1], &attr, inc_count, (void *)t2);
	pthread_create(&threads[2], &attr, inc_count, (void *)t3);
	
	// Wait for all threads to complete
	for (i=0; i <NUM_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}
	printf("Main(): Waited and joined with %d threads. Final value of count = %d. Done.\n", NUM_THREADS, count);
	
	// Clean up and exit
	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&count_mutex);
	pthread_cond_destroy(&count_threshold_cv);
	pthread_exit(NULL);
    */	
}



void *producer(void *ptr)
{
    
}

void *consumer(void *ptr)
{
    
}

 