/*
Author: Su Ming Yi
Date: 10/23/2018

This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer.
This program will serve as a reference and also provide a basis for future multi-threaded implementataion.

compile (how to compile):
icc -o lab2_pthreads 1031_main.c transform.o -pthread
gcc -o test1 1031_main.c -lpthread

usage (how to run):
time ./lab2_pthreads < PC_data_x1

input file:
PC_data_t00100, PC_data_t01000, PC_data_t05000, PC_data_t10000, PC_data_x1

output file:
none

coded by Su Ming Yi, OSU_ID: su.672, email: su.672@osu.edu

*/
#include <pthread.h>
#include <stdio.h> /* scanf, printf */
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> /* clock_t, clock, CLOCKS_PER_SEC */
#define Q_SIZE 5
#define NUM_THREADS 2
typedef struct
{
    char cmd;
    int16_t key;
} work_entry;


typedef struct
{
    work_entry *job;
    int full;
    int empty;
    int head;
    int tail;
    int final_break;
    double producer_time;
    double consumer_time;
    double user_time;
    pthread_mutex_t the_mutex;
    pthread_cond_t cond_C, cond_P;
} work_array;

/*
int transformA(int key){return key;};
int transformB(int key){return key;};
int transformC(int key){return key;};
int transformD(int key){return key;};
*/


uint16_t transformA(uint16_t input_val);
uint16_t transformB(uint16_t input_val);
uint16_t transformC(uint16_t input_val);
uint16_t transformD(uint16_t input_val);


void *produer(void *ptr);
void *consumer(void *ptr);

void queueAdd(work_array *my_data);
void queueRemove(work_array *my_data);


void *producer(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    while(my_data->final_break!=1){
        //printf("Thread producer(): Starting..\n");
        //pthread_mutex_lock(&(my_data->the_mutex));
        
        while (my_data->full==1) 
        {
		    //printf ("producer: queue FULL.\n");
		    printf("producer(): Going to wait...\n");
		    pthread_mutex_lock(&(my_data->the_mutex));
		    pthread_cond_wait (&(my_data->cond_P), &(my_data->the_mutex));
		    //printf("producer(): unlocking the mutex\n");
            //pthread_mutex_unlock(&my_data->the_mutex);
		    //printf("producer(): Condition signal received.\n");
		    //pthread_mutex_lock(&my_data->the_mutex);
	    }
	/////////////////////////////////////////////////////////////////////////////////////////
    // producer produce data
    //queueAdd(my_data);
    // declare local variables
    int i = my_data->tail;
    char input1[20],input2[20];
    char tmp_cmd;
    int tmp_t_key;
    scanf("%s\t%s", &input1, &input2);
    if((input1[0]=='X'))
    {
        printf("The end of the program.\n");
		my_data->final_break=1;
		exit(0);
    }
    int tmp_key = strtoumax(input2, NULL, 10);
    if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        tmp_cmd = input1[0];
        tmp_t_key = transformA(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        tmp_cmd = input1[0];
        tmp_t_key = transformB(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        tmp_cmd = input1[0];
        tmp_t_key = transformC(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        tmp_cmd = input1[0];
        tmp_t_key = transformD(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    // write into the buffer
    //pthread_mutex_lock(&my_data->the_mutex);
    my_data->job[i].cmd = tmp_cmd;
    my_data->job[i].key = tmp_t_key;
    pthread_mutex_unlock(&(my_data->the_mutex));    
    
    if(my_data->tail == Q_SIZE)
    {
        my_data->tail =0;
    }
    if(my_data->tail == my_data->head)
    {
        my_data->full = 1;
    }
    
    pthread_cond_broadcast (&(my_data->cond_C));    
        
        
        
     /*   
        if(my_data->empty!=1)
        {
            //printf("producer(): already put the data into queue.\n");
            //printf("producer(): unlocking the mutex\n");
            pthread_mutex_unlock(&(my_data->the_mutex));
            //printf("producer(): send the signal to consumer.\n");
            pthread_cond_broadcast (&(my_data->cond_C));
        }
    */
        //printf("producer(): unlocking the mutex\n");
        //pthread_mutex_unlock(&my_data->the_mutex);
	    //usleep (1000);
	    //printf("Thread producer(): done.\n");
	    
    }
    pthread_exit(0);
};
void *consumer(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    while(my_data->final_break!=1)
    {
        //printf("Thread Consumer(): Starting.\n");
        //pthread_mutex_lock(&(my_data->the_mutex));
        while (my_data->empty==1) 
        {
		    //printf ("consumer: queue EMPTY.\n");
		    printf ("consumer(): Going into wait...\n");
		    pthread_mutex_lock(&(my_data->the_mutex));
		    pthread_cond_wait (&(my_data->cond_C), &(my_data->the_mutex));
		    //printf("consumer(): unlocking the mutex.\n");
            //pthread_mutex_unlock(&my_data->the_mutex);
		    //printf("consumer(): Condition signal received.\n");
        }
    ////////////////////////////////////////////////////////////
    //queueRemove(my_data);
    // consumer consume data
    int i=my_data->head;
    int encoder, decoder;
    // read the data from buffer
    //pthread_mutex_lock(&(my_data->the_mutex));
    char cur_cmd = my_data->job[i].cmd;
    encoder = my_data->job[i].key;
    pthread_mutex_unlock(&my_data->the_mutex);
    if(cur_cmd=='A')
    {
        decoder = transformA(encoder);
    }
    if(cur_cmd=='B')
    {
        decoder = transformB(encoder);
    }
    if(cur_cmd=='C')
    {
        decoder = transformC(encoder);
    }
    if(cur_cmd=='D')
    {
        decoder = transformD(encoder);
    }
    printf("Q:%d\t%c\t%d\t%d\n", i, cur_cmd, encoder, decoder);

    my_data->head++;
    if(my_data->head == Q_SIZE)
	{
		my_data->head =0;
	}
	if(my_data->head == my_data->tail)
	{
		my_data->empty =1;
	}
	my_data->full =0;
        
        //printf("consumer(): unlocking the mutex.\n");
        //pthread_mutex_unlock(&(my_data->the_mutex));
        //printf("consumer(): send the signal to producer.\n");
	    pthread_cond_broadcast (&(my_data->cond_P));
        
        //usleep(1000);
        //printf("Thread consumer(): done.\n");
    }
    pthread_exit(0);
};

void queueAdd(work_array *my_data)
{
    // declare local variables
    int i = my_data->tail;
    char input1[20],input2[20];
    scanf("%s\t%s", &input1, &input2);
    //printf("Producer Q tail: %d %s %s \n", my_data->tail, input1, input2);
    if((input1[0]=='X'))
    {
        printf("The end of the program.\n");
		my_data->final_break=1;
		exit(0);
    }
    int tmp_key = strtoumax(input2, NULL, 10);
    if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        my_data->job[i].cmd = input1[0];
        my_data->job[i].key = transformA(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        my_data->job[i].cmd = input1[0];
        my_data->job[i].key = transformB(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        my_data->job[i].cmd = input1[0];
        my_data->job[i].key = transformC(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        my_data->job[i].cmd = input1[0];
        my_data->job[i].key = transformD(tmp_key);
        my_data->tail++;
        my_data->empty =0;
    }
    if(my_data->tail == Q_SIZE)
    {
        my_data->tail =0;
    }
    if(my_data->tail == my_data->head)
    {
        my_data->full = 1;
    }
    //printf("producer(): empty: %d.\n", my_data->empty);
};

void queueRemove(work_array *my_data)
{
    // consumer consume data
    int i=my_data->head;
    char cur_cmd = my_data->job[i].cmd;
    int encoder, decoder;
    encoder = my_data->job[i].key;
    if(cur_cmd=='A')
    {
        decoder = transformA(encoder);
    }
    if(cur_cmd=='B')
    {
        decoder = transformB(encoder);
    }
    if(cur_cmd=='C')
    {
        decoder = transformC(encoder);
    }
    if(cur_cmd=='D')
    {
        decoder = transformD(encoder);
    }
    printf("Q:%d\t%c\t%d\t%d\n", i, cur_cmd, encoder, decoder);

    my_data->head++;
    if(my_data->head == Q_SIZE)
	{
		my_data->head =0;
	}
	if(my_data->head == my_data->tail)
	{
		my_data->empty =1;
	}
	my_data->full =0;
};


int main(int argc, char* argv[])
{
    //printf("This is Su Ming Yi's 5441 lab2.\n");
	//printf("The size of queue is %d.\n", Q_SIZE);
	//printf("The number of threads are %d.\n", NUM_THREADS);
    
    printf("Initialize the queue.\n");
    pthread_t con, pro;
    void *status;
    
    // initialize queue
    work_array *ptr = malloc(sizeof(work_array));
    ptr->empty = 1;
    ptr->full = 0;
    ptr->head = 0;
    ptr->tail = 0;
    ptr->final_break =0;
    ptr->job = malloc(sizeof(work_entry)*Q_SIZE);
    
    
    // Initialize the mutex and condition variables
    pthread_mutex_init(&(ptr->the_mutex), NULL);	
    pthread_cond_init(&(ptr->cond_C), NULL);		
    pthread_cond_init(&(ptr->cond_P), NULL);	

    printf("Main(): Start producer() and consumer() functions.\n");

    pthread_create(&pro, NULL, producer, (void *) ptr);
    pthread_create(&con, NULL, consumer, (void *) ptr);

    pthread_join(pro, NULL);
    pthread_join(con, NULL);
    //printf("Main: completed join with thread consumer\n");
    //printf("Main(): Waited and joined with producer and consumer threads.\n");

    pthread_mutex_destroy(&(ptr->the_mutex));
    pthread_cond_destroy(&(ptr->cond_C));		
    pthread_cond_destroy(&(ptr->cond_P));		
    
    //printf("Main: program completed. Exiting.\n");
    pthread_exit(NULL);

    /*
    ptr->user_time = ptr->consumer_time + ptr->producer_time;
    printf("The producer runtime = %f seconds\n", ptr->producer_time);
    printf("The consumer runtime = %f seconds\n", ptr->consumer_time);
    printf("The total user time = %f seconds\n", ptr->user_time);
    */
    // let the console keep open until someone enter anything.
	//getchar();
    return 0;
}
