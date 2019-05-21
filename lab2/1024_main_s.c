/*
Author: Su Ming Yi
Date: 10/23/2018

This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer.
This program will serve as a reference and also provide a basis for future multi-threaded implementataion.

compile (how to compile):
icc -o test1 1024_main_s.c transform.o -lpthread
gcc -o test1 1024_main_s.c

usage (how to run):
time ./test1 < PC_data_t00100
time ./test1 < PC_data_x1

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
struct work_entry
{
    char cmd[Q_SIZE];
    int16_t key[Q_SIZE];
    int full; // item produced
    int empty; // remaining space
    int final_break;
    double producer_time;
    double consumer_time;
    double user_time;
    pthread_mutex_t the_mutex;
    pthread_cond_t condc, condp;
};



uint16_t transformA(uint16_t input_val);
uint16_t transformB(uint16_t input_val);
uint16_t transformC(uint16_t input_val);
uint16_t transformD(uint16_t input_val);

/*
int transformA(int key){return key;};
int transformB(int key){return key;};
int transformC(int key){return key;};
int transformD(int key){return key;};
*/
void *produer(void *ptr);
void *consumer(void *ptr);


void *producer(void *ptr)
{
    struct work_entry *my_data;
    my_data = (struct work_entry *) ptr;
    printf("(Procuder start) full:%d empty: %d\n", my_data->full, my_data->empty);
    /*declare useful local variables */
    while( ( my_data->empty >0) && (!my_data->final_break) ){
        pthread_mutex_lock(&my_data->the_mutex);	// protect buffer 
        int i = my_data->full;
        char input1[20],input2[20];
        scanf("%s\t%s", &input1, &input2);
        printf("Producer: %s\t%s\n", input1, input2);
        int tmp_key = strtoumax(input2, NULL, 10);
        if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            my_data->cmd[i] = input1[0];
            my_data->key[i] = transformA(tmp_key);
            my_data->empty = my_data->empty -1;
            my_data->full = my_data->full +1;
        }
        if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            my_data->cmd[i] = input1[0];
            my_data->key[i] = transformB(tmp_key);
            my_data->empty = my_data->empty -1;
            my_data->full = my_data->full +1;
        }
        if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            my_data->cmd[i] = input1[0];
            my_data->key[i] = transformC(tmp_key);
            my_data->empty = my_data->empty -1;
            my_data->full = my_data->full +1;
        }
        if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            my_data->cmd[i] = input1[0];
            my_data->key[i] = transformD(tmp_key);
            my_data->empty = my_data->empty -1;
            my_data->full = my_data->full +1;
        }
        if((input1[0]=='X'))
        {
		    my_data->final_break =1;
		    printf("The end of the program.\n");
        }
        pthread_mutex_unlock(&my_data->the_mutex);	// release the buffer 
    }
    printf("(Procuder rest) full:%d empty: %d\n", my_data->full, my_data->empty);
    if(my_data->full >0){
        pthread_cond_wait(&my_data->condp, &my_data->the_mutex); // let producer wait
        pthread_cond_signal(&my_data->condc);	// wake up consumer 
    }
    pthread_exit(0);
    /*
    pthread_mutex_lock(&the_mutex);	// protect buffer 
    // If there is something in the buffer then wait 
    pthread_cond_wait(&condp, &the_mutex);
    pthread_cond_signal(&condc);	// wake up consumer 
    pthread_mutex_unlock(&the_mutex);	// release the buffer 
    pthread_exit(0);
    */
};
void *consumer(void *ptr)
{
    struct work_entry *my_data;
    my_data = (struct work_entry *) ptr;
    
    printf("(Consumer start) empty: %d, full: %d \n", my_data->empty, my_data->full);
    
    // 10/25
    while( my_data->full>0 )
    {
    pthread_mutex_lock(&my_data->the_mutex);	// protect buffer 
    int i=my_data->empty;
    
    char cur_cmd = my_data->cmd[i];
    int encoder, decoder;
    encoder = my_data->key[i];
    
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
    printf("Consumer Q:%d\t%c\t%d\t%d\n", i, cur_cmd, encoder, decoder);

    my_data->full = my_data->full - 1;
    my_data->empty = my_data->empty +1;
    pthread_mutex_unlock(&my_data->the_mutex);	// release the buffer
    }
    
    if(my_data->empty>0)
    {
        pthread_cond_wait(&my_data->condc, &my_data->the_mutex); // let consumer rest
        pthread_cond_signal(&my_data->condp);	// wake up consumer 
    }
    
    //pthread_cond_wait(&condc, &the_mutex); // let consumer rest
    //pthread_cond_signal(&condp);	// wake up consumer 
    //pthread_mutex_unlock(&the_mutex);	// release the buffer
    //producer(my_data);
    printf("(Consumer rest) empty: %d, full: %d \n", my_data->empty, my_data->full);
    pthread_exit(0);
    /*
    pthread_mutex_lock(&the_mutex);	// protect buffer 
    pthread_cond_wait(&condc, &the_mutex);
    pthread_cond_signal(&condp);	// wake up consumer 
    pthread_mutex_unlock(&the_mutex);	// release the buffer 
    pthread_exit(0);
    */
};


int main(int argc, char* argv[])
{
    //printf("This is Su Ming Yi's 5441 lab2.\n");
	//printf("The size of queue is %d.\n", Q_SIZE);
	//printf("The number of threads are %d.\n", NUM_THREADS);
    // 10/24
    pthread_t pro, con;
    // 10/25
    struct work_entry *ptr = malloc(sizeof(struct work_entry));
    
    ptr->empty = Q_SIZE;
    ptr->full = 0;
    ptr->final_break =0;
    
    // Initialize the mutex and condition variables
    pthread_mutex_init(&ptr->the_mutex, NULL);	
    pthread_cond_init(&ptr->condc, NULL);		/* Initialize consumer condition variable */
    pthread_cond_init(&ptr->condp, NULL);		/* Initialize producer condition variable */

    // Create the threads
    /*
    while(!ptr->final_break)
    {
        pthread_create(&pro, NULL, producer, (void *) ptr);
        pthread_create(&con, NULL, consumer, (void *) ptr);
        pthread_join(&pro, NULL);
        pthread_join(&con, NULL);
    }
    */
    
    
    pthread_create(&pro, NULL, producer, (void *) ptr);
    pthread_create(&con, NULL, consumer, (void *) ptr);
    
    // Wait for the threads to finish
    // Otherwise main might run to the end
    // and kill the entire process when it exits.
    pthread_join(pro, NULL);
    pthread_join(con, NULL);
    
    // Cleanup -- would happen automatically at end of program
    pthread_mutex_destroy(&ptr->the_mutex);	/* Free up the_mutex */
    pthread_cond_destroy(&ptr->condc);		/* Free up consumer condition variable */
    pthread_cond_destroy(&ptr->condp);		/* Free up producer condition variable */
    
    ptr->user_time = ptr->consumer_time + ptr->producer_time;
    printf("The producer runtime = %f seconds\n", ptr->producer_time);
    printf("The consumer runtime = %f seconds\n", ptr->consumer_time);
    printf("The total user time = %f seconds\n", ptr->user_time);
    // let the console keep open until someone enter anything.
	getchar();
    return 0;
}

