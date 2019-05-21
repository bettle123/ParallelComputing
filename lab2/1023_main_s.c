/*
Author: Su Ming Yi
Date: 10/23/2018

This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer.
This program will serve as a reference and also provide a basis for future multi-threaded implementataion.

compile (how to compile):
icc -o test1 1023_main_s.c transform.o -lpthread
gcc -o test1 1023_main_s.c

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
    char cmd;
    int16_t key;
};
/***********************************************************/
//global variable
pthread_mutex_t the_mutex;
pthread_cond_t condc, condp;
int buffer = 0;
// If the buffer contains zero, that indicates that the buffer is empty.

int myIndex =0;
int record=-1;
int final_break =0;
int run_code = 0;
//struct work_entry work[Q_SIZE];
struct work_entry work[1];
double producer_time =0.0f;
double consumer_time =0.0f;
double user_time = 0.0f;
/*********************************************************/


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

void *producer(void *ptr)
{
    printf("Producer Start: \n");
    /*declare useful local variables */
    int i=0;
    char input1[20],input2[20];
    scanf("%s\t%s", &input1, &input2);
    printf("Producer: %s\t%s\n", input1, input2);
    int tmp_key = strtoumax(input2, NULL, 10);
    pthread_mutex_lock(&the_mutex);	/* protect buffer */
    if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[i].cmd = input1[0];
        work[i].key = transformA(tmp_key);
    }
    if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[i].cmd = input1[0];
        work[i].key = transformB(tmp_key);
    }
    if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[i].cmd = input1[0];
        work[i].key = transformC(tmp_key);
    }
    if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[i].cmd = input1[0];
        work[i].key = transformD(tmp_key);
    }
    printf("Procuder rest.\n");
    /* If there is something in the buffer then wait */
    pthread_cond_wait(&condp, &the_mutex);
    pthread_cond_signal(&condc);	/* wake up consumer */
    pthread_mutex_unlock(&the_mutex);	/* release the buffer */
    pthread_exit(0);
};
void *consumer(void *ptr)
{
    printf("Consumer Start: \n");
    int i=0;
    pthread_mutex_lock(&the_mutex);	/* protect buffer */
    int encoder, decoder;
    encoder = work[i].key;
    if(work[i].cmd=='A')
    {
        decoder = transformA(work[i].key);
    }
    if(work[i].cmd=='B')
    {
        decoder = transformB(work[i].key);
    }
    if(work[i].cmd=='C')
    {
        decoder = transformC(work[i].key);
    }
    if(work[i].cmd=='D')
    {
        decoder = transformD(work[i].key);
    }
    printf("Consumer Q:%d\t%c\t%d\t%d\n", i, work[i].cmd, encoder, decoder);
    printf("Consumer rest.\n");
    buffer =0;
    pthread_cond_wait(&condc, &the_mutex);
    pthread_cond_signal(&condp);	/* wake up consumer */
    pthread_mutex_unlock(&the_mutex);	/* release the buffer */
    pthread_exit(0);
};


int main(int argc, char* argv[])
{
    //printf("This is Su Ming Yi's 5441 lab2.\n");
	//printf("The size of queue is %d.\n", Q_SIZE);
	//printf("The number of threads are %d.\n", NUM_THREADS);
    // 10/24
    pthread_t pro, con;

    // Initialize the mutex and condition variables
    pthread_mutex_init(&the_mutex, NULL);	
    pthread_cond_init(&condc, NULL);		/* Initialize consumer condition variable */
    pthread_cond_init(&condp, NULL);		/* Initialize producer condition variable */

    // Create the threads
    pthread_create(&con, NULL, consumer, NULL);
    pthread_create(&pro, NULL, producer, NULL);
    
    // Wait for the threads to finish
    // Otherwise main might run to the end
    // and kill the entire process when it exits.
    pthread_join(&con, NULL);
    pthread_join(&pro, NULL);

    // Cleanup -- would happen automatically at end of program
    pthread_mutex_destroy(&the_mutex);	/* Free up the_mutex */
    pthread_cond_destroy(&condc);		/* Free up consumer condition variable */
    pthread_cond_destroy(&condp);		/* Free up producer condition variable */
    
    user_time = consumer_time + producer_time;
    printf("The producer runtime = %f seconds\n", producer_time);
    printf("The consumer runtime = %f seconds\n", consumer_time);
    printf("The total user time = %f seconds\n", user_time);
    // let the console keep open until someone enter anything.
	getchar();
    return 0;
}

