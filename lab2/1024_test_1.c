/*
Author: Su Ming Yi
Date: 10/23/2018

This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer.
This program will serve as a reference and also provide a basis for future multi-threaded implementataion.

compile (how to compile):
icc -o test1 1024_test_1.c transform.o
gcc -o test1 1023_main_s.c

usage (how to run):
time ./test1 < PC_data_t00100

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

void *producer(struct work_entry work[], int *myIndex, int *final_break, double *producer_time, int *run_code)
{
	clock_t time_clock;
    time_clock = clock();
    char input1[20],input2[20];
    scanf("%s\t%s", &input1, &input2);
    //printf("Producer: %s\t%s\n", input1, input2);
    int tmp_key = strtoumax(input2, NULL, 10);
    if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[*myIndex].cmd = input1[0];
        work[*myIndex].key = transformA(tmp_key);
        *run_code =1;
    }
    if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[*myIndex].cmd = input1[0];
        work[*myIndex].key = transformB(tmp_key);
        *run_code =1;
    }
    if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[*myIndex].cmd = input1[0];
        work[*myIndex].key = transformC(tmp_key);
        *run_code =1;
    }
    if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
    {
        work[*myIndex].cmd = input1[0];
        work[*myIndex].key = transformD(tmp_key);
        *run_code =1;
    }
    if(strcmp(input1, "X")==0)
    {
        *final_break = 1;
    }
	if(*myIndex==5)
	{
		*myIndex=0;
	}
    time_clock = clock() - time_clock;
    *producer_time = *producer_time +((double)time_clock)/CLOCKS_PER_SEC;
	
};
void *consumer(struct work_entry work[], int myIndex, double *consumer_time, int *run_code)
{
	clock_t time_clock2;
    time_clock2 = clock();
    int encoder, decoder;
    encoder = work[myIndex].key;
    if(work[myIndex].cmd=='A')
    {
        decoder = transformA(work[myIndex].key);
    }
    if(work[myIndex].cmd=='B')
    {
        decoder = transformB(work[myIndex].key);
    }
    if(work[myIndex].cmd=='C')
    {
        decoder = transformC(work[myIndex].key);
    }
    if(work[myIndex].cmd=='D')
    {
        decoder = transformD(work[myIndex].key);
    }
	*run_code =0;
    printf("Q:%d\t%c\t%d\t%d\n", myIndex, work[myIndex].cmd, encoder, decoder);
    time_clock2 = clock() - time_clock2;
    *consumer_time = *consumer_time + ((double)time_clock2)/CLOCKS_PER_SEC;
};


int main(int argc, char* argv[])
{
	pthread_t threads[NUM_THREADS];
	int record=-1;
	int final_break =0;
	int run_code = 0;
	struct work_entry work[Q_SIZE];
	double producer_time =0.0f;
	double consumer_time =0.0f;
	double user_time = 0.0f;
	printf("This is Su Ming Yi's 5441 lab2.\n");
	printf("The size of queue is %d.\n", Q_SIZE);
	printf("The number of threads are %d.\n", NUM_THREADS);
	
	int rc;
	long t=0;
	int myIndex =0;
    while(!final_break)
    {
		producer(work, &myIndex, &final_break, &producer_time, &run_code);
		if(run_code){
			consumer(work, myIndex, &consumer_time, &run_code);
		}
		/*
		rc = pthread_create(&threads[0], NULL, producer(work, &record, &final_break, &producer_time), (void *)0);
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
		rc = pthread_create(&threads[1], NULL, consumer(work, &record, &consumer_time), (void *)0);
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
		}
		*/
        //consumer(work, &record, &consumer_time);
		
    }
    user_time = consumer_time + producer_time;
    printf("The producer runtime = %f seconds\n", producer_time);
    printf("The consumer runtime = %f seconds\n", consumer_time);
    printf("The total user time = %f seconds\n", user_time);
    // let the console keep open until someone enter anything.
	getchar();
    return 0;
}

