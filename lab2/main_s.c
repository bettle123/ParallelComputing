/*
Author: Su Ming Yi
Date: 10/21/2018

This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer.
This program will serve as a reference and also provide a basis for future multi-threaded implementataion.

compile (how to compile):
icc -o a.out main_s.c transform.o
icc -pthread -o main_s.c transform.o

usage (how to run):
time ./a.out < PC_data_t00100

input file:
PC_data_t00100, PC_data_t01000, PC_data_t05000, PC_data_t10000, PC_data_x1

output file:
none

coded by Su Ming Yi, OSU_ID: su.672, email: su.672@osu.edu

*/


#include <stdio.h> /* printf */
#include <string.h>
#include <inttypes.h>
#include <math.h>       /* sqrt */
#include <stdlib.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
struct work_entry
{
    char cmd;
    int16_t key;
};

uint16_t transformA(uint16_t input_val);
uint16_t transformB(uint16_t input_val);
uint16_t transformC(uint16_t input_val);
uint16_t transformD(uint16_t input_val);

void producer(int q_size, struct work_entry work[], int *record, int *final_break, double *producer_time)
{
    // producer
    int index=0;
    clock_t time_clock;
    time_clock = clock();
    int i;
    char input1[20],input2[20];
    int tmp_key;
    while(index<q_size)
    {
        scanf("%s\t%s", &input1, &input2);
        if(strcmp(input1, "A")==0||strcmp(input1, "B")==0||
           strcmp(input1, "C")==0||strcmp(input1, "D")==0)
        {
            tmp_key = strtoumax(input2, NULL, 10);
            if(tmp_key>=0 && tmp_key<=1000)
            {
                work[index].cmd = input1[0];
                work[index].key = tmp_key;
                index = index+1;
                *record =index;
            }
        }
        if(strcmp(input1, "X")==0)
        {
            *final_break = 1;
            *record =index;
            index=q_size;
        }

    }
    // The end of read file
    // call transformX() function to create an encoded key.
    for(i=0;i<*record;i++)
    {
        if(work[i].cmd=='A')
        {
            work[i].key = transformA(work[i].key);
        }
        if(work[i].cmd=='B')
        {
            work[i].key = transformB(work[i].key);
        }
        if(work[i].cmd=='C')
        {
            work[i].key = transformC(work[i].key);
        }
        if(work[i].cmd=='D')
        {
            work[i].key = transformD(work[i].key);
        }
    }
    // finish this round of producer
    // now calculate the time
    time_clock = clock() - time_clock;
    *producer_time = *producer_time +((double)time_clock)/CLOCKS_PER_SEC;
}


void consumer(struct work_entry work[], int *record, double *consumer_time)
{
    // consumer
    clock_t time_clock2;
    int index =0;
    time_clock2 = clock();
    while(index<*record)
    {
        int encoder, decoder;
        encoder = work[index].key;
        if(work[index].cmd=='A')
        {
            decoder = transformA(work[index].key);
        }
        if(work[index].cmd=='B')
        {
            decoder = transformB(work[index].key);
        }
        if(work[index].cmd=='C')
        {
            decoder = transformC(work[index].key);
        }
        if(work[index].cmd=='D')
        {
            decoder = transformD(work[index].key);
        }
        printf("Q:%d\t%c\t%d\t%d\n",index,work[index].cmd, encoder, decoder);
        index = index + 1;
    }
    // finish this round of consumer
    // calculate the time
    time_clock2 = clock() - time_clock2;
    *consumer_time = *consumer_time +((double)time_clock2)/CLOCKS_PER_SEC;

}


int main(int argc, char* argv[])
{
    ////////////////////////////////////////////////////////////////
    // These are the variables that we use in this program
    int q_size =5;
    struct work_entry work[q_size];
    int record=-1;
    int final_break =0;
    double producer_time =0.0f;
    double consumer_time =0.0f;
    double user_time = 0.0f;
    ///////////////////////////////////////////////////////////////
    printf("This is Su Ming Yi's 5441 lab2.\n");
    printf("For this lab, the size of queue is %d.\n",q_size);
    while(!final_break)
    {
        producer(q_size, work, &record, &final_break, &producer_time);
        consumer(work, &record, &consumer_time);
    }
    user_time = producer_time + consumer_time;
    printf("The producer runtime = %f seconds\n", producer_time);
    printf("The consumer runtime = %f seconds\n", consumer_time);
    printf("The user time= %f\n", user_time);
    getchar();
    return 0;
}
