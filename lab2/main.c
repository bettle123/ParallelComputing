/*
Author: Su Ming Yi
Date: 10/21/2018

This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer.
This program will serve as a reference and also provide a basis for future multi-threaded implementataion.

compile (how to compile):
icc -o out main.c

usage (how to run):
time ./out

input file:
PC_data_t00100, PC_data_t01000, PC_data_t05000, PC_data_t10000, PC_data_x1

output file:
none

coded by Su Ming Yi, OSU_ID: su.672, email: su.672@osu.edu

*/


#include <stdio.h> /* printf */
#include <inttypes.h>
#include <math.h>       /* sqrt */
#include <stdlib.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
struct work_entry
{
    char cmd;
    int16_t key;
};
/*
uint16_t transformA(uint16_t input_val);
uint16_t transformB(uint16_t input_val);
uint16_t transformC(uint16_t input_val);
uint16_t transformD(uint16_t input_val);
*/
/////////////////////////////////////////////////////////////
// global variable
time_t timer;
int q_size =5;
struct work_entry work[5];
int myIndex = 0;
int record=-1;
int final_break =0;
clock_t time_clock;
clock_t time_clock2;
double producer_time =0.0f;
double consumer_time =0.0f;
double user_time = 0.0f;
//////////////////////////////////////////////////////////////
void producer(FILE *file)
{
    // producer
        time_clock = clock();
        int i;
        char input1[20],input2[20];
        int tmp_key;
        while(myIndex<q_size)
        {
            fscanf(file, "%s\t%s", &input1, &input2);

            if(strcmp(input1, "A")==0||strcmp(input1, "B")==0||
               strcmp(input1, "C")==0||strcmp(input1, "D")==0)
            {
                tmp_key = strtoumax(input2, NULL, 10);
                if(tmp_key>=0 && tmp_key<=1000)
                {
                    work[myIndex].cmd = input1[0];
                    work[myIndex].key = tmp_key;
                    myIndex = myIndex+1;
                    record =myIndex;
                }
            }

            if(strcmp(input1, "X")==0)
            {
                final_break = 1;
                record =myIndex;
                myIndex=q_size;
            }

        }
        // call transformX() function to create an encoded key.
        for(i=0;i<record;i++)
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
            //printf("Q: %d\t %c\t%d\t\n", i, work[i].cmd, work[i].key);
        }

        myIndex =0;
        time_clock = clock() - time_clock;
        producer_time = producer_time +((double)time_clock)/CLOCKS_PER_SEC;
}


void consumer()
{
// consumer
        time_clock2 = clock();
        while(myIndex<record)
        {
            //printf("myIndex: %d\n",myIndex);
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
            printf("Q:%d\t%c\t%d\t%d\n",myIndex,work[myIndex].cmd, encoder, decoder);
            myIndex = myIndex + 1;
        }
        myIndex =0;
        time_clock2 = clock() - time_clock2;
        //printf ("It took me %d clicks (%f seconds).\n",time2,((double)time2)/CLOCKS_PER_SEC);
        consumer_time = consumer_time +((double)time_clock2)/CLOCKS_PER_SEC;

}


int transformA(int key){return key;}
int transformB(int key){return key;}
int transformC(int key){return key;}
int transformD(int key){return key;}

int main(int argc, char* argv[])
{
    printf("This is Su Ming Yi's 5441 lab2!\n");
    printf("For this lab, the size of queue is %d.\n",q_size);
    //FILE *file = fopen("PC_data_x1", "r");
    FILE *file = fopen("PC_data_t00100", "r");
    if(!file) {
        printf("Could not open file");
        return 1;
    }

    while(!final_break)
    {
        producer(file);
 ///////////////////////////////////////////////////////////////////////////////////////////
        consumer();
    }
    fclose(file);
    user_time = producer_time + consumer_time;
    printf("The producer runtime = %f seconds\n", producer_time);
    printf("The consumer runtime = %f seconds\n", consumer_time);
    printf("The user time= %f\n", user_time);
    getchar();
    return 0;
}
