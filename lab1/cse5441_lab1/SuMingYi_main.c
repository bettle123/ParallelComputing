/*
This code can be compiled and run ok.
Goal:
Using a collection of invertible functions (provided), implement a serial version of producer consumer. This program will
serve as a reference and also provide a basis for future multi-threaded implementations.

usage (how to run):
icc -o a.out main.c transform.o
./a.out PC_data_t00100

input file:
test.txt

output files:
none

compile (how to compile):
icc -o a.out main.c transform.o 
./a.out PC_data_t00100


coded by Su Ming Yi, OSU_ID: su.672, email: su.672@osu.edu
date: 2018.09.24

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

/*
int transformA(int key){return key;}
int transformB(int key){return key;}
int transformC(int key){return key;}
int transformD(int key){return key;}
*/
int main(int argc, char* argv[])
{
    time_t timer;
    timer = time(NULL);
    //printf("System time: %ld\n", timer);

    int q_size =5;
    struct work_entry work[5];
    printf("Welcome to Su Ming Yi's 5441 lab1!\n");
    printf("For this lab, the size of queue is %d.\n",q_size);
    int index = 0;
    int final_break =0;
    clock_t time_clock;
    clock_t time_clock2;
    double producer_time =0.0f;
    double consumer_time =0.0f;
    double user_time = 0.0f;

    FILE *file = fopen(argv[1], "r");
    //FILE *file = fopen("PC_data_t00100", "r");
    if(!file) {
        printf("Could not open file");
    return 1;
    }

    while(!final_break)
    {
/////////////////////////////////////////////////////////////////////////////////////////
        // producer
        time_clock = clock();
        int i;
        char input1[20],input2[20];
        int tmp_key;
        int record=-1;
        while(index<q_size)
        {
            fscanf(file, "%s\t%s", &input1, &input2);
            //printf("%s\t%s\n", input1, input2);

            if(strcmp(input1, "A")==0||strcmp(input1, "B")==0||
               strcmp(input1, "C")==0||strcmp(input1, "D")==0)
            {
                tmp_key = strtoumax(input2, NULL, 10);
                if(tmp_key>=0 && tmp_key<=1000)
                {
                    //printf("Q: %d\t %s\t%s\t\n", index, input1, input2);
                    work[index].cmd = input1[0];
                    work[index].key = tmp_key;
                    //printf("Q:%d\t%c\t%d\n", index, work[index].cmd, work[index].key);
                    //printf("Element[%d] = %d\n", index, work[index].key );
                    index = index+1;
                    record =index;
                }
            }

            if(strcmp(input1, "X")==0)
            {
                final_break = 1;
                record =index;
                index=q_size;
            }

        }
        //printf("things in queue: %d\n", record);
        //printf("index: %d\n",index);
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

        index =0;
        time_clock = clock() - time_clock;
        producer_time = producer_time +((double)time_clock)/CLOCKS_PER_SEC;
 ///////////////////////////////////////////////////////////////////////////////////////////
        // consumer
        time_clock2 = clock();
        while(index<record)
        {
            //printf("index: %d\n",index);
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
        index =0;
        time_clock2 = clock() - time_clock2;
        //printf ("It took me %d clicks (%f seconds).\n",time2,((double)time2)/CLOCKS_PER_SEC);
        consumer_time = consumer_time +((double)time_clock2)/CLOCKS_PER_SEC;

    }
    fclose(file);

    user_time = producer_time + consumer_time;
    printf("The producer runtime = %f seconds\n", producer_time);
    printf("The consumer runtime = %f seconds\n", consumer_time);
    printf("The user time= %f\n", user_time);

    time_t timer2 = time(NULL);
    printf("The real time= %f\n", timer-timer2);
    printf("The end of program\n");
    //printf ("It took me %d clicks (%f seconds).\n",time,((float)time)/CLOCKS_PER_SEC);

    return 0;
}