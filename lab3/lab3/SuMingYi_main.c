/*
Author: Su Ming Yi
Date: 11/12/2018
goal: use omp to solve producer consumer problem.
How to compile it:
    icc -fopenmp -o lab3_omp SuMingYi_main.c transform.o
How to run it:
    time ./lab3_omp 4 < PC_data_x1
// 4 is the number of threads


OpenMP can only handle `for' loops

*/
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> 
#define Q_SIZE 5

uint16_t transformA(uint16_t input_val);
uint16_t transformB(uint16_t input_val);
uint16_t transformC(uint16_t input_val);
uint16_t transformD(uint16_t input_val);

int check=0;

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
} work_array;

void producer(void *ptr);
void consumer(void *ptr);

void producer(void *ptr)
{
    clock_t start, end;
    start = clock();
    
    clock_t time_clock = clock();
    work_array *my_data;
    my_data = (work_array *) ptr;
    // copy the array from global to local
    work_array *local_array = malloc(sizeof(work_array));
    local_array->job = malloc(sizeof(work_entry)*Q_SIZE);
    #pragma omp critical
    {
        local_array->empty = my_data->empty;
        local_array->full = my_data->full;
        local_array->head = my_data->head;
        local_array->tail = my_data->tail;
        local_array->final_break =my_data->final_break;
    }
    
    
    while(check==0)
    {
        if(local_array->full==1)
        {
            return;
        }
        /////////////////////////////////////////////////////////////////////////////////////////
        // producer produce data
        // declare local variables
        int i = local_array->tail;
        char input1[20],input2[20];
        char tmp_cmd;
        int tmp_t_key;
        scanf("%s\t%s", &input1, &input2);
        //printf("producer(): %s\t%s\n", input1, input2);
        if((input1[0]=='X'))
        {
            //printf("The end of the program.\n");
            local_array->final_break =1;
		    #pragma omp critical
            {
                check=1;
                my_data->final_break =local_array->final_break;
            }
            return;
        }
        int tmp_key = strtoumax(input2, NULL, 10);
        if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            tmp_t_key = transformA(tmp_key);

            local_array->tail++;
            local_array->empty =0;

        }
        if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            tmp_t_key = transformB(tmp_key);

            local_array->tail++;
            local_array->empty =0;
            
        }
        if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            tmp_t_key = transformC(tmp_key);

            local_array->tail++;
            local_array->empty =0;
            
        }
        if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            tmp_t_key = transformD(tmp_key);
        
            local_array->tail++;
            local_array->empty =0;
        }
        local_array->job[i].cmd = tmp_cmd;
        local_array->job[i].key = tmp_t_key;
        if(local_array->tail == Q_SIZE)
        {
            local_array->tail =0;
            if(local_array->tail == local_array->head)
            {
                local_array->full = 1;
            }
        }
        #pragma omp critical
        {
            my_data->empty = local_array->empty;
            my_data->full = local_array->full;
            my_data->head = local_array->head;
            my_data->tail = local_array->tail;
            my_data->final_break =local_array->final_break;
            my_data->job[i].cmd = local_array->job[i].cmd;
            my_data->job[i].key = local_array->job[i].key;
        }
        
        time_clock = clock() - time_clock;
	    local_array->producer_time = ((double)time_clock)/CLOCKS_PER_SEC;
	    #pragma omp critical
        {
	        my_data->producer_time += local_array->producer_time;
        }
        consumer(my_data);
    }
}

void consumer(void *ptr)
{
    //clock_t time_clock = clock();
    work_array *my_data;
    my_data = (work_array *) ptr;
    int i;
    // copy the array from global to local
    work_array *local_array = malloc(sizeof(work_array));
    local_array->job = malloc(sizeof(work_entry)*Q_SIZE);
    #pragma omp critical
    {
        i=my_data->head;
        local_array->empty = my_data->empty;
        local_array->full = my_data->full;
        local_array->head = my_data->head;
        local_array->tail = my_data->tail;
        local_array->final_break =my_data->final_break;
        local_array->job[i].cmd = my_data->job[i].cmd;
        local_array->job[i].key = my_data->job[i].key;
    }
    if(local_array->final_break!=1)
    {
        clock_t time_clock = clock();
        if(local_array->empty==1)
        {
            return;
        }
        ////////////////////////////////////////////////////////////
        // consumer consume data
        int encoder, decoder;
        char cur_cmd;
        if(local_array->empty!=1){
            // read the data from buffer

            cur_cmd = local_array->job[i].cmd;
            encoder = local_array->job[i].key;

        }
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
    
        local_array->head++;
        if(local_array->head == Q_SIZE)
	    {
		    local_array->head =0;
	    }
	    if(local_array->head == local_array->tail)
	    {
		    local_array->empty =1;
	    }
	    local_array->full =0;
	    // copy from local to global
	    #pragma omp critical
        {
            my_data->empty = local_array->empty;
            my_data->full = local_array->full;
            my_data->head = local_array->head;
            my_data->tail = local_array->tail;
            my_data->final_break =local_array->final_break;
        }
        time_clock = clock() - time_clock;
	    local_array->consumer_time = ((double)time_clock)/CLOCKS_PER_SEC;
	    #pragma omp critical
	    {
	        my_data->consumer_time += local_array->consumer_time;
	    }
	    producer(my_data);
    }
}

int main(int argc, char* argv[])
{
    int nthreads = atoi(argv[1]);
    if(nthreads<=0)
    {
        nthreads = 8;
        printf("Please input an avaialable number of threads.\n");
        printf("Now we run on 8 threads.\n");
    }
    int tid;
    // initialize queue
    work_array *ptr = malloc(sizeof(work_array));
    ptr->empty = 1;
    ptr->full = 0;
    ptr->head = 0;
    ptr->tail = 0;
    ptr->final_break =0;
    ptr->job = malloc(sizeof(work_entry)*Q_SIZE);
    printf("Initialize the queue successfully.\n");
    printf("Assigned number of threads = %d\n", nthreads);
    //printf("Main(): Start producer() and consumer() functions.\n");
    clock_t time_clock = clock();
    #pragma omp parallel num_threads(nthreads) private(nthreads, tid)
    {
        
        // Obtain thread number
        tid = omp_get_thread_num();
        //printf("Hello World from thread = %d\n", tid);
        int x = omp_get_num_threads();
        
        if(tid==0)
        {
            nthreads = omp_get_num_threads();
            printf("Actual number of threads created = %d\n", nthreads);
        }
        producer(ptr);
        consumer(ptr);
    }
    time_clock = clock() - time_clock;
    double total_time =((double)time_clock)/CLOCKS_PER_SEC;
    printf("The producer runtime = %f seconds\n", total_time - ptr->consumer_time/nthreads);
    //printf("The producer runtime = %f seconds\n", ptr->producer_time/nthreads);
    printf("The consumer runtime = %f seconds\n", ptr->consumer_time/nthreads);
    printf("The total user time = %f seconds\n", total_time);
    /*
    // serial version
    producer(ptr);
    consumer(ptr);
    */
    
    
    return 0;
}