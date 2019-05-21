/*
Author: Su Ming Yi
Date: 11/12/2018
goal: use cuda to solve producer consumer problem.
How to compile it:
    module load cuda/9.2.88
    nvcc -dc SuMingYi_lab4_v1.cu
    nvcc -O -o lab4_arbitrary SuMingYi_lab4_v1.o transform_cuda.o
    
How to run it:
    ./lab4_arbitrary < PC_data_x1
    
    
*/
#include "stdio.h"
#include "string.h"
#include "inttypes.h"
#include "math.h"
#include "stdlib.h"
#include "time.h" 
#define Q_SIZE 10000
#define T_Q_SIZE 5

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





////////////////////////////////////////////////////////////////////////////////////
// declare cuda function
// teacher version:

__device__ uint16_t transformA(uint16_t input_val);
__device__ uint16_t transformB(uint16_t input_val);
__device__ uint16_t transformC(uint16_t input_val);
__device__ uint16_t transformD(uint16_t input_val);


/*
// my version: to test my logic
__device__ int transformA(int input_val){ return input_val;}
__device__ int transformB(int input_val){ return input_val;}
__device__ int transformC(int input_val){ return input_val;}
__device__ int transformD(int input_val){ return input_val;}
*/
__global__ void transform(int N, char *cmd, int *before_key, int *after_key)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < N)
    {
        if(cmd[tID]=='A')
        {
            after_key[tID] = transformA(before_key[tID]); 
        }
        if(cmd[tID]=='B')
        {
            after_key[tID] = transformB(before_key[tID]); 
        }
        if(cmd[tID]=='C')
        {
            after_key[tID] = transformC(before_key[tID]); 
        }
        if(cmd[tID]=='D')
        {
            after_key[tID] = transformD(before_key[tID]); 
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////

void producer(void *ptr)
{
    clock_t time_clock = clock();
    work_array *my_data;
    my_data = (work_array *) ptr;

    while(check==0)
    {
        if(my_data->full==1)
        {
            return;
        }
        /////////////////////////////////////////////////////////////////////////////////////////
        // producer produce data
        // declare local variables
        int i = my_data->tail;
        char input1[20],input2[20];
        char tmp_cmd;
        scanf("%s\t%s", &input1, &input2);
        //printf("producer(): %s\t%s\n", input1, input2);
        if((input1[0]=='X'))
        {
            //printf("The end of the program.\n");
            my_data->final_break =1;
            check=1;
        }
        int tmp_key = strtoumax(input2, NULL, 10);
        tmp_cmd = input1[0];
        int require = 0;
        ///////////////////////////////////////////////////////////////
        if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            require = 1;
            my_data->tail++;
            my_data->empty =0;
        }
        if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            require = 1;
            my_data->tail++;
            my_data->empty =0;
            
        }
        if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            require = 1;
            my_data->tail++;
            my_data->empty =0;
            
        }
        if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            require = 1;
            my_data->tail++;
            my_data->empty =0;
        }
        if(require==1){
            my_data->job[i].cmd = tmp_cmd;
            my_data->job[i].key = tmp_key;
        }
        if(my_data->tail == Q_SIZE)
        {
            my_data->tail =0;
            if(my_data->tail == my_data->head)
            {
                my_data->full = 1;
            }
        }

    }
    /////////////////////////////////////////////////////////////
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    
    char cmd[N];
    int before_key[N], after_key[N];
    char *dev_cmd;
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    cudaMalloc((void **) &dev_cmd, N*sizeof(char));
    for(int i=0;i<N;i++)
    {
        cmd[i] = my_data->job[i].cmd;
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cmd, cmd, N*sizeof(char), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    transform<<<dimGrid, dimBlock>>>(N, dev_cmd, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        my_data->job[i].key = after_key[i];
    }
    time_clock = clock() - time_clock;
	double cur_time = ((double)time_clock)/CLOCKS_PER_SEC;
	my_data->producer_time += cur_time;
}

void consumer(void *ptr)
{
    clock_t time_clock = clock();
    work_array *my_data;
    my_data = (work_array *) ptr;
    /////////////////////////////////////////////////////////////
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    //printf("N = %d.\n", N);
    char cmd[N];
    int before_key[N], after_key[N];
    char *dev_cmd;
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    cudaMalloc((void **) &dev_cmd, N*sizeof(char));
    for(int i=0;i<N;i++)
    {
        cmd[i] = my_data->job[i].cmd;
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cmd, cmd, N*sizeof(char), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    transform<<<dimGrid, dimBlock>>>(N, dev_cmd, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        int Q = i%T_Q_SIZE;
        printf("Q:%d\t%c\t%d\t%d\n", Q, cmd[i], before_key[i], after_key[i]);
    }
    
    
    
    time_clock = clock() - time_clock;
    double cur_time = ((double)time_clock)/CLOCKS_PER_SEC;;
	my_data->consumer_time += cur_time;
}

int main(int argc, char* argv[])
{
    // initialize queue
    work_array *ptr = (work_array*) malloc(sizeof(work_array));
    ptr->empty = 1;
    ptr->full = 0;
    ptr->head = 0;
    ptr->tail = 0;
    ptr->final_break =0;
    ptr->job = (work_entry*) malloc(sizeof(work_entry)*Q_SIZE);
    printf("Initialize the queue successfully.\n");
    printf("Main(): Start producer() and consumer() functions.\n");
    clock_t time_clock = clock();
    
    
    producer(ptr);
    consumer(ptr);
    
    time_clock = clock() - time_clock;
    double total_time =((double)time_clock)/CLOCKS_PER_SEC;
    
    printf("The producer runtime = %f seconds\n", ptr->producer_time);
    printf("The consumer runtime = %f seconds\n", ptr->consumer_time);
    printf("The total user time = %f seconds\n", total_time);
    
    return 0;
}


