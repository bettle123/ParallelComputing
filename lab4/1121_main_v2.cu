/*
Author: Su Ming Yi
Date: 11/12/2018
goal: use cuda to solve producer consumer problem.
How to compile it:
    module load cuda/9.2.88
    nvcc -dc 1121_main_v2.cu
    nvcc -O -o lab4_arbitrary_2 1121_main_v2.o transform_cuda.o
    
test myself:    
    nvcc -O -o lab4_arbitrary_2 1121_main_v2.cu
    
    
How to run it:
    ./lab4_arbitrary_2 < PC_data_x1
    
    
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

void producer(void *ptr1, void *ptr2, void *ptr3, void *ptr4);
void consumer(void *ptr1, void *ptr2, void *ptr3, void *ptr4);



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

__global__ void g_transformA(int N, int *before_key, int *after_key)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < N)
    {
        after_key[tID] = transformA(before_key[tID]); 
    }
}

__global__ void g_transformB(int N, int *before_key, int *after_key)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < N)
    {
        after_key[tID] = transformB(before_key[tID]); 
    }
}

__global__ void g_transformC(int N, int *before_key, int *after_key)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < N)
    {
        after_key[tID] = transformC(before_key[tID]); 
    }
}

__global__ void g_transformD(int N, int *before_key, int *after_key)
{
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < N)
    {
        after_key[tID] = transformD(before_key[tID]); 
    }
}



/*
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
*/
/////////////////////////////////////////////////
void cuda_transformA(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformA<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        my_data->job[i].key = after_key[i];
    }
}

void cuda_transformB(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformB<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        my_data->job[i].key = after_key[i];
    }
}

void cuda_transformC(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformC<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        my_data->job[i].key = after_key[i];
    }
}

void cuda_transformD(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformD<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        my_data->job[i].key = after_key[i];
    }
}

/////////////////////////////////////////////////////////////////////////////////////
void C_cuda_transformA(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    /////////////////////////////////////////////////////////////
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformA<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        int Q = i%T_Q_SIZE;
        printf("Q:%d\t%c\t%d\t%d\n", Q, my_data->job[i].cmd, before_key[i], after_key[i]);
    }
}

void C_cuda_transformB(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    /////////////////////////////////////////////////////////////
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformB<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        int Q = i%T_Q_SIZE;
        printf("Q:%d\t%c\t%d\t%d\n", Q, my_data->job[i].cmd, before_key[i], after_key[i]);
    }
}

void C_cuda_transformC(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    /////////////////////////////////////////////////////////////
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformC<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        int Q = i%T_Q_SIZE;
        printf("Q:%d\t%c\t%d\t%d\n", Q, my_data->job[i].cmd, before_key[i], after_key[i]);
    }
}

void C_cuda_transformD(void *ptr)
{
    work_array *my_data;
    my_data = (work_array *) ptr;
    /////////////////////////////////////////////////////////////
    // cuda transfer producer input
    int N = (my_data->tail);
    // define thread hierarchy
    int num_blocks= 20;
    int num_th_per_blk = 1000;
    int before_key[N], after_key[N];
    int *dev_before_key, *dev_after_key;
    cudaMalloc((void **) &dev_before_key, N*sizeof(int));
    cudaMalloc((void **) &dev_after_key, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        before_key[i] = my_data->job[i].key;
    }
    cudaMemcpy(dev_before_key, before_key, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_after_key, after_key, N*sizeof(int), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    g_transformD<<<dimGrid, dimBlock>>>(N, dev_before_key, dev_after_key);
    
    cudaMemcpy(after_key, dev_after_key, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i=0;i<N;i++)
    {
        int Q = i%T_Q_SIZE;
        printf("Q:%d\t%c\t%d\t%d\n", Q, my_data->job[i].cmd, before_key[i], after_key[i]);
    }
}


//////////////////////////////////////////////////////////////////////////////////////

void producer(void *ptr1, void *ptr2, void *ptr3, void *ptr4)
{
    clock_t time_clock = clock();
    work_array *my_data1;
    my_data1 = (work_array *) ptr1;
    work_array *my_data2;
    my_data2 = (work_array *) ptr2;
    work_array *my_data3;
    my_data3 = (work_array *) ptr3;
    work_array *my_data4;
    my_data4 = (work_array *) ptr4;
    while(check==0)
    {
        /////////////////////////////////////////////////////////////////////////////////////////
        // producer produce data
        // declare local variables
        char input1[20],input2[20];
        char tmp_cmd;
        scanf("%s\t%s", &input1, &input2);
        //printf("producer(): %s\t%s\n", input1, input2);
        if((input1[0]=='X'))
        {
            check=1;
        }
        int tmp_key = strtoumax(input2, NULL, 10);
        tmp_cmd = input1[0];
        ///////////////////////////////////////////////////////////////
        if((input1[0]=='A')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            int i = my_data1->tail;
            my_data1->job[i].cmd = tmp_cmd;
            my_data1->job[i].key = tmp_key;
            my_data1->tail++;
        }
        if((input1[0]=='B')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            int i = my_data2->tail;
            my_data2->job[i].cmd = tmp_cmd;
            my_data2->job[i].key = tmp_key;
            my_data2->tail++;
        }
        if((input1[0]=='C')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            int i = my_data3->tail;
            my_data3->job[i].cmd = tmp_cmd;
            my_data3->job[i].key = tmp_key;
            my_data3->tail++;
        }
        if((input1[0]=='D')&&(tmp_key>=0 &&tmp_key<=1000))
        {
            tmp_cmd = input1[0];
            int i = my_data4->tail;
            my_data4->job[i].cmd = tmp_cmd;
            my_data4->job[i].key = tmp_key;
            my_data4->tail++;
        }
    }
    /*
    printf("cmd A.\n");
    for (int i=0;i<my_data1->tail;i++)
    {
        printf("Q: %c\t%d\n", my_data1->job[i].cmd, my_data1->job[i].key);
    }
    printf("cmd B.\n");
    for (int i=0;i<my_data2->tail;i++)
    {
        printf("Q: %c\t%d\n", my_data2->job[i].cmd, my_data2->job[i].key);
    }
    printf("cmd C.\n");
    for (int i=0;i<my_data3->tail;i++)
    {
        printf("Q: %c\t%d\n", my_data3->job[i].cmd, my_data3->job[i].key);
    }
    printf("cmd D.\n");
    for (int i=0;i<my_data4->tail;i++)
    {
        printf("Q: %c\t%d\n", my_data4->job[i].cmd, my_data4->job[i].key);
    }
    */
    //////////////////////////////////////////////////////////////////////
    cuda_transformA(my_data1);
    cuda_transformB(my_data2);
    cuda_transformC(my_data3);
    cuda_transformD(my_data4);
    //////////////////////////////////////////////////////////////////////
    time_clock = clock() - time_clock;
	double cur_time = ((double)time_clock)/CLOCKS_PER_SEC;
	my_data1->producer_time += cur_time;
}

void consumer(void *ptr1, void *ptr2, void *ptr3, void *ptr4)
{
    clock_t time_clock = clock();
    work_array *my_data1;
    my_data1 = (work_array *) ptr1;
    work_array *my_data2;
    my_data2 = (work_array *) ptr2;    
    work_array *my_data3;
    my_data3 = (work_array *) ptr3;    
    work_array *my_data4;
    my_data4 = (work_array *) ptr4;    
    
    //////////////////////////////////////////////////////////////////////
    C_cuda_transformA(my_data1);
    C_cuda_transformB(my_data2);
    C_cuda_transformC(my_data3);
    C_cuda_transformD(my_data4);
    //////////////////////////////////////////////////////////////////////
    
    time_clock = clock() - time_clock;
    double cur_time = ((double)time_clock)/CLOCKS_PER_SEC;;
	my_data1->consumer_time += cur_time;
}

int main(int argc, char* argv[])
{
    // initialize queue
    work_array *ptr1 = (work_array*) malloc(sizeof(work_array));
    ptr1->empty = 1;
    ptr1->full = 0;
    ptr1->head = 0;
    ptr1->tail = 0;
    ptr1->final_break =0;
    ptr1->job = (work_entry*) malloc(sizeof(work_entry)*Q_SIZE);
    
    
    
    work_array *ptr2 = (work_array*) malloc(sizeof(work_array));
    ptr2->job = (work_entry*) malloc(sizeof(work_entry)*Q_SIZE);
    ptr2->tail = 0;
    
    work_array *ptr3 = (work_array*) malloc(sizeof(work_array));
    ptr3->job = (work_entry*) malloc(sizeof(work_entry)*Q_SIZE);
    ptr3->tail = 0;
    
    work_array *ptr4 = (work_array*) malloc(sizeof(work_array));
    ptr4->job = (work_entry*) malloc(sizeof(work_entry)*Q_SIZE);
    ptr4->tail = 0;
    
    printf("Initialize the queue successfully.\n");
    printf("Main(): Start producer() and consumer() functions.\n");
    clock_t time_clock = clock();
    
    
    producer(ptr1, ptr2, ptr3, ptr4);
    consumer(ptr1, ptr2, ptr3, ptr4);
    
    time_clock = clock() - time_clock;
    double total_time =((double)time_clock)/CLOCKS_PER_SEC;
    
    printf("The producer runtime = %f seconds\n", ptr1->producer_time);
    printf("The consumer runtime = %f seconds\n", ptr1->consumer_time);
    
    printf("The total user time = %f seconds\n", total_time);
    
    return 0;
}


