/*
@Author: Su Miing Yi
@Date: 12/02/2018
@Goal: send/receive an array by MPI

How to compile it:
    mpicc -fopenmp -o lab5_mpi SuMingYi_main.c transform.o
How to run it:
    time mpirun -np 5 ./lab5_mpi < PC_data_x1

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <time.h> 
#define Q_SIZE 5

double total_time;
double producer_time, consumer_time;
int complete_id = 0;
double trans_time;



uint16_t transformA(uint16_t input_val);
uint16_t transformB(uint16_t input_val);
uint16_t transformC(uint16_t input_val);
uint16_t transformD(uint16_t input_val);

void producer(char *cmd, int *key, int *check);
void consumer(char cmd, int key, int *alpha, int *result, double *one_p_time, double *one_c_time);

void producer(char *cmd, int *key, int *check)
{
    char input1[20],input2[20];
    scanf("%s\t%s", &input1, &input2);
    //printf("producer(): %s\t%s\n", input1, input2);
    int tmp_key = strtoumax(input2, NULL, 10);
    if(((input1[0]=='A')||(input1[0]=='B')||(input1[0]=='C')||(input1[0]=='D'))&&(tmp_key>=0 &&tmp_key<=1000))
    {
        *cmd = input1[0];
        *key = tmp_key;
        *check = 1;
    }
    else if (input1[0]=='X')
    {
        
        total_time = producer_time + consumer_time + trans_time;
        printf("The producer runtime = %f seconds\n", producer_time);
        printf("The consumer runtime = %f seconds\n", consumer_time);
        printf("The transmission time = %f seconds\n", trans_time);
        printf("The total runtime = %f seconds\n", total_time);
        exit(0);
        
        return;
    }
}

void consumer(char cmd, int key, int *alpha, int *result, double *one_p_time, double *one_c_time)
{
    clock_t time_clock = clock();
    clock_t time_p;
    clock_t time_c;
    int beta;
    if(cmd=='A')
    {   
        *alpha = transformA(key);
        time_p = clock() - time_clock;
	    *one_p_time = ((double)time_p)/CLOCKS_PER_SEC;
	    time_p = clock();
        beta = (*alpha+1)%1000;
        *result = transformA(beta);
        time_c = clock() -time_p;
        *one_c_time = ((double)time_c)/CLOCKS_PER_SEC;
    }
    if(cmd=='B')
    {   
        *alpha = transformB(key);
        time_p = clock() - time_clock;
	    *one_p_time = ((double)time_p)/CLOCKS_PER_SEC;
	    time_p = clock();
        beta = (*alpha+1)%1000;
        *result = transformB(beta);
        time_c = clock() -time_p;
        *one_c_time = ((double)time_c)/CLOCKS_PER_SEC;
    }
    if(cmd=='C')
    {   
        *alpha = transformC(key);
        time_p = clock() - time_clock;
	    *one_p_time = ((double)time_p)/CLOCKS_PER_SEC;
	    time_p = clock();
        beta = (*alpha+1)%1000;
        *result = transformC(beta);
        time_c = clock() -time_p;
        *one_c_time = ((double)time_c)/CLOCKS_PER_SEC;
    }
    if(cmd=='D')
    {   
        *alpha = transformD(key);
        time_p = clock() - time_clock;
	    *one_p_time = ((double)time_p)/CLOCKS_PER_SEC;
	    time_p = clock();
        beta = (*alpha+1)%1000;
        *result = transformD(beta);
        time_c = clock() -time_p;
        *one_c_time = ((double)time_c)/CLOCKS_PER_SEC;
    }
}


int main(int argc, char *argv[])
{
    // variable for MPI
    int rank, size;
    // one unit work
    char cmd;
    int key;
    int result;
    int alpha;
    
    double one_p_time;
    double one_c_time;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int final_break=0;
    while(final_break==0){
    if (rank==0)
    {
        //printf("In rank %d and it is my master MPI process.\n", rank);
        //printf("Our communicator size = %d \n", size);
        // variable for thread
        int nthreads = 4;
        int tid;
        #pragma omp parallel num_threads(nthreads) private(nthreads, tid)
        {
            // Obtain thread id
            tid = omp_get_thread_num();
            int check=0;
            if(tid==0)
            {
                producer(&cmd, &key, &check);
                if(check==1){
                    //printf("In thread %d, obtain the work unit: %c\t%d\n", tid, cmd, key);
                    //printf("In thread %d, send work unit to process 1\n", tid);
                    clock_t time_clock = clock();
                    
                    // Send
                    MPI_Send(&cmd, 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
                    // Receive
                    MPI_Recv(&cmd, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&key, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&alpha, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&result, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_p_time, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_c_time, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    time_clock = clock() - time_clock;
	                trans_time = trans_time + ((double)time_clock)/CLOCKS_PER_SEC;
                    trans_time = trans_time - one_p_time - one_c_time;
                    producer_time = producer_time + one_p_time;
                    consumer_time = consumer_time + one_c_time;
                    //printf("In thread %d, get the results from process 1\n", tid);
                    printf("Completed task: %d, cmd = %c, key = %d, alpha = %d, result = %d\n", complete_id, cmd, key, alpha, result);
                    //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
                    #pragma omp critical
                    {
	                    complete_id++;
                    }
                }
            }
            else if(tid==1)
            {
                producer(&cmd, &key, &check);
                if(check==1){
                    //printf("In thread %d, obtain the work unit: %c\t%d\n", tid, cmd, key);
                    //printf("In thread %d, send work unit to process 1\n", tid);
                    clock_t time_clock = clock();
                    
                    // Send
                    MPI_Send(&cmd, 1, MPI_CHAR, 2, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
                    // Receive
                    MPI_Recv(&cmd, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&key, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&alpha, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&result, 1, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_p_time, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_c_time, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    time_clock = clock() - time_clock;
	                trans_time = trans_time + ((double)time_clock)/CLOCKS_PER_SEC;
                    trans_time = trans_time - one_p_time - one_c_time;
                    producer_time = producer_time + one_p_time;
                    consumer_time = consumer_time + one_c_time;
                    //printf("In thread %d, get the results from process 1\n", tid);
                    printf("Completed task: %d, cmd = %c, key = %d, alpha = %d, result = %d\n", complete_id, cmd, key, alpha, result);
                    //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
                    #pragma omp critical
                    {
	                    complete_id++;
                    }
                }
            }
            else if(tid==2)
            {
                producer(&cmd, &key, &check);
                if(check==1){
                    //printf("In thread %d, obtain the work unit: %c\t%d\n", tid, cmd, key);
                    //printf("In thread %d, send work unit to process 1\n", tid);
                    clock_t time_clock = clock();
                    
                    // Send
                    MPI_Send(&cmd, 1, MPI_CHAR, 3, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);
                    // Receive
                    MPI_Recv(&cmd, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&key, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&alpha, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&result, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_p_time, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_c_time, 1, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    time_clock = clock() - time_clock;
	                trans_time = trans_time + ((double)time_clock)/CLOCKS_PER_SEC;
                    trans_time = trans_time - one_p_time - one_c_time;
                    producer_time = producer_time + one_p_time;
                    consumer_time = consumer_time + one_c_time;
                    //printf("In thread %d, get the results from process 1\n", tid);
                    printf("Completed task: %d, cmd = %c, key = %d, alpha = %d, result = %d\n", complete_id, cmd, key, alpha, result);
                    //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
                    #pragma omp critical
                    {
	                    complete_id++;
                    }
                }
            }
            else if(tid==3)
            {
                producer(&cmd, &key, &check);
                if(check==1){
                    //printf("In thread %d, obtain the work unit: %c\t%d\n", tid, cmd, key);
                    //printf("In thread %d, send work unit to process 1\n", tid);
                    clock_t time_clock = clock();
                    
                    // Send
                    MPI_Send(&cmd, 1, MPI_CHAR, 4, 0, MPI_COMM_WORLD);
                    MPI_Send(&key, 1, MPI_INT, 4, 0, MPI_COMM_WORLD);
                    // Receive
                    MPI_Recv(&cmd, 1, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&key, 1, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&alpha, 1, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&result, 1, MPI_INT, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_p_time, 1, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&one_c_time, 1, MPI_DOUBLE, 4, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    time_clock = clock() - time_clock;
	                trans_time = trans_time + ((double)time_clock)/CLOCKS_PER_SEC;
                    trans_time = trans_time - one_p_time - one_c_time;
                    producer_time = producer_time + one_p_time;
                    consumer_time = consumer_time + one_c_time;
                    //printf("In thread %d, get the results from process 1\n", tid);
                    printf("Completed task: %d, cmd = %c, key = %d, alpha = %d, result = %d\n", complete_id, cmd, key, alpha, result);
                    //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
                    #pragma omp critical
                    {
	                    complete_id++;
                    }
                }
            }
        }
        
    }
    else if (rank==1)
    {
        //printf("In rank %d\n", rank);
        // Receive
        MPI_Recv(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        //printf("In process 1, received the work unit: %c\t%d\n", cmd, key);
        consumer(cmd, key, &alpha, &result, &one_p_time, &one_c_time);
        //printf("In process 1, alpha = %d, result = %d\n", alpha, result);
        // Send
        //printf("In process 1, send the results back to thread 0\n");
        MPI_Send(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&alpha, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
        MPI_Send(&one_p_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&one_c_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        
    }
    else if (rank==2)
    {
        //printf("In rank %d\n", rank);
        // Receive
        MPI_Recv(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        //printf("In process 1, received the work unit: %c\t%d\n", cmd, key);
        consumer(cmd, key, &alpha, &result, &one_p_time, &one_c_time);
        //printf("In process 1, alpha = %d, result = %d\n", alpha, result);
        // Send
        //printf("In process 2, send the results back to thread 1\n");
        MPI_Send(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&alpha, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
        MPI_Send(&one_p_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&one_c_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank==3)
    {
        //printf("In rank %d\n", rank);
        // Receive
        MPI_Recv(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        //printf("In process 1, received the work unit: %c\t%d\n", cmd, key);
        consumer(cmd, key, &alpha, &result, &one_p_time, &one_c_time);
        //printf("In process 1, alpha = %d, result = %d\n", alpha, result);
        // Send
        //printf("In process 3, send the results back to thread 2\n");
        MPI_Send(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&alpha, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
        MPI_Send(&one_p_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&one_c_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank==4)
    {
        //printf("In rank %d\n", rank);
        // Receive
        MPI_Recv(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        //printf("In process 1, received the work unit: %c\t%d\n", cmd, key);
        consumer(cmd, key, &alpha, &result, &one_p_time, &one_c_time);
        //printf("In process 1, alpha = %d, result = %d\n", alpha, result);
        // Send
        //printf("In process 4, send the results back to thread 3\n");
        MPI_Send(&cmd, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&key, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&alpha, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        //printf("One producer time = %f, One consumer time = %f.\n", one_p_time, one_c_time);
        MPI_Send(&one_p_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&one_c_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    
    }
    MPI_Finalize();
    return 0;
}

