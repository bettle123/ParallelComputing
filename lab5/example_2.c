/*
@Author: Su Miing Yi
@Date: 12/02/2018
@Goal: send/receive an array by MPI

How to compile it:
    mpicc example_2.c -o example_2
How to run it:
    time mpirun -np 2 ./example_2

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size, msg;
    int buffer_size=5;
    float *buffer = (float *)malloc(sizeof(float)*buffer_size);
    
    
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // process 0 sending to process 1
    if (rank==0)
    {
        printf("In rank %d\n", rank);
        printf("communicator size = %d \n", size);
        // set buffer values
        for (int i=0;i<5;i++)
        {
            buffer[i] = i;
        }
        MPI_Send(buffer, buffer_size, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank==1)
    {
        printf("In rank %d\n", rank);
        MPI_Recv(buffer, buffer_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Received buffer\n");
        for (int i=0;i<5;i++)
        {
            printf("buffer[%d] = %0.1f\n", i, buffer[i]);
        }
    }
    MPI_Finalize();
    
    
    return 0;
}

