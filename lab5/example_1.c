/*
@Author: Su Miing Yi
@Date: 12/02/2018
@Goal: send/receive a single value by MPI

How to compile it:
    mpicc example_1.c -o example_1
How to run it:
    mpirun -np 4 ./example_1

*/

#include <mpi.h>
#include <stdio.h>



int main(int argc, char *argv[])
{
    int rank, size, msg;
    double x = 0.0f;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // process 0 sending to process 1
    if (rank==0)
    {
        x = 2;
        printf("x= %f\n", x);
        printf("communicator size = %d \n", size);
        msg = 42;
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(&x, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank==1)
    {
        MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Received %d\n", msg);
        MPI_Recv(&x, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        x= x+3;
        printf("x= %f\n", x);
    }
    MPI_Finalize();
    
    
    return 0;
}

