/*
@Author: Su, Ming Yi
@Date: 12/01/2018
@Goal: hello MPI

How to compile it:
    mpicc hello_world.c -o hello_world
How to run it:
    mpirun -np 4 ./hello_world

the number of processes in the specified communicator is 4.

*/

#include <mpi.h>
#include <stdio.h>

int main (int argc, char *argv[])
{
    //printf("argc = %d, argv = %s \n", argc, argv[0]);
    int rank, size;
    MPI_Init(&argc, &argv);
    // int MPI_Init( int *argc, char ***argv )
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Comm_rank(COMMUNICATOR, int *rank);
    // MPI_Comm_size(COMMUNICATOR, int *size);
    printf("I am rank %d, of communicator size %d \n", rank, size);
    MPI_Finalize();
    
    return 0;
}