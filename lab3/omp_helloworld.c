/*
Author: Su Ming Yi
Date: 11/11/2018
Goal: OpenMP hello world example

How to compile it:
    icc -fopenmp -o omp_hello omp_helloworld.c
How to run it:
    ./omp_hello
/////////////////////////////////////////////////////////////////
omp_get_thread_num(): get the thread number from omp
omp_set_num_threads(4): set the number of threads for omp

#pragma omp parallel private(nthreads, tid)
{
    ## Things that we want to do parallel
}


*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int nthreads, tid;
    // how to set threads number in omp
    omp_set_num_threads(4);
    // Fork a team of threads giving them their own copies of variables
    #pragma omp parallel private(nthreads, tid)
    {
        // Obtain thread number
        tid = omp_get_thread_num();
        
        if(tid==0)
        { 
            int x = omp_get_num_threads();
            printf("Number of threads = %d\n", x);
        }
        printf("Hello World from thread = %d\n", tid);
        printf("omp_get_num_threads() = %d\n", x);
    }
    // All threads join master thread and disband
    
    return 0;
}