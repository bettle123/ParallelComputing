/*
Author: Su, Ming Yi
Date: 11/11/2018
Goal: 
    learn omp
    parallel the pi calculation
    example 1, to calculate the pi
How to compile it:
    icc -fopenmp -o omp_pi omp_pi.c
How to run it:
    time ./omp_pi


*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
# define NUM_THREADS 10

int main()
{
    int num_steps = 100;
    int i, nthreads;
    double pi=0.0;
    double sum[NUM_THREADS];
    double step = 1.0/(double) num_steps;
    
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        printf("hello world from thread: %d\n", id);
        
        if(id==0)
        {
            nthreads = nthrds;
        }
        sum[id] = 0.0;
        for (i=id;i<num_steps;i+=nthrds)
        {
            x = (i+0.5)*step;
            sum[id] += 4.0/(1.0+x*x);
        }
    }
    printf("After omp.\n");
    for(int j=0;j<nthreads;j++)
    {
        pi+=sum[j]*step;
        printf("sum[%d] = %f\n", j, sum[j]);
    }
    
    printf("pi = %f\n", pi);
    
    
    
    return 0;
}