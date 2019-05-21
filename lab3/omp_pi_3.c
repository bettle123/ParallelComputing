/*
Author: Su, Ming Yi
Date: 11/11/2018
Goal: 
    learn omp
    parallel the pi calculation
    example 1, to calculate the pi
    No false sharing version
How to compile it:
    icc -fopenmp -o omp_pi_3 omp_pi_3.c
How to run it:
    time ./omp_pi_3


*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
# define NUM_THREADS 8

int main()
{
    int num_steps = 10;
    int i, nthreads;
    double step = 1.0/(double) num_steps;
    omp_set_num_threads(NUM_THREADS);
    double x, pi, sum=0.0;
    #pragma omp parallel for private(x) reduction(+:sum)
    for (i=0;i<num_steps;i++)
    {
        x = (i+0.5)*step;
        sum +=4.0/(1.0+x*x);
    }
    pi+=sum*step;
    
    printf("pi = %f\n", pi);
    
    
    
    return 0;
}