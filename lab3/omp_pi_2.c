/*
Author: Su, Ming Yi
Date: 11/11/2018
Goal: 
    learn omp
    parallel the pi calculation
    example 1, to calculate the pi
    No false sharing version
How to compile it:
    icc -fopenmp -o omp_pi_2 omp_pi_2.c
How to run it:
    time ./omp_pi_2


*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
# define NUM_THREADS 2

int main()
{
    int num_steps = 10;
    int i, nthreads;
    double pi=0.0;
    double step = 1.0/(double) num_steps;
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel
    {
        int i, id, nthrds;
        double x, sum;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        printf("hello world from thread: %d\n", id);
        if(id==0){  nthreads = nthrds;}
        
        int spt = num_steps/nthrds;
        printf("each threads do %d jobs\n.", spt);
        sum= 0.0;
        for(i=id*spt;i<(id+1)*spt;i++)
        {
            x = (i+0.5)*step;
            sum += 4.0/(1.0+x*x);
        }
        printf("sum in thread %d = %f\n", id, sum);
        #pragma omp critical
        {
            pi += sum*step;
        }
        /*
        for (i=id;i<num_steps;i+=nthrds)
        {
            x = (i+0.5)*step;
            sum[id] += 4.0/(1.0+x*x);
        }
        */
    }
    /*
    printf("After omp.\n");
    for(int j=0;j<nthreads;j++)
    {
        pi+=sum[j]*step;
        //printf("sum[%d] = %f\n", j, sum[j]);
    }
    */
    printf("pi = %f\n", pi);
    
    
    
    return 0;
}