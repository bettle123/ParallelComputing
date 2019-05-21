/*
Author: Su Ming Yi
Date: 12/02/2018
Goal: Use openMP for a while loop

How to compile it:
    icc -fopenmp -o example_while example_while.c 
How to run it:
    ./example_while
*/


#include <stdio.h>
#include <omp.h>

void main (void)
{
    int i = 0;
    #pragma omp parallel
    while (i < 10)
    {
        printf ("Hello. I am thread %d and i is %d\n", omp_get_thread_num(), i);
        #pragma omp atomic
        i++;
    }
}