/*
Author: Su, Ming Yi
Date: 11/11/2018
Goal: 
    learn omp
    example 1, to calculate the pi
How to compile it:
    icc -o test serial_pi.c
How to run it:
    time ./test


*/

#include <stdio.h>
#include <stdlib.h>

int main()
{
    int num_steps = 100000;
    double step;
    
    int i;
    double x, pi, sum =0.0;
    step = 1.0/(double) num_steps;
    
    for(i=0;i<num_steps;i++)
    {
        x = (i + 0.5)*step;
        sum = sum + 4.0/(1.0 + x*x);
    }
    pi = step*sum;
    
    printf("pi = %f\n", pi);
    
    
    
    return 0;
}