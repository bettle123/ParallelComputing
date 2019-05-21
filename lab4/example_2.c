/*
Author: Su Ming Yi
Date: 11/16/2018
Goal: Two vectors sum up standard C example 
How to compile it:
    icc -o example_2 example_2.c
How to run it:
    ./example_2
*/

#include "stdio.h"
#define N 10

void add(int *a, int *b, int *c)
{
    int tID = 0;
    while(tID<N)
    {
        c[tID] = a[tID] + b[tID];
        tID++;
    }
}

int main()
{
    int a[N], b[N], c[N];
    
    // Fill Arrays
    for (int i=0;i<N;i++)
    {
        a[i] = i;
        b[i] = N-i;
    }
    add(a, b, c);
    for(int i=0;i<N;i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    return 0;
    
}