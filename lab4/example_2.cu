/*

Author: Su, Ming Yi
Date: 11/16/2018
Goal: Learn to use cuda to sum up two vectors

How to compile it:
    nvcc -O -o example_2 example_2.cu
How to run it:
    ./example_2


*/

#include "stdio.h"
#define N 10

__global__ void add(int *a, int *b, int *c, char *d)
{
    int tID = blockIdx.x;
    if(tID < N)
    {
        c[tID] = a[tID] + b[tID];
        printf("c[%d] = %d\n", tID, c[tID]);
    }
}

int main()
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    char *dev_d;
    cudaMalloc((void **) &dev_a, N*sizeof(int));
    cudaMalloc((void **) &dev_b, N*sizeof(int));
    cudaMalloc((void **) &dev_c, N*sizeof(int));
    for(int i=0;i<N;i++)
    {
        a[i] = i;
        b[i] = i;
    }
    cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
    
    add<<<N,1>>>(dev_a, dev_b, dev_c, dev_d);
    
    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i=0;i<N;i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }
    return 0;
}