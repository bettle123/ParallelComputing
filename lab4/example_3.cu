/*
Author: Su Ming Yi
Date: 11/16/2018
Goal:
    Add 2D array by cuda
How to compile it:
    nvcc -O -o example_3 example_3.cu
How to run it:
    ./example_3


*/
#include "stdio.h"
#define COLUMNS 3
#define ROWS 2

__global__ void add(int *a, int *b, int *c)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int i = (COLUMNS*y) + x;
    c[i] = a[i] + b[i];
}

int main()
{
    int a[ROWS][COLUMNS], b[ROWS][COLUMNS], c[ROWS][COLUMNS];
    int *dev_a, *dev_b, *dev_c;
    
    
    // allocate Memory size in gpu
    cudaMalloc((void **) &dev_a, ROWS*COLUMNS*sizeof(int));
    cudaMalloc((void **) &dev_b, ROWS*COLUMNS*sizeof(int));
    cudaMalloc((void **) &dev_c, ROWS*COLUMNS*sizeof(int));
    
    // assign the value in 2D array
    for (int y=0; y<ROWS; y++)
    {
        for (int x=0; x<COLUMNS;x++)
        {
            a[y][x] = x;
            b[y][x] = y;
        }
    }
    
    // copy the memory from Host to Device
    cudaMemcpy(dev_a, a, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, ROWS*COLUMNS*sizeof(int), cudaMemcpyHostToDevice);
    
    // Do not understand
    dim3 grid(COLUMNS, ROWS);
    add<<<grid,1>>>(dev_a, dev_b, dev_c);
    
    // copy the memory from Device to Host
    cudaMemcpy(c, dev_c, ROWS*COLUMNS*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Output the result
    for (int y=0; y<ROWS;y++)
    {
        for( int x =0; x<COLUMNS;x++)
        {
            printf("[%d][%d] = %d", y, x, c[y][x]); 
        }
        printf("\n");
    }
    
    
    return 0;
}
