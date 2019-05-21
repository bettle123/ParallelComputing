/*
Author: Su Ming Yi
Date: 11/16/2018
Goal: use CUDA to sum up two numbers
Because we do not have the gpu resource of OSC now,
we cannot get the correct portion from gpu and host.


How to compile it:
    qsub -I -l walltime=00:59:00 -l nodes=1:gpus=1,mem=4gb -A PAS0027
    module load cuda
    nvcc -o example_1 example_1.cu
How to run it:
    ./example_1


*/

#include "stdio.h"

__global__ void add(int a, int b, int *c)
{
    *c = a+b;
}

int main()
{
    int a, b, c;
    int *dev_c;
    a = 3; 
    b = 4;
    cudaMalloc((void**)&dev_c, sizeof(int));
    add<<<1,1>>>(a,b,dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d + %d is %d \n", a, b, c);
    cudaFree(dev_c);
    return 0;
}
