/*
Author: Su Ming Yi
Date: 11/16/2018
Goal: Learn CUDA hello world

How to compile it:
    qsub -I -l walltime=00:01:00 -l nodes=1:gpus=1,mem=4gb -A PAS0027
    module load cuda
    nvcc -o hello hello_world.cu
How to run it:
    ./hello


*/
#include "stdio.h"

int main()
{
    printf("Hello world.\n");
    return 0;
}
