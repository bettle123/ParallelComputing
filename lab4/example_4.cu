/*


Author: Su, Ming Yi
Date: 11/18/2018
Goal: use cuda to reverse matrix

How to compile it:
    module load cuda
    nvcc -o example_4 example_4.cu
How to run it:
    ./example_4


*/


#include "stdio.h"


// kernel-find linearized threadId, and set A[id] = tid

// use "__global__ void" type for base cuda kernel
__global__ void initArray(int *A)
{
    printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    A[tid] = tid;
}


int main()
{
    printf("Hello World.\n");
    
    int *h_a; // pointer for host memory
    int *d_a; // pointer for device memory
    
    // define thread hierarchy
    int num_blocks= 8;
    int num_th_per_blk = 8;
    
    // allocate host and device memory
    size_t memSize;
    
    // for this example, assume we will have one thread per data item
    memSize = num_blocks*num_th_per_blk * sizeof(int);
    h_a= (int*)malloc(memSize);
    cudaMalloc((void **) &d_a, memSize);
    
    // launch kernel
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_th_per_blk);
    initArray<<<dimGrid, dimBlock>>>(d_a);
    
    // retrieve results
    cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);
    
    
    for (int i=0;i<num_blocks;i++)
    {
        for(int j=0;j<num_th_per_blk;j++)
        {
            printf("h_a[%d][%d] = %d ", i, j, h_a[i+j*num_th_per_blk]);
        }
        printf("\n");
    }
    
    
    
    
    return 0;
}