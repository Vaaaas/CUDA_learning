#include <stdio.h>

__global__ void hello_from_gpu(void){
    const int grid_size_x = gridDim.x;
    const int grid_size_y = gridDim.y;
    const int grid_size_z = gridDim.z;
    const int block_size_x = blockDim.x;
    const int block_size_y = blockDim.y;
    const int block_size_z = blockDim.z;

    const int b_x = blockIdx.x + 1;
    const int b_y = blockIdx.y + 1;
    const int b_z = blockIdx.z + 1;
    const int t_x = threadIdx.x + 1;
    const int t_y = threadIdx.y + 1;
    const int t_z = threadIdx.z + 1;

    printf("Hello World from block (%d/%d, %d/%d, %d/%d) and thread (%d/%d, %d/%d, %d/%d)\n",
            b_z, grid_size_z, b_y, grid_size_y, b_x, grid_size_x,
            t_z, block_size_z, t_y, block_size_y, t_x, block_size_x);
}

int main(void){
    const dim3 grid_size(2, 4, 5);
    const dim3 block_size(3, 5, 4);
    hello_from_gpu<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();
    return 0;
}