

__global__ void dla_kernel(int*grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
}


void dla(int* d_grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long seed) {

}