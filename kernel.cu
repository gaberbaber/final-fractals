#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>

//error check
#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while (0)

//cluster check
__device__ __forceinline__ bool near_cluster(const int* grid, int x, int y, int N) {
    //false if 4 neighboring pixels are not cluster
    //true if any 4 are cluster
    return ((grid[(y-1)*N + x] != 0)||
            (grid[(y+1)*N + x] != 0)||
            (grid[(y)*N + (x-1)] != 0)||
            (grid[(y)*N + (x-1)] != 0));
}

//device kernel: each thread simulates one particle
__global__ void dla_kernel(int*grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_PARTICLES) {
        return;
    }

    //curand setup - cuda's on-device RNG
    curandState state;
    curand_init(seed, tid, 0, &state);

    //start at 1 pixel inside random border spot
    int x, y;
    int side = curand(&state) & 3;          //picks one of four edges
    if (side == 0) { x = 1 + (curand(&state) % (N - 2)); y = 1; }
    if (side == 1) { x = N - 2; y = 1 + (curand(&state) % (N - 2)); }
    if (side == 2) { x = 1 + (curand(&state) % (N - 2)); y = N - 2; }
    if (side == 3) { x = 1; y = 1 + (curand(&state) % (N - 2)); }

    //DEBUG: trying starting position right next to seed
    //x = N/2;
    //y = N/2 + 1;

    //random walk until sticks or hits max steps
    for (int steps = 0; steps < MAX_STEPS; steps++) {
        //attempt to stick
        if (near_cluster(grid, x, y, N)) {
            //atomic compare and swap so only one thread claims pixel
            if (atomicCAS(&grid[y * N + x], 0, tid + 1) == 0) {
                return;     //successful stick
            }
        }

        //take a step
        int d = curand(&state) & 3;
        x += (d == 2) - (d == 3);        // +/-1 on x
        y += (d == 0) - (d == 1);        // +/-1 on y

        // keep inside edges
        if (x < 1) x = 1; else if (x > N - 2) x = N - 2;
        if (y < 1) y = 1; else if (y > N - 2) y = N - 2;
    }
    
}

//host wrapper
void dla(int* d_grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long long seed) {
    int threads = 256;
    int blocks = (NUM_PARTICLES + threads - 1) / threads;
    dla_kernel<<<blocks, threads>>>(d_grid, N, NUM_PARTICLES, MAX_STEPS, seed);
    CUDA_CHECK(cudaGetLastError());       // catch invalid launch / invalid device function
    CUDA_CHECK(cudaDeviceSynchronize());  // catch runtime errors
}