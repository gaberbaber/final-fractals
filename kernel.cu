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

//global cluster # counter
__device__ int d_stick_counter; //seed is 1, collective clustered particles are 2, 3, ...

//cluster check with border check
__device__ __forceinline__ bool near_cluster(const int* grid, int x, int y, int N) {
    //false if 4 neighboring pixels are not cluster
    //true if any 4 are cluster
    /*
    return ((grid[(y-1)*N + x] != 0)||
            (grid[(y+1)*N + x] != 0)||
            (grid[(y)*N + (x-1)] != 0)||
            (grid[(y)*N + (x+1)] != 0));
    */
    
    if (y > 0       && grid[(y-1) * N + x] != 0) {return true;}// up
    if (y + 1 < N   && grid[(y+1) * N + x] != 0) {return true;}// down
    if (x > 0       && grid[y * N + (x-1)] != 0) {return true;}// left
    if (x + 1 < N   && grid[y * N + (x+1)] != 0) {return true;}// right

    return false;
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

    //spawn on the border
    int x, y;
    int side = curand(&state) & 3;          //picks one of four edges
    if (side == 0) {        //top edge
        x = curand(&state) % N; y = 0; 
    }else if (side == 1) {  //right edge
        x = N - 1; y = curand(&state) % N; 
    }else if (side == 2) {  //bottom edge
        x = curand(&state) % N; y = N - 1; 
    }else{                  //left edge
        x = 0; y = curand(&state) % N;
    }

    //DEBUG: trying starting position right next to seed
    //x = N/2;
    //y = N/2 + 1;

    //random walk until sticks or hits max steps
    for (int steps = 0; steps < MAX_STEPS; steps++) {
        //attempt to stick
        if (near_cluster(grid, x, y, N)) {
            int idx = y * N + x;
            //atomic compare and swap so only one thread claims pixel
            if (atomicCAS(&grid[idx], 0, 1) == 0) {
                //order
                int val = atomicAdd(&d_stick_counter, 1) +1;
                grid[idx] = val;
                return;     //successful stick
            }
        }

        //take a step
        int d = curand(&state) & 3;
        x += (d == 2) - (d == 3);        // +/-1 on x
        y += (d == 0) - (d == 1);        // +/-1 on y

        //kill particle if it falls off edge
        if (x < 0 || x >= N || y < 0 || y >= N) {
            return; //particle escape
        }
    }
    
}

//host wrapper
void dla(int* d_grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long long seed) {
    int init = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_stick_counter, &init, sizeof(int)));
    
    int threads = 256;
    int blocks = (NUM_PARTICLES + threads - 1) / threads;
    dla_kernel<<<blocks, threads>>>(d_grid, N, NUM_PARTICLES, MAX_STEPS, seed);
    CUDA_CHECK(cudaGetLastError());       // catch invalid launch / invalid device function
    CUDA_CHECK(cudaDeviceSynchronize());  // catch runtime errors
}