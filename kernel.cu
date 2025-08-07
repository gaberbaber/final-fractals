#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>

//device kernel: each thread simulates one particle
__global__ void dla_kernel(int*grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_PARTICLES) {
        return;
    }

    //curand setup - cuda's on-device RNG
    curandState state;
    curand_init(seed, tid, 0, &state);

    //start at random border
    int x, y;
    int edge = curand(&state) % 4;          //picks one of four edges
    switch (edge) {
        case 0: x = 0; y = curand(&state) % N; break;
        case 1: x = N-1; y = curand(&state) % N; break;
        case 2: x = curand(&state) % N; y = 0; break;
        case 3: x = curand(&state) % N; y = N-1; break;
    }

    //DEBUG: trying starting position right next to seed
    x = N/2;
    y = N/2 + 1;

    //random walk until it sticks or escapes
    int steps = 0;
    bool stuck = false;
    while (!stuck && steps < MAX_STEPS) {
        //random direction
        int d = curand(&state) % 4;
        x += (d == 2) - (d == 3);
        y += (d == 0) - (d == 1);

        //kill particle if OOB
        if (x < 0 || x >= N || y < 0 || y >= N) {
            break;
        }

        //adjacency check
        bool adjacent = false;
        int offsets[4][2] = {{0,1}, {0,-1}, {1,0}, {-1,0}};
        for (int i = 0; i < 4; i++) {
            int nx = x + offsets[i][0];
            int ny = y + offsets[i][1];
            if (nx >= 0 && nx < N && ny >= 0 && ny < N) {
                if (grid[ny*N + nx] != 0) {
                    adjacent = true;
                    break;
                }
            }
        }

        //stick if it's adjacent
        if (adjacent) {
            //atomic compare and swap so that only one thread can stick to that pixel
            if (atomicCAS(&grid[y*N + x], 0, tid+1) == 0) {
                stuck = true;
            }
        }
        ++steps;
    }
}

//host wrapper
void dla(int* d_grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long seed) {
    int threads = 256;
    int blocks = (NUM_PARTICLES + threads - 1) / threads;
    dla_kernel<<<blocks, threads>>>(d_grid, N, NUM_PARTICLES, MAX_STEPS, seed);
    cudaDeviceSynchronize();
}