//#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

//error check
#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while (0)

//new rng instead of curand (heavy init costs)
// mutates the seedmix state within register
__device__ __forceinline__ uint32_t xorshift32(uint32_t &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

//gives each thread a different, nonzero starting state
__device__ __forceinline__ uint32_t seed_mix(unsigned long long seed, int tid) {
    uint32_t x = (uint32_t)(seed) ^ (0x9E3779B9u * (uint32_t)(tid + 1));
    //hash scramble
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    if (x == 0) x = 1u;
    return x;
}

__device__ __forceinline__ int rand4(uint32_t &s) {
    return (int)((xorshift32(s) >> 24) & 3u);
}

__device__ __forceinline__ int randN(uint32_t &s, int n) {
    return (int)(xorshift32(s) % (uint32_t)n);
}


//global cluster # counter
__device__ int d_stick_counter; //seed is 1, collective clustered particles are 2, 3, ...
//global work counter for persistent threads
__device__ int d_next_id;


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

    /*
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
    */
    
    //register-only rng
    uint32_t rng = seed_mix(seed, tid);
    int x, y;
    int side = rand4(rng);      //picks one of four edges
    if (side == 0) {            //top edge
        x = randN(rng, N);  y = 0; 
    }else if (side == 1) {      //right edge
        x = N - 1;          y = randN(rng, N); 
    }else if (side == 2) {      //bottom edge
        x = randN(rng, N);  y = N - 1; 
    }else{                      //left edge
        x = 0;              y = randN(rng, N);
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
        int d = rand4(rng);
        x += (d == 2) - (d == 3);        // +/-1 on x
        y += (d == 0) - (d == 1);        // +/-1 on y

        //kill particle if it falls off edge
        if (x < 0 || x >= N || y < 0 || y >= N) {
            return; //particle escape
        }
    }
    
}

//new kernel that dynamically distributes work
__global__ void dla_kernel2(int* grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long long seed) {
    
    //persistent work loop
    while (true) {
        //next particle id
        int pid = atomicAdd(&d_next_id, 1);
        if (pid >= NUM_PARTICLES) {
            break; //all particles assigned
        }

        // per-particle rng seed (not per thread like prev)
        uint32_t rng = seed_mix(seed, pid);

        //border spawn
        int x, y;
        int side = rand4(rng);      //picks one of four edges
        if (side == 0) {            //top edge
            x = randN(rng, N);  y = 0; 
        }else if (side == 1) {      //right edge
            x = N - 1;          y = randN(rng, N); 
        }else if (side == 2) {      //bottom edge
            x = randN(rng, N);  y = N - 1; 
        }else{                      //left edge
            x = 0;              y = randN(rng, N);
        }

        //random walk
        for (int steps = 0; steps < MAX_STEPS; steps++) {
            if (near_cluster(grid, x, y, N)) {
                int idx = y * N + x;
                if (atomicCAS(&grid[idx], 0, 1) == 0) {
                    int val = atomicAdd(&d_stick_counter, 1) + 1;
                    grid[idx] = val;
                    break;      //finish this particle
                }
            }
            int d = rand4(rng);
            x += (d == 2) - (d == 3);
            y += (d == 0) - (d == 1);

            if ((unsigned)x >= (unsigned)N || (unsigned)y >= (unsigned)N) {
                break; //particle escape
            }
        }
        //loop back and claim another pid
    }
}

//host wrapper
void dla(int* d_grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long long seed) {
    //counter init
    int init = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_stick_counter, &init, sizeof(int)));
    //work counter init
    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_next_id, &zero, sizeof(int)));

    int threads = 256;
    int blocks = (NUM_PARTICLES + threads - 1) / threads;
    dla_kernel2<<<blocks, threads>>>(d_grid, N, NUM_PARTICLES, MAX_STEPS, seed);
    CUDA_CHECK(cudaGetLastError());       // catch invalid launch / invalid device function
    CUDA_CHECK(cudaDeviceSynchronize());  // catch runtime errors
}