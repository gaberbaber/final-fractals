#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
//found open source code for making pngs
#include "lodepng.h"
//took Dr. Schubert's support for the timings
#include "support.h"

//error check
#define CUDA_CHECK(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    exit(1); \
  } \
} while (0)


//kernel function
void dla(int* d_grid, int N, int NUM_PARTICLES, int MAX_STEPS, unsigned long long seed);

//test case paramters
const int test_N[4]         = {101, 201, 401, 801};       //odd so there is a center
const int test_particles[4] = {25000, 300000, 2000000, 12000000};
const int test_maxsteps[4] = {50000, 100000, 200000, 500000};
const char* test_filenames[4] = {"dla_gpu_101.png", "dla_gpu_201.png", "dla_gpu_401.png", "dla_gpu_801.png"};

int main() {
    
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    printf("CUDA devices visible: %d\n", devCount);
    if (devCount == 0) {
    fprintf(stderr, "No CUDA device found.\n");
    return 1;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device 0: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    CUDA_CHECK(cudaSetDevice(0));

    //four test cases
    for (int test = 0; test < 4; test++) {
        int N = test_N[test];
        int NUM_PARTICLES = test_particles[test];
        int MAX_STEPS = test_maxsteps[test];
        const char* outname = test_filenames[test];

        printf("\n===== GPU DLA Test %d: N=%d, Particles=%d, MaxSteps=%d =====\n", test+1, N, NUM_PARTICLES, MAX_STEPS);
        
        //grid host mem (zeroed)
        int* h_grid = (int*)calloc(N*N, sizeof(int));

        //seed at the center
        h_grid[(N/2)*N + (N/2)] = 1;

        //grid device mem
        int* d_grid = nullptr;
        cudaMalloc(&d_grid, N*N*sizeof(int));
        cudaMemcpy(d_grid, h_grid, N*N*sizeof(int), cudaMemcpyHostToDevice);
        

        Timer timer;
        //start timer
        startTime(&timer);

        //launch cuda dla
        dla(d_grid, N, NUM_PARTICLES, MAX_STEPS, (unsigned long long)time(NULL));
        cudaDeviceSynchronize();
        stopTime(&timer);
        printf("GPU DLA simulation time for Test %d: %f s\n", test+1, elapsedTime(timer));

        //copy result
        cudaMemcpy(h_grid, d_grid, N*N*sizeof(int), cudaMemcpyDeviceToHost);
        
        //seed check debug
        printf("DEBUG: Center cell value after kernel: %d\n", h_grid[(N/2)*N + (N/2)]);
        //cluster size check debug
        int stuck_particles = 0;
        for (int i = 0; i < N*N; ++i){
            if (h_grid[i] != 0) stuck_particles++;
        }
        printf("DEBUG: Total nonzero (stuck) particles after kernel: %d\n", stuck_particles);



        //PRINT OUTPUT (changed from main_cpu because 1D grid)
        //each pixel is 4 bytes: R, G, B, A
        unsigned char* image = (unsigned char*)malloc(N * N * 4);

        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = 4 * (j * N + i);
                int v = h_grid[j*N + i];

                if (v == 0) {
                    //empty: white pixel
                    image[idx+0] = 255;
                    image[idx+1] = 255;
                    image[idx+2] = 255;
                    image[idx+3] = 255;
                } else {
                    //map number of particles to rainbow
                    float t = (float)(v-1) / NUM_PARTICLES;
                    //function for rainbow
                    float r, g, b;
                    //hue from 0 (red) to .75 (violet)
                    float hue = 0.75 * (1.0f - t); //red to violet
                    //convert hue to rgb
                    int h = (int)(hue * 6);
                    float f = hue * 6 - h;
                    float q = 1 - f;
                    switch (h % 6) {
                        case 0: r = 1; g = f; b = 0; break;
                        case 1: r = q; g = 1; b = 0; break;
                        case 2: r = 0; g = 1; b = f; break;
                        case 3: r = 0; g = q; b = 1; break;
                        case 4: r = f; g = 0; b = 1; break;
                        case 5: r = 1; g = 0; b = q; break;
                    }
                    image[idx+0] = (unsigned char)(r * 255);
                    image[idx+1] = (unsigned char)(g * 255);
                    image[idx+2] = (unsigned char)(b * 255);
                    image[idx+3] = 255;
                }
            }
        }

        //lodepng to make dla_gpu_NNN.png
        unsigned error = lodepng_encode32_file(outname, image, N, N);
        if (error) {
            printf("PNG Encoder error %u: %s\n", error, lodepng_error_text(error));
        } else {
            printf("DLA PNG written to %s\n", outname);
        }

        //free mem
        cudaFree(d_grid);
        free(image);
        free(h_grid);

    }

    return 0;
}