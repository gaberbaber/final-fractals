#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//found open source code for making pngs
#include "lodepng.h"
//took Dr. Schubert's support for the timings
#include "support.h"


//test case paramters
const int test_N[4]         = {101, 201, 401, 801};       //odd so there is a center
const int test_particles[4] = {25000, 300000, 2000000, 12000000};
const int test_maxsteps[4] = {50000, 100000, 200000, 500000};
const char* test_filenames[4] = {"dla_cpu_101.png", "dla_cpu_201.png", "dla_cpu_401.png", "dla_cpu_801.png"};

//direction vectors for random walk = dx*i_hat + dy*j_hat
int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};

//check if cell is in bounds
int in_bounds(int x, int y, int N) {
    return (x >= 0 && x < N) && (y >= 0 && y < N);
}

//check if cell is adjacent to cluster
int is_adjacent_to_cluster(int x, int y, int** grid, int N) {
    //check any next step for any cluster
    for (int d = 0; d < 4; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (in_bounds(nx, ny, N) && grid[nx][ny] != 0) {
            return 1;
        }
    }
    return 0;
}


//randomly place new particle at border
void random_border_position(int* x, int* y, int N) {
    int edge = rand() % 4; 
    switch (edge) {
        case 0: *x = 0;             // top
                *y = rand() % N;
                break;
        case 1: *x = N-1;           // bottom
                *y = rand() % N;
                break;
        case 2: *x = rand() % N;    // left
                *y = 0;
                break;
        case 3: *x = rand() % N;    // right
                *y = N-1;
                break;
    }
}



int main() {
    //seed rng for varied output
    srand(time(NULL));

    //four test cases
    for (int test = 0; test < 4; test++) {
        int N = test_N[test];
        int NUM_PARTICLES = test_particles[test];
        int MAX_STEPS = test_maxsteps[test];
        const char* outname = test_filenames[test];

        printf("\n===== CPU DLA Test %d: N=%d, Particles=%d, MaxSteps=%d =====\n", test+1, N, NUM_PARTICLES, MAX_STEPS);
        
        //malloc grid
        //2D grid: 0 is empty, 1 is seed, rest of all real numbers = particle #
        int** grid = (int**)malloc(N * sizeof(int*));
        for (int i = 0; i < N; i++) {
            grid[i] = (int*)calloc(N, sizeof(int));
        }

        //seed at the center
        int cx = N/2, cy = N/2;
        grid[cx][cy] = 1;

        Timer timer;
        //start timer
        startTime(&timer);

        //for every particle
        for (int p = 0; p < NUM_PARTICLES; p++) {
            int x, y;
            
            //start with a particle on edge
            random_border_position(&x, &y, N);
            int steps = 0;

            while (1) {
                //particle randomly moves
                int d = rand() % 4;
                x += dx[d];
                y += dy[d];
                steps++;

                //break if it gets out of bounds
                if (!in_bounds(x, y, N)) {
                    break;                  //particle escaped
                }

                //check if it is adjacent to cluster
                if (is_adjacent_to_cluster(x, y, grid, N)) {
                    grid[x][y] = p + 1;     //records what # particle sticks
                    break;                  //the cluster grows!
                }

                //kill the unlucky particles (prevent infinite loops)
                if (steps > MAX_STEPS) {
                    break;
                }

            }
        }

        stopTime(&timer);
        printf("CPU DLA simulation time for Test %d: %f s\n", test+1, elapsedTime(timer));

        //PRINT OUTPUT
        //each pixel is 4 bytes: R, G, B, A
        unsigned char* image = (unsigned char*)malloc(N * N * 4);

        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                int idx = 4 * (j * N + i);
                int v = grid[i][j];

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

        //lodepng to make dla_cpu_NNN.png
        unsigned error = lodepng_encode32_file(outname, image, N, N);
        if (error) {
            printf("PNG Encoder error %u: %s\n", error, lodepng_error_text(error));
        } else {
            printf("DLA PNG written to %s\n", outname);
        }

        //free mem
        free(image);
        for (int i = 0; i < N; i++) {
            free(grid[i]);
        }
        free(grid);

    }

    return 0;
}