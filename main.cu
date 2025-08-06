#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//found open source code for making pngs
#include "lodepng.h"

#define N 201       //odd so there is a center
#define NUM_PARTICLES 350000


//2D grid: xxxxx(0 is empty, 1 is part of the cluster)xxxxx
// ^^^^^^^old logic; new: 0 is empty, 1 is seed, rest of all real numbers = particle #
int grid[N][N];

//direction vectors for random walk = dx*i_hat + dy*j_hat
int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};

//check if cell is in bounds
int in_bounds(int x, int y) {
    return (x >= 0 && x < N) && (y >= 0 && y < N);
}

//check if cell is adjacent to cluster
int is_adjacent_to_cluster(int x, int y) {
    //check any next step for any cluster
    for (int d = 0; d < 4; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (in_bounds(nx, ny) && grid[nx][ny] != 0) {
            return 1;
        }
    }
    return 0;
}


//randomly place new particle at border
void random_border_position(int* x, int* y) {
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

    //seed at the center
    int cx = N/2, cy = N/2;
    grid[cx][cy] = 1;

    //for every particle
    for (int p = 0; p < NUM_PARTICLES; p++) {
        int x, y;
        
        //start with a particle on edge
        random_border_position(&x, &y);
        int steps = 0;

        while (1) {
            //particle randomly moves
            int d = rand() % 4;
            x += dx[d];
            y += dy[d];
            steps++;

            //break if it gets out of bounds
            if (!in_bounds(x, y)) {
                break;                  //particle escaped
            }

            //check if it is adjacent to cluster
            if (is_adjacent_to_cluster(x, y)) {
                grid[x][y] = p + 1;     //records what # particle sticks
                break;                  //the cluster grows!
            }

            //kill the unlucky particles (prevent infinite loops)
            if (steps > 10000) {
                break;
            }

        }
    }

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
                    case 0: r = 1; g = f; b = 0;
                            break;
                    case 1: r = q; g = 1; b = 0;
                            break;
                    case 2: r = 0; g = 1; b = f;
                            break;
                    case 3: r = 0; g = q; b = 1;
                            break;
                    case 4: r = f; g = 0; b = 1;
                            break;
                    case 5: r = 1; g = 0; b = q;
                            break;
                }
                image[idx+0] = (unsigned char)(r * 255);
                image[idx+1] = (unsigned char)(g * 255);
                image[idx+2] = (unsigned char)(b * 255);
                image[idx+3] = 255;
            }
        }
    }
    
    //lodepng to make dla.png
    unsigned error = lodepng_encode32_file("dla.png", image, N, N);
    if (error) {
        printf("PNG Encoder error %u: %s\n", error, lodepng_error_text(error));
    } else {
        printf("DLA PNG written to dla.png\n");
    }

    //free mem
    free(image);


    return 0;
}