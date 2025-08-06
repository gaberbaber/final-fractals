#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 201       //odd so there is a center
#define NUM_PARTICLES 500


//2D grid: 0 is empty, 1 is part of the cluster
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
        if (in_bounds(nx, ny) && grid[nx][ny] == 1) {
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

    //seed at the center
    int cx = N/2, cy = N/2;
    grid[cx][cy] = 1;

    //for every particle
    for (int p = 0; p < NUM_PARTICLES; p++) {
        int x, y;
        
        //start with a particle on edge
        random_border_position(&x, &y);

        while (1) {
            //particle randomly moves
            int d = rand % 4;
            x += dx[d];
            y += dy[d];

            //break if it gets out of bounds
            if (!in_bounds(x, y)) {
                break;                  //particle escaped
            }

            //check if it is adjacent to cluster
            if (is_adjacent_to_cluster(x, y)) {
                grid[x, y] = 1;
                break;                  //the cluster grows!
            }

            //kill the unlucky particles (prevent infinite loops)


        }
    }

    //print output



    return 0;
}