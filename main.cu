#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 201       //odd so there is a center
#define NUM_PARTICLES 500


//2D grid: 0 is empty, 1 is part of the cluster
int grid[N][N];

//direction vectors for random walk



//check if cell is in bounds

//check if cell is adjacent to cluster

//randomly place new particle at border




int main() {

    //seed at the center
    int cx = N/2, cy = N/2;
    grid[cx][cy] = 1;

    //on every particle
    for (int p = 0; p < NUM_PARTICLES; p++) {
        int x, y;
        
        //start with a particle on edge

        while (1) {
            //particle randomly moves

            //break if it gets out of bounds

            //check if it is adjacent to cluster

            //kill the unlucky particles (prevent infinite loops)

        }
    }

    //print output



    return 0;
}