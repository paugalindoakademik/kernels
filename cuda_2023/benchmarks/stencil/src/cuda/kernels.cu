#include <iostream>
#include "kernels.h"

#define TILE_SIZE 16
#define Index3D(_i,_j,_k) ( (_i) + nx * ((_j) + ny*(_k)) )

// Select a kernel implementation. Comment/uncomment the line below to switch between the simple and tiled versions
#define SIMPLE

__global__ void simple2D_stencil(float fac, const float *in, float *out, int nx, int ny, int nz)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    if( i>0 && j>0 &&(i<nx-1) &&(j<ny-1) ){//ignore the border cells
        for(int k=1;k<nz-1;k++){
            out[Index3D (i, j, k)] =
            in[Index3D (i, j, k + 1)] +
            in[Index3D (i, j, k - 1)] +
            in[Index3D (i, j + 1, k)] +
            in[Index3D (i, j - 1, k)] +
            in[Index3D (i + 1, j, k)] +
            in[Index3D (i - 1, j, k)]
            - 6.0f * in[Index3D (i, j, k)] / (fac*fac);
        }
    }
}

//Make sure you synchronize threads when necessary
__global__ void block2D_stencil(float fac, const float *in, float *out, int nx, int ny, int nz)
{
    // Aliases for readability
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dx = blockDim.x;
    int dy = blockDim.y;

    // Global x and y indexes of the thread
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j = blockIdx.y*blockDim.y+threadIdx.y;

    //Fill all the missing indexes in the code:
    __shared__ float sh_in[][];

    sh_in[tx][ty] = in[Index3D()];


    //Iterate over xy planes in the z dimension
    for () {
        //Load the element from the next xy plane
        float front = in[];
        //Check global boundaries
        if(i>0 && j>0 && (i<nx-1) &&(j<ny-1)) {
            //Load the neighboring elements
            float left  =
            float right =
            float top   =
            float bottom =
            float back =
            float current =

            out[Index3D()] = back + front + left + right + top + bottom - 6.0f * current / (fac * fac);
        }


        //Update sh_in for the next iteration
        sh_in[tx][ty] = 
    }
}

void execute_kernel(float fac, const float* d_in, float* d_out, int nx, int ny, int nz) 
{
    int tx = TILE_SIZE;
    int ty = TILE_SIZE;
    // Define grid and block sizes
    dim3 block ();
    dim3 grid ();

#ifdef SIMPLE
    std::cout << std::endl << "Using the SIMPLE kernel implementation" << std::endl;
    simple2D_stencil<<<grid, block>>>(fac, d_in, d_out, nx, ny, nz);
#else
    std::cout << std::endl << "Using the TILED kernel implementation" << std::endl;
    block2D_stencil<<<grid, block>>>(fac, d_in, d_out, nx, ny, nz);
#endif
    CUERR;
}
