/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 * The algorithm is based on CUDA sgemm code from Vasily Volkov
 * at UC Berkeley.
 */

#include <iostream>

#include "sgemm_kernel.h"

// Returns x/y ceiled to the next (upper) integer when x doesn't evenly divide by y (uses integer arithmetic trickery)
#define div_and_ceil(x,y) (((x) - 1)/(y) + 1)

// Select a kernel implementation. Comment/uncomment the line below to switch between the simple and tiled versions
#define SIMPLE

// Parameters of tile sizes (only needed for the tiled implementation)
#define TILE_SZ_M 128
#define TILE_SZ_N 16
#define TILE_SZ_K (TILE_SZ_M/TILE_SZ_N) // keep this ratio to ease B_s loading (i.e. # of elements in B_s == # of threads in a thread block)

#ifdef SIMPLE
// simple sgemm kernel implementation.
// Note: A and C are stored in memory in column major order, and B is stored in row major.
//   m -> #rows of A
//   n -> #cols of B
//   k -> #rows of B
__global__ void sgemmNT_naive( const float *A, const float *B, float* C, int m, int n, int k )
{

}

#else

// sgemm kernel implementation with shared memory and register tiling.
// Note: A and C are stored in memory in column major order, and B is stored in row major.
__global__ void sgemmNT_tiled( const float *A, const float *B, float* C, int m, int n, int k )
{
    // Shared memory allocation to store a tile of B
    __shared__ float B_s [][];

    // Macros for accessing flattened matrices
    #define A(row,col) A[row + (col)*m]
    #define B(row,col) B[(row)*n + (col)]
    #define C(row,col) C[row + (col)*m]

    // Compute thread's global row index (for A and C)
    const unsigned int im = 
    // Compute the global column index of the first column processed by the thread (for B and C)   
    const unsigned int in = 

    // Privatization of output variables. Each thread computes a row of a tile of C.
    float c_reg[];

    // Initialize output values
    for() {
        c_reg[] = 0;
    }

    // Loop over the input tiles following the K dimension
    for() {
        // Compute the coordinates of the element of B_s loaded by each thread 
        const unsigned int iB = 
        const unsigned int jB = 
        // Load the current tile of B into shared memory. Ensure all threads finished before proceeding.
        if () {
            B_s[iB][jB] = 
        } else {
            B_s[iB][jB] =
        }

        // Loop over the columns of A's tile and the rows of B's tile
        for() {
            // Load current element of A matrix into the register
            float a_reg;
            if() {
                a_reg = 
            } else {
                a_reg = 
            }
            // Loop over the columns of B_s and update the output elements assigned to the thread
            for() {
                c_reg[] += 
            }
        }
        // Ensure all threads finished before proceeding.

    }
    
    // Store the result to C
    for() {
        if() {
            C(,) =
        }
    }
}
#endif

void sgemm( const float *A, const float *B, float* C, int m, int n, int k )
{
#ifdef SIMPLE
    std::cout << std::endl << "Using the SIMPLE kernel implementation" << std::endl;

    const unsigned block_sz = 16;

    dim3 grid(/* fill grid dimensions */), threads( block_sz, block_sz );
    sgemmNT_naive<<<grid, threads>>>( A, B, C, m, n, k);

#else
    std::cout << std::endl << "Using the TILED kernel implementation" << std::endl;

    dim3 grid(/* fill grid dimensions */), threads(/* fill block dimensions */);
    sgemmNT_tiled<<<grid, threads>>>( A, B, C, m, n, k);

#endif
    CHECK_ERROR();
}

