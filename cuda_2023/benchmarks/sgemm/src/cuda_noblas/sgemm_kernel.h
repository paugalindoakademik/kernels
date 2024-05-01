/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#ifndef SGEMM_KERNEL_H
#define SGEMM_KERNEL_H

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 * The algorithm is based on CUDA sgemm code from Vasily Volkov
 * at UC Berkeley.
 */

#include <stdio.h>

#define CHECK_ERROR() do { \
  cudaError_t err = cudaGetLastError(); \
  if( cudaSuccess != err) { \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
	 __FILE__, __LINE__, cudaGetErrorString( err) ); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

__global__ void simple_sgemm_kernel( const float *A, const float *B, float* C, int m, int n, int k );
__global__ void tiled_sgemm_kernel( const float *A, const float *B, float* C, int m, int n, int k );

void sgemm( const float *A, const float *B, float* C, int m, int n, int k );

#endif
