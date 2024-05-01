
#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>

#define CUERR do { \
  cudaError_t err = cudaGetLastError(); \
  if( cudaSuccess != err) { \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
	 __FILE__, __LINE__, cudaGetErrorString( err) ); \
    exit(EXIT_FAILURE); \
  } \
} while(0)

void execute_kernel(float fac, const float* d_in, float* d_out, int nx, int ny, int nz); 

#endif

