/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr   This version maintained by: Nasser Anssari (anssari1@illinois.edu)
 ***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "kernel.h"

    
int main(int argc, char *argv[])
{
    struct pb_TimerSet timers;
    struct pb_Parameters *parameters;

    parameters = pb_ReadParameters(&argc, argv);
    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    unsigned vec_size;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;

    /* Initialize input matrixes */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    readVector(parameters->inpFiles[0], &A_h, &vec_size);
    readVector(parameters->inpFiles[1], &B_h, &vec_size);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /* Allocate host memory */
    C_h = (float *)malloc(vec_size * sizeof(float));
    if(C_h == NULL) FATAL("Unable to allocate host");

    /********************************************************************
    Allocate device memory for the input/output vectors
    ********************************************************************/





    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    /********************************************************************
    Copy the input vectors from the host memory to the device memory
    ********************************************************************/





    cuda_ret = cudaMemset(C_d, 0, vec_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    /********************************************************************
    Initialize thread block and kernel grid dimensions
    ********************************************************************/





    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    /********************************************************************
    Invoke CUDA kernel
    ********************************************************************/





    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    if (parameters->outFile) {
    	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    /********************************************************************
    Copy the result back to the host
    ********************************************************************/





	/* Write the result */
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	writeVector(parameters->outFile, C_h, vec_size);
    }

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    /********************************************************************
    Free device memory allocations
    ********************************************************************/





    free(A_h); free(B_h); free(C_h);

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);

    return 0;
}
