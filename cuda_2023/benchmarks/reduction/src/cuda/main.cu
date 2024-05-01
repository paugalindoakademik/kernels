/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr   This version maintained by: Nasser Anssari (anssari1@illinois.edu)
 ***************************************************************************/

#include <stdio.h>
#include <parboil.h>

#include "file.h"
#include "kernel.h"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) 
{
    struct pb_TimerSet timers;
    struct pb_Parameters *parameters;

    parameters = pb_ReadParameters(&argc, argv);
    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned in_elements, out_elements;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;
    int i;

    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    readVector(parameters->inpFiles[0], &in_h, &in_elements);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    out_elements = in_elements / (BLOCK_SIZE<<1);
    if(in_elements % (BLOCK_SIZE<<1)) out_elements++;

    out_h = (float*)malloc(out_elements * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host");

    /********************************************************************
    Allocate device memory for the input/output vectors
    ********************************************************************/





    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    /********************************************************************
    Copy the input vector from the host memory to the device memory
    ********************************************************************/





    cuda_ret = cudaMemset(out_d, 0, out_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    /********************************************************************
    Initialize thread block and kernel grid dimensions
    ********************************************************************/

    printf("Dimensions: grid(%d,%d,%d)   block(%d,%d,%d)\n", dim_grid.x, dim_grid.y, dim_grid.z, dim_block.x, dim_block.y, dim_block.z);    

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





        /********************************************************************
        Reduce output vector on the host
        ********************************************************************/
        for(i=1; i<out_elements; i++) {
            out_h[0] += out_h[i];
        }

        /* Write the result */
        pb_SwitchToTimer(&timers, pb_TimerID_IO);
        writeVector(parameters->outFile, out_h, 1);
    } 

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /********************************************************************
    Free device memory allocations
    ********************************************************************/





    free(in_h); free(out_h);

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);

    return 0;
}
