#include <parboil.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "file.h"
#include "kernel.h"
#define BLOCKS 32
#define THREADS 256

//#define PRIVATIZATION
#ifdef PRIVATIZATION
#define histo_kernel histo_privatization
#else
#define histo_kernel histo_simple
#endif

int main(int argc, char *argv[])
{
    struct pb_TimerSet timers;
    struct pb_Parameters *parameters;

    parameters = pb_ReadParameters(&argc, argv);
    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    unsigned char *buffer_h, *buffer_d;
    unsigned int *histo_h, *histo_d;
    unsigned size;
    cudaError_t err;
    dim3 dim_grid, dim_block;

    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    readVector(parameters->inpFiles[0], &buffer_h, &size);
    
    /*Allocate a buffer to hold the histogram*/
    histo_h =(unsigned int*) malloc(MAX_VAL * sizeof(unsigned int));
    if(histo_h == NULL) FATAL("Unable to allocate host");
    
    /********************************************************************
    Allocate device memory
    ********************************************************************/
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    err = cudaMalloc((void**)&buffer_d, size);
    if(err != cudaSuccess) FATAL("Unable to allocate device memory");
    err = cudaMalloc((void**)&histo_d, MAX_VAL * sizeof(unsigned int));
    if(err != cudaSuccess) FATAL("Unable to allocate device memory");
    
    /********************************************************************
    Copy the input data from the host memory to the device memory
    and reset the 
    ********************************************************************/
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    err = cudaMemcpy(buffer_d, buffer_h, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) FATAL("Unable to copy memory to the device");
    err = cudaMemset(histo_d, 0 , MAX_VAL * sizeof(unsigned int)) ;
    if(err != cudaSuccess) FATAL("Unable to initialize device memory");

    /********************************************************************
    Perform histogramming
    ********************************************************************/
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    histo_kernel<<<BLOCKS, THREADS>>>(buffer_d, size, histo_d);
    err = cudaDeviceSynchronize();

    /********************************************************************
    Copy the resulting histogram back to the host
    ********************************************************************/
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    cudaMemcpy(histo_h, histo_d, MAX_VAL * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /********************************************************************
    Perform the histogram in the CPU (golden version)
    ********************************************************************/
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    unsigned histo_g[MAX_VAL];
    memset(histo_g, 0, MAX_VAL * sizeof(unsigned int));
    for(int i=0; i<size; i++)
        ++histo_g[buffer_h[i]];
    
    /********************************************************************
    Write back the result and perform the cleanup
    ********************************************************************/

    writeVector(parameters->outFile, histo_h, MAX_VAL);
    cudaFree(histo_d);
    cudaFree(buffer_d);
    free(histo_h);
    free(buffer_h);

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);

    return 0;
}
