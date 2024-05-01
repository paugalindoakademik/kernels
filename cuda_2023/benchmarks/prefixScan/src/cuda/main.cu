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

#include "file.h"
#include "kernel.h"
#include "common.h"

int main(int argc, char* argv[])
{
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;

	parameters = pb_ReadParameters(&argc, argv);
	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);	

	float *in_h, *out_h;
	float *in_d, *out_d;
	unsigned num_elements;
	cudaError_t cuda_ret;

	/* Allocate and initialize input vector */
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	readVector(parameters->inpFiles[0], &in_h, &num_elements);

	/* Allocate and initialize output vector */
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	out_h = (float*)calloc(num_elements, sizeof(float));
	if(out_h == NULL) FATAL("Unable to allocate host");

	/* Allocate device memory */
	cuda_ret = cudaMalloc((void**)&in_d, num_elements*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
	cuda_ret = cudaMalloc((void**)&out_d, num_elements*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

	/*Copy input vectot from host to device */
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	cuda_ret = cudaMemcpy(in_d, in_h, num_elements*sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

	cuda_ret = cudaMemset(out_d, 0, num_elements*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

	pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
	/* Modify the body of this function to complete the functionality of the scan
	   on the deivce */
	preScan(out_d, in_d, num_elements);
	
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

	if(parameters->outFile) {
		/* Copy the result back to the host */
		pb_SwitchToTimer(&timers, pb_TimerID_COPY);
		cuda_ret = cudaMemcpy(out_h, out_d, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
		if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

		/* Write the result */
		pb_SwitchToTimer(&timers,  pb_TimerID_IO);
		writeVector(parameters->outFile, out_h, num_elements);
	}

	/* Free memory aloocations */
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	cudaFree(in_d); cudaFree(out_d);
	free(in_h); free(out_h);

	pb_SwitchToTimer(&timers, pb_TimerID_NONE);
	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;
}
	
			 
