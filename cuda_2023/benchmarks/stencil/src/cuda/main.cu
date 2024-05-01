
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "kernels.h"

static int generate_data(float *A0, int nx,int ny,int nz) 
{	
	srand(54321);
	int s=0;
	for(int i=0;i<nz;i++)
	{
		for(int j=0;j<ny;j++)
		{
			for(int k=0;k<nx;k++)
			{
				A0[s] = (rand() / (float) RAND_MAX);
				s++;
			}
		}
	}
	return 0;
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
	float fac;
	
	if (argc<4) 
    {
      printf("Usage: probe nx ny nz\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;

	//host data
	float *h_A0;
	float *h_Anext;
	
	//device
	float *d_A0;
	float *d_Anext;

	//load data from files
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	// 	inputData(parameters->inpFiles[0], &nz, &ny, &nz);
	
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	size=nx*ny*nz;
	printf ("SIZE = %dx%dx%d\n", nx, ny, nz);
	
	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);

	generate_data(h_A0, nx,ny,nz);
	fac = h_A0[0];
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_A0, size*sizeof(float));
	CUERR;
	cudaMalloc((void **)&d_Anext, size*sizeof(float));
	CUERR;
	cudaMemset(d_Anext,0,size*sizeof(float));
	CUERR;

	//memory copy
	cudaMemcpy(d_A0, h_A0, size*sizeof(float), cudaMemcpyHostToDevice);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

	execute_kernel(fac, d_A0, d_Anext, nx, ny, nz);
	
	cudaDeviceSynchronize();
    CUERR;

	pb_SwitchToTimer(&timers, pb_TimerID_COPY);

	cudaMemcpy(h_Anext, d_Anext,size*sizeof(float), cudaMemcpyDeviceToHost);
	CUERR;

    cudaFree(d_A0);
	CUERR;
    cudaFree(d_Anext);
	CUERR;
 
	if (parameters->outFile) {
		pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	free (h_A0);
	free (h_Anext);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;

}
