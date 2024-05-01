/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <parboil.h>
#include <iostream>

#include "io.h"
#include "sgemm_kernel.h"

int
main (int argc, char *argv[]) {

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  float *dA, *dB, *dC;
  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);

  // load B^T, note swapped dimensions
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );

  A_sz = matArow*matAcol*sizeof(float);
  B_sz = matBrow*matBcol*sizeof(float);
  C_sz = matArow*matBcol*sizeof(float);

  // CUDA memory allocation
  std::vector<float> matC(matArow*matBcol);
  cudaMalloc((void**)&dA, A_sz);
  CHECK_ERROR();
  cudaMalloc((void**)&dB, B_sz);
  CHECK_ERROR();
  cudaMalloc((void**)&dC, C_sz);
  CHECK_ERROR();

  // Copy A and B^T into device memory
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  cudaMemcpy(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice); 
  CHECK_ERROR();
  cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 
  CHECK_ERROR();

  pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );

  sgemm(dA, dB, dC, matArow, matBcol, matAcol);

  if (params->outFile) {
    pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    cudaMemcpy(&matC.front(), dC, C_sz, cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    /* Write C to file */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_KERNEL]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  
  cudaFree(dA);
  CHECK_ERROR();
  cudaFree(dB);
  CHECK_ERROR();
  cudaFree(dC);
  CHECK_ERROR();
  return 0;
}
