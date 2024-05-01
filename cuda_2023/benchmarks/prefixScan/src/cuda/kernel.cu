/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr   This version maintained by: Nasser Anssari (anssari1@illinois.edu)
 ***************************************************************************/

#include <iostream>

#include "common.h"

#define BLOCK_SIZE 512

__global__ void preScanKernel(float *out, float *in, unsigned size, float *sum)
{
}


__global__ void addKernel(float *out, float *sum, unsigned size)
{
}

/********************************************************************
Modify the body of this function to complete the functionality of
    the scan on the device
You may need multiple kernel calls; write your kernels in this file
    and call them from this function
********************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{





}
