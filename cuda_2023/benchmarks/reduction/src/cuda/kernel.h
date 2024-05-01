/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr   This version maintained by: Nasser Anssari (anssari1@illinois.edu)
 ***************************************************************************/

#ifndef KERNEL_H
#define KERNEL_H

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size);

#endif
