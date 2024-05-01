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

__global__ void vecAdd(float *C, const float * __restrict__ A, const float * __restrict__ B, const unsigned size);

#endif
