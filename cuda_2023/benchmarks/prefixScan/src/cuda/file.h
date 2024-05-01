/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr   This version maintained by: Nasser Anssari (anssari1@illinois.edu)
 ***************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#ifdef __cplusplus
extern "C" {
#endif
void readVector (const char* fName, float **vec_h, unsigned *size);
void writeVector(const char* fName, float *vec_h, unsigned size);

#ifdef __cplusplus
}
#endif

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
