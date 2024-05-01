/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *cr   This version maintained by: Nasser Anssari (anssari1@illinois.edu)
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "file.h"

void readVector(const char *fName, float **vec_h, unsigned *size)
{
    FILE* fp = fopen(fName, "rb");

    if (fp == NULL) FATAL("Cannot open input file");

    fread(size, sizeof(unsigned), 1, fp);
    *vec_h = (float*)malloc(*size * sizeof(float));
    if(*vec_h == NULL) FATAL("Unable to allocate host");
    fread(*vec_h, sizeof(float), *size, fp);
    fclose(fp);
}


void writeVector(const char *fName, float *vec_h, unsigned size)
{
    FILE* fp = fopen(fName, "wb");
    if (fp == NULL) FATAL("Cannot open output file");
    fwrite(&size, sizeof(unsigned), 1, fp);
    fwrite(vec_h, sizeof(float), size, fp);
    fclose(fp);
}
