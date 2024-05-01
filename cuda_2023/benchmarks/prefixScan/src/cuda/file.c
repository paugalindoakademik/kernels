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
#include "common.h"

void readVector(const char *fName, float **vec_h, unsigned *size)
{
    size_t read_items = 0;
    FILE* fp = fopen(fName, "rb");

    if (fp == NULL) FATAL("Cannot open input file");

    read_items = fread(size, sizeof(unsigned), 1, fp);
    if (read_items != 1) FATAL("Error while reading file");

    *vec_h = (float*)malloc(*size * sizeof(float));
    if(*vec_h == NULL) FATAL("Unable to allocate host");

    read_items = fread(*vec_h, sizeof(float), *size, fp);
    if (read_items != *size) FATAL("Error while reading file");

    fclose(fp);
}


void writeVector(const char *fName, float *vec_h, unsigned size)
{
    size_t written_items = 0;
    FILE* fp = fopen(fName, "w");
    if (fp == NULL) FATAL("Cannot open output file");
    written_items = fwrite(&size, sizeof(unsigned), 1, fp);
    if (written_items != 1) FATAL("Error while writting file");

    written_items = fwrite(vec_h, sizeof(float), size, fp);
    if (written_items != size) FATAL("Error while writting file");

    fclose(fp);
}
