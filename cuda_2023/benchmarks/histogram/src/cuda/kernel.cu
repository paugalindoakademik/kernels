#include "kernel.h"

__global__ void histo_simple(unsigned char *buffer, long size, unsigned *histo)
{
    //Write kernel code
}

__global__ void histo_privatization(unsigned char *buffer, long size, unsigned *histo)
{
    //Write kernel code
    __shared__  unsigned temp[MAX_VAL]; //Will hold the private histogram

}
