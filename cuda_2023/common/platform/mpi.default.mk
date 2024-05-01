# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# gcc (default)
CC = mpicc
PLATFORM_CFLAGS = -O3
  
CXX = mpicxx
PLATFORM_CXXFLAGS = -O3
  
LINKER = mpicxx
#PLATFORM_LDFLAGS = -lmpi

NVCC = $(CUDA_PATH)/bin/nvcc
