# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# c.default is the base along with CUDA configuration in this setting
include $(PARBOIL_ROOT)/common/platform/c.default.mk

# Paths
CUDAHOME=$(CUDA_PATH)

# Programs

CC = $(MPICC)
CXX = $(MPICXX)

CUDACC=$(CUDAHOME)/bin/nvcc
CUDALINK=$(CXX)

# Flags
PLATFORM_CUDACFLAGS = -O3 -g
ifneq ($(MPI_INC_PATH),)
PLATFORM_CUDACFLAGS += -I$(MPI_INC_PATH)
endif

PLATFORM_CUDALDFLAGS = -lm -lpthread -lcudart
ifneq ($(MPI_LIB_PATH),)
PLATFORM_CUDALDFLAGS += -L$(MPI_LIB_PATH)
endif


