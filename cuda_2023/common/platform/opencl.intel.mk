# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# Uncomment below two lines and configure if you want to use a platform
# other than global one

#OPENCL_PATH=/openclpath
#OPENCL_LIB_PATH=$(OPENCL_PATH)/lib/x86_64

# gcc (default)
CC = gcc
PLATFORM_CFLAGS= -O3
  
CXX = g++
PLATFORM_CXXFLAGS= -O3
  
LINKER = g++
PLATFORM_LDFLAGS= -lm -lpthread

.PHONY: KERNELS
KERNELS :
	
LIBOPENCL=-lOpenCL

.PHONY: resolvelibOpenCL

resolvelibOpenCL: $(BIN)
	@echo "Resolving OpenCL library..."
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(OPENCL_LIB_PATH) ldd $(BIN) | grep OpenCL

PLATFORM_NAME="Intel"

