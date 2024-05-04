# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda-10.0/bin/nvcc
# the build target executable:
TARGET = matix_multiplication
# FLAGS 
# CC compiler flags:
CFLAGS  = -O3
# NVCC compiler flags
NVCCFLAGS = -arch compute_72 -code sm_72 -O3
# OPENCL FLAGS
OPFLAGS = -I/opt/rocm/opencl/include/   -L/opt/rocm/opencl/lib/ -lOpenCL  
# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 
# Data type by default is double and for now can be -DINT -DFLOAT
DATAPYPE = -DFLOAT
# Data type for GPU by default is double and for now can be -DINT -DFLOAT -DFLOAT16
DATAPYPEGPU = -DFLOAT16
# FOLDERS
# OPENCL FOLDER
OPFOLDER = ./opencl/
# CPU FOLDER
CPUFOLDER = ./cpu/
# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/

# COMPILER MAIN
all:
	@echo "YOU NEED TO SELECT CUDA OR OPENCL -> make CUDA / make OpenCL / make CUDA-opt / make OpenCL-opt / make CUDA-lib / make OpenCL-lib"
# End Main
# Shortcuts
.PHONY: all-bin
all-bin: opencl opencl-opt
.PHONY: all-opencl
all-opencl: opencl opencl-opt
.PHONY: OpenCL
OpenCL: opencl
.PHONY: OpenCL-opt
OpenCL-opt: opencl-opt
# End Shortcuts
# CPU part
lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp
	$(CC) $(ENDIANFLAGS) $(DATAPYPE) -c $(CPUFOLDER)lib_cpu.cpp -o $(CPUFOLDER)lib_cpu.o $(CFLAGS) 
# End CPU 

# OpenCL Part
opencl:  main_opencl

lib_opencl.o: $(OPFOLDER)lib_opencl.cpp 
	$(CC) $(DATAPYPE) -DOPENCL -c $(OPFOLDER)lib_opencl.cpp -o $(OPFOLDER)lib_opencl.o $(CFLAGS) $(OPFLAGS)

main_opencl: main.cpp lib_opencl.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) $(DATAPYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_opencl $(CFLAGS) $(OPFLAGS) 

# End OpenCL

# OpenCL Part optimized
opencl-opt:  main_opencl_opt

lib_opencl_opt.o: $(OPFOLDER)lib_opencl_opt.cpp 
	$(CC) $(DATAPYPE) -DOPENCL -c $(OPFOLDER)lib_opencl_opt.cpp -o $(OPFOLDER)lib_opencl_opt.o $(CFLAGS) $(OPFLAGS)

main_opencl_opt: main.cpp lib_opencl_opt.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) $(DATAPYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_opt.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_opt $(CFLAGS) $(OPFLAGS) 

# End OpenCL optimized

# Clean
.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_opencl
	rm -rf $(OUTPUTFOLDER)$(TARGET)_cuda
	rm -rf $(OUTPUTFOLDER)$(TARGET)_opencl_lib
	rm -rf $(OUTPUTFOLDER)$(TARGET)_cuda_lib
	rm -rf $(OUTPUTFOLDER)$(TARGET)_opencl_opt
	rm -rf $(OUTPUTFOLDER)$(TARGET)_cuda_opt
