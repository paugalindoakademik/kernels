# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# Uncomment below two lines and configure if you want to use a platform
# other than global one

# gcc (default)
CC = ${INTEL_COMPOSER_PATH}/bin/intel64/icc
PLATFORM_CFLAGS = -g
  
CXX = ${INTEL_COMPOSER_PATH}/bin/intel64/icpc
PLATFORM_CXXFLAGS = -g

LINKER = ${INTEL_COMPOSER_PATH}/bin/intel64/icc
PLATFORM_LDFLAGS = -lm -lpthread -Wl,-rpath,'$$ORIGIN:${INTEL_COMPOSER_PATH}/compiler/lib/intel64' -g

CCL = mopencl
PLATFORM_CCLFLAGS = --keep

ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

SHIP_PRECOMPILED_KERNELS_DIR=$(BUILDDIR)

KERNEL_OBJS=$(addprefix $(BUILDDIR)/,$(KERNEL_FILES))

.PHONY: KERNELS
KERNELS : ${KERNEL_OBJS}  

$(BUILDDIR)/%.cl : $(SRCDIR)/%.cl
	$(CCL) -o $(SHIP_PRECOMPILED_KERNELS_DIR) ${PLATFORM_CCLFLAGS} ${APP_CCLFLAGS} $< 
   
.PHONY: resolvelibOpenCL
resolvelibOpenCL :

LIBOPENCL=-lmxpa_runtime -ldl
