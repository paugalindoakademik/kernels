# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include
LANG_CXXFLAGS=$(LANG_CFLAGS)
#LANG_LDFLAGS=$(LIBMPI) -L$(MPI_LIB_PATH)
LANG_LDFLAGS=-L$(LIB_PATH)

CFLAGS=$(LANG_CFLAGS) $(PLATFORM_CFLAGS) $(APP_CFLAGS)
CXXFLAGS=$(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS) $(APP_CXXFLAGS)
LDFLAGS=$(LANG_LDFLAGS) $(PLATFORM_LDFLAGS) $(APP_LDFLAGS)

# Rules common to all makefiles

########################################
# Functions
########################################

# Add BUILDDIR as a prefix to each element of $1
INBUILDDIR=$(addprefix $(BUILDDIR)/,$(1))

# Add SRCDIR as a prefix to each element of $1
INSRCDIR=$(addprefix $(SRCDIR)/,$(1))


########################################
# Environment variable check
########################################

# The second-last directory in the $(BUILDDIR) path
# must have the name "build".  This reduces the risk of terrible
# accidents if paths are not set up correctly.
ifeq ("$(notdir $(BUILDDIR))", "")
$(error $$BUILDDIR is not set correctly)
endif

ifneq ("$(notdir $(patsubst %/,%,$(dir $(BUILDDIR))))", "build")
$(error $$BUILDDIR is not set correctly)
endif

.PHONY: default run debug clean

########################################
# Derived variables
########################################

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

OBJS = $(call INBUILDDIR,$(SRCDIR_OBJS))

########################################
# Rules
########################################

default: $(BUILDDIR) $(BIN)

run:
	@mpdstartup
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(LIB_PATH):$(MPI_LIB_PATH) mpirun -n $(MPI_PROCS) $(BIN) $(ARGS)
	@echo

debug:
	@mpdstartup
	@$(shell echo $(RUNTIME_ENV)) LD_LIBRARY_PATH=$(LIB_PATH):$(MPI_LIB_PATH) mpirun -gdb -n $(MPI_PROCS) $(BIN)

clean:
	rm -f $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi
	rm -f $(RUNDIR)/*
	if [ -d $(RUNDIR) ]; then rmdir $(RUNDIR); fi

#$(BIN) : $(OBJS) $(PARBOIL_ROOT)/common/lib/libparboil.a
#	$(CXX) $^ -o $@ $(LDFLAGS)


$(BIN) : $(OBJS) $(BUILDDIR)/parboil.o $(BUILDDIR)/args.o
	$(LINKER) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/parboil.o : $(PARBOIL_ROOT)/common/src/parboil.c ${BUILDDIR}
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/args.o : $(PARBOIL_ROOT)/common/src/args.c ${BUILDDIR}
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@
