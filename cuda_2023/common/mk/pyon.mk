# (c) 2007 The Board of Trustees of the University of Illinois.

# Default language wide options

LANG_CFLAGS=-I$(PARBOIL_ROOT)/common/include -I$(PYONHOME)/share/pyon-0.1/include -I/usr/include/gc
LANG_CXXFLAGS=$(LANG_CFLAGS)
LANG_LDFLAGS=-L$(PYONHOME)/share/pyon-0.1 -L$(PARBOIL_ROOT)/common/lib \
	-lparboil -lpyonrts -lgc -lpthread -lm

CFLAGS=$(LANG_CFLAGS) $(PLATFORM_CFLAGS) $(APP_CFLAGS)
CXXFLAGS=$(LANG_CXXFLAGS) $(PLATFORM_CXXFLAGS) $(APP_CXXFLAGS) -I$(BUILDDIR)
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

.PHONY: default run debug clean no_pyon

ifeq ($(PYONHOME),)
FAILSAFE=no_pyon
else 
FAILSAFE=
endif

########################################
# Derived variables
########################################

PYON_OBJS=$(call INBUILDDIR, $(addsuffix .o,$(PYON_SRCS)))

PYON_APP_HEADERS=$(call INBUILDDIR, $(addsuffix _cxx.h,$(PYON_SRCS)))

OBJS = $(PYON_OBJS) $(call INBUILDDIR,$(SRCDIR_OBJS))

ifeq ($(DEBUGGER),)
DEBUGGER=gdb
endif

########################################
# Rules
########################################

default: $(FAILSAFE) $(BUILDDIR) $(BIN)

run:
	@echo $(PYON_APP_HEADERS)
	@$(shell echo $(RUNTIME_ENV)) ldd $(BIN)
	@$(shell echo $(RUNTIME_ENV)) $(BIN) $(ARGS)

debug:
	@$(shell echo $(RUNTIME_ENV)) ldd $(BIN)
	@$(shell echo $(RUNTIME_ENV)) $(DEBUGGER) --args $(BIN) $(ARGS)

clean:
	rm -f $(BUILDDIR)/*
	if [ -d $(BUILDDIR) ]; then rmdir $(BUILDDIR); fi
	rm -f $(RUNDIR)/*
	if [ -d $(RUNDIR) ]; then rmdir $(RUNDIR); fi

$(BIN) : $(OBJS) $(PARBOIL_ROOT)/common/lib/libparboil.a
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILDDIR) :
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o : $(SRCDIR)/%.c $(PYON_APP_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o : $(SRCDIR)/%.pyon
	$(PYONHOME)/bin/pyon -o $@ $<

# This rule is for dependences only
$(BUILDDIR)/%_cxx.h : $(BUILDDIR)/%.o $(SRCDIR)/%.pyon
	true

no_pyon:
	@echo "PYONHOME is not set. Open $(PARBOIL_ROOT)/common/Makefile.conf to set default value."
	@echo "You may use $(PLATFORM_MK) if you want a platform specific configurations."
	@exit 1

