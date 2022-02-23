# Python stuff
CFLAGS += $(shell python3-config --cflags)
LDFLAGS += $(shell python3-config --ldflags)
CFLAGS += -I$(shell python -c 'import numpy as np; print(np.get_include())')

# Fmt
LDLIBS += -lfmt

# C++ stuff
CPPFLAGS += -std=c++17
CXXFLAGS += -Wall
CXXFLAGS += -fopenmp
LDFLAGS += -fopenmp


# Remove undesired flags
CFLAGS   := $(filter-out -DNDEBUG, $(CFLAGS))
CPPFLAGS := $(filter-out -DNDEBUG, $(CPPFLAGS))
CXXFLAGS := $(filter-out -DNDEBUG, $(CXXFLAGS))


# Compilation mode
OPTIMIZE ?= yes
DEBUG ?= no
DEBUGSYMBOLS ?= no
ifeq ($(OPTIMIZE),yes)
    CXXFLAGS += -flto -O3 -march=native
    LDFLAGS += -flto -O3 -march=native
else ifeq ($(OPTIMIZE),no)
else
    $(error "bad value for OPTIMIZE: $(OPTIMIZE); must be 'yes' or 'no' (default='yes').")
endif
ifeq ($(DEBUG),no)
    CXXFLAGS += -DNDEBUG
else ifeq ($(DEBUG),yes)
else
    $(error "bad value for DEBUG: $(DEBUG); must be 'yes' or 'no' (default='yes').")
endif
ifeq ($(DEBUGSYMBOLS),yes)
    CXXFLAGS += -g
    LDFLAGS += -g
else ifeq ($(DEBUGSYMBOLS),no)
else
    $(error "bad value for DEBUGSYMBOLS: $(DEBUGSYMBOLS); must be 'yes' or 'no' (default='no').")
endif


# Get files to compile
CPPS = $(wildcard *.cpp)
HEADERS = $(wildcard *.hpp)
OBJECTS = $(patsubst %.cpp,.build/%.o,$(CPPS))

# Phonies
.PHONY: all
all: model.so

.PHONY: clean
clean:
	rm -rf model.so .build/

# C++ Compilation targets
model.so: $(OBJECTS)
	$(CXX) -Wall -shared -o $@ $^ $(LDFLAGS) $(LDLIBS)

.build/%.o: %.cpp $(HEADERS) | .build/
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $< -fPIC

# Build directory management
.build/:
	mkdir -p $@
