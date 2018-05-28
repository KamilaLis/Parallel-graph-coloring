CC=g++
NVCC=nvcc
CUDAFLAGS= -g -G 
CFLAGS = -std=c++11
GENCODE_FLAGS += -gencode arch=compute_35,code=sm_35
LIBDIRS=-L/usr/local/cuda-9.1/lib64
INCDIRS=-I/usr/local/cuda-9.1/samples/common/inc -I/usr/local/cuda-9.1/include -I include


# Gencode arguments
SMS ?= 30 35 37 50 52 60 61 70

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


# Target rules
all: build

build: parallel sequential

parallel.o: src/parallel.cu
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

coloring_cpu.o:src/graph_coloring/coloring_cpu.cpp
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

parallel: parallel.o coloring_cpu.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBDIRS)

sequential: src/sequential.cpp src/graph_coloring/coloring_cpu.cpp
	$(CC) $(CFLAGS) -o sequential src/sequential.cpp src/graph_coloring/coloring_cpu.cpp $(INCDIRS) 

clean:
	rm -f parallel sequential parallel.o coloring_cpu.o