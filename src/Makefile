CC=g
NVCC=nvcc
CUDAFLAGS= -g -G 
GENCODE_FLAGS += -gencode arch=compute_35,code=sm_35
LIBDIRS=-L/usr/local/cuda-9.1/lib64
INCDIRS=-I/usr/local/cuda-9.1/samples/common/inc -I/usr/local/cuda-9.1/include

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

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif


# Target rules
all: build

build: template

template.o:template.cu
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

template_cpu.o:template_cpu.cpp
	$(NVCC) $(INCDIRS) $(CUDAFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

template: template.o template_cpu.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LIBDIRS)

run: build
	$(EXEC) ./template

clean:
	rm -f template template.o template_cpu.o