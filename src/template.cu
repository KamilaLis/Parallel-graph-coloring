/*
 * Katarzyna Dziewulska, Kamila Lis
 * Kolorowanie grafu metoda LF
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
// includes nvGRAPH
#include "nvgraph.h"
#include <thrust/count.h>


void colorLF();
void runTest(int argc, char **argv);
void check_status(nvgraphStatus_t status);
int count_occur(int a[], int num_elements, int value);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float *g_idata, float *g_odata)
{
	printf("in testKernel...\n\n");
	// shared memory
    // the size is determined by the host application
    extern  __shared__  float sdata[];

    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // read in input data from global memory
    sdata[tid] = g_idata[tid];
    __syncthreads();

    // perform some computations
    sdata[tid] = (float) num_threads * sdata[tid];
    __syncthreads();

    // write data to global memory
    g_odata[tid] = sdata[tid];
}



__global__ void
colorLFkernel(int n, int c, int* source_offsets, int* destination_indices,
			  int* colors, int* randoms, int* out_colors)
{
	const int idx = threadIdx.x+blockIdx.x*blockDim.x;
//	printf("index: %d\n",idx);
//	printf("threats in block: %d\n", blockDim.x);

	bool f=true; // true if f you have max random

	if(idx < n){
		// ignore nodes colored earlier
		if ((colors[idx] != -1)) return;

		int ir = randoms[idx];
//		printf("my random: %d\n", ir);

		// look at neighbors to check their random number
		for (int k = source_offsets[idx]; k < source_offsets[idx+1]; k++) {
		// ignore nodes colored earlier (and yourself)
		int j = destination_indices[k];
		int jc = colors[j];
		if ((jc != -1) || (idx == j)) continue;

		int jr = randoms[j];
//		printf("neighbour random:%d\n", jr);
		if (ir <= jr) f=false;
	}
	__syncthreads();

	// assign color if you have the maximum random number
	if (f) colors[idx] = c;
	out_colors[idx] = colors[idx];
	__syncthreads();
	}
//	int i;
//    for (i = 0; i<n; i++)  printf("%d\n",colors[i]); printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
//    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
//    int cuda_device = 0;
//    cuda_device = findCudaDevice(argc, (const char **)argv);
//    cudaDeviceProp deviceProp;
//    checkCudaErrors(cudaGetDevice(&cuda_device));
//    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
//    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
//           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);
//    if (deviceProp.major < 3)
//    {
//        printf("> nvGraph requires device SM 3.0+\n");
//        printf("> Waiving.\n");
//        exit(EXIT_WAIVED);
//    }
//
//    StopWatchInterface *timer = 0;
//    sdkCreateTimer(&timer);
//    sdkStartTimer(&timer);

    // let's color
	colorLF();

//    sdkStopTimer(&timer);
//    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
//    sdkDeleteTimer(&timer);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
colorLF()
{
	const size_t  n = 5, q = 8;
	int *source_offsets_h, *destination_indices_h;
	int i, c, *colors_h, *randoms;

    // inicjalizacja zmiennych CPU (host)
    source_offsets_h = (int*) malloc((n+1)*sizeof(int));
    destination_indices_h = (int*) malloc(q*sizeof(int));
    colors_h = (int*) malloc((n)*sizeof(int));
    randoms = (int*) malloc((n)*sizeof(int));

    // allocate mem for the result on host side
    int *out_colors_h = (int*) malloc((n)*sizeof(int));


    source_offsets_h [0] = 0;
    source_offsets_h [1] = 3;
    source_offsets_h [2] = 5;
    source_offsets_h [3] = 7;
    source_offsets_h [4] = 8;
    source_offsets_h [5] = 8;

    destination_indices_h [0] = 2;
    destination_indices_h [1] = 1;
    destination_indices_h [2] = 3;
    destination_indices_h [3] = 2;
    destination_indices_h [4] = 3;
    destination_indices_h [5] = 3;
    destination_indices_h [6] = 4;
    destination_indices_h [7] = 4;

    colors_h[0] = -1;
    colors_h [1] = -1;
    colors_h [2] = -1;
    colors_h [3] = -1;
    colors_h [4] = -1;

    randoms [0] = 0;
    randoms [1] = 1;
    randoms [2] = 2;
    randoms [3] = 3;
    randoms [4] = 4;

    // inicjalizacja zmiennych GPU (device)
    int *source_offsets_d, *destination_indices_d;
    int *colors_d, *randoms_d;
    int *out_colors_d;

    checkCudaErrors(cudaMalloc((void **) &source_offsets_d, (n+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &destination_indices_d, q*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &colors_d, (n)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &out_colors_d, (n)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &randoms_d, (n)*sizeof(int)));

    // kopiowanie na GPU
    checkCudaErrors(cudaMemcpy(source_offsets_d, source_offsets_h, (n+1)*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(destination_indices_d, destination_indices_h, q*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colors_d, colors_h, (n)*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(randoms_d, randoms, (n)*sizeof(int),
                               cudaMemcpyHostToDevice));

    // liczba watkow i blokow
	int num_threads = 1024;
    int num_blocks = (n/num_threads)+1;

    c=0;
    // algorytm docelowy:
    while(c <= n)
    {
    	colorLFkernel<<<1,n>>>(n, c, source_offsets_d, destination_indices_d,
    						   colors_d, randoms_d, out_colors_d);

    	++c;
        // copy result from device to host
        checkCudaErrors(cudaMemcpy(out_colors_h, out_colors_d, (n)*sizeof(int),
                                   cudaMemcpyDeviceToHost));
        if(count_occur(out_colors_h, n, -1) == 0) break;
    }
	// zaczekaj na wyniki obliczen GPU
	cudaDeviceSynchronize();

    // wyswietlenie wyniku
    printf("out_colors_h\n");
    for (i = 0; i<n; i++)  printf("%d\n",out_colors_h[i]); printf("\n");
    printf("\nDone!\n");

    // sprzatanie
    free(source_offsets_h);
    free(destination_indices_h);
    free(randoms);
    free(colors_h);
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int num_threads = 32;
    unsigned int mem_size = sizeof(float) * num_threads;

    // allocate host memory
    float *h_idata = (float *) malloc(mem_size);

    // initalize the memory
    for (unsigned int i = 0; i < num_threads; ++i)
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size,
                               cudaMemcpyHostToDevice));

    // allocate device memory for result
    float *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(mem_size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,
                               cudaMemcpyDeviceToHost));

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // compute reference solution
    float *reference = (float *) malloc(mem_size);
    computeGold(reference, h_idata, num_threads);

    // check result
    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    {
        // write file for regression test
        sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
    }
    else
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected solution
        bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
    }

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
void
check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//!
////////////////////////////////////////////////////////////////////////////////
int count_occur(int a[], int num_elements, int value)
{
    int i, count = 0;
    for (i = 0; i < num_elements; i++)
    {
        if (a[i] == value)
        {
            ++count;
        }
    }
    return (count);
}
