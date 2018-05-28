/*
 * Katarzyna Dziewulska, Kamila Lis
 * Równoległe kolorowanie grafu metoda LF (Largest First)
 */

#define MAX_THREATS_PER_BLOCK 1024

// system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>


// struktura grafu
struct graphCSR_st {
  int nvertices;
  int nedges;
  int *source_offsets;
  int *destination_indices;
};
typedef struct graphCSR_st *graphCSR_t;

void colorLF(graphCSR_t graph);
int count_occur(int a[], int num_elements, int value);
int maxValue(int a[], int num_elements);
graphCSR_t read_graph_DIMACS_ascii(char *file);

////////////////////////////////////////////////////////////////////////////////
//! Funkcja kolorowania wykonywana na GPU
//! (wykonywana dla kazdego wierzcholka przez oddzielny watek)
//! @param n                liczba wierzcholkow grafu
//! @param c                aktualny (najmniejszy) kolor
//! @param source_offsets   -> patrz struktura grafu
//! @param destination_indices
//! @param colors           kolory do tej pory przydzielone wierzcholkom
//! @param randoms          liczby losowe przydzielone wierzcholkom
//! @param degrees          stopnie wierzcholkow
//! @param out_colors       wartosc zwracana - kolory przydzielone wierzcholkom
////////////////////////////////////////////////////////////////////////////////
__global__ void
colorLFkernel(int n, int c, int* source_offsets, int* destination_indices,
			  int* colors, int* randoms, int* degrees, int* out_colors)
{
	//dodane 
    int neighbours_colors_flags* = (int*) malloc((n)*sizeof(int)) //zainicjalizować zerami;
    
    const int idx = threadIdx.x+blockIdx.x*blockDim.x;

	bool has_max_deg=true;

	if(idx < n){
		// ignoruj, jesli wierzcholek jest juz pokolorowany
		if ((colors[idx] != -1)) return;

		int ir = randoms[idx];
		int ideg = degrees[idx];

		// sprawdz stopnie sasiadow
		for (int k = source_offsets[idx]; k < source_offsets[idx+1]; k++) {
			// ignoruj pokolorowane wierzcholki i siebie
			int j = destination_indices[k];
			int jc = colors[j];
			if ((jc != -1) || (idx == j)) continue;
			if (ideg < degrees[j]) has_max_deg=false;
			if (ideg == degrees[j]){
				if (ir <= randoms[j]) has_max_deg=false;
			}
		}
	__syncthreads();

	// przydziel kolor

	if (has_max_deg) {
    //znajdź wektor kolorów sąsiadów
        for(int a = 0; a < ideg; a++){
            neighbours_colors_flags[colors_h[destination_indices_h[source_offsets_h[idx]+a]]]=1;
        }
        for(int b = 0; b < n; b++){
            if(neighbours_colors_flags[b] == 0)colors[idx] = b; 
        }    

    colors[idx] = c;
	out_colors[idx] = colors[idx];

}
	__syncthreads();
	}
}

////////////////////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    // wczytaj nazwe pliku .col
    char* f;
    if(!getCmdLineArgumentString(argc, (const char **)argv,"file=",&f)){
        printf("> Path to file required as command-line argument.\n");
        printf("> Waiving.\n");
        exit(EXIT_WAIVED);
    }

    // znajdz GPU
    int cuda_device = 0;
    cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));
    printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // wczytaj graf
	graphCSR_t graph = read_graph_DIMACS_ascii(f);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

	colorLF(graph);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Funkcja kolorująca graf wykonywana na CPU 
//! (uruchamia funkcje wykonywana na GPU -> colorLFkernel)
//! @param graph    struktura grafu do kolorowania
////////////////////////////////////////////////////////////////////////////////
void
colorLF(graphCSR_t graph)
{
	const int n = graph->nvertices,
			  q = graph->nedges;
	int *source_offsets_h = graph->source_offsets,
		*destination_indices_h = graph->destination_indices;

	int i, c, *colors_h, *out_colors_h, *randoms, *degrees_h;
	int num_threads, num_blocks;

    // inicjalizacja zmiennych CPU (host)
    colors_h = (int*) malloc((n)*sizeof(int));
    randoms = (int*) malloc((n)*sizeof(int));
    degrees_h = (int*) malloc((n)*sizeof(int));
    out_colors_h = (int*) malloc((n)*sizeof(int));


    for(i = 0; i < n; i++){
    	// niepokolorowane wierzcholki - wstepnie wszystkie -1
    	colors_h[i] = -1;
    	// przydzielenie wartosci losowej - na razie indeks
    	randoms[i] = i;
    	// okreslenie stopni wierzcholkow
    	degrees_h[i] = source_offsets_h[i+1]-source_offsets_h[i];
    }

    // inicjalizacja zmiennych GPU (device)
    int *source_offsets_d, *destination_indices_d;
    int *colors_d, *randoms_d, *degrees_d;
    int *out_colors_d;

    checkCudaErrors(cudaMalloc((void **) &source_offsets_d, (n+1)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &destination_indices_d, (2*q)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &colors_d, (n)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &out_colors_d, (n)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &randoms_d, (n)*sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &degrees_d, (n)*sizeof(int)));

    // kopiowanie na GPU
    checkCudaErrors(cudaMemcpy(source_offsets_d, source_offsets_h, (n+1)*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(destination_indices_d, destination_indices_h, 2*q*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colors_d, colors_h, (n)*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(randoms_d, randoms, (n)*sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(degrees_d, degrees_h, (n)*sizeof(int),
                               cudaMemcpyHostToDevice));

    // liczba watkow i blokow
    if(n > MAX_THREATS_PER_BLOCK){
    	num_threads = MAX_THREATS_PER_BLOCK;
        num_blocks = (n/num_threads)+1;
    }
    else{
    	num_threads = n;
        num_blocks = 1;
    }


    c=0;
    // algorytm docelowy:
    printf("> Coloring...\n");
    while(c <= n)
    {
    	colorLFkernel<<<num_blocks,num_threads>>>(n,c,
    											 source_offsets_d,
    											 destination_indices_d,
    											 colors_d,
    											 randoms_d,
    											 degrees_d,
    											 out_colors_d);

    	++c;
        // kopiuj wynik z GPU
        checkCudaErrors(cudaMemcpy(out_colors_h, out_colors_d, (n)*sizeof(int),
                                   cudaMemcpyDeviceToHost));
        if(count_occur(out_colors_h, n, -1) == 0) break;
    }
	// zaczekaj na wyniki obliczen GPU
	cudaDeviceSynchronize();

    // wyswietlenie wyniku
    printf("Computed colors:\n");
    for (i = 0; i<n; i++)  printf("%d\n",out_colors_h[i]); printf("\n");
    printf("Number of used colors: %d\n", maxValue(out_colors_h, n));
    printf("\n> Done!\n");

    // sprzatanie
    free(source_offsets_h);
    free(destination_indices_h);
    free(randoms);
    free(colors_h);
    checkCudaErrors(cudaFree(source_offsets_d));
    checkCudaErrors(cudaFree(destination_indices_d));
    checkCudaErrors(cudaFree(colors_d));
    checkCudaErrors(cudaFree(randoms_d));
    checkCudaErrors(cudaFree(degrees_d));
    checkCudaErrors(cudaFree(out_colors_d));
}