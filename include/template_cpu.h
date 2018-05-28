#ifndef TEMPLATE_CPU_H
#define TEMPLATE_CPU_H

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_PREAMBLE 10000

// struktura grafu
struct graphCSR_st {
  int nvertices;
  int nedges;
  int *source_offsets;
  int *destination_indices;
};
typedef struct graphCSR_st *graphCSR_t;


graphCSR_t read_graph_DIMACS_ascii(char *file);
int get_params();
int count_occur(int a[], int num_elements, int value);
bool checkIfCorrect(graphCSR_st* graph, int *colors_h, int *degrees);

#endif