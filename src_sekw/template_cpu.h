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

static char Preamble[MAX_PREAMBLE];
int Nr_vert, Nr_edges;


graphCSR_t read_graph_DIMACS_ascii(char *file);
int get_params();
int count_occur(int a[], int num_elements, int value);
bool checkIfCorrect(graphCSR_st* graph, int *colors_h, int *degrees);

#include "template_cpu.h"

// Funkcja sprawdzająca poprawność kolorowania
bool checkIfCorrect(graphCSR_st* graph, int *colors_h, int *degrees){
	
	bool correct = true;
	int d;

	for(int i = 0; i < graph->nvertices; i++){
		d = degrees[i];

		//sprawdza każdego kolor każdego sąsiada dla każdego z wierzchołków
		for(int a = 0; a < d; a++){
    		if(colors_h[graph->destination_indices[graph->source_offsets[i]+a]] == colors_h[i]) correct = false;
    	}
	}
	return correct;
}

graphCSR_t read_graph_DIMACS_ascii(char *file)
{
	int c, oc;
	char * pp = Preamble;
	int i,j, nnz=0, old_i=0;
    int line_idx = 0, offset_idx = 0;
	FILE *fp;
	graphCSR_t graph = (graphCSR_t) malloc(sizeof(struct graphCSR_st));
	int *source_offsets_h, *destination_indices_h;

	if ( (fp=fopen(file,"r"))==NULL )
	  { printf("ERROR: Cannot open infile\n"); exit(10); }

	for(oc = '\0' ;(c = fgetc(fp)) != EOF && (oc != '\n' || c != 'e')
		; oc = *pp++ = c);

	ungetc(c, fp);
	*pp = '\0';
	get_params();

    source_offsets_h = (int*) malloc((Nr_vert+1)*sizeof(int));
    destination_indices_h = (int*) malloc((2*Nr_edges)*sizeof(int));
    source_offsets_h[0] = 0;
    source_offsets_h[Nr_vert] = 2*Nr_edges;

    while ((c = fgetc(fp)) != EOF){
    	switch (c)
    	{
    		case 'e':
    			if (!fscanf(fp, "%d %d", &i, &j))
    			{ printf("ERROR: corrupted inputfile\n"); exit(10); }
    			// poniewaz graf jest nieskierowany
    			// tablica krawedzi jest 2x wieksza
    			destination_indices_h[line_idx] = j-1;
    			if((i-1)!=old_i){
    				offset_idx++;
    				source_offsets_h[offset_idx] = source_offsets_h[offset_idx-1]+nnz;
    				nnz = 0;
    				old_i = i-1;
    			}
    			nnz++;
    			line_idx++;
    			break;
    		case '\n':
    		default: break;
		}
	}

	offset_idx++;
	source_offsets_h[offset_idx] = source_offsets_h[offset_idx-1]+nnz;
	fclose(fp);

//	printf("source_offsets_h\n");
//	for (i = 0; i<Nr_vert+1; i++)  printf("%d\n",source_offsets_h[i]); printf("\n");
//	printf("destination_indices_h\n");
//	for (i = 0; i<2*Nr_edges; i++)  printf("%d\n",destination_indices_h[i]); printf("\n");

	graph->nvertices = Nr_vert;
	graph->nedges = Nr_edges;
	graph->source_offsets = source_offsets_h;
	graph->destination_indices = destination_indices_h;

	printf("> Graph loaded.\n");
	return graph;
}


////////////////////////////////////////////////////////////////////////////////
//! getting Nr_vert and Nr_edge from the preamble string "p ??? num num"
////////////////////////////////////////////////////////////////////////////////
int get_params()
{
	char c, *tmp;
	char * pp = Preamble;
	int stop = 0;
	tmp = (char *)calloc(100, sizeof(char));

	Nr_vert = Nr_edges = 0;

	while (!stop && (c = *pp++) != '\0'){
		switch (c)
		  {
			case 'c':
			  while ((c = *pp++) != '\n' && c != '\0');
			  break;

			case 'p':
			  sscanf(pp, "%s %d %d\n", tmp, &Nr_vert, &Nr_edges);
			  stop = 1;
			  break;

			default:
			  break;
		  }
	}

	free(tmp);

	if (Nr_vert == 0 || Nr_edges == 0)
	  return 0;  /* error */
	else
	  return 1;

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

#endif