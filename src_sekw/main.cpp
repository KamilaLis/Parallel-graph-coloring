//Kolorowanie Largest-First Sekwencyjne grafu nieskierowanego

#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include "template_cpu.h"


bool checkIfCorrect(graphCSR_st* graph, int *colors_h, int *degrees){
	
	bool correct = true;
	int d;

	for(int i = 0; i < graph->nvertices; i++){
		d = degrees[i];
		//vector kolorów sąsiadów
		for(int a = 0; a < d; a++){
    		if(colors_h[graph->destination_indices[graph->source_offsets[i]+a]] == colors_h[i]) correct = false;
    	}
	}
	return correct;
}

int main(int argc, char** argv)
{
	if(argc < 1){
		std::cout<<"Nie podano pliku.";
	}

	char *file = argv[1];
	graphCSR_st* graph = read_graph_DIMACS_ascii(file);

	const int n = graph->nvertices;
	const int q = graph->nedges;
	//tablica source_offsets_h jest rozmiaru n - liczba wierzchołków
	int *source_offsets_h = graph->source_offsets;
	// tablica destination_indices_h jest rozmiaru 2q = 2 * liczba krawedzi
	int *destination_indices_h = graph->destination_indices;
	int *colors_h = new int[n];
	int *degrees = new int[n]; 

	std::vector< std::pair <int,int> > degrees_h;

	for( int i = 0; i < n; i++){
    	// niepokolorowane wierzcholki - wstepnie wszystkie -1
    	colors_h[i] = -1;
    	// okreslenie stopni wierzcholkow
    	degrees_h.push_back( std::make_pair(i,source_offsets_h[i+1]-source_offsets_h[i]) );
    	degrees[i] = source_offsets_h[i+1]-source_offsets_h[i];
    }

    //sortowanie nierosnąco według stopni wierzchołków
    std::sort(degrees_h.begin(), degrees_h.end(), [](const std::pair <int,int>& a, const std::pair <int,int>& b) {return a.second > b.second; });
	
	/*
    for(int i = 0; i < 2*q; i++){
    	std::cout<< destination_indices_h[i]<<std::endl;
    }*/

    std::vector<int> neighbours_colors;
    int color;
    int j,d;
    //Kolorowanie LF
    for(int i = 0; i < n; i++){
    	//kolorujemy wierzchołki j - numer wierzchołka i index w niektórych tablicach
    	j = degrees_h[i].first;
    	d = degrees_h[i].second;
    	color = 0;
    	//znajdź wektor kolorów sąsiadów
    	for(int a = 0; a < d; a++){
    		neighbours_colors.push_back(colors_h[destination_indices_h[source_offsets_h[j]+a]]);
    	}
    	std::sort(neighbours_colors.begin(),neighbours_colors.end());
    	neighbours_colors.erase( unique( neighbours_colors.begin(), neighbours_colors.end() ), neighbours_colors.end() );

    	for(int v : neighbours_colors){
    		std::cout<<"colory "<<v<<" ";
    		if(color == -1) continue;
    		if(color < v) break;
    		color = v+1;
    	}
    	std::cout<<std::endl;
    	colors_h[j] = color; 
    	neighbours_colors.clear();

    	std::cout<<"znalezione kolory:"<<std::endl;
        for(int i = 0; i < n; i++){
    	std::cout<< colors_h[i]<<std::endl;
    }

    }

    std::cout<<"czy poprawnie "<<checkIfCorrect(graph, colors_h, degrees);
  return 0;
}
