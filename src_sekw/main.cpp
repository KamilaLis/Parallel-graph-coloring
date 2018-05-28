//Katarzyna Dziewulska, Kamila Lis
//Kolorowanie Largest-First Sekwencyjne grafu nieskierowanego

#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include "template_cpu.h"

int main(int argc, char** argv)
{
	if(argc < 2){
		std::cout<<"Nie podano ścieżki do pliku z grafem.";
		return 0;
	}

	//wczytaj graf z pliku do formatu csr
	char *file = argv[1];
	graphCSR_st* graph = read_graph_DIMACS_ascii(file);

	const int n = graph->nvertices;
	const int q = graph->nedges;

	//tablica source_offsets_h jest rozmiaru n - liczba wierzchołków
	int *source_offsets_h = graph->source_offsets;

	// tablica destination_indices_h jest rozmiaru 2q = 2 * liczba krawedzi
	int *destination_indices_h = graph->destination_indices;
	
	//tablica wynikowa kolorowania
	int *colors_h = new int[n];
	//tablica stopni wierzchołków
	int *degrees = new int[n]; 
	//vector stopni wierzchołków sparowany z indeksem wierzchołka do sortowania
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


    //vector stopni sąsiadów wierzchołka
    std::vector<int> neighbours_colors;
    int color;
    int j,d;

    //Kolorowanie LF
    for(int i = 0; i < n; i++){

    	//kolorujemy wierzchołki j od najwiekszego stopnia - numer wierzchołka j
    	j = degrees_h[i].first;
    	d = degrees_h[i].second;
    	color = 0;

    	//znajdź wektor kolorów sąsiadów
    	for(int a = 0; a < d; a++){
    		neighbours_colors.push_back(colors_h[destination_indices_h[source_offsets_h[j]+a]]);
    	}

    	//Jako, że kolory to liczby posortu vector rosnąco i usuń powtarzające się kolory sąsiadów
    	std::sort(neighbours_colors.begin(),neighbours_colors.end());
    	neighbours_colors.erase( unique( neighbours_colors.begin(), neighbours_colors.end() ), neighbours_colors.end() );

    	//Szukaj najmniejszego koloru, który nie występuje u sąsiada
    	for(int v : neighbours_colors){
    		if(color == -1) continue;
    		if(color < v) break;
    		color = v+1;
    	}

    	//przypisz znaleziony kolor do wierzchołka
    	colors_h[j] = color; 

    	//wyczyść vecotor sąsiadów
    	neighbours_colors.clear();

    }


    //obliczenie liczby kolorów
    std::vector< int > colors;
    for(int i = 0; i<n; i++){
    	colors.push_back(colors_h[i]);
    }
    std::sort(colors.begin(),colors.end());
    colors.erase( unique( colors.begin(), colors.end() ), colors.end() );
    std::cout<<"Czy graf został pokolorowany poprawnie: "<<checkIfCorrect(graph, colors_h, degrees)<<std::endl;
    std::cout<<"Liczba kolorów użytych do kolorowania: "<<colors.size()<<std::endl;
  return 0;
}
