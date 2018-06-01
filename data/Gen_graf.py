#!/usr/bin/python
import sys
import re
import numpy as np
from random import randint
file_w = open(str(sys.argv[1]),'w')
vertices = int(sys.argv[2])
edges = int(sys.argv[3])

file_w.write("p edge " + str(vertices)+" "+str(edges)+"\n")

max_edges = vertices*vertices-vertices
if edges>max_edges:
	print("Za duza liczba wierzcholokow, maks liczba to: "+str(max_edges))
	sys.exit()

matrix = np.zeros((vertices,vertices), dtype=int)
for i in range(0,edges):
	rand_1 = randint(0, vertices-1)
	rand_2 = randint(0, vertices-1)
	while(matrix[rand_1][rand_2] == 1 or rand_2==rand_1 or matrix[rand_2][rand_1] == 1):
		rand_1 = randint(0, vertices-1)
		rand_2 = randint(0, vertices-1)
	matrix[rand_1][rand_2] = 1
	matrix[rand_2][rand_1] = 1

for i in range(0,vertices):
	for j in range(0,vertices):
		if matrix[i][j] == 1:
			file_w.write("e " +str(i+1)+" "+str(j+1)+"\n")

print(matrix)
file_w.close()
