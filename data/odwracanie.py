#!/bin/sh
import sys
import re
file_r = open(str(sys.argv[1]),'a+')

lines=file_r.readlines()

tablica = re.findall(r'\d+', lines[2])
print(tablica)
edges = int(tablica[1])*2
nodes = int(tablica[0])

for i in range(4,len(lines)):
	tablica = re.findall(r'\d+', lines[i])
	file_r.write(str(tablica[1])+" "+str(tablica[0])+"\n") 

file_r.close()
fil = open(str(sys.argv[1])+"2",'a')
file_r = open(str(sys.argv[1]),'r')
lines=file_r.readlines()

fil.write("p edge "+str(nodes)+" "+str(edges) + "\n")
for i in range(4,len(lines)):
	fil.write("e " + lines[i])

fil.close()
file_r.close()
