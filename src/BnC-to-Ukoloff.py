import networkx as nx
import sys
import os
import re
from pathlib import Path

FIRST_NODE = 1

def parser(infile):
    edge_list = []
    pattern1 = re.compile("x\[(\d+),(\d+)\]\s+1")
    pattern2 = re.compile("# Objective value = (\d+)")
    with open(infile,'r') as f:
        line = f.readline()
        while line:
            match1 = re.match(pattern1, line)
            match2 = re.match(pattern2, line)
            if match1:
                t = (int(match1.groups()[0]),int(match1.groups()[1]))
                edge_list.append(t)
            if match2:
                obj = int(match2.groups()[0])
            line = f.readline()
            
    return obj, edge_list        

def solution_builder(infile):
    G=nx.DiGraph()
    obj, edge_list=parser(infile)
    G.add_edges_from(edge_list)
    cycles = nx.simple_cycles(G)
    return obj, list(cycles)[0]
    
def rewind_a_cycle(tour):
    L = len(tour)
    tries = 0
    while tries < L:
        if tour[0] == FIRST_NODE:
            return tour
        tries += 1
        tour = tour[1:] + [tour[0]]
    raise Exception(f'*** error: tour does not visit node {FIRST_NODE}')
    

def write_solution(infile, heurfile, outfile):
    obj, tour = solution_builder(infile)
    tour = rewind_a_cycle(tour)
    with open(heurfile,'r') as f:
        with open(outfile,'w') as w:
            line = f.readline()
            while line:
                if line.startswith("Comment"):
                    w.write(f'Comment          : an exact solution obtained by BnC\n')
                elif line.startswith("Solver Time"):
                    w.write(f'Solver Time      : see corresponding output file\n')
                elif line.startswith("Tour Cost"):
                    w.write(f'Tour Cost        : {obj}\n')
                elif line.startswith("Tour"):
                    w.write(f'Tour             : {tour}\n')
                else:
                    w.write(line)
                line = f.readline()


def convert_file(input_dir, filename, heur_dir, output_dir):
    if not filename.endswith(".sol"):
        print(f'Illegal filename {filename}')
        return
    
    splitted = filename.split('_')
    name = '_'.join(splitted[:-1])

    infile = input_dir + '/' + filename
    heurfile = heur_dir + '/' + name + '.PCGLNS.res'
    outfile = output_dir + '/' + name + '.result.txt' 
    
    write_solution(infile, heurfile, outfile)


def convert_dir(input_dir, heur_dir, output_dir):
    for filename in os.listdir(input_dir):
        print("Processing " + filename + "...")
        convert_file(input_dir, filename, heur_dir, output_dir)

if __name__ == '__main__':
	try:
		for arg in sys.argv:
			if '=' in arg:
				parts = arg.split('=')
				if parts[0] == '--input' or parts[0] == '-i':
					input_dir = parts[1]
				if parts[0] == '--output' or parts[0] == '-o':
					output_dir = parts[1]
				if parts[0] == '--heuristic' or parts[0] == '-h':
					heur_dir = parts[1]
	except:
		print('SYNTAX: python BnC-to-Ukoloff.py -i=<Gurobi solutions dir> -o=<Converted solutions (to .txt for Ukoloff)> -h=<Dir with PCGLNS heuristic solutions>')

	convert_dir(input_dir,heur_dir,output_dir) 