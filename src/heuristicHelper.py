import ast
import subprocess as sp
import os
from numpy import random

#####
###  Interface package to PCGLNS heuristic
###  
###


HEURISTIC_PATH = 'heuristic/'
HEURISTIC_SOLUTION_SUFFIX = '.PCGLNS.res'

SCRIPT_PATH_NAME = 'PCGLNS/runPCGLNS.jl'
SOURCE_PCGLNS_PATH ='input/'
SOURCE_PCGLNS_SUFFIX='.pcglns'
PCGLNS_ENGINE = 'julia'

SEED = -1


def getUB(task_name):
	n, m, tour, obj = parseResult(task_name)
	return obj  

def parseResult(task_name, path = HEURISTIC_PATH):
	with open(f'{path}{task_name}{HEURISTIC_SOLUTION_SUFFIX}') as f:
		ln = f.readline().strip()
		while ln:
			if ln.startswith('Vertices'):
				n = int(ln.split(':')[1].strip())
			if ln.startswith('Sets'):
				m = int(ln.split(':')[1].strip())
			elif ln.startswith('Tour Cost'):
				obj = int(round(float(ln.split(':')[1].strip()),0))
			elif ln.startswith('Tour'):
				tour_str = ln.split(':')[1].strip()
				if ']]' in tour_str:
					tour_str = tour_str[:-1]
				tour = ast.literal_eval(tour_str)
				tour.append(tour[0])
				break
			ln = f.readline()
	return n,m,tour, obj


def getMIPS(task_name, path = HEURISTIC_PATH):
	n, m, tour, obj = parseResult(task_name, path)
	x = {}
	z = {}
	# z = {v: 0 for v in range(1, n+1)}
	# x = {(v,u): 0 for v in range(1, n+1) for u in range(1, n+1)}

	for i in range(m):
		x[tour[i],tour[i+1]] = 1
		z[tour[i]] = 1
	return x,z,obj

def runHeuristic(task_name, source_path = SOURCE_PCGLNS_PATH, path = HEURISTIC_PATH, script_path = SCRIPT_PATH_NAME, seed = SEED, recalc = False, verboselvl = 3):
	result_filename = f'{path}{task_name}{HEURISTIC_SOLUTION_SUFFIX}'
	#sp.run([f'{PCGLNS_ENGINE}', f'{script_path}',f'{source_path}{task_name}{SOURCE_PCGLNS_SUFFIX}', f'-seed={seed}',  f'-output={result_filename}'])
	if recalc or not os.path.isfile(result_filename):
	 	sp.run([f'{PCGLNS_ENGINE}', f'{script_path}',f'{source_path}{task_name}{SOURCE_PCGLNS_SUFFIX}', f'-seed={seed}', f'-verbose={verboselvl}',  f'-output={result_filename}'])


def main():
	task_name = 'p3xl_1-m'
	runHeuristic(task_name)

	n,m,tour,obj = parseResult(task_name) 
	print(f'Instance {task_name}, n={n}, m={m}')
	print(f'obj={obj}, tour={tour}')

	x, z = getMIPS(task_name)
	print(f'x=\n{x}')
	print(f'z=\n{z}')

if __name__ == '__main__':
	main()
