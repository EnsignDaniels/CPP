import numpy as np
import networkx as nx
from itertools import combinations

def get_line_contains_idx(substr, lines):
    idx = -1
    for line in lines:
        if line.startswith(substr):
            idx = lines.index(line)
    
    assert idx>-1, f'file contains no {substr}'
    return idx

def get_number_of_nodes(lines):
	idx = get_line_contains_idx('DIMENSION',lines)
	n = int(lines[idx].split(':')[1])
	return n

def get_number_of_clusters(lines):
	idx = get_line_contains_idx('GTSP_SETS',lines)
	m = int(lines[idx].split(':')[1])
	return m

def get_weight_matrix(lines, n):
	idx = get_line_contains_idx('EDGE_WEIGHT_SECTION',lines) + 1
	array = [[int(item) for item in lines[i].split()] for i in range(idx, idx + n)]
	matrix = np.array(array, dtype=int)
	return matrix

def get_clusters(lines, m):
	idx = get_line_contains_idx('GTSP_SET_SECTION', lines) + 1
	clusters = {}
	for i in range(idx, idx + m):
		data = [int(item) for item in lines[i].split(' ')[:-1]]
		key = data[0]
		value = data[1:]
		clusters[key] = value
	return clusters

def get_successors(lines, m):
	idx_start = get_line_contains_idx('GTSP_SET_ORDERING', lines) + 1
	idx_up_to = get_line_contains_idx('START_GROUP_SECTION', lines)

	succ = {1: [i for i in range(2, m + 1)]}
	for i in range(idx_start, idx_up_to):
		data = [int(item) for item in lines[i].split(' ')[:-1]]
		key = data[0]
		value = data[1:]
		succ[key] = value
	return succ

def create_weighted_digraph(n, matrix):
	edge_tuples = [(i + 1, j + 1, matrix[i,j]) for i in range(n) for j in range(n) if i != j]
	G = nx.DiGraph()
	G.add_weighted_edges_from(edge_tuples)
	return G

def create_partial_order(succ):
	order = nx.DiGraph()
	edge_tuples = [(key, item) for key in succ.keys() for item in succ[key]]
	order.add_edges_from(edge_tuples)
	return order

def reduce_the_order(order):
	reduced = nx.transitive_reduction(order)
	return reduced

def ordered_dict_to_str(dct):
	keys = sorted(dct.keys())
	line = ', '.join([f'({k}: {sorted(dct[k])})' for k in keys])
	return line


def parseFile(filename, verbose = False):
	with open(filename, 'r') as f:
		lines = f.read().split('\n')

	n=get_number_of_nodes(lines)
	m=get_number_of_clusters(lines)

	weights = get_weight_matrix(lines, n)
	clusters = get_clusters(lines, m)
	succ = get_successors(lines, m)

	G = create_weighted_digraph(n, weights)
	order = create_partial_order(succ)
	reduced_order = reduce_the_order(order)


	if verbose:
		print('Initial data\n===========')
		print(f'n={n}, m={m}')
		print(f'weight matrix of {weights.shape[0]} x {weights.shape[1]}:')
		print(weights)
		print(f'some arcs of G:')
		print(','.join([f'({u},{v},{w["weight"]})' for (u,v,w) in G.edges(data=True) if (u,v) in [(1,2),(1,3),(n,n-2),(n,n-1)]]))
		print(f'clusters:\n{clusters}')
		print(f'successors:\n{ordered_dict_to_str(succ)}')
		print(f'transitively reduced order:\n{reduced_order.edges}')
		print(f'transitively closed order:\n{order.edges}')
		print(f'PC density:\n{len(order.edges)}')
		
		


	return n,m, weights, G, clusters, succ, reduced_order, order

def getInstance(filename):
	n,m, weights, G, clusters, succ, reduced_order, order = parseFile(filename)
	return G, clusters, reduced_order


if __name__ == '__main__':
	parseFile('PCGLNS/PCGLNS_PCGTSP/rbg378a.pcglns', True)