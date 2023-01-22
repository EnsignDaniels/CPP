import networkx as nx
from itertools import combinations

######
###  Input:
### 			G - networkx DiGraph (without self-loops)
###	  	 clusters - dict clusterID -> [nodes]  	
###
###  Output:
###				G without intra-cluster arcs
######

def excludeInnerArcs(G, clusters):
	for c in clusters:
		for e in combinations(clusters[c],2):
			to_exclude = [e,e[::-1]]
			G.remove_edges_from(to_exclude)
	return G

####### Removing first set of arcs according to the paper, 
####### i.e. arcs (1,3), (1,6) that
####### shortcuts the initial ordering
###			Input:
### 			G - networkx DiGraph (without self-loops)
###	  	 clusters - dict clusterID -> [nodes]
###  	tree - initial tree
###		tree_closure - partial ordering tree
### 	Output:  G without shortcuts


def excludePCViolation1(G, clusters, tree, tree_closure):
	for c1, c2 in combinations(clusters,2):
		if tree_closure.has_edge(c1,c2) and not tree.has_edge(c1,c2):
			to_exclude = [(nd1,nd2) for nd1 in clusters[c1] for nd2 in clusters[c2]]
			G.remove_edges_from(to_exclude)
		if tree_closure.has_edge(c2,c1) and not tree.has_edge(c2,c1):
			to_exclude = [(nd2,nd1) for nd1 in clusters[c1] for nd2 in clusters[c2]]
			G.remove_edges_from(to_exclude)
	return G


####### Removing second set of arcs according to the paper,
###			Input:  
### 			G - networkx DiGraph (without self-loops)
###	  	 clusters - dict clusterID -> [nodes]
###  	tree - initial tree
###		tree_closure - partial ordering tree
###		first_cluster_id - id of the first cluster (default is 1)
### 	Output:  G without back routing
### 	we cannot return to the source before we have reached the leaf in our ordering tree

def excludePCViolation2(G, clusters, tree_closure, first_cluster_id=1):
	for c2 in clusters:
		if c2!=1 and len(list(tree_closure.successors(c2)))>0:
			to_exclude = [(nd2,nd1) for nd1 in clusters[first_cluster_id] for nd2 in clusters[c2]]
			G.remove_edges_from(to_exclude)
	return G

####### Removing third set of arcs according to the paper, 
###		Input:
### 			G - networkx DiGraph (without self-loops)
###	  	 clusters - dict clusterID -> [nodes]
###  	tree - initial tree
###		tree_closure - partial ordering tree
### 	Output:  G without ordering violations
### 	we cannot return to the source before we have reached the leaf in our ordering tree

def excludePCViolation3(G, clusters, tree_closure):
	for c1, c2 in combinations(clusters,2):
		if tree_closure.has_edge(c2,c1):
			to_exclude = [(nd1,nd2) for nd1 in clusters[c1] for nd2 in clusters[c2]]
			G.remove_edges_from(to_exclude)
		if tree_closure.has_edge(c1,c2) and c1 != 1:
			to_exclude = [(nd2,nd1) for nd1 in clusters[c1] for nd2 in clusters[c2]]
			G.remove_edges_from(to_exclude)

	return G





def preprocessInstance(G,clusters, tree):
	G_prime = excludeInnerArcs(G,clusters)
	tree_closure = nx.transitive_closure_dag(tree)
	G_prime = excludePCViolation1(G_prime, clusters, tree, tree_closure)
	G_prime = excludePCViolation2(G_prime, clusters, tree_closure)
	G_prime = excludePCViolation3(G_prime, clusters, tree_closure)
	
	return G_prime, clusters, tree, tree_closure


