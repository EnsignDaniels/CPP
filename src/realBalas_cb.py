import networkx as nx
from itertools import combinations
from gurobiMTZHelper import addRealBalas_cuts
from parameters import *
import numpy as np


# pi_node_dict = {}
# sigma_node_dict = {}
# pi_sigma_node_dict = {}

# global_cuts = set()

# def getPiAuxGraphNodes(cluster, clusters, tree_closure, depot_cluster):
#     global pi_node_dict
#     if not cluster in pi_node_dict:
#         pi_node_dict[cluster] = [cl for cl in clusters if not cl in tree_closure.predecessors(cluster)] + [depot_cluster]
#     return pi_node_dict[cluster]

# def getSigmaAuxGraphNodes(cluster, clusters, tree_closure):
#     global sigma_node_dict
#     if not cluster in sigma_node_dict:
#         sigma_node_dict[cluster] = [cl for cl in clusters if not cl in tree_closure.successors(cluster)]
#     return sigma_node_dict[cluster]

# def getPiSigmaAuxGraphNodes(cluster1, cluster2, clusters, tree_closure):
#     global pi_sigma_node_dict
#     if not (cluster1, cluster2) in pi_sigma_node_dict:
#         pi_sigma_node_dict[cluster1, cluster2] = [cl for cl in clusters if not cl in tree_closure.predecessors(cluster1) and not cl in tree_closure.successors(cluster2)]
#     return pi_sigma_node_dict[cluster1, cluster2]


global_auxGraphsPi = {}
global_auxGraphsSigma = {}
global_auxGraphsPiSigma = {}



def createClusterGraphPi(cluster, clusters, tree_closure, wrapped_u, depot_cluster):
    auxgraph = None
    if cluster in global_auxGraphsPi:
        auxgraph = global_auxGraphsPi[cluster]
        for e in auxgraph.edges(data=True):
            e[2]['capacity'] = wrapped_u[e[:-1]]
    else:  
        cl_star = [cl for cl in clusters if not cl in tree_closure.predecessors(cluster)] + [depot_cluster]
        auxgraph = nx.DiGraph()
        auxgraph.add_nodes_from(cl_star)
        #################################################
        edges = [(e[0],e[1],{'capacity': wrapped_u[e]}) for e in wrapped_u if (e[0] in cl_star) and (e[1] in cl_star)]
        auxgraph.add_edges_from(edges)
        #################################################
        # for e in wrapped_u.keys():
        #     if (e[0] in cl_star) and (e[1] in cl_star):
        #         auxgraph.add_edge(*e, capacity = wrapped_u[e])
        #################################################
        # for e in combinations(auxgraph.nodes, 2):
        #     if e in wrapped_u:
        #         auxgraph.add_edge(*e, capacity=wrapped_u[e])
        #     e = e[::-1]
        #     if e in wrapped_u:
        #         auxgraph.add_edge(*e, capacity=wrapped_u[e])
        global_auxGraphsPi[cluster] = auxgraph
    return auxgraph



def createClusterGraphSigma(cluster, clusters, tree_closure, wrapped_u):
    auxgraph = None
    if cluster in global_auxGraphsSigma:
        auxgraph = global_auxGraphsSigma[cluster]
        for e in auxgraph.edges(data=True):
            e[2]['capacity'] = wrapped_u[e[:-1]]
    else:
        cl_star = [cl for cl in clusters if not cl in tree_closure.successors(cluster)]
        auxgraph = nx.DiGraph()
        auxgraph.add_nodes_from(cl_star)
        ##################################################
        edges = [(e[0],e[1],{'capacity': wrapped_u[e]}) for e in wrapped_u if (e[0] in cl_star) and (e[1] in cl_star)]
        auxgraph.add_edges_from(edges)
        ##################################################
        # for e in wrapped_u.keys():
        #     if (e[0] in cl_star) and (e[1] in cl_star):
        #         auxgraph.add_edge(*e, capacity = wrapped_u[e])
        ##################################################
        # for e in combinations(auxgraph.nodes, 2):
        #     if e in wrapped_u:
        #         auxgraph.add_edge(*e, capacity=wrapped_u[e])
        #     e = e[::-1]
        #     if e in wrapped_u:
        #         auxgraph.add_edge(*e, capacity=wrapped_u[e])
        global_auxGraphsSigma[cluster] = auxgraph
    return auxgraph



def createClusterGraphPiSigma(cluster1, cluster2, clusters, tree_closure, wrapped_u):
    auxgraph = None
    if (cluster1,cluster2) in global_auxGraphsPiSigma:
        auxgraph = global_auxGraphsPiSigma[(cluster1,cluster2)]
        for e in auxgraph.edges(data=True):
            e[2]['capacity'] = wrapped_u[e[:-1]]
    else:        
        cl_star = [cl for cl in clusters if not cl in tree_closure.predecessors(cluster1) and not cl in tree_closure.successors(cluster2)]
        auxgraph = nx.DiGraph()
        auxgraph.add_nodes_from(cl_star)
        ##################################################
        edges = [(e[0],e[1],{'capacity': wrapped_u[e]}) for e in wrapped_u if (e[0] in cl_star) and (e[1] in cl_star)]
        auxgraph.add_edges_from(edges)
        ##################################################
        # for e in wrapped_u.keys():
        #     if (e[0] in cl_star) and (e[1] in cl_star):
        #         auxgraph.add_edge(*e, capacity = wrapped_u[e])
        ##################################################
        # for e in combinations(auxgraph.nodes, 2):
        #     if e in wrapped_u:
        #         auxgraph.add_edge(*e, capacity=wrapped_u[e])
        #     e = e[::-1]
        #     if e in wrapped_u:
        #         auxgraph.add_edge(*e, capacity=wrapped_u[e])
        global_auxGraphsPiSigma[(cluster1,cluster2)] = auxgraph
    return auxgraph
    
    

def createConstraintPi(auxgraph,cluster, tree_closure, ampl, depot_cluster):
    cut_val, part = nx.minimum_cut(auxgraph,cluster,depot_cluster)
    if cut_val < 1 - ampl * EPSILON:
        U,V = part
        Pi_U = set([cl for cl in U if set(tree_closure.successors(cl)).intersection(U)])    
        # cut = tuple(sorted(list(U - Pi_U))), tuple(sorted(list(V - Pi_U)))
        return tuple(U - Pi_U), tuple(V - Pi_U)
    else:
        return None          
    
def createConstraintSigma(auxgraph,cluster, tree_closure, ampl, depot_cluster):
    cut_val, part = nx.minimum_cut(auxgraph,depot_cluster,cluster)
    if cut_val < 1 - ampl * EPSILON:
        U,V = part
        Sigma_V = set([cl for cl in V if set(tree_closure.predecessors(cl)).intersection(V)])
        return tuple(U - Sigma_V), tuple(V - Sigma_V)
    else:
        return None         


def createConstraintPiSigma(auxgraph, cluster1, cluster2, ampl):
    cut_val, part = nx.minimum_cut(auxgraph, cluster1, cluster2)
    if cut_val < 1 - ampl * EPSILON:
        U, V = part
        return tuple(U), tuple(V)
    else:
        return None

def init_ratings(model, depot_cluster=1):
    clusters = model._clusters
    tree = model._tree
    model._PiRates = {cl: 1 for cl in clusters if cl != depot_cluster}
    model._SigmaRates = {cl: 1 for cl in clusters if cl != depot_cluster}
    model._PiSigmaRates = {(c1,c2): 1 for (c1,c2) in tree.edges if c1 != depot_cluster}
    #print(model._PiRates, model._SigmaRates, model._PiSigmaRates)


###### Clusters should not contain depot  ########### 
def singlecluster_sampling(clusters, rates, samplesize):
    rates_array = np.array([rates[c] for c in clusters])
    rates_array = rates_array / np.sum(rates_array)
    return np.random.choice(clusters, samplesize, replace=False, p=rates_array)

###### Order edges should not be incident to depot ###########
def doublecluster_sampling(edges, rates, samplesize):
    rates_array = np.array([rates[e] for e in edges])
    rates_array = rates_array / np.sum(rates_array)
    edges_index = list(range(len(edges)))
    sample = np.random.choice(edges_index, samplesize, replace=False, p=rates_array)
    return [edges[index] for index in sample]



def generatePi_cuts(model,ampl=1, depot_cluster=1, need_sampling=False):
    clusters = model._clusters
    tree_closure = model._tree_closure
    wrapped_u = model.cbGetNodeRel(model._u)
    rates = model._PiRates
    cluslist = [c for c in clusters if c != depot_cluster]
    if need_sampling:
        cluslist = singlecluster_sampling(cluslist, rates, PI_SAMPLE_SIZE)

    
    cuts = set()
    for clus in cluslist:
        aux_G = createClusterGraphPi(clus, clusters, tree_closure, wrapped_u, depot_cluster)
        cut = createConstraintPi(aux_G, clus, tree_closure, ampl, depot_cluster)
        if cut:
            rates[clus] += PI_RATES_STEP
            cuts.add(cut)
    number_of_cuts = len(cuts)
    #return cuts, number_of_cuts
    new_model = model
    if cuts:
        new_model = addRealBalas_cuts(model,cuts)
    return new_model, number_of_cuts



def generateSigma_cuts(model,ampl=1,depot_cluster=1, need_sampling=False):
    clusters = model._clusters
    tree_closure = model._tree_closure
    wrapped_u = model.cbGetNodeRel(model._u)
    rates = model._SigmaRates
    cluslist = [c for c in clusters if c != depot_cluster]
    if need_sampling:
        cluslist = singlecluster_sampling(cluslist, rates, SIGMA_SAMPLE_SIZE)

    cuts = set()
    for clus in cluslist:
        aux_G = createClusterGraphSigma(clus, clusters, tree_closure, wrapped_u)
        cut = createConstraintSigma(aux_G, clus, tree_closure, ampl, depot_cluster)
        if cut:
            rates[clus] += SIGMA_RATES_STEP
            cuts.add(cut)
    number_of_cuts = len(cuts)
    #return cuts, number_of_cuts
    new_model = model
    if cuts:
        new_model = addRealBalas_cuts(model,cuts)
    return new_model, number_of_cuts


def generatePiSigma_cuts(model,ampl=1, depot_cluster=1, need_sampling=False):
    clusters = model._clusters
    tree = model._tree
    tree_closure = model._tree_closure
    rates = model._PiSigmaRates
    edges = [(c1,c2) for (c1,c2) in tree.edges if c1 != depot_cluster]
    if need_sampling:
        edges = doublecluster_sampling(edges, rates, PI_SIGMA_SAMPLE_SIZE)


    wrapped_u = model.cbGetNodeRel(model._u)
    cuts = set()
    for cluster1, cluster2 in edges:
        # if cluster1 == depot_cluster:
        #     continue
        aux_G = createClusterGraphPiSigma(cluster1, cluster2, clusters, tree_closure, wrapped_u)
        cut = createConstraintPiSigma(aux_G, cluster1, cluster2,ampl)
        if cut:
            rates[(cluster1,cluster2)] += PI_SIGMA_RATES_STEP
            cuts.add(cut)
    number_of_cuts = len(cuts)
    #return cuts, number_of_cuts
    new_model = model
    if cuts:
        new_model = addRealBalas_cuts(model,cuts)
    return new_model, number_of_cuts














