import networkx as nx

from gurobiMTZHelper import addGBCP_cuts
from parameters import *
import numpy as np
import math
from itertools import islice

from sampler_v2 import get_address_pair_sample, update_rates, set_random_edge_costs


SOURCE = 's'
TARGET = 't'
MAX_CAPACITY = 1000

def init_GBCP_rates(model, tree, depot_cluster = 1):
    tree = model._tree
    T = tree.copy()
    T.remove_node(depot_cluster)
    
    roots =  [v for v,d in T.in_degree()  if d == 0]
    leaves = [v for v,d in T.out_degree() if d == 0]

    roots =  [r for r in roots  if r not in leaves]
    leaves = [l for l in leaves if l not in roots]

    ap_rates = {(r,l):1 for r in roots for l in leaves if nx.has_path(T,r,l)}
    ap_paths = {(r,l):[path for path in islice(nx.all_simple_paths(T, r, l), GBCP_MAX_PATH_COUNT) if len(path) >= 3 and len(path) % 2 > 0] for (r,l) in ap_rates}

    print(f'{len(ap_paths)} address pairs initially')
    # for key in ap_paths:
    #     print(key, ap_paths[key])

    ap_paths = {key: val for (key,val) in ap_paths.items() if val}
    print(f'{len(ap_paths)} address pairs after filtering')
    # print(ap_paths.keys())


    ap_rates = {key: val for (key,val) in ap_rates.items() if key in ap_paths}
            
    model._GBCP_rates = ap_rates
    model._GBCP_paths = ap_paths
    print(f'{len(ap_rates)} possible address pairs found finally')
    print(f'{sum(len(ap_paths[pair]) for pair in ap_paths)} total possible cuts')


def createClusterGraph(clusters,  wrapped_u):
    auxgraph = nx.DiGraph()
    auxgraph.add_nodes_from(clusters)
    auxgraph.add_nodes_from([SOURCE,TARGET])
    
        
    for e in wrapped_u:
        auxgraph.add_edge(*e, capacity=wrapped_u[e])

    return auxgraph

###########################
### Only for the ft54.4 test purposes
### Delete before production   
    
#def testGBCP_cut(model, p, q, path, cut):
#    test_u = model._test_u
#    S, barS, rhs = cut

#    if sum(test_u[p,q] for p in S for q in barS) < rhs:
#        print('*** invalid GBCP cut:')
#        print(f'*** S = {S}, barS  = {barS}, rhs = {rhs}')
#        print(f'*** path = {path}')
#############################


def create_GBCP_Constraint(auxgraph, p,q, paths, depot_cluster, model):
    
    def make_a_cut(auxgraph,source, dest, lhs, rhs,  thresh):
        auxG = auxgraph.copy()
        #auxG.remove_nodes_from(nodes_to_out)
        
        for v in lhs:
            auxG.add_edge(source,v, capacity = MAX_CAPACITY)
        for v in rhs:
            auxG.add_edge(v,dest, capacity = MAX_CAPACITY)
        
        cut_val, part = nx.minimum_cut(auxG,source,dest)
        if cut_val < thresh:
            S,barS = part
            S.remove(source)
            barS.remove(dest)
            
            if set(left_side).issubset(S) and set(right_side).issubset(barS):
                return tuple(S), tuple(barS), thresh
        return None  
    
    
    cuts = set()
    for path in paths:
        p_len = len(path)
        left_side = path[::2]
        right_side = path[1::2] + [depot_cluster]
        #print(f'Path: {path}, LHS: {left_side}, RHS: {right_side}')
        thresh = math.ceil(p_len / 2)
        cut = make_a_cut(auxgraph, SOURCE, TARGET, left_side, right_side, thresh)
        
        if cut:
            cuts.add(cut)
        ############ ft53.4 test ###########
            #testGBCP_cut(model, p, q, path, cut)    
        ####################################
            
    if cuts:
        return cuts
    else:
        return None


def generate_GBCP_cuts(model, depot_cluster=1):

    clusters = model._clusters
    
    #print('in GBCP')
    tree_closure = model._tree_closure
    wrapped_u = model.cbGetNodeRel(model._u)
    epoch = model._epoch

    rates = model._GBCP_rates
    paths = model._GBCP_paths
    l_rates = len(rates)
    if NEED_GBCP_CUTS_SAMPLING and l_rates > 0:    
        iter = get_address_pair_sample(epoch, rates, FRACTION_GBCP_CUTS)
    else:
        iter = rates 

    aux_G = createClusterGraph(clusters, wrapped_u)
    
    counter = 0
    cuts = set()
    #print('GBCP main loop')
    for p,q in iter:
        #print(p,q)
        counter += len(paths[p,q])
        cuts_to_add  = create_GBCP_Constraint(aux_G, p, q, paths[(p,q)],  depot_cluster, model)
        if cuts_to_add:
            #print(p,q,cut)
            cuts = cuts | cuts_to_add
            if NEED_GBCP_CUTS_SAMPLING:
                update_rates(model._GBCP_rates, (p,q), RATES_GBCP_CUTS_STEP)
    number_of_cuts = len(cuts)
    #return cuts, number_of_cuts 
    new_model = model
    if cuts:
        new_model = addGBCP_cuts(model,cuts)
    return new_model, number_of_cuts, counter  

















