import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import permutations, combinations
from parameters import *
from math import ceil


def createExactModel(model_name, G, clusters, tree,  mips = None, first_cluster = 1):
    with gp.Env(empty = True) as env:
        env.setParam('LogToConsole', 1)
        env.start()
        
        model=gp.Model(model_name, env=env)
        
        n = len(G)
        n_list = list(G)
        n_dict = {n_list[idx]: idx for idx in range(n)}
        m = len(clusters)
        
        tree_closure = nx.transitive_closure_dag(tree)
        edgedict = {t[:2]: t[2] for t in G.edges(data='weight')}
 

        clust_pairs = {(c1,c2):None for c1,c2 in permutations(clusters,2)}
        x_ij, cost = gp.multidict(edgedict)
        z_i,_ = gp.multidict(n_dict)
        u_pq,_ = gp.multidict(clust_pairs)
        y_pq,_ = gp.multidict(clust_pairs)

        UU_p,_ = gp.multidict({c: None for c in clusters if c != first_cluster})
        #obj_val = 0.0
        obj_idx = {idx: None for idx in range(1)}
        obj_val,_ = gp.multidict(obj_idx) 
 
        ### VARIABLES
        x = model.addVars(x_ij, vtype=GRB.BINARY, name='x')
        z = model.addVars(z_i, vtype=GRB.BINARY, name='z')
        y = model.addVars(y_pq, name='y')
        u = model.addVars(u_pq, name='u')
        UU = model.addVars(UU_p, name = 'UU')

        obj = model.addVars(obj_val, name='C')

               
        ### OBJECTIVE
        #objective = model.setObjective(x.prod(cost), GRB.MINIMIZE)
        objective = model.setObjective(obj[0], GRB.MINIMIZE)

        ### CONSTRAINTS
        model.addConstr(x.prod(cost) - obj[0] == 0, f'(1_objective)')

        ### Constraint 2
        for c in clusters:
            model.addConstr(sum(z[i] for i in clusters[c]) == 1, f'(2_{c})')

        ### Constraint 3
        for i in G:
            model.addConstr(sum(x[i,j] for j in G.successors(i)) == z[i], f'(3_{i})')

        ### Constraint 4
        for i in G:
            model.addConstr(sum(x[j,i] for j in G.predecessors(i)) == z[i], f'(4_{i})')

        ### Constraint 5 (1+2)
        for p in clusters:
            model.addConstr(sum(u[p,q] for q in clusters if q != p) == 1, f'(5_1_{p})')
            model.addConstr(sum(u[q,p] for q in clusters if q != p) == 1, f'(5_2_{p})')

        ### Constraint 6

        for p,q in permutations(clusters,2):
            model.addConstr(u[p,q] == sum(x[i,j] for i in clusters[p] for j in clusters[q] if G.has_edge(i,j)), f'(6_{p}_{q})')

        ### Constraint 7 MTZ-DL
        for p,q, in permutations(clusters,2):
            if first_cluster in (p,q):
                continue
            model.addConstr((m - 1) * u[p,q] + (m - 3) * u[q,p] + UU[p] - UU[q] <= m - 2, f'(7_{p}_{q}_MTZ-DL)')
        
        ### Constraint 8 MTZ-DL
        for p in clusters:
            if first_cluster == p:
                continue
            model.addConstr(-UU[p] + (m - 3) * u[p,1] + sum(u[q,p] for q in clusters if not q in (first_cluster, p)) <= 0, f'(8_{p}_{q}_MTZ-DL)')
        
        ### Constraint 9 MTZ-DL
        for p in clusters:
            if first_cluster == p:
                continue
            model.addConstr(UU[p] + (m - 3) * u[1,p] + sum(u[p,q] for q in clusters if not q in (first_cluster, p)) <= m - 2, f'(9_{p}_{q}_MTZ-DL)')
            
        ### Constraint 10 MTZ-DL
        for p,q in tree.edges:
            if first_cluster in (p,q):
                continue
            model.addConstr(UU[p] <= UU[q] - 1, f'(10_{p}_{q}_MTZ-DL)')

        ###############################################################
        #  Excluded from the MTZ model
        #
        # ### Constraint 7

        # clust_keys = list(clusters.keys())
        # for p_idx in range(m):
        #     p = clust_keys[p_idx] 
        #     if p == first_cluster:
        #         continue
        #     for q_idx in range(p_idx+1,m):
        #         q = clust_keys[q_idx]
        #         if q == first_cluster:
        #             continue
        #         for r_idx in range(p_idx+1,m):
        #             r = clust_keys[r_idx]
        #             if r_idx == q_idx or r == first_cluster:
        #                 continue
        #             model.addConstr(y[p,q] + u[q,p]  + y[q,r] + y[r,p] <= 2, f'(7_{p}_{q}_{r})')

        # ### Constraint 8

        # for p, q in permutations(clusters, 2):
        #     if first_cluster in (p,q):
        #         continue
        #     model.addConstr(u[p,q] - y[p,q] <= 0, f'(8_{p}_{q})')

        # ### Constraint 9

        # for p, q in combinations(clusters, 2):
        #     if first_cluster in (p,q):
        #         continue
        #     model.addConstr(y[p,q] + y[q,p] == 1, f'(9_{p}_{q})')


        # ### Constraint 10

        # for p, q in permutations(clusters, 2):
        #     if first_cluster in (p,q):
        #         continue
        #     if tree_closure.has_edge(p,q):
        #         model.addConstr(y[p,q] == 1, f'(10_{p}_{q})')
                
        ### constraint 11 - moved to callbacks
                
        #if NEED_BALAS_UTILITIES:
        #    for p, q in combinations(clusters, 2):
        #        if first_cluster in (p,q):
        #            continue
        #        Pi_p = [pi_p for pi_p in tree_closure.predecessors(p) if pi_p != first_cluster]
        #        Sigma_q = list(tree_closure.successors(q))
        #        if len(Pi_p) > 0 and len(Sigma_q) > 0:
        #            #print(p,q,Pi_p, Sigma_q)
        #            model.addConstr(u[p,q] + u[q,p] + sum(u[pi_p,sig_q] for pi_p in Pi_p for sig_q in Sigma_q) <= 1, f'(11_1_{p}_{q})')
        #            model.addConstr(u[p,q] + u[q,p] + sum(u[sig_q,pi_p] for pi_p in Pi_p for sig_q in Sigma_q) <= 1, f'(11_2_{p}_{q})')
                 
                 

        ### MIP start
        if mips:
            for key in x:
                x[key].start = 0
            for key in z:
                z[key].start = 0

            for key, val in mips['x'].items():
                x[key].start = val
            for key, val in mips['z'].items():
                z[key].start = val

        ### incoporate variables into the model
        model._x = x
        model._z = z
        model._u = u
        model._y = y
        model._obj = obj
        model._UU = UU
        #model._cost = cost
    return model, x, y, u, z 


def addIncumbent(model, X,Z):
    x = model._x
    z = model._z

    for i,j in x:
        if (i,j) in X:
            model.cbSetSolution(x[i,j], X[(i,j)])
        else:
            model.cbSetSolution(x[i,j], 0)
    for v in z:
        if v in Z:
            model.cbSetSolution(z[v],Z[v])


def optimizeModel(model, time_limit, threads, callback = None, lazyCallback = None):
    model.setParam(GRB.Param.TimeLimit,time_limit)
    model.setParam(GRB.Param.Threads, threads)
    model.setParam(GRB.Param.MIPFocus, MIP_FOCUS)
    model.setParam(GRB.Param.CutPasses,MAXIMUM_NUMBER_OF_CUTS)
    if not CAN_LEAVE_ROOT:
        model.setParam(GRB.Param.NodeLimit, 1)

    if callback:
        if not GUROBI_CUTS:
            model.setParam(GRB.Param.Cuts, 0)
        #model.setParam(GRB.Param.LazyConstraints, 1)
        if not USE_GUROBI_HEURISTICS:
            model.setParam(GRB.Param.Heuristics, 0)
        model.setParam(GRB.Param.PreCrush, 1)
        model.optimize(callback)
    else:
        if not USE_GUROBI_HEURISTICS:
            model.setParam(GRB.Param.Heuristics, 0)
            model.setParam(GRB.Param.PreCrush, 1)
        model.optimize()
        
    while model.status == GRB.INTERRUPTED:
        if model._reason == REASON_ROOT_EVAC:
            print('Root interruption code received, disabling Gurobi cuts')
            model.setParam(GRB.Param.Cuts, 0)
            if callback:
                model.optimize(callback)
            else:
                model.optimize()
        elif model._reason == REASON_ENDGAME:
            print('Endgame phase engaged - processing tree, all cuts are disabled')
            model.setParam(GRB.Param.Cuts, 0)
            model.setParam(GRB.Param.MIPFocus, 2)
            model.setParam(GRB.Param.LazyConstraints, 1)
            model.optimize(lazyCallback)
    assert model.status == GRB.OPTIMAL, f'Optimum value has not been found, status code:{model.status}'
    return model.status

# def addGSEC_cuts(model,cut_counter):
#     u = model._u
#     for U,V in cut_counter:
#         model.cbCut(sum(u[p,q] for p in U for q in V) >= 1)
#     return model

# def addBalas_cuts(model,cut_counter, Pi_or_Sigma):
#     G = model._G
#     x = model._x
#     z = model._z
#     for U,V, node in cut_counter:
#         model.cbCut(sum(x[i,j] for i in U for j in V if G.has_edge(i,j)) - z[node] >= 0)
#     return model


# def addBalasPS_cuts(model,cuts):
#     G = model._G
#     x = model._x
#     z = model._z
#     for u,v in cuts:
#         I,J = cuts[(u,v)]
#         model.cbCut(sum(x[i,j] for i in I for j in J if G.has_edge(i,j)) - z[u] - z[v] >= -1)
#     return model


# def addBalasPS_cluster_cuts(model,cuts):
#     G = model._G
#     x = model._x
#     for u,v in cuts:
#         I,J = cuts[(u,v)]
#         model.cbCut(sum(x[i,j] for i in I for j in J if G.has_edge(i,j)) >= 1)
#     return model

# def addBalasPSfull_cluster_cuts(model,cuts):
#     u = model._u
#     for c1,c2 in cuts:
#         P,Q = cuts[(c1,c2)]
#         model.cbCut(sum(u[p,q] for p in P for q in Q if (p,q) in u) >= 1)
#     return model

def addRealBalas_cuts(model, cuts):
    u = model._u
    for P,Q in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) >= 1)
    return model

def addGBCP_cuts(model, cuts):
    u = model._u
    for P,Q,rhs in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) >= rhs)
    return model
    
def addGouveia25_cuts(model, cuts):
    u = model._u
    for p,q,r,s in cuts:
        model.cbCut(u[p,q] + u[q,p] + u[r,s] + u[s,r] <= 1)
    return model

def addGouveia26_cuts(model, cuts):
    u = model._u
    for p,q,r,q_suc in cuts:
        model.cbCut(u[p,q] + u[q,p] + sum(u[r,s] for s in q_suc if r != s) <= 1)
    return model
    
def addGouveia27_cuts(model, cuts):
    u = model._u
    for p,q,r,q_suc in cuts:
        model.cbCut(u[p,q] + u[q,p] + sum(u[s,r] for s in q_suc if r != s) <= 1)
    return model

def addGouveia28_cuts(model, cuts):
    u = model._u
    for p,q,s,p_pre in cuts:
        model.cbCut(u[p,q] + u[q,p] + sum(u[r,s] for r in p_pre if r != s) <= 1)
    return model

def addGouveia29_cuts(model, cuts):
    u = model._u
    for p,q,s,p_pre in cuts:
        model.cbCut(u[p,q] + u[q,p] + sum(u[s,r] for r in p_pre if r != s) <= 1)
    return model

    
def addGDDL_cuts(model, cuts):
    u = model._u
    y = model._y
    for P, Q, p, q, r in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) - y[r,q] - y[p,r] >= 0)
    return model

def addSimple_cuts(model, cuts):
    u = model._u
    y = model._y
    for P, Q, p, q in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) - y[p,q] >= 0)
    return model

def add_2path_cuts(model, cuts):
    u = model._u
    y = model._y
    for P, Q, p, q, r in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) - y[p,q] - y[q,r] >= -1)
    return model

def add_3vGDDL_cuts(model, cuts):
    u = model._u
    y = model._y
    for P, Q, p, q, r, s in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) - y[p,q] - y[q,r] - y[r,s] >= -1)
    return model
    
def add_4vGDDL_cuts(model, cuts):
    u = model._u
    y = model._y
    for P, Q, p, q, k, r, s in cuts:
        model.cbCut(sum(u[p,q] for p in P for q in Q) - y[p,q] - y[q,k] - y[k,r] - y[r,s] >= -2)
    return model


def addObjectiveRoundCut(model,LB):
    obj = model._obj
    wrapped_obj = model.cbGetNodeRel(model._obj)
    ceiledLB = ceil(LB)
    #print(f'C={wrapped_obj[0]}, LB = {ceiledLB}') 
    model.cbCut(obj[0] >= ceiledLB)
    return model

def addObjectiveRoundLazy(model,LB):
    obj = model._obj
    wrapped_obj = model.cbGetNodeRel(model._obj)
    ceiledLB = ceil(LB)
    #print(f'C={wrapped_obj[0]}, LB = {ceiledLB}') 
    model.cbLazy(obj[0] >= ceiledLB)
    return model


def main(model_name, G, clusters, tree, tree_closure):
    mips = {}
    mips['x'], mips['z'] = getMIPS(model_name)
    model, x, y, u, z = createExactModel(f'{model_name}-exact', G, clusters, tree, mips)


    model.write(f'models/{model_name}.lp')

    status = optimizeModel(model)
    print(status)



if __name__ == '__main__':
    from fromPCGLNS import getInstance
    from preprocess import preprocessInstance
    from heuristicHelper import getMIPS
    

    model_name='ESC12'
    G, clusters, tree = getInstance(f'PCGLNS/PCGLNS_PCGTSP/{model_name}.pcglns')
    G, clusters, tree, tree_closure = preprocessInstance(G,clusters,tree)


    main(model_name, G, clusters, tree, tree_closure)

