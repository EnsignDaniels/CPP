import networkx as nx
from itertools import combinations

from gurobiMTZHelper import addGouveia25_cuts, addGouveia26_cuts, addGouveia27_cuts, addGouveia28_cuts, addGouveia29_cuts
from parameters import *
import numpy as np
import math

from sampler_v2 import get_25_29_sample, update_rates


SOURCE = 's'
TARGET = 't'
MAX_CAPACITY = 1000


def init_Gouveia_25_29_rates(model, depot_cluster = 1):
    tree_closure = model._tree_closure
    clusters = [c for c in model._clusters if c != depot_cluster ]
    rates25={}
    rates26={}
    rates27={}
    rates28={}
    rates29={}
    
    for p,q in combinations(clusters, 2):
        p_pre = set(tree_closure.predecessors(p)) - {depot_cluster, q}
        q_suc = set(tree_closure.successors(q)) - {p}
        
        rates25.update({(p,q,r,s): 1  for r in p_pre for s in q_suc if r != s })
        rates26.update({(p,q,r):1 for r in p_pre})
        rates27.update({(p,q,r):1 for r in p_pre})
        rates28.update({(p,q,s):1 for s in q_suc})
        rates29.update({(p,q,s):1 for s in q_suc})
    
    model._rates25 = rates25
    model._rates26 = rates26
    model._rates27 = rates27
    model._rates28 = rates28
    model._rates29 = rates29


def generate_25_cuts(model, depot_cluster = 1):
    rates = model._rates25
    epoch = model._epoch
    wrapped_u = model.cbGetNodeRel(model._u)
    tree_closure = model._tree_closure
    
    cuts = set()
    out_of = 0
    number_of_cuts = 0
    if NEED_GOUVEIA25_CUTS_SAMPLING:
        sample = get_25_29_sample(epoch, rates, FRACTION_GOUVEIA25_CUTS)
    else:
        sample = rates
        
    for p,q,r,s in sample:
        out_of += 1
        if wrapped_u[p,q] + wrapped_u[q,p] + wrapped_u[r,s] + wrapped_u[s,r] > 1:
            cuts.add((p,q,r,s))
            number_of_cuts += 1
            if NEED_GOUVEIA25_CUTS_SAMPLING:
                update_rates(rates, (p,q,r,s), RATES_GOUVEIA25_CUTS_STEP)
    if cuts:
        addGouveia25_cuts(model, cuts)
    return model, number_of_cuts, out_of 
    #return cuts, number_of_cuts 
        
    

def generate_26_cuts(model, depot_cluster = 1):
    rates = model._rates26
    epoch = model._epoch
    wrapped_u = model.cbGetNodeRel(model._u)
    tree_closure = model._tree_closure
    
    cuts = set()
    out_of = 0
    number_of_cuts = 0
    if NEED_GOUVEIA26_CUTS_SAMPLING:
        sample = get_25_29_sample(epoch, rates, FRACTION_GOUVEIA26_CUTS )
    else:
        sample = rates
        
    for p,q,r in sample:
        q_suc = tuple(tree_closure.successors(q))
        out_of += 1
        if wrapped_u[p,q] + wrapped_u[q,p] + sum(wrapped_u[r,s] for s in q_suc if r != s) > 1:
            cuts.add((p,q,r,q_suc))
            number_of_cuts += 1
            if NEED_GOUVEIA26_CUTS_SAMPLING:
                update_rates(rates, (p,q,r), RATES_GOUVEIA26_CUTS_STEP)
    if cuts:
        addGouveia26_cuts(model, cuts)
    return model, number_of_cuts, out_of
    #return cuts, number_of_cuts 


def generate_27_cuts(model, depot_cluster = 1):
    rates = model._rates26
    epoch = model._epoch
    wrapped_u = model.cbGetNodeRel(model._u)
    tree_closure = model._tree_closure
    
    cuts = set()
    out_of = 0
    number_of_cuts = 0
    if NEED_GOUVEIA27_CUTS_SAMPLING:
        sample = get_25_29_sample(epoch, rates, FRACTION_GOUVEIA27_CUTS)
    else:
        sample = rates
        
    for p,q,r in sample:
        q_suc =tuple( tree_closure.successors(q))
        out_of += 1
        if wrapped_u[p,q] + wrapped_u[q,p] + sum(wrapped_u[s,r] for s in q_suc if r != s) > 1:
            cuts.add((p,q,r,q_suc))
            number_of_cuts += 1
            if NEED_GOUVEIA27_CUTS_SAMPLING:
                update_rates(rates, (p,q,r), RATES_GOUVEIA27_CUTS_STEP)
    if cuts:
        addGouveia27_cuts(model, cuts)
    return model, number_of_cuts, out_of
    #return cuts, number_of_cuts 


def generate_28_cuts(model, depot_cluster = 1):
    rates = model._rates28 
    epoch = model._epoch
    wrapped_u = model.cbGetNodeRel(model._u)
    tree_closure = model._tree_closure
    
    cuts = set()
    out_of = 0
    number_of_cuts = 0
    if NEED_GOUVEIA28_CUTS_SAMPLING:
        sample = get_25_29_sample(epoch, rates, FRACTION_GOUVEIA28_CUTS)
    else:
        sample = rates
        
    for p,q,s in sample:
        p_pre = tuple(set(tree_closure.predecessors(p)) - {depot_cluster})
        out_of += 1
        if wrapped_u[p,q] + wrapped_u[q,p] + sum(wrapped_u[r,s] for r in p_pre if r != s) > 1:
            cuts.add((p,q,s,p_pre))
            number_of_cuts += 1
            if NEED_GOUVEIA28_CUTS_SAMPLING:
                update_rates(rates, (p,q,s), RATES_GOUVEIA28_CUTS_STEP)
    if cuts:
        addGouveia28_cuts(model, cuts)
    return model, number_of_cuts, out_of
    #return cuts, number_of_cuts 

def generate_29_cuts(model, depot_cluster = 1):
    rates = model._rates29 
    epoch = model._epoch
    wrapped_u = model.cbGetNodeRel(model._u)
    tree_closure = model._tree_closure
    
    cuts = set()
    out_of = 0
    number_of_cuts = 0
    if NEED_GOUVEIA29_CUTS_SAMPLING:
        sample = get_25_29_sample(epoch, rates, FRACTION_GOUVEIA29_CUTS)
    else:
        sample = rates
        
    for p,q,s in sample:
        p_pre = tuple(set(tree_closure.predecessors(p)) - {depot_cluster})
        out_of += 1
        if wrapped_u[p,q] + wrapped_u[q,p] + sum(wrapped_u[s,r] for r in p_pre if r != s) > 1:
            cuts.add((p,q,s,p_pre))
            number_of_cuts += 1
            if NEED_GOUVEIA29_CUTS_SAMPLING:
                update_rates(rates, (p,q,s), RATES_GOUVEIA29_CUTS_STEP)
    if cuts:
        addGouveia29_cuts(model, cuts)
    return model, number_of_cuts, out_of
    #return cuts, number_of_cuts 


