import sys
import networkx as nx
from itertools import permutations, combinations
from math import log10

from fromPCGLNS import getInstance
from preprocess import preprocessInstance
from heuristicHelper import getMIPS, runHeuristic

from gurobipy import GRB

from gurobiMTZHelper import createExactModel, optimizeModel, addObjectiveRoundCut, addObjectiveRoundLazy, addIncumbent

#from heuristicHelper import getUB, getMIPS
from realBalas_cb import generatePi_cuts, generateSigma_cuts, generatePiSigma_cuts, init_ratings
#from GDDL_STR_cb import generateGDDL_cuts
#from simplecut_STR_cb import generateSimple_cuts
#from Twopath_STR_cb import generate_2path_cuts
#from ThreevGDDL_STR_cb import generate_3vGDDL_cuts
#from FourvGDDL_STR_cb import generate_4vGDDL_cuts
from GBCP_STR_cb import init_GBCP_rates, generate_GBCP_cuts
from Gouveia_cuts_cb import init_Gouveia_25_29_rates, generate_25_cuts, generate_26_cuts, generate_27_cuts, generate_28_cuts, generate_29_cuts

from sampler_v2 import init_seeds


from parameters import *
from math import ceil

from parallelPCGLNS import init, getRecord, getRecordSolution



model_name = 'ESC25'
path_name = 'PCGLNS/PCGLNS_PCGTSP/'
PCGLNS_EXT='.pcglns'
MODELS_PATH='models/'
SOLUTIONS_PATH='sols/'


use_callbacks = True
time_limit = 36000
threads = 3
	

def cutGeneratorCallback(model, where):
    #return
    if where == GRB.Callback.MIP:
        BEST_UB = model.cbGet(GRB.Callback.MIP_OBJBST)
        BEST_LB = max(1, model.cbGet(GRB.Callback.MIP_OBJBND))
        model._GAP= (BEST_UB - BEST_LB) / BEST_LB
        model._BEST_UB = BEST_UB
        cut_cnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
        #print(f'Cut CNT = {cut_cnt}')
        #runtime = model.cbGet(GRB.Callback.RUNTIME)

    if where == GRB.Callback.MIPNODE:
        nodecnt = int(model.cbGet(GRB.callback.MIPNODE_NODCNT))
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        
        if PARALLEL_HEURISTIC:
        	pretenderUB = getRecord()
        	if pretenderUB < model._BEST_UB:
        		X,Z = getRecordSolution()
        		model._BEST_UB = pretenderUB
        		addIncumbent(model, X, Z)

        #old_node, need_cut = model._cutswitcher
        #if (old_node == nodecnt) and not need_cut and CANCEL_OBSOLETE_CUTS:
        #    return
        
        if nodecnt == 0 and runtime > ROOT_TIME_LIMIT:
        	model._reason = REASON_ROOT_EVAC
        	model.terminate()
        elif model._GAP <= ENDGAME_GAP:
        	model._reason = REASON_ENDGAME
        	model.terminate()
        LB = model.cbGet(GRB.callback.MIPNODE_OBJBND)
        need_sampling = True if nodecnt >= SAMPLING_THRESHOLD else False
        #print(f'Node {nodecnt}, current LB = {LB}, previous LB ={model._prevBestLB}')
        
        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
        	#_ = addObjectiveRoundCut(model,LB)
        	#return
        	#_ = addObjectiveRoundCut(model,LB)
        	#print('in user cut')
        	amplificator = 1 / log10(nodecnt + 100)
        	if model._GAP > CUT_OFF_GAP:
        		if ENABLE_BALAS_CUTS:
        			number_of_BalasPi_cuts, number_of_BalasSigma_cuts, number_of_BalasPS_cuts = 0, 0, 0
        			_, number_of_BalasPi_cuts = generatePi_cuts(model,amplificator, DEPOT_CLUSTER, need_sampling)
        			_, number_of_BalasSigma_cuts = generateSigma_cuts(model,amplificator, DEPOT_CLUSTER, need_sampling)
        			_, number_of_BalasPS_cuts = generatePiSigma_cuts(model, amplificator, DEPOT_CLUSTER, need_sampling)
        		#if CANCEL_OBSOLETE_CUTS and (number_of_BalasPi_cuts + number_of_BalasSigma_cuts + number_of_BalasPS_cuts == 0):
        			#model._cutswitcher = (nodecnt, False)
        			#print(f'Node {nodecnt}: custom cuts switched off')
        			if number_of_BalasPi_cuts + number_of_BalasSigma_cuts + number_of_BalasPS_cuts > 0:
        				print(f'Cb   {nodecnt}, Pi {number_of_BalasPi_cuts}, Sigma {number_of_BalasSigma_cuts}, Pi+Sigma {number_of_BalasPS_cuts}')
        		#if nodecnt % 100 == 0:
        			#print(f'Callback: Node {nodecnt}, Pi rates {model._PiRates},\n Sigma {model._SigmaRates},\n Pi+Sigma {model._PiSigmaRates}')
        		
        		# Generalized BCP cuts
        		if ENABLE_GBCP_CUTS:
        			_, number_of_GBCP_cuts, out_of = generate_GBCP_cuts(model, DEPOT_CLUSTER)
        			if number_of_GBCP_cuts > 0:
        				print(f'Cb   {nodecnt}, Strengthened GBCP   cuts: {number_of_GBCP_cuts:4} out of {out_of:8}')
        			
        		# Gouveia 25-29 cuts
        		if ENABLE_GOUVEIA_25_29_CUTS:
        			_, number_of_25_cuts, out_of = generate_25_cuts(model, DEPOT_CLUSTER)
        			if number_of_25_cuts > 0:
        				print(f'Cb   {nodecnt}, Gouveia  25   cuts: {number_of_25_cuts:4} out of {out_of:8}')
        			_, number_of_26_cuts, out_of = generate_26_cuts(model, DEPOT_CLUSTER)
        			if number_of_26_cuts:
        				print(f'Cb   {nodecnt}, Gouveia  26   cuts: {number_of_26_cuts:4} out of {out_of:8}')
        			
        			_, number_of_27_cuts, out_of = generate_27_cuts(model, DEPOT_CLUSTER)
        			if number_of_27_cuts:
        				print(f'Cb   {nodecnt}, Gouveia  27   cuts: {number_of_27_cuts:4} out of {out_of:8}')
        			
        			_, number_of_28_cuts, out_of = generate_28_cuts(model, DEPOT_CLUSTER)
        			if number_of_28_cuts:
        				print(f'Cb   {nodecnt}, Gouveia  28   cuts: {number_of_28_cuts:4} out of {out_of:8}')
        			
        			_, number_of_29_cuts, out_of = generate_29_cuts(model, DEPOT_CLUSTER)
        			if number_of_29_cuts:
        				print(f'Cb   {nodecnt}, Gouveia  29   cuts: {number_of_29_cuts:4} out of {out_of:8}')
        			
        		###################################################################
        		# Excluded from the MTZ experiment 	
        		###################################################################
        		# F model
        		# if ENABLE_GDDL_CUTS: #and (not ENABLE_BALAS_CUTS or LB < model._prevBestLB * BEST_BOUND_RATE):
        		# 	_, number_of_GDDL_cuts, out_of = generateGDDL_cuts(model, DEPOT_CLUSTER)
        		# 	size_of_ratings = sys.getsizeof(model._GDDL_rates) / 10**6
        		# 	print(f'Cb   {nodecnt}, Strengthened GDDL   cuts: {number_of_GDDL_cuts:4} out of {out_of:8} | size {size_of_ratings:.4f} MB')
        		# 	if NEED_GDDL_CUTS_SAMPLING and size_of_ratings > MAXIMUM_SIZE_OF_RATINGS_GB:
        		# 		model._GDDL_rates = {} 
        		# if ENABLE_SIMPLE_CUTS:# and (not ENABLE_BALAS_CUTS or LB < model._prevBestLB * BEST_BOUND_RATE):
        		# 	#print(f'Trying to separate simple cuts')
        		# 	_, number_of_Simple_cuts, out_of = generateSimple_cuts(model, DEPOT_CLUSTER)
        		# 	zise_of_ratings = sys.getsizeof(model._Simple_rates) / 10**6
        		# 	print(f'Cb   {nodecnt}, Strengthened Simple cuts: {number_of_Simple_cuts:4} out 0f {out_of:8} | size {size_of_ratings:.4f} MB')
        		# 	if NEED_SIMPLE_CUTS_SAMPLING and size_of_ratings > MAXIMUM_SIZE_OF_RATINGS_GB:
        		# 		model,_Simple_rates = {}
        			
        		# #M3 model
        		# if ENABLE_2PATH_CUTS: #and (ENABLE_3vGDDL_CUTS or LB < model._prevBestLB * BEST_BOUND_RATE):  
        		# 	#print(f'Trying to separate 2-path cuts')
        		# 	_, number_of_2path_cuts, out_of = generate_2path_cuts(model, DEPOT_CLUSTER)
        		# 	size_of_ratings = sys.getsizeof(model._2path_rates) / 10**6
        		# 	print(f'Cb   {nodecnt}, Strengthened 2-path cuts: {number_of_2path_cuts:4} out 0f {out_of:8} | size {size_of_ratings:.4f} MB')
        		# 	if NEED_2PATH_CUTS_SAMPLING and size_of_ratings > MAXIMUM_SIZE_OF_RATINGS_GB:
        		# 		model._2path_rates = {}
        			
        		# #M4 model
        		# if ENABLE_3vGDDL_CUTS: #and (ENABLE_4vGDDL_CUTS or LB < model._prevBestLB * BEST_BOUND_RATE): 
        		# 	#print(f'KU-KU, LB={LB}, prevLB * Rate ={model._prevBestLB * BEST_BOUND_RATE}')
        		# 	_, number_of_3vGDDL_cuts, out_of = generate_3vGDDL_cuts(model, DEPOT_CLUSTER)
        		# 	#size_of_ratings = sys.getsizeof(model._3vGDDL_rates) / 10**9
        		# 	print(f'Cb   {nodecnt}, Strengthened 3vGDDL cuts: {number_of_3vGDDL_cuts:4} out 0f {out_of:8} | ')
        		# 	#if NEED_3vGDDL_CUTS_SAMPLING and size_of_ratings > MAXIMUM_SIZE_OF_RATINGS_GB:
        		# 	#	model._3vGDDL_rates = {}
        			
        		# #M5 model
        		# if ENABLE_4vGDDL_CUTS: #and LB < model._prevBestLB * BEST_BOUND_RATE:  
        		# 	#print(f'KU-KU, LB={LB}, prevLB * Rate ={model._prevBestLB * BEST_BOUND_RATE}')
        		# 	#print('Here should be 4vGDDLs')
        		# 	_, number_of_4vGDDL_cuts, out_of = generate_4vGDDL_cuts(model, DEPOT_CLUSTER)
        		# 	#size_of_ratings = sys.getsizeof(model._4vGDDL_rates) / 10**9
        		# 	print(f'Cb   {nodecnt}, Strengthened 4vGDDL cuts: {number_of_4vGDDL_cuts:4} out 0f {out_of:8} | ')
        		# 	#if NEED_4vGDDL_CUTS_SAMPLING and size_of_ratings > MAXIMUM_SIZE_OF_RATINGS_GB:
        		# 	#	model._4vGDDL_rates = {}
        model._prevBestLB = LB
        model._epoch = model._epoch + 1

def lazyCallback(model, where):

    if where == GRB.Callback.MIPNODE:
        LB = model.cbGet(GRB.callback.MIPNODE_OBJBND)

        status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status == GRB.OPTIMAL:
        	_ = addObjectiveRoundLazy(model,LB)



def main(need_callbacks = True):
	print('Branch-and-Cut PCGTSP solver')
	print('Strengthened Gouveia formulations')
	print(f'Task name:\t{model_name}')
	print(f'Time limit:\t{time_limit} sec')
	print(f'Threads:\t{threads}')
	printparameters()

	G, clusters, tree = getInstance(f'{path_name}{model_name}{PCGLNS_EXT}')
	G, clusters, tree, tree_closure = preprocessInstance(G,clusters,tree)

	
	if NEED_MIP_START:
		runHeuristic(model_name, path_name)
		mips = {}
		mips['x'], mips['z'], obj = getMIPS(model_name)
		model, x, y, u, z = createExactModel(f'{model_name}-exact', G, clusters, tree, mips)
		print('PCGLNS MIP start is incorporated')
	else:
		model, x, y, u, z = createExactModel(f'{model_name}-exact', G, clusters, tree)

############################	
	model._G = G
	model._clusters = clusters
	model._tree = tree
	model._tree_closure = tree_closure
	
	init_ratings(model,DEPOT_CLUSTER)
	
	model._cutswitcher = (0, need_callbacks)
	model._prevBestLB = 0
	model._epoch = 1

	clusters = model._clusters
	clus = [c for c in clusters if c != DEPOT_CLUSTER]
	print('Compact model is ready, trying to init sampling for M2 - M5')
	need_init_seeds = True

	# if NEED_GDDL_CUTS_SAMPLING and ENABLE_GDDL_CUTS:
	# 	#model._GDDL_rates = {c: 1 for c in permutations(clus,3)}
	# 	model._GDDL_rates = {}
	# 	need_init_seeds = True
	# if NEED_SIMPLE_CUTS_SAMPLING and ENABLE_SIMPLE_CUTS:
	# 	#model._Simple_rates = {c: 1 for c in permutations(clus,2)}
	# 	model._Simple_rates = {}
	# 	need_init_seeds = True
	# if NEED_2PATH_CUTS_SAMPLING and ENABLE_2PATH_CUTS:
	# 	#model._2path_rates = {c: 1 for c in permutations(clus,3)}
	# 	model._2path_rates = {}
	# 	need_init_seeds = True
	# if NEED_3vGDDL_CUTS_SAMPLING and ENABLE_3vGDDL_CUTS:
	# 	#model._3vGDDL_rates = {c: 1 for c in permutations(clus,4)}
	# 	model._3vGDDL_rates = {}
	# 	need_init_seeds = True
	# if NEED_4vGDDL_CUTS_SAMPLING and ENABLE_4vGDDL_CUTS:
	# 	#model._4vGDDL_rates = {c: 1 for c in permutations(clus,5)}
	# 	model._4vGDDL_rates = {}
	# 	need_init_seeds = True
	
	if need_init_seeds:
		init_seeds()
		
	if ENABLE_GBCP_CUTS:
		print('Initialize generalized BCP cuts')
		init_GBCP_rates(model, tree, DEPOT_CLUSTER)
		
	if ENABLE_GOUVEIA_25_29_CUTS:
		print('Initialize Gouveia 25-29 cuts')
		init_Gouveia_25_29_rates(model, DEPOT_CLUSTER)
############################
	print('Model is ready')

	
	model.write(f'{MODELS_PATH}{model_name}_exact.lp')
	print(f'model written to {MODELS_PATH}{model_name}_exact.lp')
	if PARALLEL_HEURISTIC:
		init(model_name,obj,source_path=path_name)
	
	try:
		if need_callbacks:
			status = optimizeModel(model, time_limit, threads - 1, cutGeneratorCallback, lazyCallback)
		else:
			status = optimizeModel(model, time_limit, threads)
		if status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
			model.write(f'{SOLUTIONS_PATH}{model_name}_raw.sol')
	except AssertionError as msg:
		print(msg)




if __name__ == '__main__':
	try:
		for arg in sys.argv:
			if arg == '--without_callbacks' or arg == '-c':
				use_callbacks = False
			if '=' in arg:
				parts = arg.split('=')
				if parts[0] == '--input' or parts[0] == '-i':
					model_name = parts[1]
				if parts[0] == '--path' or parts[0] == '-p':
					path_name = f'{parts[1]}/'
				if parts[0] == '--time_limit' or parts[0] == '-t':
					time_limit = int(parts[1])
				if parts[0] == '--threads' or parts[0] == '-th':
					threads = int(parts[1])
	except:
		print('SYNTAX: python BnC_cb.py -i=<model_name> [-c or --without_callbacks] [-t or -time_limit=<in secs>] [-th or --threads=<# of threads>] [-p or --path=<path to GLNS sources>] ')

	main(use_callbacks)


