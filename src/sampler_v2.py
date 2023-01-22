from sklearn.utils.random import sample_without_replacement
import numpy as np
from itertools import permutations, islice
#from math import prod
from functools import reduce
from operator import mul


MASTER_SEED = 1701
SEEDS_AMOUNT= 1000
MAXIMUM_SEED = 10000
SCALE = 10
seeds = None

# def init_Gouveia_cuts_ratings(model, depot_cluster = 1):
# 	clusters = model._clusters
# 	clus = [c for c in clusters if c != depot_cluster]
# 	#model._GDDL_rates = {c: 1 for c in permutations(clus,3)}
# 	#model._Simple_rates = {c: 1 for c in permutations(clus,2)}
# 	#model._2Path_rates = {c: 1 for c in permutations(clus,3)}
# 	#model._3vGDDL_rates = {c: 1 for c in permutations(clus,4)}
# 	#model._4vGDDL_rates = {c: 1 for c in permutations(clus,5)}
# 	model._GDDL_rates = {}
# 	model._Simple_rates = {}
# 	model._2Path_rates = {}
# 	model._3vGDDL_rates = {}
# 	model._4vGDDL_rates = {} 

def prod(iterable):
    return reduce(mul, iterable, 1)


def init_seeds():
	global seeds
	seeds = sample_without_replacement(MAXIMUM_SEED, SEEDS_AMOUNT, random_state = MASTER_SEED)

###############
# Function samples a fraction subset from the set of ordered combinations of the given set
# according to the probability measure given by the dictionary rates
#
# rates
#		key: a combination, e.g. (p,q,r)
#		val: rating of the key


def get_a_sample(epoch, groundset, combs, rates, fraction):
	np.random.seed(seeds[epoch % SEEDS_AMOUNT])

	rates_array = np.array([(rates[e] if e in rates else 1) for e in combs])
	prob = rates_array / np.sum(rates_array)
	L = len(rates_array)
	gs_index = list(range(L))

	sample_size = round(L * fraction / 100)

	sample_index=np.random.choice(gs_index, sample_size, replace=False, p=prob)
	return [combs[i] for i in sample_index]

###################################################
#   Random sampling from huge populations
###################################################
def first_digit_in_k_tuple_from_index(digits,M,k,idx):
    in_row = prod(M-1-i for i in range(k-1))
    first = digits[idx // in_row]
    return first, idx % in_row
    
def k_tuple_from_index(M,k,idx):
    digits = list(range(1,M+1))
    result = []
    for pos in range(k):
        dgt, idx = first_digit_in_k_tuple_from_index(digits, M, k, idx)
        result.append(dgt)
        digits.remove(dgt)
        M -= 1
        k -= 1
    return tuple(result)

def get_sample_from_a_huge_population(epoch, groundset, k, fraction):
	M = len(groundset)
	n_pop = prod(M - i for i in range(k))
	sample_size = int(n_pop * fraction // 100)

	sample_index = sample_without_replacement(n_pop, sample_size, random_state = seeds[epoch % SEEDS_AMOUNT], method = 'auto')
	return [k_tuple_from_index(M,k,idx) for idx in sample_index]


def update_rates(rates,key,step):
    if key in rates:
        rates[key] += step
    else:
        rates[key] = step + 1


def set_random_edge_costs(epoch, T):
    np.random.seed(MASTER_SEED)
    
    maxval = len(T.edges)
    for e in T.edges(data=True):
        e[2]['cost'] = np.random.randint(maxval * SCALE)
    return T


def get_address_pair_sample(epoch, rates, fraction):
    np.random.seed(MASTER_SEED)
    sample_size = round(len(rates) * fraction / 100)
    
    n_pop = len(rates)
    pair_list = [p for p in rates]
    sum_rates = sum(rates.values())
    prob = [rates[p] / sum_rates for p in pair_list]
    sample_index = np.random.choice(list(range(n_pop)), sample_size, replace = False, p =  prob)
    sample = [pair_list[idx] for idx in sample_index]
    return sample

def get_25_29_sample(epoch,rates, fraction):
    np.random.seed(MASTER_SEED)
    sample_size = round(len(rates) * fraction / 100)
    
    n_pop = len(rates)
    r_list = [p for p in rates]
    sum_rates = sum(rates.values())
    prob = [rates[p] / sum_rates for p in r_list]
    sample_index = np.random.choice(list(range(n_pop)), sample_size, replace = False, p =  prob)
    sample = [r_list[idx] for idx in sample_index]
    return sample


def test_list():
	groundset=[1,2,3,4,5,6,7,8,9]
	comb_len = 3
	combs = [c for c in permutations(groundset,comb_len)]
	L = len(combs)
	rates = {c: 1 for c in combs}
	fraction = 1
	sample = get_a_sample(groundset, combs, rates, fraction)

	print(f'groundset = {groundset}')
	print(f'comb_len = {comb_len}')
	print(f'total number of combinations {L}')
	print(f'fraction = {fraction}')
	print(f'sample = {sample}')

def test_huge():
	groundset = list(range(2,400))
	M = len(groundset)
	k = 5
	fraction = 1e-6
	n_pop = prod(M - i for i in range(k))

	sample = get_sample_from_a_huge_population(groundset, k, fraction)

	print(f'groundset = {groundset}')
	print(f'comb_len = {k}')
	print(f'total number of combinations {n_pop}')
	print(f'sample size = {int(n_pop * fraction / 100)}')
	print(f'fraction = {fraction}')
	print(f'sample = {sample}')


def main():
	test_huge()
	# test_compinations()



if __name__ == '__main__':
	main()