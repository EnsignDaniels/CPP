from multiprocessing import Manager, Process, Event
import multiprocessing as mp
from numpy import random
import time
import shutil
from heuristicHelper import runHeuristic, getMIPS, SOURCE_PCGLNS_SUFFIX, HEURISTIC_SOLUTION_SUFFIX, HEURISTIC_PATH, SCRIPT_PATH_NAME, SOURCE_PCGLNS_PATH, SOURCE_PCGLNS_PATH

HEURISTIC_TEMP_PATH = f'{HEURISTIC_PATH}tmp/'

dct = None

SLEEP_TIME = 0
MAX_SEED = 1000

def init(task_name, ub, path = HEURISTIC_PATH, tmp_path = HEURISTIC_TEMP_PATH, source_path = SOURCE_PCGLNS_PATH, script_path_name = SCRIPT_PATH_NAME):
	global dct
	manager = mp.Manager()
	dct = manager.dict()
	term_event = mp.Event()

	worker = mp.Process(target=do_work, name='myWorker', args=(dct, task_name, path, tmp_path, source_path, script_path_name), daemon = True)
	worker.start()
	dct['record'] = ub
	dct['x'] = 'N/A'
	dct['z'] = 'N/A'
	return worker

def getRecord():
	return dct['record']

def getRecordSolution():
	return dct['x'], dct['z']



def do_work(dct, task_name, path, tmp_path, source_path, script_path_name):
	#print('================================================================================')
	print(f'==================\nHello from the PCGLNS worker, task name = {task_name}, UB={dct["record"]}, path={path}, tmp_path={tmp_path}\n==================')
	#print('================================================================================')
	epoch = -1
	while True:
		epoch += 1
		seed = random.randint(1, MAX_SEED)
		#print(f'epoch={epoch}, seed={seed}')
		time.sleep(SLEEP_TIME)
		runHeuristic(task_name, source_path, tmp_path, script_path_name, seed, recalc=True, verboselvl=0)
		x,z,obj = getMIPS(task_name, tmp_path)
		#print(f'epoch={epoch}, seed={seed}, obj={obj}')
		#print(f'OBJ = {obj}')
		if obj < dct['record']:
			dct['record'] = obj
			dct['x'] = x
			dct['z'] = z
			shutil.copyfile(f'{tmp_path}{task_name}{HEURISTIC_SOLUTION_SUFFIX}', f'{path}{task_name}{HEURISTIC_SOLUTION_SUFFIX}')
			print(f'PCGLNS: epoch {epoch}, new incumbent found, UB={obj}')


def test1():
	print('test1 p3xl_1_m, seed = 1000')
	runHeuristic('p3xl_1-m', 'input/', 'heuristic/tmp/', 'PCGLNS/runPCGLNS.jl', seed=1000, recalc=True, verboselvl=0)

if __name__ == '__main__':
	test1()