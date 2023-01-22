import multiprocessing as mp
from parallelPCGLNS import init, getRecord
import time

HEURISTIC_PATH = 'heuristic/'
HEURISTIC_SOLUTION_SUFFIX = '.PCGLNS.res'
TEMP_HEURISTIC_PATH = 'heuristic/tmp/'

SCRIPT_PATH_NAME = 'PCGLNS/runPCGLNS.jl'
SOURCE_PCGLNS_PATH ='input/'
SOURCE_PCGLNS_SUFFIX='.pcglns'

UB = 1000000

def main():
	print(f'Hello from the client')
	worker=init('p3xl_1-m', UB, HEURISTIC_PATH, TEMP_HEURISTIC_PATH, SOURCE_PCGLNS_PATH, SCRIPT_PATH_NAME)
	print(f'worker process {worker.pid} was started')

	for ce in range(50):
		rec, x, z = getRecord()
		print(f'Client poll for the current record -> {rec}')
		time.sleep(3)

	# worker.join()



if __name__ == '__main__':
	main()