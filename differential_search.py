from heys import HeysCipher
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial, wraps
from math import prod, ceil
from itertools import product, repeat
import ast
import os
import logging
from helpers import timeit, calc_range_window
from config import Config
import argparse

logging.basicConfig(filename='lab_logs.log', format='%(message)s', level=logging.INFO)

config = Config()


def diff_s(a, b):
	return sum(int(HeysCipher(Config().s_block, Config().s_block_rev).S(x) ^ HeysCipher(Config().s_block, Config().s_block_rev).S(x ^ a) == b) for x in range(16)) / 16

def diff_probs_table_s(print_mode = False):
	diff = np.zeros((16, 16))
	for alpha in range(16):
		for beta in range(16):
			diff[alpha][beta] = diff_s(alpha, beta)
	if print_mode == True:
		print(pd.DataFrame(diff))
	return diff

config.diff_s_block_table = diff_probs_table_s(False)

@timeit(display_args=True)
def delta_worker(args):
	ai_start, ai_end, bi_start, bi_end = args
	delta_worker_list = []
	for ai in range(ai_start, ai_end):
		alpha_i_list = []
		for bi in range(bi_start, bi_end):
			dpf = prod([config.diff_s_block_table[ai >> 4*i & 0xf][bi >> 4*i & 0xf] for i in range(4)])
			if dpf != 0:				
				alpha_i_list.append((HeysCipher(config.s_block, config.s_block_rev).L(bi), dpf))
		delta_worker_list.append(alpha_i_list)
	return delta_worker_list

@timeit(display_args=False)
def delta(processes: int = None, chunksize: int = 1):
	differentials = {}
	if os.path.exists('f.txt'):
		with open('f.txt', 'r') as file:
			lines = file.readlines()
		for i in range(1 << 16):
			try:
				differentials.update({i: dict(ast.literal_eval(lines[i]))})
			except IndexError:
				break
	else:
		i = 0
		if not processes:
			processes = mp.cpu_count()-2
		start_end_window_4_args = [(ai_start, ai_end, bi_start, bi_end) for ai_start, ai_end in calc_range_window(0, 1 << 16, 1000) for bi_start, bi_end in calc_range_window(0, 1 << 16, 1000)]
		with mp.Pool(processes=processes) as pool:
			for result in pool.imap(delta_worker, start_end_window_4_args, chunksize=chunksize):
				for r in result:
					if r != []:
						with open('f.txt', 'a') as file:
							file.write(str(r)+'\n')
					differentials.update({i: dict(r)})
					i+=1
	return differentials
	
def differential_search(alpha, diffs, q = 0.00008, r = 6):
	Gamma0 = {alpha: 1}
	for t in range(1, r):
		Gammat = {}
		for bi in Gamma0:
			for gj in diffs[bi]:
				if gj in Gammat:
					Gammat[gj] += Gamma0[bi]*diffs[bi][gj]
				else:
					Gammat[gj] = Gamma0[bi]*diffs[bi][gj]
		Gammat.update({key: value for key, value in Gammat.items() if value > q})    
		Gamma0 = Gammat
	return Gammat


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Hmmmm..."
	)
	 
	parser.add_argument(
		"--processes", type=int, help="Number of processes to run", default=10
	)

	parser.add_argument(
		"--step-size", type=int, help="Stop search of L at this value", default=100000
	)

	args = parser.parse_args()
	differentials = delta(processes=10, chunksize=2000)
	