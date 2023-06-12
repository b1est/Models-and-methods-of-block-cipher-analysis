from heys import HeysCipher
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial, wraps
from math import prod
from itertools import product, repeat
import ast
import os
import logging
import time
from config import Config

logging.basicConfig(filename='lab_logs.log', format='%(message)s', level=logging.INFO)




def calc_range_window(range_start, range_end, step):
	start = range_start
	while start < range_end:
		end = start + step
		if end > range_end:
			end = range_end
		yield start, end
		start = end + 1

def timeit(display_args):
	def decorator(func):
		@wraps(func)
		def timeit_wrapper(*args, **kwargs):
			start_time = time.perf_counter()
			result = func(*args, **kwargs)
			end_time = time.perf_counter()
			total_time = end_time - start_time
			msg = func.__name__
		
			msg += f"({args[0][:-1]})" if display_args else "()"
			full_output_msg = f"Taken time for execution of the function {msg} in seconds - {total_time:.4f}."
			logging.info(full_output_msg)
			print(full_output_msg)
			return result

		return timeit_wrapper

	return decorator

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
@timeit(display_args=True)
def delta_worker(args):
	ai_start, ai_end, bi_start, bi_end, config = args
	for ai in range(ai_start, ai_end):
		alpha_i_list = []
		for bi in range(bi_start, bi_end):
			dpf = prod([config.diff_s_block_table[ai >> 4*i & 0xf][bi >> 4*i & 0xf] for i in range(4)])
			if dpf != 0:
				# logging.degug(f"Found non-zero dpf: {dpf} for ai={ai}, bi={bi}")
				
				alpha_i_list.append((HeysCipher(config.s_block, config.s_block_rev).L(bi), dpf))
		if alpha_i_list != []:
			with open(f'f.txt', 'a') as file:
				file.write(str(alpha_i_list)+'\n')

@timeit(display_args=False)
def delta():
	differentials = {}
	if os.path.exists('f.txt'):
		with open('f.txt', 'r') as file:
			lines = file.readlines()
		for i in range(2**16):
			differentials.update({i: dict(ast.literal_eval(lines[i]))})
	else:
	
		num_processes = mp.cpu_count()-2
		start_end_window_2_args = list(calc_range_window(0, 2**16, 1000))
		pool = mp.Pool(processes=num_processes)
		sub_processes = []
		
		for ai_start, ai_end in start_end_window_2_args:
			for bi_start, bi_end in start_end_window_2_args:
				args = (ai_start, ai_end, bi_start, bi_end, Config())
				process = pool.map_async(delta_worker, [args])
				sub_processes.append(process)
		pool.close()
		pool.join()
		with open('f.txt', 'r') as file:	
			lines = file.readlines()
		for i in range(2**16):
			differentials.update({i: dict(ast.literal_eval(lines[i]))})

	
	return differentials
	



if __name__ == "__main__":
	config = Config()
	config.s_block = [10, 9, 13, 6, 14, 11, 4, 5, 15, 1, 3, 12, 7, 0, 8, 2]
	config.s_block_rev = [13, 9, 15, 10, 6, 7, 3, 12, 14, 1, 0, 5, 11, 2, 4, 8]
	config.diff_s_block_table = diff_probs_table_s(False)
	differentials = delta()