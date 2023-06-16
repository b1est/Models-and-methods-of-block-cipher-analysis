from heys import HeysCipher
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial, wraps
from math import prod, ceil
from itertools import product, repeat
import os
import logging
from helpers import timeit, calc_range_window, chunk_list, fastest_reader
from config import Config
import argparse
import pickle
import time
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
def delta(processes: int, step_size: int, chunksize: int):
    if os.path.exists('f.pkl'):
        with open('f.pkl', 'rb') as f:
           differentials = pickle.load(f)
    else:
        i = 0
        differentials = []
        if not processes:
            processes = mp.cpu_count()-2
        start_end_window_4_args = [(ai_start, ai_end, bi_start, bi_end) for ai_start, ai_end in calc_range_window(0, 1 << 16, step_size) for bi_start, bi_end in calc_range_window(0, 1 << 16, step_size)]
        with mp.Pool(processes=processes) as pool:
            for result in pool.imap(delta_worker, start_end_window_4_args, chunksize=chunksize):
                for r in result:
                    if r != []:
                        differentials.append((i, tuple(r)))
                        i+=1
        with open('f.pkl', 'wb') as f:
            pickle.dump(differentials, f)
    return get_dict_dict_(differentials)

@timeit(display_args=False) 
def differential_search(alpha, diffs, q = 0.00008, r = 6):
    InitialGamma = {alpha: 1}
    for t in range(1, r):
        GammaIndex = {}
        for bi in InitialGamma:
            for gj in diffs[bi]:
                if gj in GammaIndex:
                    GammaIndex[gj] += InitialGamma[bi]*diffs[bi][gj]
                else:
                    GammaIndex[gj] = InitialGamma[bi]*diffs[bi][gj]
        GammaIndex.update({key: value for key, value in GammaIndex.items() if value > q})    
        InitialGamma = GammaIndex
    return GammaIndex

# def get_sorted_differentials(differentials):
# 	tmp_differentials = []
# 	for key, value in differentials.items():
# 		for value_k, value_v in value.items():
# 			tmp_differentials.append(((key, value_k), value_v))
# 	sorted_differentials = sorted(tmp_differentials, key = lambda x: x[1], reverse = True)
# 	print(differentials)
# 	print(sorted_differentials)
# 	return sorted_differentials

def get_dict_dict_(obj: list[list] or list[tuple] or tuple[list] or tuple[tuple]) -> dict:
    return dict(map(lambda x: (x[0], dict(x[1])), obj))

@timeit(display_args=False)
def get_dp_worker(alpha_lst, diff):
    diff_cand_dict = {}
    for alpha in alpha_lst:
        difs = differential_search(alpha, diff)
        if difs != {}:
            for b in difs:
                diff_cand_dict[(alpha, b)] = tuple(difs)
    return diff_cand_dict

@timeit(display_args=False)
def get_dp(differentials, processes = 10):
    if os.path.exists('dp.pkl'):
        with open('dp.pkl', 'rb') as file:
           difs_dict = pickle.load(file)
    else:
        alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0)>=3]
        alpha_array_chunks = chunk_list(alpha_array, processes)
        difs_dict = {}
        sub_processes = []
        with mp.Pool(processes=processes) as pool:
        # for alpha in alpha_array:
        #     difs = differential_search(alpha, differentials)
        #     if difs != {}:
        #         for b in difs:
        #             print(f'alpha = {alpha}, beta = {b}: DP({alpha}, {b}) = {difs[b]}')
        #             difs_list.append((alpha, tuple(difs)))
            for chunk in alpha_array_chunks:
                r = pool.apply_async(get_dp_worker, [chunk, differentials])
                sub_processes.append(r)
            for sp in sub_processes:
                difs_dict.update(sp.get())
        with open('dp.pkl', 'wb') as file:
           pickle.dump(difs_dict, file)
    return difs_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculation of the table of differential probabilities of a round transformation."
    )
     
    parser.add_argument(
        "--processes", type=int, help="Number of processes to run", default=mp.cpu_count()-4
    )

    parser.add_argument(
        "--step-size", type=int, help="Length of intervals into which the range will be split.", default=1000
    )

    parser.add_argument(
        "--chunksize", type=int, help="Keyword argument that can be used to specify the size of chunks in which to process input data when using the pool.imap() function. This can be useful for improving performance when processing large amounts of data.", default=1
    )
    args = parser.parse_args()

    differentials = delta(processes=args.processes, step_size = args.step_size, chunksize=args.chunksize)

    get_dp(differentials)