from heys import HeysCipher
from collections import Counter
from config import Config
from multiprocessing import cpu_count, Pool, Manager
from helpers import timeit
import logging
import os, sys
import pickle
from pathlib import Path
from linear_approximations_search import linear_approximations_search as linsearch
from linear_approximations_search import scalar_mul as mul
from subprocess import Popen, PIPE 
import numpy as np

logging.basicConfig(filename='lab_logs.log', format='%(message)s', level=logging.INFO)


config = Config()


@timeit(display_args=True)
def linsearch_worker(alpha):
    save_file = Path(f'./saves/differentials/{alpha}.pkl')
    differentials = []
    differentials_alpha = linsearch(alpha)
    if differentials_alpha != {}:
        for beta in differentials_alpha:
                differentials.append(((alpha, beta), differentials_alpha[beta]))
        with open(save_file, 'wb') as file:
            pickle.dump(differentials, file)
    return differentials

def create_statistical_materials(text_quantity):
    # for text in range(text_quantity):
    #     input_file = Path(f'./saves/materials/pt_{text}.bin')
    #     output_file = Path(f'./saves/materials/ct_{text}.bin')


    #     with open(input_file, 'wb') as file:
    #         file.write(int(text).to_bytes(2, 'little'))
        
        
    #     if sys.platform == "linux" or sys.platform == "linux2":
    #         os.system('chmod +x heys.bin')
    #         os.system(f'./heys.bin e 01 {input_file} {output_file}')
    #     elif sys.platform == "darwin":
    #         os.system('chmod +x heys.bin')
    #         os.system(f'./heys.bin e 01 {input_file} {output_file}')
    #     elif sys.platform == "win32":
    #         Popen(f"Heys e 01 {input_file} {output_file}", stdin = PIPE).communicate('\n'.encode())
    #         os.system(f"Heys e 01 {input_file} {output_file}")
    
    texts = read(text_quantity)
    return texts
    

def read(text_quantity):
    texts = [(int.from_bytes(open(f'./saves/materials/pt_{text}.bin', 'rb').read(), 'little'), int.from_bytes(open(f'./saves/materials/ct_{text}.bin', 'rb').read(), 'little')) for text in range(text_quantity)]
    
    return texts



@timeit(display_args=False)
def m2(args):
    keys = {}
    alpha, beta, texts = args
    for k in range(1 << 16):
        u_k = 0
        for x, y in texts:
            x1 = HeysCipher(config.s_block, config.s_block_rev).round(x, k)
            if mul(alpha, x1)^mul(beta, y) == 0:
                u_k += 1
            else:
                u_k -= 1
            keys[k] = abs(u_k)
    keys_ind = sorted(keys, reverse = True)
    return keys[:100]

    

if __name__ == '__main__':
    keys = Manager().dict()

    if not os.listdir(Path('./saves/approximations')):
        approximations = []
        alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0)>=3]

        if len(alpha_array) >= cpu_count():
            num_processes = cpu_count() - 2
        else:
            num_processes = len(alpha_array)

        with Pool(processes=num_processes) as pool:
            for alpha_diff_search_res in pool.map(linsearch_worker, alpha_array):
                if alpha_diff_search_res != []:
                    for ri in alpha_diff_search_res:
                        approximations.append(ri)

        with open(Path(f'./saves/approximations/all.pkl'), 'wb') as f:
                    pickle.dump(approximations, f)
    else:    
        if Path('./saves/approximations/all.pkl').exists():
            with open(f'./saves/approximations/all.pkl', 'rb') as f:
                approximations = pickle.load(f)
        else:
            approximations = []
            for filename in os.listdir(Path('./saves/approximations/')):
                with open(Path(f'./saves/approximations/{filename}'), 'rb') as f:
                    for ri in pickle.load(f): 
                        approximations.append(ri)
            with open(Path(f'./saves/approximations/all.pkl'), 'wb') as f:
                    pickle.dump(approximations, f)
    
    approximations = sorted(approximations, key = lambda x: x[1], reverse = True)
    texts = create_statistical_materials(config.texts)
    num_processes = cpu_count() - 4
    for ab, p in approximations:
        args =[(ab[0], ab[1], texts)]
        with Pool(processes=num_processes) as pool:
            keys = pool.map(m2, args)
    
        # maximum_frequency_keys = []
        # for k,v in keys.items():
        #     if v == max(keys.values()):
        #         maximum_frequency_keys.append(k)

    # num_processes = cpu_count() - 4
    # 
    #     alpha, beta = ab
    #     create_statistical_materials(config.texts, alpha)
    #     texts, muted = read(config.texts, alpha)
    #     args = [(texts, muted, beta, k, keys) for  k in config.keys]
    #     with Pool(processes=num_processes) as pool:
    #         pool.map(last_round_attack, args)
        
    #     maximum_frequency_keys = []
    #     for k,v in keys.items():
    #         if v == max(keys.values()):
    #             maximum_frequency_keys.append(k)

    #     maximum_frequency_keys.sort()

    #     logging.info(f'Iteration: {iteration+1}/nKeys with max frequency: {maximum_frequency_keys}/n---------------------------')
    #     print(f'Iteration: {iteration+1}/nKeys with max frequency: {maximum_frequency_keys}/n---------------------------')
    #     if  len(maximum_frequency_keys) == 1:
    #         break