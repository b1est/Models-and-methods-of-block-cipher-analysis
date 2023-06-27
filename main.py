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
import operator

logging.basicConfig(filename='lab_logs.log', format='%(message)s', level=logging.INFO)


config = Config()


@timeit(display_args=True)
def linsearch_worker(alpha):
    save_file = Path(f'./saves/approximations/{alpha}.pkl')
    approximations = []
    approximations_alpha = linsearch(alpha)
    if approximations_alpha != {}:
        for beta in approximations_alpha:
                approximations.append(((alpha, beta), approximations_alpha[beta]))
        with open(save_file, 'wb') as file:
            pickle.dump(approximations, file)
    return approximations

def create_statistical_materials(text_quantity):
    for text in range(text_quantity):
        input_file = Path(f'./saves/materials/pt_{text}.bin')
        output_file = Path(f'./saves/materials/ct_{text}.bin')


        with open(input_file, 'wb') as file:
            file.write(int(text).to_bytes(2, 'little'))
        
        
        if sys.platform == "linux" or sys.platform == "linux2":
            Popen(f"./heys.bin e 01 {input_file} {output_file}", stdin = PIPE, stderr=True).communicate()
        elif sys.platform == "darwin":
            Popen(f"./heys.bin e 01 {input_file} {output_file}", stdin = PIPE, stderr=True).communicate()
        elif sys.platform == "win32":
            Popen(f"Heys e 01 {input_file} {output_file}", stdin = PIPE, stderr=True).communicate('\n'.encode())

def read(text_quantity):
    texts = [(int.from_bytes(open(f'./saves/materials/pt_{text}.bin', 'rb').read(), 'little'), int.from_bytes(open(f'./saves/materials/ct_{text}.bin', 'rb').read(), 'little')) for text in range(text_quantity)]
    
    return texts



@timeit(display_args=False)
def m2(args):
    keys = {}
    alpha, beta = args
    texts = read(config.texts)
    for k in range(1 << 16):
        u_k = 0
        for x, y in texts:
            x1 = HeysCipher(config.s_block, config.s_block_rev).round(x, k)
            if mul(alpha, x1)^mul(beta, y) == 0:
                u_k += 1
            else:
                u_k -= 1
            keys[k] = abs(u_k)
    keys = [k for k, v in tuple(sorted(keys.items(), key=operator.itemgetter(1), reverse = True))[:100]]
    keys_m2.append(keys)

    

if __name__ == '__main__':
    if not Path('./saves/approximations').is_dir():
        Path("./saves/approximations").mkdir(parents=True)
    if not Path('./saves/materials').is_dir():
        Path("./saves/materials").mkdir(parents=True)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        os.system('chmod +x heys.bin')
    if not os.listdir(Path('./saves/approximations')):
        approximations = []
        alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0)>=3]

        if len(alpha_array) >= cpu_count():
            num_processes = cpu_count() - 2
        else:
            num_processes = len(alpha_array)

        with Pool(processes=num_processes) as pool:
            for alpha_lin_search_res in pool.map(linsearch_worker, alpha_array):
                if alpha_lin_search_res != []:
                    for ri in alpha_lin_search_res:
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
    appr_list_ab = [(ab[0], ab[1]) for ab, p in approximations]
    if len(os.listdir(Path('./saves/materials')))//2 != config.texts:
        create_statistical_materials(config.texts)
    num_processes = cpu_count() - 4
    
    
    if not Path('./saves/keys_m2.pkl').exists():
        keys_m2 = Manager().list()
        with Pool(processes=num_processes) as pool:
            pool.map(m2, appr_list_ab[:50])
        
        with open(Path('./saves/keys_m2.pkl'), 'wb') as f:
            pickle.dump(keys_m2, f)
    else:
        with open(Path('./saves/keys_m2.pkl'), 'rb') as f:
            keys_m2 = pickle.load(f)
            
    keys = {}
    for k in keys_m2:
        for ki in k:
            if ki in keys:
                keys[ki] += 1
            else:
                keys[ki] = 1
    keys = dict(Counter(keys).most_common(10))
    logging.info(keys)
    for k, v in keys.items():       # dict(sorted(keys.items(), key=operator.itemgetter(1), reverse=True)[:10]).items():
        print(f'key = {k}\nstat = {v}')