from heys import HeysCipher
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

logging.basicConfig(filename='lab_logs.log', format='%(message)s', level=logging.INFO)

config = Config()

@timeit(display_args=True)
def linsearch_worker(alpha):
    save_file = Path(f'./saves/approximations/{alpha}.pkl')
    approximations = list()
    approximations_alpha = linsearch(alpha)
    if approximations_alpha != {}:
        for beta in approximations_alpha:
                approximations.append(((alpha, beta), approximations_alpha[beta]))
        with open(save_file, 'wb') as file:
            pickle.dump(approximations, file)
    return approximations

def read(text_quantity):
    texts = []
    for text in range(text_quantity):
        texts.append((int.from_bytes(open(f'./saves/materials/pt_{text}.bin', 'rb').read(), 'little'), int.from_bytes(open(f'./saves/materials/ct_{text}.bin', 'rb').read(), 'little')))
    return texts

def create_statistical_materials(text_quantity):
    if len(os.listdir(saves_materials_path))//2 != config.texts:
        for text in range(text_quantity):
            input_file = Path(f'./saves/materials/pt_{text}.bin')
            output_file = Path(f'./saves/materials/ct_{text}.bin')
            with open(input_file, 'wb') as file:
                file.write(int(text).to_bytes(2, 'little')) 
            if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
                Popen(['./heys.bin', 'e', '01', input_file, output_file], stdin = PIPE, stderr=True).communicate()
            elif sys.platform == 'win32':
                Popen(f'Heys e 01 {input_file} {output_file}', stdin = PIPE, stderr=True).communicate('\n'.encode())
    return read(text_quantity)

def get_saves_m2_and_ab(path: Path or str, appr = None):
    if isinstance(path, str):
        path = Path(path)
    dir_list = os.listdir(path)
    if not dir_list:
        return None
    else:
        res = []
        if not appr:
            ab = [tuple(map(int, file[:-4].split('-'))) for file in dir_list]
            for filename in dir_list:
                with open(path/filename, 'rb') as file:
                    res.append(pickle.load(file))
        else:
            ab = []
            for i in appr:
                try:
                    with open(path/f'{i[0]}-{i[1]}.pkl', 'rb') as file:
                        res.append(pickle.load(file))
                except FileNotFoundError:
                    ab.append(i)
        return ab, res

@timeit(display_args=False)
def m2(args):
    keys = dict()
    alpha, beta, keys_m2, texts = args
    path_m2_file = Path('./saves/m2') / f'{alpha}-{beta}.pkl'
    for k in range(1 << 16):
        u_k = 0
        for x, y in texts:
            x1 = HeysCipher(config.s_block, config.s_block_rev).round(x, k)
            if mul(alpha, x1)^mul(beta, y) == 0:
                u_k += 1
            else:
                u_k -= 1
        keys[k] = abs(u_k)
    # _, keys = zip(*sorted(zip(keys.values(), range(1 << 16)), reverse = True))
    keys = sorted(keys, key=keys.get, reverse=True)
    keys_m2.append(keys[:100])
    with open(path_m2_file, 'wb') as file:
        pickle.dump(keys[:100], file)

if __name__ == '__main__':
    saves_path = Path('./saves')
    saves_keys_m2_path = saves_path / 'keys_m2.pkl'
    saves_m2_path = saves_path / 'm2'
    saves_approximations_path = saves_path / 'approximations'
    saves_approximations_all_path = saves_approximations_path / 'all.pkl'
    saves_materials_path = saves_path / 'materials'

    if not saves_approximations_path.is_dir():
        saves_approximations_path.mkdir(parents=True)

    if not saves_materials_path.is_dir():
        saves_materials_path.mkdir(parents=True)

    if not saves_m2_path.is_dir():
        saves_m2_path.mkdir(parents=True)

    if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
        os.system('chmod +x heys.bin')

    if not os.listdir(saves_approximations_path):
        approximations = list()
        alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0)>=3]
        if len(alpha_array) >= cpu_count():
            num_processes = max(cpu_count()-2, 1)
        else:
            num_processes = len(alpha_array)
        
        with Pool(processes=num_processes) as pool:
            for alpha_lin_search_res in pool.map(linsearch_worker, alpha_array):
                if alpha_lin_search_res != []:
                    for ri in alpha_lin_search_res:
                        approximations.append(ri)

        with open(saves_approximations_all_path, 'wb') as f:
                    pickle.dump(approximations, f)
    else:    
        if saves_approximations_all_path.exists():
            with open(saves_approximations_all_path, 'rb') as f:
                approximations = pickle.load(f)
        else:
            approximations = list()
            for filename in os.listdir(saves_approximations_path):
                with open(saves_approximations_path/filename, 'rb') as f:
                    for ri in pickle.load(f): 
                        approximations.append(ri)
            with open(saves_approximations_all_path, 'wb') as f:
                    pickle.dump(approximations, f)
    
    if not saves_keys_m2_path.exists():
        approximations = [(ab[0], ab[1]) for ab, p in sorted(approximations, key = lambda x: x[1], reverse = True)]

        if len(approximations) > 300:
            approximations = approximations[:300]

        texts = create_statistical_materials(config.texts)

        res = get_saves_m2_and_ab(saves_m2_path, approximations)

        keys_m2 = None

        if res != None:
            approximations, keys_m2 = res

        num_processes = max(cpu_count()-2, 1)

        if keys_m2:
            keys_m2 = Manager().list(keys_m2)
        else:
            keys_m2 = Manager().list()

        appr_list_ab = [(ab[0], ab[1], keys_m2, texts) for ab in approximations]

        if appr_list_ab:
            with Pool(processes=num_processes) as pool:
                pool.map(m2, appr_list_ab)

        with open(saves_keys_m2_path, 'wb') as f:
            pickle.dump(list(keys_m2), f)
    else:
        with open(saves_keys_m2_path, 'rb') as file:
            keys_m2 = pickle.load(file)
        
    keys = dict()
    for k in keys_m2:
        for ki in k:
            if ki in keys:
                keys[ki] += 1
            else:
                keys[ki] = 1
    # with open('1.txt', 'w') as file:
    #     file.write(str(keys))
    # # print(max(keys, key=keys.get))
    # print(list(zip(*sorted(keys.items(), key=lambda x: x[1],reverse=True)))[1])
    # print(sorted(keys, key=keys.get, reverse = True) == [k for k,v in  sorted(keys.items(), key=lambda x: x[1],reverse=True)])
    
    # stat, candidates = zip(*sorted(zip(keys.items(), range(1 << 16)), reverse = True))
    # print(list(map(hex, candidates[:10])), candidates[:10], stat[:10], sorted(stat, key= lambda x: x[1], reverse=True))
    candidates = sorted(keys, key=keys.get, reverse=True)
    logging.info(list(map(hex, candidates[:10])))
    logging.info(candidates[:10])
