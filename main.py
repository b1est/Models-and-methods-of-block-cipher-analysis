from heys import HeysCipher
from collections import Counter
from config import Config
from multiprocessing import cpu_count, Pool, Manager
from helpers import timeit
import logging
import os, sys
import pickle
from pathlib import Path

logging.basicConfig(filename='lab_logs.log', format='%(message)s', level=logging.INFO)


config = Config()



def delta(beta):
    gs = [HeysCipher(config.s_block, config.s_block_rev).round(x, 0) ^ HeysCipher(config.s_block, config.s_block_rev).round(x ^ beta, 0) for x in range(1 << 16)]
    return dict([(gamma, count / (1 << 16)) for gamma, count in Counter(gs).items()])


@timeit(display_args=True) 
def differential_search(alpha, p = [1, 0.1, 0.003, 0.001, 0.0002, 0.00012], r = 6):
    InitialGamma = {alpha: 1}
    for t in range(1, r):
        GammaIndex = {}
        for bi in InitialGamma:
            delta_iter = delta(bi)
            for gj in delta_iter:
                if gj in GammaIndex:
                    GammaIndex[gj] += InitialGamma[bi]*delta_iter[gj]
                else:
                    GammaIndex[gj] = InitialGamma[bi]*delta_iter[gj]
        
        keys_for_delete = [key for key in GammaIndex if GammaIndex[key] <= p[t]]
        for key in keys_for_delete:
            GammaIndex.pop(key)
        InitialGamma = GammaIndex.copy()
    return GammaIndex

@timeit(display_args=True)
def diff_search_alpha_worker(alpha):
    save_file = Path(f'./saves/differentials/{alpha}.pkl')
    differentials = []
    differentials_alpha = differential_search(alpha)
    if differentials_alpha != {}:
        for beta in differentials_alpha:
                differentials.append(((alpha, beta), differentials_alpha[beta]))
        with open(save_file, 'wb') as file:
            pickle.dump(differentials, file)
    return differentials

def create_statistical_materials(text_quantity, alpha):
    for text in range(text_quantity):
        input_file = Path(f'./saves/materials/pt_{text}_{alpha}.bin')
        output_file = Path(f'./saves/materials/ct_{text}_{alpha}.bin')
        input_file_d = Path(f'./saves/materials/pt_muted_{text}_{alpha}.bin')
        output_file_d = Path(f'./saves/materials/ct_muted_{text}_{alpha}.bin')

        with open(input_file, 'wb') as file:
            file.write(int(text).to_bytes(2, 'little'))
        with open(input_file_d, 'wb') as file:
            file.write(int(text ^ alpha).to_bytes(2, 'little'))
        
        if sys.platform == "linux" or sys.platform == "linux2":
            os.system('chmod +x heys.bin')
            os.system(f'./heys.bin e 01 {input_file} {output_file}')
            os.system(f"./heys.bin e 01 {input_file_d} {output_file_d}")
        elif sys.platform == "darwin":
            os.system('chmod +x heys.bin')
            os.system(f'./heys.bin e 01 {input_file} {output_file}')
            os.system(f"./heys.bin e 01 {input_file_d} {output_file_d}")
        elif sys.platform == "win32":
            os.system(f"Heys e 01 {input_file} {output_file}")
            os.system(f"Heys e 01 {input_file_d} {output_file_d}")
    

def read(texts, alpha):
    ct_texts = [int.from_bytes(open(f'./saves/materials/ct_{text}_{alpha}.bin', 'rb').read(), 'little') for text in range(texts)]
    ct_muted = [int.from_bytes(open(f'./saves/materials/ct_muted_{text}_{alpha}.bin', 'rb').read(), 'little') for text in range(texts)]
    return ct_texts, ct_muted




def last_round_attack(args):
        
        text, muted, beta, key, keys = args
        for c1, c2 in zip(text, muted):
            pos_beta = HeysCipher(config.s_block, config.s_block_rev).round_rev(c1, key) ^ HeysCipher(config.s_block, config.s_block_rev).round_rev(c2, key)
            if pos_beta == beta:
                if key in keys:
                    keys[key] += 1
                else:
                    keys[key] = 1

    

if __name__ == '__main__':
    keys = Manager().dict()
    if not Path('./saves/differentials').is_dir():
        Path("./saves/differentials").mkdir(parents=True)
    if not Path('./saves/materials').is_dir():
        Path("./saves/materials").mkdir(parents=True)
    if not os.listdir(Path('./saves/differentials')):
        differentials = []
        alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0)>=3]

        if len(alpha_array) >= cpu_count():
            num_processes = cpu_count() - 2
        else:
            num_processes = len(alpha_array)

        with Pool(processes=num_processes) as pool:
            for alpha_diff_search_res in pool.map(diff_search_alpha_worker, alpha_array):
                if alpha_diff_search_res != []:
                    for ri in alpha_diff_search_res:
                        differentials.append(ri)

        with open(Path(f'./saves/differentials/all.pkl'), 'wb') as f:
                    pickle.dump(differentials, f)
    else:    
        if Path('./saves/differentials/all.pkl').exists():
            with open(f'./saves/differentials/all.pkl', 'rb') as f:
                differentials = pickle.load(f)
        else:
            differentials = []
            for filename in os.listdir(Path('./saves/differentials/')):
                with open(Path(f'./saves/differentials/{filename}'), 'rb') as f:
                    for ri in pickle.load(f): 
                        differentials.append(ri)
            with open(Path(f'./saves/differentials/all.pkl'), 'wb') as f:
                    pickle.dump(differentials, f)
    
    
    differentials = sorted(differentials, key = lambda x: x[1], reverse = True)
    
    num_processes = cpu_count() - 4
    for iteration, (ab, p) in enumerate(differentials):
        alpha, beta = ab
        create_statistical_materials(config.texts, alpha)
        texts, muted = read(config.texts, alpha)
        args = [(texts, muted, beta, k, keys) for  k in config.keys]
        with Pool(processes=num_processes) as pool:
            pool.map(last_round_attack, args)
        
        maximum_frequency_keys = []
        for k,v in keys.items():
            if v == max(keys.values()):
                maximum_frequency_keys.append(k)

        maximum_frequency_keys.sort()

        logging.info(f'Iteration: {iteration+1}/nKeys with max frequency: {maximum_frequency_keys}/n---------------------------')
        print(f'Iteration: {iteration+1}/nKeys with max frequency: {maximum_frequency_keys}/n---------------------------')
        if  len(maximum_frequency_keys) == 1:
            break