from heys import HeysCipher
from config import Config
from multiprocessing import cpu_count, Pool
from helpers import timeit
import logging
import os
import sys
import pickle
from pathlib import Path
from linear_approximations_search import linear_approximations_search as linsearch
from linear_approximations_search import scalar_mul as mul
from subprocess import Popen, PIPE
from math import ceil
from collections import Counter
import time


logging.basicConfig(filename='lab_logs.log',
                    format='%(message)s', level=logging.INFO)

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


@timeit(display_args=False)
def get_approximations(save_path: Path, num_of_appr=None):
    approximations_file = save_path / "approximations.pkl"
    saves_approximations_path = saves_path / 'approximations'

    if not saves_approximations_path.is_dir():
        saves_approximations_path.mkdir(parents=True)

    if approximations_file.exists():
        with open(approximations_file, 'rb') as f:
            approximations = pickle.load(f)
    else:
        if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
            os.system('chmod +x heys.bin')

        if not os.listdir(saves_approximations_path):

            approximations = list()
            alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(
                4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0) >= 3]
            if len(alpha_array) >= cpu_count()-2:
                num_processes = max(cpu_count()-2, 1)
            else:
                num_processes = len(alpha_array)

            with Pool(processes=num_processes) as pool:
                for alpha_lin_search_res in pool.map(linsearch_worker, alpha_array):
                    if alpha_lin_search_res != []:
                        for ri in alpha_lin_search_res:
                            approximations.append(ri)
        else:
            approximations = list()
            for filename in os.listdir(saves_approximations_path):
                with open(saves_approximations_path/filename, 'rb') as f:
                    for ri in pickle.load(f):
                        approximations.append(ri)

    approximations = sorted(approximations, key=lambda x: x[1], reverse=True)

    with open(approximations_file, 'wb') as f:
        pickle.dump(approximations, f)

    if num_of_appr:
        approximations = approximations[:num_of_appr]
        config.texts = int(1 / approximations[-1][1])

    approximations = [(ab[0], ab[1]) for ab, p in approximations]

    return approximations


def create_statistical_materials(save_path: Path):
    texts_path = save_path / "texts.pkl"
    saves_materials_path = saves_path / 'materials'
    if not saves_materials_path.is_dir():
        saves_materials_path.mkdir(parents=True)
    if texts_path.exists():
        with open(texts_path, 'rb') as f:
            texts = pickle.load(f)
        if len(texts) != config.texts:
            if len(texts) > config.texts:
                texts = texts[:config.texts]
            else:
                last_plaintext = texts[-1][0]
                for text in range(last_plaintext+1, config.texts):
                    input_file = saves_materials_path / f'pt_{text}.bin'
                    output_file = saves_materials_path / f'ct_{text}.bin'
                    with open(input_file, 'wb') as file:
                        file.write(text.to_bytes(2, byteorder='little'))
                    if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
                        os.system('chmod +x heys.bin')
                        Popen(['./heys.bin', 'e', '01', input_file, output_file],
                              stdin=PIPE, stderr=True).communicate()
                    elif sys.platform == 'win32':
                        Popen(f'Heys e 01 {input_file} {output_file}', stdin=PIPE, stderr=True).communicate(
                            '\n'.encode())

                    texts.append((int.from_bytes(open(f'./saves/materials/pt_{text}.bin', 'rb').read(), byteorder='little'), int.from_bytes(
                        open(f'./saves/materials/ct_{text}.bin', 'rb').read(), byteorder='little')))

                with open(texts_path, 'wb') as f:
                    pickle.dump(texts, f)
    else:
        texts = []
        for text in range(config.texts):
            input_file = Path(f'./saves/materials/pt_{text}.bin')
            output_file = Path(f'./saves/materials/ct_{text}.bin')
            with open(input_file, 'wb') as file:
                file.write(text.to_bytes(2, byteorder='little'))
            if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
                os.system('chmod +x heys.bin')
                Popen(['./heys.bin', 'e', '01', input_file, output_file],
                      stdin=PIPE, stderr=True).communicate()
            elif sys.platform == 'win32':
                Popen(f'Heys e 01 {input_file} {output_file}',
                      stdin=PIPE, stderr=True).communicate('\n'.encode())

            texts.append((int.from_bytes(open(f'./saves/materials/pt_{text}.bin', 'rb').read(), byteorder='little'), int.from_bytes(
                open(f'./saves/materials/ct_{text}.bin', 'rb').read(), byteorder='little')))

        with open(texts_path, 'wb') as f:
            pickle.dump(texts, f)

    return texts


@timeit(display_args=True)
def calculate_uk(alpha, beta, k):
    with open('./saves/texts.pkl', 'rb') as f:
        texts = pickle.load(f)
    uk = 0
    cipher = HeysCipher(config.s_block, config.s_block_rev)
    for x, y in texts:
        x1 = cipher.round(x, k)
        if mul(alpha, x1) ^ mul(beta, y) == 0:
            uk += 1
        else:
            uk -= 1
    return ((alpha, beta), (abs(uk), k))


@timeit(display_args=False)
def m2(params, approximations):
    saves_path = Path('./saves')
    uk_file_path = saves_path / "uk.pkl"
    m2_file_path = saves_path / "m2.pkl"
    if m2_file_path.exists():
        with open(m2_file_path, 'rb') as f:
            return pickle.load(f)
    elif uk_file_path.exists():
        with open(uk_file_path, 'rb') as f:
            uk = pickle.load(f)
        for a, b in approximations:
            k = keys[(a, b)]
            _, k = zip(*sorted(k, reverse=True))
            keys[(a, b)] = k[:100]
        keys = list(keys.values())
        with open(m2_file_path, 'wb') as f:
            pickle.dump(keys, f)
        return keys
    else:
        keys = {(a, b): [] for a, b in approximations}
        with Pool() as pool:
            result = pool.starmap_async(
                calculate_uk, params, chunksize=ceil(len(params) / len(pool._pool)))
            for r in result.get():
                ab, uk = r
                keys[ab].append(uk)

        with open(saves_path / "uk.pkl", "wb") as f:
            pickle.dump(keys, f)

        for a, b in approximations:
            k = keys[(a, b)]
            _, k = zip(*sorted(k, reverse=True))
            keys[(a, b)] = k[:100]

        keys = list(keys.values())

        with open(saves_path / "m2.pkl", "wb") as f:
            pickle.dump(keys, f)

        return keys


if __name__ == '__main__':
    saves_path = Path('./saves')

    if not saves_path.is_dir():
        saves_path.mkdir(parents=True)

    approximations = get_approximations(saves_path)

    texts = create_statistical_materials(saves_path)

    params = [(alpha, beta, k)
              for alpha, beta in approximations for k in range(1 << 16)]

    logging.info('Attack has started')
    print('Attack has started')
    start_time = time.time()
    
    m2_result = m2(params, approximations)

    keys = [(v, k) for k, v in Counter([item for sub_list in m2_result for item in sub_list]).items()]

    statistic, candidates = zip(*sorted(keys, reverse=True))
    candidates = list(map(lambda x: x[2:], map(hex, candidates)))
    total_time = time.time() - start_time

    logging.info(f'Attack is successful, time: {total_time:.4f} seconds')
    print(f'Attack is successful, time: {total_time:.4f} seconds')
    logging.info(candidates[:10])
    logging.info(statistic[:10])

