from heys import HeysCipher
from config import Config
from multiprocessing import cpu_count, Pool
from helpers import timeit
import logging
import os, sys
import pickle
from pathlib import Path
from linear_approximations_search import linear_approximations_search as linsearch
from linear_approximations_search import scalar_mul as mul
from subprocess import Popen, PIPE 
from math import ceil

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

@timeit(display_args=False)
def get_approximations(save_path: Path):
	approximations_file = save_path / "approximations.pkl"
	saves_approximations_path = saves_path / 'approximations'


	if not saves_approximations_path.is_dir():
		saves_approximations_path.mkdir(parents=True)
		
	if approximations_file.exists():
		with open(approximations_file, 'rb') as f:
			return pickle.load(f)
	else:
		if sys.platform == 'linux' or sys.platform == 'linux2' or sys.platform == 'darwin':
			os.system('chmod +x heys.bin')

		if not os.listdir(saves_approximations_path):

			approximations = list()
			alpha_array = [alpha[1] for alpha in [([alpha >> 4 * i & 0xf for i in range(4)], alpha) for alpha in range(1, 1 << 16)] if alpha[0].count(0)>=3]
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

		approximations = sorted(approximations, key = lambda x: x[1], reverse = True)
		logging.info(approximations)
		if input(f"Would you like to change amount of pairs plaintext-ciphertext (now = {config.texts}) to {int(1 / approximations[-1][1])}? (y/n): ") == "y":
			config.texts = int(1 / approximations[-1][1])
		approximations = [(ab[0], ab[1]) for ab, p in approximations]
		logging.info(approximations)
		with open(approximations_file, 'wb') as f:
			pickle.dump(approximations, f)
		return approximations


def read(text_quantity):
	if Path('./saves/materials/texts.pkl').exists():
		with open('./saves/materials/texts.pkl', 'rb') as f:
			texts = pickle.load(f)
	else:
		texts = []
		for text in range(text_quantity):
			texts.append((int.from_bytes(open(f'./saves/materials/pt_{text}.bin', 'rb').read(), byteorder='little'), int.from_bytes(open(f'./saves/materials/ct_{text}.bin', 'rb').read(), byteorder='little')))
	return texts

def create_statistical_materials(text_quantity):
	if len(os.listdir(saves_materials_path))//2 != config.texts:
		for text in range(text_quantity):
			input_file = Path(f'./saves/materials/pt_{text}.bin')
			output_file = Path(f'./saves/materials/ct_{text}.bin')
			with open(input_file, 'wb') as file:
				file.write(text.to_bytes(2, byteorder='little')) 
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

def calculate_uk(alpha, beta, k):
	with open('./saves/materials/texts.pkl', 'rb') as f:
		texts = pickle.load(f)
	uk = 0
	cipher = HeysCipher(config.s_block, config.s_block_rev)
	for x, y in texts:
		x1 = cipher.round(x, k)
		if mul(alpha, x1)^mul(beta, y) == 0:
			uk+=1
		else:
			uk-=1
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
			result = pool.starmap_async(calculate_uk, params, chunksize=ceil(len(params) / len(pool._pool)))
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
			if len(alpha_array) >= cpu_count()-2:
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
		approximations = sorted(approximations, key = lambda x: x[1], reverse = True)
		# config.texts = int(1 / approximations[-1][1])
		approximations = [(ab[0], ab[1]) for ab, p in approximations]
		all_sorted_path = saves_approximations_path / 'all_sorted.pkl'
		if not all_sorted_path.exists():
			with open(all_sorted_path, 'wb') as f:
				pickle.dump(approximations, f)

		for alpha, beta in approximations:
			path_m2_uk_folder_path = Path(f'./saves/m2/uk/{alpha}-{beta}')
				
			if not path_m2_uk_folder_path.is_dir():
				path_m2_uk_folder_path.mkdir(parents=True)
		
		texts = create_statistical_materials(config.texts)

		params = [(alpha, beta, k) for alpha, beta in approximations for k in range(1 << 16)]

		keys = m2(params)

		# m2 var
		










			# start = time.time()
			# for r in pool.starmap(calculate_uk, params, chunksize=1):
			# 	ab, k, uk = r
			# 	keys[ab].append((uk, k))

			# print(f"Time for chunksize = 1 - {time.time()-start}")

			# start = time.time()
			# for r in pool.starmap(calculate_uk, params, chunksize=None):
			# 	ab, k, uk = r
			# 	keys[ab].append((uk, k))

			# print(f"Time for chunksize = None (default) - {time.time()-start}")



		# for alpha, beta in approximations:
		# 	path_m2_uk_folder_path = Path(f'./saves/m2/uk/{alpha}-{beta}')

		# 	for k in range(1 << 16):
		# 		path_m2_uk_file_path = path_m2_uk_folder_path / f'{k}.pkl'
		# 		uk = u_k(texts, alpha, beta, k)
		# 		with open(path_m2_uk_file_path, 'wb') as f:
		# 			pickle.dump(uk, f)

		# alpha, beta = approximations[0]
		# for alpha, beta in approximations:
		# 	for k in range(1 << 16):
		# 		uk = u_k(texts, alpha, beta, k)
		# 		uk2 = u_k2(texts, alpha, beta, k)
		# 		print(uk, uk2)
		# 		if uk != uk2:
		# 			print("hmmm")
		# uk_precalc(14, 0, 1 << 16, 1000)

	#     saved_keys_m2 = get_saves_m2_and_ab(saves_m2_path, approximations)

	#     keys_m2 = None

	#     if saved_keys_m2 != None:
	#         approximations, keys_m2 = saved_keys_m2

	#     num_processes = max(cpu_count()-2, 1)

	#     if keys_m2:
	#         keys_m2 = Manager().list(keys_m2)
	#     else:
	#         keys_m2 = Manager().list()

	#     appr_list_ab = [(ab[0], ab[1], keys_m2, texts) for ab in approximations]
	#     #  chunksize=len(appr_list_ab)/num_processes
	#     if appr_list_ab:
	#         with Pool(processes=num_processes) as pool:
	#             pool.map(m2, appr_list_ab)

	#     with open(saves_keys_m2_path, 'wb') as f:
	#         pickle.dump(list(keys_m2), f)
	# else:
	#     with open(saves_keys_m2_path, 'rb') as file:
	#         keys_m2 = pickle.load(file)
		
	# keys = dict()
	# for k in keys_m2:
	#     for ki in k:
	#         if ki in keys:
	#             keys[ki] += 1
	#         else:
	#             keys[ki] = 1
	# # with open('1.txt', 'w') as file:
	# #     file.write(str(keys))
	# # # print(max(keys, key=keys.get))
	# # print(list(zip(*sorted(keys.items(), key=lambda x: x[1],reverse=True)))[1])
	# # print(sorted(keys, key=keys.get, reverse = True) == [k for k,v in  sorted(keys.items(), key=lambda x: x[1],reverse=True)])
	
	# # stat, candidates = zip(*sorted(zip(keys.items(), range(1 << 16)), reverse = True))
	# # print(list(map(hex, candidates[:10])), candidates[:10], stat[:10], sorted(stat, key= lambda x: x[1], reverse=True))
	# candidates = sorted(keys, key=keys.get, reverse=True)
	# logging.info(list(map(hex, candidates[:10])))
	# logging.info(candidates[:10])
