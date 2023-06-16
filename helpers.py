import time
from functools import wraps
import logging
import ast


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
		
			msg += f"({args})" if display_args else "()"
			full_output_msg = f"Taken time for execution of the function {msg} in seconds - {total_time:.4f}."
			logging.info(full_output_msg)
			print(full_output_msg)
			return result

		return timeit_wrapper

	return decorator


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def fastest_reader(filename = 'f.txt'):
	i = 0
	differentials = {}
	with open(filename) as file:
		for line in file:
			differentials.update({i: dict(ast.literal_eval(line))})
			i+=1
	differentials = list(map(lambda x: (x[0], list(x[1].items())), differentials.items()))
	return differentials