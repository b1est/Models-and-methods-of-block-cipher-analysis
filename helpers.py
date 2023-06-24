import time
from functools import wraps
import logging

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
