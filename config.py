import numpy as np

class Singleton(object):
	_instance = None

	def __new__(cls, *args, **kwargs):
		if not isinstance(cls._instance, cls):
			cls._instance = object.__new__(cls, *args, **kwargs)
		return cls._instance


class Config(Singleton):
	s_block :  list
	s_block_rev: list
	diff_probs_table_s: np.ndarray
	