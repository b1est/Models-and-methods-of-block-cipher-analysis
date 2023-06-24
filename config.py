import numpy as np
from pathlib import Path

class Singleton(object):
	_instance = None

	def __new__(cls, *args, **kwargs):
		if not isinstance(cls._instance, cls):
			cls._instance = object.__new__(cls, *args, **kwargs)
		return cls._instance


class Config(Singleton):
	s_block :  list = [10, 9, 13, 6, 14, 11, 4, 5, 15, 1, 3, 12, 7, 0, 8, 2]
	s_block_rev: list = [13, 9, 15, 10, 6, 7, 3, 12, 14, 1, 0, 5, 11, 2, 4, 8]
	texts: int = 2000
	keys: list = [i for i in range(1 << 16)]