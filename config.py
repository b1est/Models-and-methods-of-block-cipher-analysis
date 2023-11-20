class Singleton(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


class Config(Singleton):
    s_block:  list = [0xA, 9, 0xD, 6, 0xE, 0xB, 4, 5, 0xF, 1, 3, 0xC, 7, 0, 8, 2]
    s_block_rev: list = [0xD, 9, 0xF, 0xA, 6, 7, 3, 0xC, 0xE, 1, 0, 5, 0xB, 2, 4, 8]
    texts: int = 9999
    rounds: int = 6
