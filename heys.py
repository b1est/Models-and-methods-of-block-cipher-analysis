class HeysCipher:
    def __init__(self, s_block: list, s_block_rev: list, keys: int or list = 0, rounds = 6):
        self.rounds = rounds
        self.keys = self.expand_key(keys)
        self.s_block = s_block
        self.s_block_rev = s_block_rev

    def S(self, x: int, encrypt: bool = True) -> int:
        s = []
        new_x = 0
        for i in range(4):
            si = x & 0xF
            s.append(si)
            x >>= 4
        s_block = self.s_block if encrypt == True else self.s_block_rev
        for i in range(4):
            new_x ^= (s_block[s[i]] << 4 * i)
        return new_x

    def expand_key(self, key: int or list) -> list:
        if isinstance(key, list):
            return key
        else:
            round_keys = []
            for i in range(self.rounds+1):
                round_key = key & 0xFFFF
                round_keys.append(round_key)
                key >>= 16
            return round_keys

    def L(self, num: int) -> int:
        result = 0

        for j in range(4):
            for i in range(4):
                result |= (num >> (4 * j + i) & 1) << (4 * i + j)

        return result
    
    def round(self, x: int, key: int) -> int:
        return self.L(self.S(x ^ key, True))
    
    def round_rev(self, x: int, key: int) -> int: 
        return self.S(self.L(x ^ key), False)

    def encrypt(self, num: int) -> int:
        for i in range(self.rounds):
            num = self.round(num, self.keys[i])
        return num ^ self.keys[self.rounds]
    
    def decrypt(self, num: int) -> int:
        for i in range(self.rounds):
            num = self.round_rev(num, self.keys[::-1][i])
        return num ^ self.keys[self.rounds]