class HeysCipher:
    def __init__(self, s_block: list, s_block_rev: list, keys: int or list = 0 ):
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
            for i in range(7):
                round_key = key & 0xFFFF
                round_keys.append(round_key)
                key >>= 16
            return round_keys

    def L(self, num: int) -> int:
        bit_positions = {0: 0, 1: 4, 2: 8, 3: 12, 4: 1, 5: 5, 6: 9, 7: 13, 8: 2, 9: 6, 10: 10, 11: 14, 12: 3, 13: 7, 14: 11, 15: 15}
        result = 0

        for i, j in bit_positions.items():
            num_shifted = num >> i
            ith_bit = num_shifted & 0b1
            ith_bit_shifted = ith_bit << j
            result |= ith_bit_shifted

        return result
    
    def round(self, x: int, key: int, enc: bool = True) -> int:
        if enc == True:
            y = x ^ key
            s = self.S(y, enc)
            round_output = self.L(s)
        else:
            l = self.L(x)
            s = self.S(l, enc)
            round_output = s ^ key
        return round_output

    def encrypt(self, num: int) -> int:
        for i in range(6):
            if i == 0:
                r = self.round(num, self.keys[i])
            else:
                r = self.round(r, self.keys[i])    
        r = r ^ self.keys[6]
        return r
    
    def decrypt(self, num: int) -> int:
        for i in range(6):
            if i == 0:
                r = self.round(num, self.keys[i], False)
            else:
                r = self.round(r, self.keys[i], False)    
        r = r ^ self.keys[6]
        return r


if __name__ == '__main__':
        num = 0x3331
        keys = 0
        s_block = [10, 9, 13, 6, 14, 11, 4, 5, 15, 1, 3, 12, 7, 0, 8, 2]
        s_block_rev = [13, 9, 15, 10, 6, 7, 3, 12, 14, 1, 0, 5, 11, 2, 4, 8]
        cipher = HeysCipher(s_block, s_block_rev, keys)
        encrypted = cipher.encrypt(num)
        decr = cipher.decrypt(encrypted)
        print(decr, num)