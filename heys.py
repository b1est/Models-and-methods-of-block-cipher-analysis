import random
import config


class HeysCipher:
    def __init__(self, s_block: list, s_block_rev: list, keys: list or int = 0, rounds: int = 6):
        self.rounds = rounds
        self.keys = self.expand_key(keys)
        self.s_block = s_block
        self.s_block_rev = s_block_rev

    def S(self, msg: int, encrypt: bool = True) -> int:
        s_block = self.s_block if encrypt == True else self.s_block_rev
        res = 0
        for j in range(4):
            res ^= (s_block[(msg >> (4-j-1)*4) & 0b1111] << (4-j-1)*4)
        return res

    def expand_key(self, key: int or list) -> list:
        if isinstance(key, list):
            return key
        else:
            round_keys = list()
            for i in range(self.rounds+1):
                round_key = key & 0xFFFF
                round_keys.append(round_key)
                key >>= 16
            round_keys.reverse()
            return round_keys

    def L(self, num: int) -> int:
        result = 0
        for j in range(4):
            for i in range(4):
                bit = (num >> (4*j+i)) & 0b1
                result ^= (bit << (j+4*i))
        return result

    def round(self, x: int, key: int) -> int:
        return self.L(self.S(x ^ key, True))

    def round_rev(self, x: int, key: int) -> int:
        return self.S(self.L(x), False) ^ key

    def encrypt(self, num: int) -> int:
        for i in range(self.rounds):
            num = self.round(num, self.keys[i])
        return num ^ self.keys[self.rounds]

    def decrypt(self, num: int) -> int:
        num ^= self.keys[::-1][0]
        for i in range(1, self.rounds+1):
            num = self.round_rev(num, self.keys[::-1][i])
        return num


def generate_keys(rounds):
    return [random.randrange(0, 1 << 16) for _ in range(rounds+1)]


if __name__ == "__main__":

    config_obj = config.Config()
    s_block = config_obj.s_block
    s_block_rev = config_obj.s_block_rev
    rounds = config_obj.rounds

    plain_text = random.randrange(0, 1 << 16)

    keys = generate_keys(rounds)

    key_string = ''
    for k in keys:
        key_string += hex(k)[2:]

    print(f"Key = {key_string}, Message = {hex(plain_text)[2:]}")

    cipher = HeysCipher(s_block, s_block_rev, keys, rounds)

    encrypted = cipher.encrypt(plain_text)

    print(f"Encrypted = {hex(encrypted)[2:]}")

    decrypted = cipher.decrypt(encrypted)

    print(f"Decrypted = {hex(decrypted)[2:]}")
