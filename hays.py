

def S(x: int, s_block: list) -> int:
    
    s = []
    newx = 0
    # print(f"x_s = {bin(x)[2:]}")
    for i in range(4):
        si = x & 0xF
        s.append(si)
        x >>= 4
    
    # for i in s:
    #     print(f"{bin(i)[2:]} --> {hex(S[i])[2:].upper()} ({bin(S[i])[2:]})")


    for i in range(4):
        newx ^= (s_block[s[i]] << 4*i)
    return newx

# Функція розбиття ключа на раундові підключі
def expand_key(key: int | list) -> list:
    if isinstance(key, list):
        for i, k in zip(range(7), key):
            if i < 6:
                print(f"K{i} = {hex(k)[2:].upper()}", end = ' ')
            else:
                print(f"K{i} = {hex(k)[2:].upper()}")
        
        return key
    else:
        round_keys = []
        for i in range(7):
            round_key = key & 0xFFFF
            round_keys.append(round_key)
            key >>= 16
    return round_keys

# input =     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# output =    [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]

# for i, j in zip(input, output):
#     print(f"{i - 1} - {j - 1}", end = ", ")

def L(num: int) -> int:
    # Define a dictionary to map each bit to its new position
    bit_positions = {0: 0, 1: 4, 2: 8, 3: 12, 4: 1, 5: 5, 6: 9, 7: 13, 8: 2, 9: 6, 10: 10, 11: 14, 12: 3, 13: 7, 14: 11, 15: 15}

    # Create a new variable to store the result
    result = 0

    # Loop over each bit position and move the bit to its new position
    for i, j in bit_positions.items():
        # Right-shift the input number by i positions to make the i-th bit the least significant bit.
        num_shifted = num >> i
        # Mask the rightmost bit to get the i-th bit.
        ith_bit = num_shifted & 0b1
        # Left-shift the i-th bit by j-i positions to move it to the j-th position.
        ith_bit_shifted = ith_bit << j
        # OR the shifted i-th bit with the result to set the j-th bit to the value of the i-th bit.
        result |= ith_bit_shifted

    return result

def round(x: int, key: int, s_block: list) -> int:
    y = x ^ key
    s = S(y, s_block)
    l = L(s)
    print(f"x = {hex(x)[2:].upper()} = {bin(x)[2:].upper()}\nkey = {hex(key)[2:].upper()} = {bin(key)[2:].upper()}\nx xor key = {hex(y)[2:].upper()} = {bin(y)[2:].upper()}\nS(x xor key) = {hex(s)[2:].upper()} = {bin(s)[2:].upper()}\nL(S(x xor key)) = {hex(l)[2:].upper()} = {bin(l)[2:].upper()}")
    return l

def encrypt(num: int, keys: int | list, s_block: list) -> int:
    keys = expand_key(keys)
    print(f"Plaintext = {hex(num)[2:].upper()} = {bin(num)[2:].upper()}")
    
    for i in range(6):
        print(f"-- Round {i} --")
        if i == 0:
            r = round(num, keys[i], s_block)
        else:
            r = round(r, keys[i], s_block)
    print("-- Final --")
    print(f"x = {hex(r)[2:].upper()} = {bin(r)[2:].upper()}\nkey = {hex(keys[6])[2:].upper()} = {bin(keys[6])[2:].upper()}\n")
    r = r ^ keys[6]
    print(f"Ciphertext = {hex(r)[2:].upper()} = {bin(r)[2:].upper()}")
    return r


num = 0x3331
keys = [0x3435, 0x3620, 0x2038, 0x3936, 0x3720, 0x2033, 0x3130]
s_block = [0xA, 0x9, 0xD, 0x6, 0xE, 0xB, 0x4, 0x5, 0xF, 0x1, 0x3, 0xC, 0x7, 0x0, 0x8, 0x2]

encrypt(num, keys, s_block)




