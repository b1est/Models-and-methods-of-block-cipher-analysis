import numpy as np
from multiprocess import Pool
from  config import Config
from heys import HeysCipher
from helpers import timeit

config = Config()

def scalar_mul(alpha, beta):
    return (alpha & beta).bit_count() & 1


def get_linear_approximations():
    table_s = np.zeros((16, 16))
    for alpha in range(16):
        for beta in range(16):
            table_s[alpha][beta] = (sum([(-1) ** (scalar_mul(alpha, x) ^ scalar_mul(beta, HeysCipher(config.s_block, config.s_block_rev).S(x ^ beta, True))) for x in range(16)]) / 16) ** 2
    return table_s

def LP_a(alpha):
    lp = get_linear_approximations()
    a = [(alpha >> 4 * i) & 15 for i in range(4)]
    beta = [np.where(lp[a[i]] != 0)[0] for i in range(4)]
   
    lpa = {}
    for b1 in beta[0]:
        for b2 in beta[1]:
            for b3 in beta[2]:
                for b4 in beta[3]:
                    lpa[b4 + (b3 << 4) + (b2 << 8) + (b1 << 12)] = lp[b1][a[0]] * lp[b2][a[1]] * lp[b3][a[2]] * lp[b4][a[3]]

    return lpa

@timeit(display_args=True)
def linear_approximations_search(alpha, r = 6, p = 0.00012):
    InitialGamma = {alpha: 1}
    for t in range(1, r):
        GammaIndex = {}
        for bi in InitialGamma:
            delta = LP_a(bi)
            for gj in delta:
                if gj in GammaIndex:
                    GammaIndex[gj] += InitialGamma[bi]*delta[gj]
                else:
                    GammaIndex[gj] = InitialGamma[bi]*delta[gj]
        keys_for_delete = [key for key in GammaIndex if GammaIndex[key] <= p]
        for key in keys_for_delete:
            GammaIndex.pop(key)
        InitialGamma = GammaIndex.copy()  
    return GammaIndex
