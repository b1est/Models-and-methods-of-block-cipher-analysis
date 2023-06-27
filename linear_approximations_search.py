import numpy as np
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
def linear_approximations_search(alpha, r = 6, p = 0.00001):
    Gamma = [dict() for i in range(r)]
    Gamma[0].update({alpha : 1})
    LPs = dict()
    for t in range(1, r):
        GammaIndex = {}
        for bi in Gamma[t-1]:
            lp = LPs.get(bi, LP_a(bi))
            LPs[bi] = lp
            p_i = Gamma[t-1][bi]
            for pp in lp:
                GammaIndex[pp] = GammaIndex.get(pp, 0) + lp[pp] * p_i
        
        Gamma[t].update({HeysCipher(config.s_block, config.s_block_rev).L(beta): GammaIndex[beta] for beta in GammaIndex if GammaIndex[beta] > p})
    return Gamma[-1]
