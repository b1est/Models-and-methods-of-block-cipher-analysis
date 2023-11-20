import numpy as np
from config import Config
from heys import HeysCipher
from helpers import timeit

config = Config()


def scalar_mul(alpha, beta):
    return (alpha & beta).bit_count() & 1


def get_linear_approximations():
    table_s = np.zeros((16, 16))
    for alpha in range(16):
        for beta in range(16):
            table_s[alpha][beta] = (sum([(-1) ** (scalar_mul(alpha, x) ^ scalar_mul(beta, HeysCipher(
                config.s_block, config.s_block_rev).S(x ^ beta, True))) for x in range(16)]) / 16) ** 2
    return np.array(table_s).transpose()


def LP_a(alpha):
    lp = get_linear_approximations()
    a = [(alpha >> 4 * i) & 0xf for i in range(4)]
    beta = []
    for j in range(4):
        b = [i for i in range(16) if lp[i][a[j]] != 0]
        beta.append(b)
    lpa = dict()
    for b1 in beta[0]:
        for b2 in beta[1]:
            for b3 in beta[2]:
                for b4 in beta[3]:
                    lpa[(b4 << 12) + (b3 << 8) + (b2 << 4) + b1] = lp[b1][a[0]
                                                                          ] * lp[b2][a[1]] * lp[b3][a[2]] * lp[b4][a[3]]
    return lpa


@timeit(display_args=True)
def linear_approximations_search(alpha, r=6, p=0.0001):
    Gamma = [dict() for i in range(r)]
    Gamma[0].update({alpha: 1})
    LPs = dict()
    for t in range(1, r):
        GammaIndex = dict()
        for bi in Gamma[t-1]:
            LPs[bi] = LPs.get(bi, LP_a(bi))
            for pp in LPs[bi]:
                GammaIndex[pp] = GammaIndex.get(
                    pp, 0) + LPs[bi][pp] * Gamma[t-1][bi]
        Gamma[t].update({HeysCipher(config.s_block, config.s_block_rev).L(
            beta): GammaIndex[beta] for beta in GammaIndex if GammaIndex[beta] > p})
    return Gamma[-1]
