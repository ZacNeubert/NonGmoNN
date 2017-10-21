import numpy as np
from random import choice


def gendat(c, N):
    m0 = [[-0.132, 0.320, 1.672, 2.230, 1.217, -0.819, 3.629, 0.8210, 1.808, 0.1700],
          [-0.711, -1.726, 0.139, 1.151, -0.373, -1.573, -0.243, -0.5220, -0.511, 0.5330]]
    m1 = [[-1.169, 0.813, -0.859, -0.608, -0.832, 2.015, 0.173, 1.432, 0.743, 1.0328],
          [2.065, 2.441, 0.247, 1.806, 1.286, 0.928, 1.923, 0.1299, 1.847, -0.052]]

    x = []
    for i in range(N):
        idx = choice(range(0, 10))
        if c == 0:
            m = [m0[0][idx], m0[1][idx]]
        elif c == 1:
            m = [m1[0][idx], m1[1][idx]]
        else:
            raise Exception('Z: Bad Class')
        item = np.array(m) + np.random.randn(2,) / np.math.sqrt(5)
        x.append(item.tolist())
    return x


if __name__ == '__main__':
    print(gendat(0, 5))
    print(gendat(1, 5))
