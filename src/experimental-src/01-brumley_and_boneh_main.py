import time
import matplotlib.pyplot as plt
import numpy as np

import utilities

from typing import List


class Device:
    def __init__(self, half_k: int = 8, seed: int = None):
        self.p, self.q = utilities.gen_rsa_factors(half_k, seed=seed)
        self.n = self.p * self.q
        self.R = 2
    def get_montgomery_coefficient(self):
        return self.R
    def get_modulus(self):
        return self.n
    def _get_factors(self):
        return self.p, self.q

    def _decryption(self, u: int):
        g = u * self.R % n

        delay = 0

        karatsuba = 10
        non_karatsuba = 100

        few_montgomery = 10
        many_montgomery = 1000

        print('inner g:', g)
        print('inner q:', self.q)
        print('inner bin_g:', utilities.binarize(g))

        if g < self.q:
            delay = many_montgomery + karatsuba
        else:  # self.q >= g
            delay = few_montgomery + non_karatsuba

        std_delay = 5
        delay = abs(np.random.normal(delay, std_delay))
        microseconds = delay / 1e+6
        time.sleep(microseconds)
        return microseconds

    def run(self, u: int) -> float:
        return self._decryption(u)


### start - initialization
half_k = 32
device = Device(half_k=half_k, seed=43)
n = device.get_modulus()
R = device.get_montgomery_coefficient()
R_inv = pow(R, -1, n)  # utilities.modulo_inverse(R, n)

bin_g = [0] * half_k
bin_g[0] = 1
bin_g[-1] = 1
bin_gi = bin_g
g = utilities.binarize_inverse(bin_g)

threshold = 4e-4
deltas = []
### end - initialization

for i in range(1, half_k - 1):

    u = g * R_inv % n

    t0 = time.time()
    print(device.run(u), flush=True)
    t1 = time.time()
    elapsed_g = t1 - t0

    bin_gi[i] = 1
    gi = utilities.binarize_inverse(bin_gi)
    ui = gi * R_inv % n

    t0 = time.time()
    print(device.run(ui), flush=True)
    t1 = time.time()
    elapsed_gi = t1 - t0

    delta = abs(elapsed_g - elapsed_gi)
    deltas.append(delta)

    print('bit #{}, delta: {}, g: {}, gi: {}'.format(i + 1, delta, g, gi))
    print('bit #{}, threshold: {}'.format(i + 1, threshold))
    print('bin_g', utilities.binarize(g))
    print('bin_gi', utilities.binarize(gi))

    if delta > threshold:
        bin_gi[i] = 0
        gi = utilities.binarize_inverse(bin_gi)
    else:  # delta <= threshold
        g = gi

print('bin_g: {}'.format(bin_g))
print('bin_gi: {}'.format(bin_gi))
p, q = device._get_factors()
print('p: {}, q: {}, n: {}, q % n: {}'.format(p, q, n, q % n))
print('q: {}'.format(q))
bin_q = utilities.binarize(q)
print('bin_q: {}'.format(bin_q))

fig, ax = plt.subplots(figsize=(8, 5))
twinx = ax.twinx()
x_values = range(1, half_k - 1)
ax.plot(x_values, deltas, c='red', label='deltas')
twinx.scatter(x_values, bin_q[1:-1], c='blue', label='q-digits')
fig.tight_layout()
fig.legend()
plt.show()

