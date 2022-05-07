import time
import math
import matplotlib.pyplot as plt
import numpy as np

import utilities

from typing import List, Tuple


class Device:
    def __init__(
        self,
        num_bits: int = 16,
        seed: int = None,
        blinding: bool = False
    ) -> None:
        self.p, self.q = utilities.gen_rsa_factors(
            half_k=num_bits // 2,
            seed=seed
        )
        self.n = self.p * self.q
        self.R = 2
        self.blinding = blinding

    def get_montgomery_coefficient(self) -> int:
        return self.R

    def get_modulus(self) -> int:
        return self.n

    def _get_factors(self) -> Tuple[int, int]:
        return self.p, self.q

    def _decryption(self, u: int) -> float:
        if self.blinding:
            # generate a random integer r coprime with self.n
            r = utilities.gen_odd(1, self.n - 1)
            while sum(utilities.gcd(r, self.n)) != 1:
                r = utilities.gen_odd(1, self.n - 1)
            u = u * r

        g = u * self.R % self.n

        q_delay = 0
        p_delay = 0

        karatsuba = 10
        non_karatsuba = 100

        few_montgomery = 10
        many_montgomery = 1000

        if g < self.q:
            q_delay = many_montgomery + karatsuba
        else:  # g >= self.q
            q_delay = few_montgomery + non_karatsuba

        if g < self.p:
            p_delay = many_montgomery + karatsuba
        else:  # g >= self.p
            p_delay = few_montgomery + non_karatsuba

        std_delay = 5
        mean_delay = q_delay + p_delay
        delay = abs(np.random.normal(mean_delay, std_delay))
        microseconds = delay / 1e6
        time.sleep(microseconds)

        # not essential code block
        # if self.blinding:
        #     r_inv = pow(r, -1, self.n)
        #     g = g * r_inv % self.n)

        return microseconds

    def run(self, u: int) -> float:
        return self._decryption(u)


class Attacker:
    def __init__(
        self,
        device: Device,
        num_bits_per_factor: int = None,
        modulus: int = None,
        montgomery_coefficient: int = None,
    ):
        self.device = device
        self.n = (
            modulus
            if modulus is not None
            else device.get_modulus()
        )
        self.R = (
            montgomery_coefficient
            if montgomery_coefficient is not None
            else device.get_montgomery_coefficient()
        )
        self.R_inv = pow(self.R, -1, self.n)

        self.half_k = (
            num_bits_per_factor
            if num_bits_per_factor is not None
            else int((2 ** math.ceil(math.log2(math.log2(self.n)))) // 2)
        )

        self.threshold = None
        self.deltas = None

    def guess(self, threshold: float = 4e-4) -> int:
        bin_g = [0] * self.half_k
        bin_g[0] = 1
        bin_g[-1] = 1
        bin_gi = bin_g
        g = utilities.binarize_inverse(bin_g)

        self.threshold = threshold
        self.deltas = []

        for i in range(1, self.half_k - 1):

            u = g * self.R_inv % self.n

            t0 = time.time()
#             print(self.device.run(u), flush=True)
            self.device.run(u)
            t1 = time.time()
            elapsed_g = t1 - t0

            bin_gi[i] = 1
            gi = utilities.binarize_inverse(bin_gi)
            ui = gi * self.R_inv % self.n

            t0 = time.time()
#             print(self.device.run(ui), flush=True)
            self.device.run(ui)
            t1 = time.time()
            elapsed_gi = t1 - t0

            delta = elapsed_gi - elapsed_g
            self.deltas.append(delta)

#             print("bit #{}, delta: {}, g: {}, gi: {}".format(i + 1, delta, g, gi))
#             print("bit #{}, threshold: {}".format(i + 1, self.threshold))
#             print("bin_g", utilities.binarize(g))
#             print("bin_gi", utilities.binarize(gi))

            if abs(delta) > self.threshold:
                bin_gi[i] = 0
                gi = utilities.binarize_inverse(bin_gi)
            else:  # delta <= self.threshold
                g = gi
        return g

    def plot_last_guess(self, savefig_path=None, figsize=None):
        if self.deltas is None:
            raise Exception("At least a guess must be performed.")

        _, q = self.device._get_factors()
        bin_q = utilities.binarize(q)

        figsize = figsize if figsize is not None else (8, 5)
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel('Bits guessed of factor q', fontsize=18)
        ax.set_ylabel('Time variations', fontsize=18)

        np_deltas = np.asarray(self.deltas)
        abs_deltas = np.abs(np_deltas)
        x_values = np.arange(1, self.half_k - 1)
        filter_bit_0 = abs_deltas > self.threshold
        filter_bit_1 = ~filter_bit_0
        ax.scatter(x_values[filter_bit_0], np_deltas[filter_bit_0], c='green', label='bit-0')
        ax.scatter(x_values[filter_bit_1], np_deltas[filter_bit_1], c='orange', label='bit-1')
        ax.plot(x_values[filter_bit_0], np_deltas[filter_bit_0], c='green', alpha=.5)
        ax.plot(x_values[filter_bit_1], np_deltas[filter_bit_1], c='orange', alpha=.5)

        # twinx = ax.twinx()
        # twinx.scatter(x_values, bin_q[1:-1], c="blue", label="q-digits", alpha=0.5)
        fig.legend()

        if savefig_path is not None:
            fig.savefig(savefig_path)

        plt.show()
        return fig

if __name__ == '__main__':
    num_bits = 64
    seed = 42

    device = Device(num_bits=num_bits, seed=seed, blinding=False)
    attacker = Attacker(device)
    g = attacker.guess()
    p, q = device._get_factors()
    bin_g = utilities.binarize(g)
    bin_q = utilities.binarize(q)
    bin_p = utilities.binarize(p)
    print('g: {}, bin_g: {}'.format(g, bin_g))
    print('q: {}, bin_q: {}'.format(q, bin_q))
    print('p: {}, bin_p: {}'.format(p, bin_p))
    attacker.plot_last_guess('../presentation/figures/attack_without_blinding')

    another_device = Device(num_bits=num_bits, seed=seed, blinding=True)
    attacker = Attacker(another_device)
    g = attacker.guess()
    p, q = device._get_factors()
    bin_g = utilities.binarize(g)
    bin_q = utilities.binarize(q)
    bin_p = utilities.binarize(p)
    print('g: {}, bin_g: {}'.format(g, bin_g))
    print('q: {}, bin_q: {}'.format(q, bin_q))
    print('p: {}, bin_p: {}'.format(p, bin_p))
    attacker.plot_last_guess('../presentation/figures/attack_with_blinding')

