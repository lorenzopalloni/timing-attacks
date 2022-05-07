import time
import math
import matplotlib.pyplot as plt
import numpy as np

import utilities

from typing import List, Tuple


class Device:
    def __init__(
        self, num_bits: int = 16, seed: int = None, blinding: bool = False
    ) -> None:
        self.p, self.q = utilities.gen_rsa_factors(half_k=num_bits // 2, seed=seed)
        self.n = self.p * self.q
        self.R = 2 ** (num_bits // 2)
        self.blinding = blinding
        self._window_size = 5
        self._word_size = 8

    def get_montgomery_coefficient(self) -> int:
        return self.R

    def get_modulus(self) -> int:
        return self.n

    def _get_factors(self) -> Tuple[int, int]:
        return self.p, self.q

    def _decryption(self, u: int) -> float:
        g = u * self.R % self.n

        q_delay = 0
        p_delay = 0

        ## start - self.q
        # sliding windows
        expected_multiplications_prob = (
            math.log2(self.q)  # assuming log(q) ~ log(d)
            / (2**self._window_size * (self._window_size + 1))
        )
        expected_multiplications = np.random.poisson(expected_multiplications_prob)
        # expected_multiplications = 100

        # montgomery
        reduction_prob = (g % self.q) / (2 * self.R)
        expected_reductions = reduction_prob * expected_multiplications
        reduction_delay = 0.5
        q_delay += expected_reductions * reduction_delay

        # karatsuba
        karatsuba_delay = 1
        non_karatsuba_delay = karatsuba_delay * 1.0005
        # if abs((g % self.q) - self.q) < 2**self._word_size:
        if abs(math.log2(g) - math.log2(self.q)) < self._word_size:
            q_delay += karatsuba_delay
        else:
            q_delay += non_karatsuba_delay
        ## end - self.q

        ## start - self.p
        # sliding windows
        expected_multiplications_prob = (
            math.log2(self.p)  # assuming log(p) ~ log(d)
            / (2**self._window_size * (self._window_size + 1))
        )
        expected_multiplications = np.random.poisson(expected_multiplications_prob)
        # expected_multiplications = 100

        # montgomery
        reduction_prob = (g % self.p) / (2 * self.R)
        expected_reductions = reduction_prob * expected_multiplications
        reduction_delay = 0.5
        q_delay += expected_reductions * reduction_delay

        # karatsuba
        karatsuba_delay = 1
        non_karatsuba_delay = karatsuba_delay * 1.05
        # if abs((g % self.p) - self.p) < 2**self._word_size:
        if abs(math.log2(g) - math.log2(self.p)) < self._word_size:
            q_delay += karatsuba_delay
        else:
            q_delay += non_karatsuba_delay
        ## end - self.p

        # print('g:', g)
        # print('self.q:', self.q)
        # print('g % self.q:', g % self.q)
        # print('reduction_prob:', reduction_prob)
        # print('montgomery_reductions:', expected_reductions)
        # print('expected_multiplications:', expected_multiplications)
        # print('q_delay:', q_delay)
        # print('-' * 10)

        mean_delay = q_delay + p_delay
        microseconds = mean_delay  # / 1e6
        # time.sleep(microseconds)

        return microseconds

### - START - original implementation
#     def _decryption(self, u: int) -> float:
#         if self.blinding:
#             # generate a random integer r coprime with self.n
#             r = utilities.gen_odd(1, self.n - 1)
#             while sum(utilities.gcd(r, self.n)) != 1:
#                 r = utilities.gen_odd(1, self.n - 1)
#             u = u * r
#
#         g = u * self.R % self.n
#
#         q_delay = 0
#         p_delay = 0
#
#         karatsuba = 10
#         non_karatsuba = 100
#
#         few_montgomery = 10
#         many_montgomery = 1000
#
#         if g < self.q:
#             q_delay = many_montgomery + karatsuba
#         else:  # g >= self.q
#             q_delay = few_montgomery + non_karatsuba
#
#         if g < self.p:
#             p_delay = many_montgomery + karatsuba
#         else:  # g >= self.p
#             p_delay = few_montgomery + non_karatsuba
#
#         std_delay = 5
#         mean_delay = q_delay + p_delay
#         delay = abs(np.random.normal(mean_delay, std_delay))
#         microseconds = delay  # / 1e6
#         time.sleep(microseconds)
#
#         # not essential code block
#         # if self.blinding:
#         #     r_inv = pow(r, -1, self.n)
#         #     g = g * r_inv % self.n)
#
#         return microseconds
### - END - original implementation


#     def _decryption(self, u: int) -> float:
#         if self.blinding:
#             # generate a random integer r coprime with self.n
#             r = utilities.gen_odd(1, self.n - 1)
#             while sum(utilities.gcd(r, self.n)) != 1:
#                 r = utilities.gen_odd(1, self.n - 1)
#             u = u * r
#
#         g = u * self.R % self.n
#
#         q_delay = 0
#         p_delay = 0
#
#         # montgomery
#         reduction_prob = (g % self.q) / (2 * self.R)
#         # window_size = 5
#         # expected_num_multiplications = (
#         #     math.log2(self.q)  # assuming log(q), log(d) with the same number of bits
#         #     / (2**window_size * (window_size + 1))
#         # )
#         expected_multiplications = 1000 # assuming log(q), log(d) with the same number of bits
#         expected_reductions = reduction_prob * expected_multiplications
#         reduction_delay = 1e-2
#         q_delay += expected_reductions * reduction_delay
#
#         # # karatsuba
#         # system_word_size = 3
#         # karatsuba_delay = 1 * q_delay
#         # non_karatsuba_delay = 10 * q_delay
#         # if abs((g % self.q) - self.q) < 2**system_word_size:
#         #     q_delay += karatsuba_delay
#         # else:
#         #     q_delay += non_karatsuba_delay
#
#         # print('expected_num_multiplications:', expected_num_multiplications)
#         print('g:', g)
#         print('g % self.q:', g % self.q)
#         print('reduction_prob:', reduction_prob)
#         # print('self.q:', self.q)
#         print('montgomery_reductions:', expected_reductions)
#         print('q_delay:', q_delay)
#         print('-' * 10)
#
# #         if g < self.q:
# #             q_delay = many_montgomery + karatsuba
# #         else:  # g >= self.q
# #             q_delay = few_montgomery + non_karatsuba
# #
# #         if g < self.p:
# #             p_delay = many_montgomery + karatsuba
# #         else:  # g >= self.p
# #             p_delay = few_montgomery + non_karatsuba
#
#         mean_delay = q_delay + p_delay
#         # std_delay = math.sqrt(mean_delay)
#         # delay = abs(np.random.normal(mean_delay, std_delay))
#         microseconds = mean_delay  # / 1e6
#         time.sleep(microseconds)
#
#         # not essential code block
#         # if self.blinding:
#         #     r_inv = pow(r, -1, self.n)
#         #     g = g * r_inv % self.n)
#
#         return microseconds

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
        self.n = modulus if modulus is not None else device.get_modulus()
        self.R = (
            montgomery_coefficient
            if montgomery_coefficient is not None
            else device.get_montgomery_coefficient()
        )
        self.R_inv = pow(self.R, -1, self.n)
        print('self.R_inv:', self.R_inv)

        self.half_k = (
            num_bits_per_factor
            if num_bits_per_factor is not None
            else int((2 ** math.ceil(math.log2(math.log2(self.n)))) // 2)
        )

        self.threshold = None
        self.deltas = None

    def guess(
        self,
        threshold: float = 4e-4,
        num_samples: int = 10,
        neighborhood_size: int = 1000,
    ) -> int:
        bin_g = [0] * self.half_k
        bin_g[0] = 1  # first bit is always 1
        bin_g[-1] = 1  # last bit is always 1 (odd)
        bin_gi = bin_g
        g = utilities.binarize_inverse(bin_g)

        self.threshold = threshold
        self.deltas = np.zeros(self.half_k - 2)  # omit first and last bits

        for i in range(1, self.half_k - 1):
            all_Tg = np.zeros((num_samples, neighborhood_size))
            all_Tgi = np.zeros((num_samples, neighborhood_size))
            for j in range(num_samples):
                for k in range(neighborhood_size):
                    u = (g + k) * self.R_inv % self.n

                    bin_gi[i] = 1
                    gi = utilities.binarize_inverse(bin_gi)
                    ui = gi * self.R_inv % self.n

                    all_Tg[j, k] = self.device.run(u)
                    all_Tgi[j, k] = self.device.run(ui)
            # print(all_Tg)
            # print(all_Tgi)
            Tg = np.sum(np.median(all_Tg, axis=0))
            Tgi = np.sum(np.median(all_Tgi, axis=0))
            delta = Tg - Tgi
            self.deltas[i - 1] = delta

            # u = g * self.R_inv % self.n

            # # t0 = time.time()
            # # # print(self.device.run(u), flush=True)
            # # self.device.run(u)
            # # t1 = time.time()
            # # elapsed_g = t1 - t0

            # bin_gi[i] = 1
            # gi = utilities.binarize_inverse(bin_gi)
            # ui = gi * self.R_inv % self.n

            # # t0 = time.time()
            # # # print(self.device.run(ui), flush=True)
            # # self.device.run(ui)
            # # t1 = time.time()
            # # elapsed_gi = t1 - t0

            # elapsed_g = self.device.run(u)
            # elapsed_gi = self.device.run(ui)

            # delta = elapsed_g - elapsed_gi
            # self.deltas.append(delta)

            print("-" * 68)
            print("bit #{}, delta: {}, g: {}, gi: {}".format(i, delta, g, gi))
            print("bit #{}, threshold: {}".format(i, self.threshold))
            print("bin_g", utilities.binarize(g))
            print("bin_gi", utilities.binarize(gi))
            _, q = self.device._get_factors()
            print("bin_q", utilities.binarize(q))
            print("-" * 88)

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
        ax.grid(alpha=0.5)

        ax.set_xlabel("Bits guessed of factor q", fontsize=18)
        ax.set_ylabel("Time variations", fontsize=18)

        np_deltas = np.asarray(self.deltas)
        abs_deltas = np.abs(np_deltas)
        x_values = np.arange(1, self.half_k - 1)
        filter_bit_0 = abs_deltas > self.threshold
        filter_bit_1 = ~filter_bit_0
        ax.scatter(
            x_values[filter_bit_0], np_deltas[filter_bit_0], c="green", label="bit-0"
        )
        ax.scatter(
            x_values[filter_bit_1], np_deltas[filter_bit_1], c="orange", label="bit-1"
        )
        ax.plot(x_values[filter_bit_0], np_deltas[filter_bit_0], c="green", alpha=0.5)
        ax.plot(x_values[filter_bit_1], np_deltas[filter_bit_1], c="orange", alpha=0.5)

        # twinx = ax.twinx()
        # twinx.scatter(x_values, bin_q[1:-1], c="blue", label="q-digits", alpha=0.5)
        fig.legend()

        if savefig_path is not None:
            fig.savefig(savefig_path)

        plt.show()
        return fig


if __name__ == "__main__":
    num_bits = 64
    seed = 42

    device = Device(num_bits=num_bits, seed=seed, blinding=False)
    attacker = Attacker(device)
    g = attacker.guess(threshold=25, num_samples=5, neighborhood_size=10000)
    p, q = device._get_factors()
    bin_g = utilities.binarize(g)
    bin_q = utilities.binarize(q)
    bin_p = utilities.binarize(p)
    print("bin_g: {}".format(bin_g))
    print("bin_q: {}".format(bin_q))
    # print("p: {}, bin_p: {}".format(p, bin_p))
    attacker.plot_last_guess("../presentation/figures/attack_without_blinding")

    # another_device = Device(num_bits=num_bits, seed=seed, blinding=True)
    # attacker = Attacker(another_device)
    # g = attacker.guess()
    # p, q = device._get_factors()
    # bin_g = utilities.binarize(g)
    # bin_q = utilities.binarize(q)
    # bin_p = utilities.binarize(p)
    # print("g: {}, bin_g: {}".format(g, bin_g))
    # print("q: {}, bin_q: {}".format(q, bin_q))
    # print("p: {}, bin_p: {}".format(p, bin_p))
    # attacker.plot_last_guess("../presentation/figures/attack_with_blinding")
