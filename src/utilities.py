import sys
import math
import random
from typing import Tuple, List


def is_probably_prime(guess: int, tries: int = 4) -> bool:
    for _ in range(tries):
        if rabin_test(guess) == True:
            return False
    return True


def gen_prime(low: int, high: int, seed: int = None, tries: int = 3) -> int:
    guess = gen_odd(low, high, seed)
    if seed is None:
        seed = random.randrange(sys.maxsize)
    counter = 0
    while not is_probably_prime(guess, tries):
        guess = gen_odd(low, high, seed + counter)
        counter += 1
    return guess


def binarize(a: int) -> List[int]:
    """Returns the binary representation of `a` as a list of integers."""
    return list(map(int, bin(a)[2:]))


def binarize_inverse(a: List[int]) -> int:
    """Returns the binary representation of `a` as a list of integers."""
    return int('0b' + ''.join(map(str, a)), 2)


def mod_exp(a: int, m: int, n: int) -> int:
    """Fast modular exponentiation that computes a^m mod n."""
    d = 1
    bm = binarize(m)
    for i in bm:
        d = d * d % n
        if i == 1:
            d = d * a % n
    return d


def check_quadratic_residuals(n: int, seq: List[int]) -> bool:
    """Return True if the given sequence satisfies the quadratic residuals theorem."""
    for i in range(len(seq) - 1, -1, -1):
        x = seq[i - 1]
        xx = seq[i]
        if xx == 1 and (x not in [1, (n - 1)]):
            return False
    return True


def get_m_and_r(n: int) -> Tuple[int, int]:
    """Given n odd, it finds (r, m): (n - 1) / (2 ** r) == m, with m integer."""
    for r in range(math.floor(math.log2(n - 1)), 0, -1):
        m = (n - 1) / (2 ** r)
        if m % 1 == 0:
            return int(m), r
    return (n - 1), 1


def gen_rabin_sequence(n: int, x: int, m: int, r: int) -> List[int]:
    """Generates the Rabin sequence."""
    seq = [mod_exp(x, m, n)]
    for i in range(1, r + 1):
        seq.append(seq[i - 1] ** 2 % n)
    return seq


def rabin_test(n: int, x: int = None, verbose: bool = False) -> bool:
    """If True, then n is composite. Otherwise, n might be prime."""
    if x is None:
        x = gen_odd(1, n - 1)
    m, r = get_m_and_r(n)
    seq = gen_rabin_sequence(n=n, x=x, m=m, r=r)
    if verbose:
        print("n: {}, x: {}, m: {}, r: {}, seq: {}".format(n, x, m, r, seq))
    if seq[r] != 1:  # Fermat's little theorem
        return True
    return not check_quadratic_residuals(n, seq)


def gen_odd(low: int, high: int, seed: int = None) -> int:
    """Generates an odd random integer in [low, high]."""
    low = low - 1 if low % 2 == 0 else low
    high = high - 1 if high % 2 == 0 else high

    if seed is not None:
        random.seed(seed)

    return 2 * random.randint(low // 2, high // 2) + 1


def gcd(a: int, b: int) -> Tuple[int, int]:
    """Greatest Common Divisor between a and b."""
    while a != 0 and b != 0:
        a, b = gcd(b, a % b)
    return a, b


def gen_rsa_factors(half_k: int = 8, seed: int = None) -> Tuple[int, int]:
    """Generates two distinct prime numbers."""
    if seed is None:
        seed = random.randrange(sys.maxsize)
    p = gen_prime(2 ** (half_k - 1), 2 ** half_k, seed=seed)
    q = gen_prime(2 ** (half_k - 1), 2 ** half_k, seed=seed + 1)
    counter = 2
    while q == p:
        q = gen_prime(2 ** (half_k - 1), 2 ** half_k, seed=seed + counter)
        counter += 1
    if p > q:
        return p, q
    return q, p

if __name__ == "__main__":
    with open("prime_numbers_lower_than_2000.txt") as f:
        text_some_prime_numbers = f.read()
    some_prime_numbers = list(map(int, text_some_prime_numbers.split(",")))
    some_not_prime_numbers = [
        3 * 11 * 17,  # first Carmichael's number
        3 * 11 * 17 * 19,
        3 * 11,
        17 * 41,
    ]
    some_numbers = some_prime_numbers + some_not_prime_numbers

    flag = True
    for n in some_prime_numbers:
        if rabin_test(n) != False:
            flag = False
            print("Something went wrong with {}".format(n))
    for n in some_not_prime_numbers:
        if rabin_test(n) != True:
            flag = False
            print("Something went wrong with {}".format(n))
    if flag:
        print("Everything should be fine with Rabin test!")

    k = 10
    a_prime = gen_prime(2 ** (k - 1), 2 ** k)
    print(a_prime)
    assert a_prime in some_prime_numbers
