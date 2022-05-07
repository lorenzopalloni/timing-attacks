import time
from typing import List

def binarize(a: int) -> List[int]:
    """Returns the binary representation of `a` as a list of integers."""
    return list(map(int, bin(a)[2:]))

def mod_exp(a: int, m: int, n: int) -> int:
    """Fast modular exponentiation that computes a^m mod n."""
    d = 1
    bm = binarize(m)
    for i in bm:
        d = d * d % n
        if i == 1:
            d = d * a % n
    return d

def original_gen_rabin_sequence(n: int, x: int, m: int, r: int) -> List[int]:
    """Generates the Rabin sequence."""
    seq = [x ** m % n]
    for i in range(1, r + 1):
        seq.append(seq[i - 1] ** 2 % n)
    return seq

def gen_rabin_sequence(n: int, x: int, m: int, r: int) -> List[int]:
    """Generates the Rabin sequence."""
    seq = [mod_exp(x, m, n)]
    for i in range(1, r + 1):
        seq.append(seq[i - 1] ** 2 % n)
    return seq

if __name__ == '__main__':
    n = 887533
    x = 308889
    m = 221883
    r = 2

    t0 = time.time()
    original_gen_rabin_sequence(n=n, x=x, m=m, r=r)
    t1 = time.time()
    time_elapsed_orig = t1 - t0
    print('Time elapsed: {}'.format(time_elapsed_orig))

    t0 = time.time()
    gen_rabin_sequence(n=n, x=x, m=m, r=r)
    t1 = time.time()
    time_elapsed_optim = t1 - t0
    print('Time elapsed: {}'.format(time_elapsed_optim))

    print('Speed up: {:.4f}'.format(time_elapsed_orig / time_elapsed_optim))

