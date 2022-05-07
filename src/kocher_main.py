import numpy as np
# import matplotlib.pyplot as plt

from TimingAttackModule import TimingAttack

n = 2 ** 32

random_seed = 42
np.random.seed(random_seed)
ta = TimingAttack()

num_bits = 64
all_num_ciphertexts = list(map(int, np.logspace(4., 2.5, num_bits - 1)))
# all_num_ciphertexts = [3000] * (num_bits - 1)  # fixed number of ciphertexts
guess = [1] * num_bits

for k, num_ciphertexts in enumerate(all_num_ciphertexts):
    print('# bit: {:d}/{:d}\tNumber of ciphertexts: {:d}'.format(
        k + 2, 64, num_ciphertexts
    ), end='')

    some_ciphertexts = [
        np.random.randint(1, n - 1)
        for _ in range(num_ciphertexts)
    ]

    delta0 = np.zeros(num_ciphertexts)
    delta1 = np.zeros(num_ciphertexts)

    for i, ciphertext in enumerate(some_ciphertexts):
        T = ta.victimdevice(ciphertext)
        T0 = ta.attackerdevice(ciphertext, guess[:(k + 1)] + [0])
        T1 = ta.attackerdevice(ciphertext, guess[:(k + 1)] + [1])
        delta0[i] = T - T0
        delta1[i] = T - T1

    std0 = np.std(delta0)
    std1 = np.std(delta1)

    if std0 < std1:
        guess[k + 1] = 0
    else:
        guess[k + 1] = 1

    print('\tRatio of recovered bits:', ta.test(guess))
print(ta.test(guess, verbose=True))

