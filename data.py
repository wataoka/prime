import sys
import numpy as np
import pickle


def get_primes(N=100000):
    natural_number = np.arange(N) + 1.0
    natural_number[0] = np.nan
    for i in range(1, int(np.floor(N/2))):
        natural_number[(natural_number!=natural_number[i]) & (natural_number%natural_number[i]==0)] = np.nan
    prime_number = natural_number[~np.isnan(natural_number)]
    prime_number = prime_number.astype(int)
    return prime_number




def create(N=100000):
    digits = 6
    question = []
    ans = []
    x = np.zeros((N, digits, 10), dtype=np.integer)
    y = np.zeros((N, 1, 2), dtype=np.integer)
    primes = get_primes()

    tmp = 0
    max = len(primes)
    for i in range(1, N):
        question.append("{0:06d}".format(i))

        if not(tmp==max) and primes[tmp] == i:
            ans.append('1')
            tmp += 1
        else:
            ans.append('0')

    chars = '0123456789'
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indeices_char = dict((i, c) for i, c in enumerate(chars))

    for i in range(N-1):
        for t, char in enumerate(question[i]):
            x[i, t, char_indices[char]] = 1
        for t, char in enumerate(ans[i]):
            y[i, t, char_indices[char]] = 1

    data = {}
    data['x'] = x
    data['y'] = y
    return data


data = create()
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)
