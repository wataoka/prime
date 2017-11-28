import sys
import numpy as np
np.set_printoptions(threshold=np.inf)   # 配列の ... 表示を抑制
N = int(sys.argv[1])+1                  # 引数の上限値を含める
is_prime = np.ones((N,), dtype=bool)
is_prime[:2]=0
N_max = int(np.sqrt(N))
for j in range(2,N_max+1):
    if(is_prime[j]):
        is_prime[j*j::j] = False        # 素数jの間隔で消していく
primes = np.arange(N)[is_prime]
print(primes,'\ncount:',primes.size)
