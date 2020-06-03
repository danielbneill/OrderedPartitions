import proto
import numpy as np

rng = np.random.RandomState(14)

n = 100
T = 10

a = proto.FArray()
b = proto.FArray()
a = rng.uniform(low=-10.0, high=10.0, size=n)
b = rng.uniform(low=1.0, high=10.0, size=n)

info = proto.sweep_parallel(n, T, a, b)

a = -1 * a
info_neg = proto.sweep_parallel(n, T, a, b)
