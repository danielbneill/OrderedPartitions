import proto
import numpy as np

rng = np.random.RandomState(14)

n = 1000
T = 10

a = proto.FArray()
b = proto.FArray()
a = rng.uniform(low=1.0, high=10.0, size=n)
b = rng.uniform(low=1.0, high=10.0, size=n)

info = proto.sweep_parallel(n, T, a, b)
