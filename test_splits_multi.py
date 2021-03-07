import os
import sys
import numpy as np
import pickle
import multiprocessing
from scipy.special import comb
from functools import partial
from itertools import chain, islice, combinations
import matplotlib.pyplot as plot
from scipy.spatial import ConvexHull, Delaunay

SEED = 3531
rng = np.random.RandomState(SEED)

def subsets(ns):
    return list(chain(*[[[list(x)] for x in combinations(range(ns), i)] for i in range(1,ns+1)]))

def knuth_partition(ns, m):
    if m == 1:
        return [[ns]]
    
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

def Bell_n_k(n, k):
    ''' Number of partitions of  1,...,n} into
        k subsets, a restricted Bell number
    '''
    if (n == 0 or k == 0 or k > n): 
        return 0
    if (k == 1 or k == n): 
        return 1
      
    return (k * Bell_n_k(n - 1, k) + 
                Bell_n_k(n - 1, k - 1))

def _Mon_n_k(n, k):
    return comb(n-1, k-1, exact=True)

def double_power_fn(a11,a12,a21,a22,b1,b2,p):
    a11_ = np.sum(a11[p])
    a12_ = np.sum(a12[p])
    a21_ = np.sum(a21[p])
    a22_ = np.sum(a22[p])
    b1_  = np.sum(b1[p])
    b2_  = np.sum(b2[p])

    A = np.array([[a11_,a12_],[a21_,a22_]])
    b = np.array([b1_, b2_])
    # XXX
    # r = np.dot(b.T, np.dot(np.linalg.inv(A), b))
    r = np.dot(b.T, np.dot(np.linalg.inv(A), b))

    return r
    
def slice_partitions(partitions):
    # Have to consume it; can't split work on generator
    partitions = list(partitions)
    num_partitions = len(partitions)
    
    bin_ends = list(range(0,num_partitions,int(num_partitions/NUM_WORKERS)))
    bin_ends = bin_ends + [num_partitions] if num_partitions/NUM_WORKERS else bin_ends
    islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))

    rng.shuffle(partitions)
    slices = [list(islice(partitions, *ind)) for ind in islice_on]
    return slices

def reduce(return_values, fn):
    return fn(return_values, key=lambda x: x[0])

class Task(object):
    def __init__(self, a11, a12, a21, a22, b1, b2, partition, power=2, cond=max, score_fn=double_power_fn, fnArgs=()):
        self.partition = partition
        self.cond = cond
        self.score_fn = score_fn
        self.task = partial(self._task, a11, a12, a21, a22, b1, b2, partition)

    def __call__(self):
        return self.task(self.partition)

    def _task(self, a11, a12, a21, a22, b1, b2, partitions, report_each=1000):

        if self.cond == min:
            max_sum = float('inf')
        else:
            max_sum = float('-inf')            
        
        arg_max = -1
        
        for ind,part in enumerate(partitions):
            val = 0
            part_val = [0] * len(part)
            # print('PARTITION: {!r}'.format(part))
            for part_ind, p in enumerate(part):
                fnArgs = (a11,a12, a21, a22, b1, b2,p)
                part_sum = self.score_fn(*fnArgs)
                part_val[part_ind] = part_sum
                val += part_sum
                # print('    SUBSET: {!r} SUBSET SCORE: {:4.4f}'.format(p, part_sum))
            if self.cond(val, max_sum) == val:
                max_sum = val
                arg_max = part
            # print('    PARTITION SCORE: {:4.4f}'.format(val))
        print('MAX PARTITION SCORE: {:4.4f}, MAX_PARTITION: {}'.format(max_sum, list(arg_max)))
        return (max_sum, arg_max)

def optimize(a11,
             a12,
             a21,
             a22,
             b1,
             b2,
             PARTITION_SIZE,
             NUM_WORKERS,
             CONSEC_ONLY=False,
             cond=max):

    partitions = list(knuth_partition(range(0, len(a11)), PARTITION_SIZE))
    if CONSEC_ONLY:
        partitions = [p for p in partitions if all(np.diff([x for x in chain.from_iterable(p)]) == 1)]
    allResults = [Task(a11, a12, a21, a22, b1, b2, partitions, cond=cond, score_fn=SCORE_FN)()]
    
    r_max = reduce(allResults, cond)

    return r_max

def _pos_dev_all(a11,a12,a21,a22):
    if (a11 is None) and (a12 is None) and (a21 is None) and (a22 is None):
        return False
    for (a11_el,a12_el,a21_el,a22_el) in zip(a11,a12,a21,a22):
        if a11_el*a22_el-a12_el*a21_el <= 0:
            return False
    return True

# Can't find correct priority function
def _priority(a11,a12,a21,a22,b1,b2):
    # A = np.array([[a11,a12],[a21,a22]])
    # b = np.array([b1,b2])
    # return np.linalg.norm(np.dot(A, b))
    return a11/a12

def _verify_extreme(r_max_raw, D):
    from itertools import combinations
    combins = combinations(range(DIMENSION), 2)
    for combin in combins:
        try:
            a = D[list(combin)][0,:]
            b = D[list(combin)][1,:]
        except Exception as e:
            import pdb; pdb.set_trace()
        sortind = list(np.argsort(a/b))
        all_consec = True
        for p in r_max_raw[1]:
            ind = [sortind.index(i) for i in p]
            print('{} {}'.format(combin, ind))
            if not np.all(np.diff(sorted(ind)) == 1):
                all_consec = False
        if all_consec:
            return True, combin
    combins = combinations(range(DIMENSION), 2)        
    for combin in combins:
        try:
            b = D[list(combin)][0,:]
            a = D[list(combin)][1,:]
        except Exception as e:
            import pdb; pdb.set_trace()
        sortind = list(np.argsort(a/b))
        all_consec = True
        for p in r_max_raw[1]:
            ind = [sortind.index(i) for i in p]
            print('{} {}'.format(combin, ind))
            if not np.all(np.diff(sorted(ind)) == 1):
                all_consec = False
        if all_consec:
            return True, combin
    import pdb;pdb.set_trace()
    return False, None
                

# Maximal ordered partition demonstration
if __name__ == '__main__':
    NUM_POINTS =        int(sys.argv[1]) or 3          # N
    PARTITION_SIZE =    int(sys.argv[2]) or 2          # T
    DIMENSION = 6                                      # For the time being

    NUM_WORKERS = min(NUM_POINTS, multiprocessing.cpu_count() - 1)
    SCORE_FN = double_power_fn
    
    num_partitions = Bell_n_k(NUM_POINTS, PARTITION_SIZE)
    num_mon_partitions = _Mon_n_k(NUM_POINTS, PARTITION_SIZE)
    partitions = knuth_partition(range(0, NUM_POINTS), PARTITION_SIZE)
    
    slices = slice_partitions(partitions)

    trial = 0
    bad_cases = 0
    while True:

        a11,a12,a21,a22 = None,None,None,None
        while (not _pos_dev_all(a11,a12,a21,a22)):
               a11 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))
               a12 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))
               a21 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))
               a22 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))
                              
        
        b1 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))
        b2 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))

        sortind = np.argsort([_priority(a11_el,a12_el,a21_el,a22_el,b1_el,b2_el)
                              for (a11_el,a12_el,a21_el,a22_el,b1_el,b2_el) in zip(a11,a12,a21,a22,b1,b2)])
        a11 = a11[sortind]
        a12 = a12[sortind]
        a21 = a21[sortind]
        a22 = a22[sortind]
        b1 = b1[sortind]
        b2 = b2[sortind]

        D = np.array([a11])
        D = np.concatenate([D,[a12]],axis=0)
        D = np.concatenate([D,[a21]],axis=0)
        D = np.concatenate([D,[a22]],axis=0)
        D = np.concatenate([D,[b1]],axis=0)
        D = np.concatenate([D,[b2]],axis=0)

        r_max_raw = optimize(a11, a12, a21, a21, b1, b2, PARTITION_SIZE, NUM_WORKERS)

        print(_verify_extreme(r_max_raw, D))
        
        try:
            assert False
            # assert all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1)
            # assert np.max([o[0] for o in con_optim_all]) >= np.max([o[0] for o in optim_all])
            _verify_extreme(r_max_raw)
        except AssertionError as e:
            continue
            # optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER) for i in range(1, 1+len(a0))]

            # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)
            # ss = [list(range(0,i)) for i in range(1,len(a0))] + [list(range(i,len(a0))) for i in range(len(a0)-1,-1,-1)]
            # _ = [print((p,SCORE_FN(a0,b0,POWER,p))) for p in ss]                        
            # res = Task(a0,b0,ss,power=POWER,cond=max,score_fn=SCORE_FN)()
    
        trial += 1
