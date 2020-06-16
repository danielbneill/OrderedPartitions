import numpy as np
import pickle
import multiprocessing
from scipy.special import comb
from functools import partial
from itertools import chain, islice, combinations

SEED = 124
rng = np.random.RandomState(SEED)

def subsets(ns):
    return list(chain(*[[[list(x)] for x in combinations(range(ns), i)] for i in range(1,ns+1)]))

def knuth_partition(ns, m):
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
    ''' Number of partitions of {1,...,n} into
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

def test_exceptions(a_file):
    b_file = '_'.join(['b'] + a_file.split('_')[1:])
    rmax_file = '_'.join(['rmax'] + a_file.split('_')[1:])    
    with open(a_file, 'rb') as f:
        a = pickle.load(f)
    with open(b_file, 'rb') as f:
        b = pickle.load(f)
    with open(rmax_file, 'rb') as f:
        rmax = pickle.load(f)
    return (a, b, rmax)

def reduce(return_values, fn):
    return fn(return_values, key=lambda x: x[0])

class EndTask(object):
    pass

class Task(object):
    def __init__(self, a, b, partition, power=2, cond=max):
        self.partition = partition
        self.cond = cond
        self.task = partial(self._task, a, b, power)

    def __call__(self):
        return self.task(self.partition)

    def _task(self, a, b, power, partitions, report_each=1000):

        if self.cond == min:
            max_sum = float('inf')
        else:
            max_sum = float('-inf')            
        
        arg_max = -1
        
        for ind,part in enumerate(partitions):

            val = 0
            part_val = [0] * len(part)
            print('INDEX: {} PARTITION: {!r}'.format(ind, part))
            for part_ind, p in enumerate(part):
                part_sum = np.sum(a[p])**power/np.sum(b[p])
                part_val[part_ind] = part_sum
                val += part_sum
                print('    PART_INDEX: {} SUBSET: {!r} PART_VAL: {}'.format(part_ind, p, part_sum))
            if self.cond(val, max_sum) == val:
                max_sum = val
                arg_max = part
            print('    FINAL VAL: {}'.format(val))
        print('MAX_SUM: {}, MAX_PART: {!r}'.format(max_sum, arg_max))
        print()
        import pdb
        pdb.set_trace()
        return (max_sum, arg_max)

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            task = self.task_queue.get()
            # print('{} : Fetched task of type {}'.format(proc_name, type(task)))
            if isinstance(task, EndTask):
                # print('Exiting: {}'.format(proc_name))
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

# LTSS demonstration
if __name__ == '__fake_news__':
    NUM_POINTS = 8
    POWER = 3.6
    NUM_WORKERS = multiprocessing.cpu_count() - 1
    INT_LIST= range(0, NUM_POINTS)
    
    partitions = subsets(NUM_POINTS)

    slices = slice_partitions(partitions)

    trial = 0
    while True:
        a00 = rng.uniform(low=1.0,  high=50.0, size=int(NUM_POINTS/2))
        a01 = rng.uniform(low=1.0,  high=50.0, size=NUM_POINTS-int(NUM_POINTS/2))    
        b00 = rng.uniform(low=1.0,  high=50.0, size=int(NUM_POINTS/2))
        b01 = rng.uniform(low=50.0, high=100.0, size=NUM_POINTS-int(NUM_POINTS/2))    
        
        a0 = np.concatenate([a00, a01])
        b0 = np.concatenate([b00, b01])
        
        ind = np.argsort(a0/b0)
        (a,b) = (seq[ind] for seq in (a0,b0))
        
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(tasks, results) for i in range(NUM_WORKERS)]
        num_slices = len(slices)

        if len(partitions) > 100000:
            for worker in workers:
                worker.start()

            for i,slice in enumerate(slices):
                tasks.put(Task(a, b, slice, power=3))

            for i in range(NUM_WORKERS):
                tasks.put(EndTask())

            tasks.join()
        
            allResults = list()
            slices_left = num_slices
            while not results.empty():
                result = results.get(block=True)
                allResults.append(result)
                slices_left -= 1
        else:
            allResults = [Task(a, b, partitions, power=POWER)()]
            
        r_max = reduce(allResults, max)

        try:
            assert all(np.diff(list(chain.from_iterable(r_max[1]))) == 1)
            assert -1+NUM_POINTS in r_max[1][0]
        except AssertionError as e:
            with open('_'.join(['a', str(SEED), str(trial), str(PARTITION_SIZE)]), 'wb') as f:
                pickle.dump(a, f)
            with open('_'.join(['b', str(SEED), str(trial), str(PARTITION_SIZE)]), 'wb') as f:
                pickle.dump(b, f)
            with open('_'.join(['rmax', str(SEED), str(trial), str(PARTITION_SIZE)]), 'wb') as f:
                pickle.dump(r_max, f)

        print('TRIAL: {} : max: {:4.6f} prtn: {!r}'.format(trial, *r_max))
    
        trial += 1

def optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, cond=max):
    ind = np.argsort(a0/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))
        
    if num_mon_partitions > 100:
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(tasks, results) for i in range(NUM_WORKERS)]
        num_slices = len(slices)
        
        for worker in workers:
            worker.start()

        for i,slice in enumerate(slices):
            tasks.put(Task(a, b, slice, power=POWER, cond=cond))

        for i in range(NUM_WORKERS):
            tasks.put(EndTask())

        tasks.join()
        
        allResults = list()
        slices_left = num_slices
        while not results.empty():
            result = results.get(block=True)
            allResults.append(result)
            slices_left -= 1
    else:
        partitions = list(knuth_partition(range(0, len(a)), PARTITION_SIZE))
        allResults = [Task(a, b, partitions, power=POWER, cond=cond)()]
    
            
    r_max = reduce(allResults, cond)

    # summands = [np.sum(a[p])**2/np.sum(b[p]) for p in r_max[1]]
    # parts = [ind[el] for el in [p for p in r_max[1]]]
    
    return r_max

if __name__ == '__main__':
    NUM_POINTS = 3
    PARTITION_SIZE = 2
    POWER = 2.2
    NUM_WORKERS = min(NUM_POINTS, multiprocessing.cpu_count() - 1)
    
    num_partitions = Bell_n_k(NUM_POINTS, PARTITION_SIZE)
    num_mon_partitions = _Mon_n_k(NUM_POINTS, PARTITION_SIZE)
    partitions = knuth_partition(range(0, NUM_POINTS), PARTITION_SIZE)
    
    slices = slice_partitions(partitions)

    trial = 0
    bad_cases = 0
    while True:
        # a0 = rng.choice(range(1,11), NUM_POINTS, True)
        # b0 = rng.choice(range(1,11), NUM_POINTS, True)

        a0 = rng.uniform(low=1.0, high=10.0, size=int(NUM_POINTS))
        b0 = rng.uniform(low=1., high=10.0, size=int(NUM_POINTS))

        a0 = np.round(a0, 1)
        b0 = np.round(b0, 1)

        a0 = np.array([0.3373581 , 0.23898464, 0.2959268 ])
        b0 = np.array([0.59529115, 0.42148619, 0.5216534 ])

        # XXX
        # Counterexample for $a \in \(-\infty, \infty\)$
        # (NUM_POINTS=4, PARTITION_SIZE=3, POWER=1)
        # (NUM_POINTS=4, PARTITION_SIZE=3, POWER=3)        
        # a0 = np.array([-6., -8, -4., -10.])
        # b0 = np.array([1., 3., 3., 10.])
        # (NUM_POINTS=4, PARTITION_SIZE=3, POWER=4)                
        # a0 = np.array([-10., 7., 7., .5])
        # b0 = np.array([10., 5., 1.5, .25])
        # a0 = np.array([8., 2., 9., 2.])
        # b0 = np.array([9., 2., 6., 2.])
        
        # Counterexample for $a \in \(0, \infty\)$
        # (NUM_POINTS=4, PARTITION_SIZE=3, POWER=1)
        # a0 = np.array([0.28000646, 6.97468258, 8.51210092, 7.83160823])
        # b0 = np.array([5.4085982 , 0.8478136 , 1.03353282, 2.89178694])
        # (NUM_POINTS=4, PARTITION_SIZE=3, POWER=3)        
        # a0 = np.array([5.18425727, 9.67691875, 2.7966528 , 5.61807798])
        # b0 = np.array([8.05647034, 9.22466034, 3.90976294, 7.13989   ])
        # (NUM_POINTS=4, PARTITION_SIZE=3, POWER=4)
        # a0 = np.array([2.200595  , 3.61049965, 7.39838389, 9.9645569 ])
        # b0 = np.array([3.16346978, 1.3654458 , 3.8398001 , 3.20519284])
        # a0 = np.array([6.59506835, 4.67466026, 8.83880866, 7.1776444 ])
        # b0 = np.array([5.88871027, 2.82525707, 4.85830833, 1.02209203])

        # Counterexample for $a \in \(-\infty, \infty\)$ without length 1 subset
        # (NUM_POINTS=4, PARTITION_SIZE=2, POWER=1)
        # a0 = np.array([-1.54236861,  3.26701667, -1.0798663 ,  9.04501844])
        # b0 = np.array([1.53368709, 3.11402404, 1.53708585, 7.59637874])
        # (NUM_POINTS=4, PARTITION_SIZE=2, POWER=3)
        # a0 = np.array([-1.02883599, -0.62152327, -8.83708256,  6.33207405])
        # b0 = np.array([2.89341254, 6.51091349, 1.53705617, 5.28271074])
        # (NUM_POINTS=5, PARTITION_SIZE=2, POWER=2) - close
        # a0 = np.array([-9.92934036255, -4.26020002365, -9.73161029816, -8.90641021729, -0.671411991119])
        # b0 = np.array([6.59366989136, 4.78853988647, 8.3982004071e-06, 6.82870006561, 1.29790997505])
        
        # r_max_abs = optimize(np.abs(a0), b0, PARTITION_SIZE, POWER, NUM_WORKERS)
        # r_max_neg = optimize(-a0, -b0, PARTITION_SIZE, POWER, NUM_WORKERS, cond=min)        
        r_max_raw = optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS)
        
        if True:
            print('TRIAL: {} : max_raw: {:4.6f} pttn: {!r}'.format(trial, *r_max_raw))
            # print('     : {} : max_neg: {:4.6f} pttn: {!r}'.format(trial, *r_max_neg))            
            # print('     : {} : max_abs: {:4.6f} pttn: {!r}'.format(trial, *r_max_abs))

        try:
            assert all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1)
        except AssertionError as e:
            # if any([len(x)==1 for x in r_max_abs[1]]):
            #     continue
            import pdb
            pdb.set_trace()
            with open('_'.join(['./violations/a', str(SEED),
                                str(trial),
                                str(PARTITION_SIZE)]), 'wb') as f:
                pickle.dump(a0, f)
            with open('_'.join(['./violations/b', str(SEED),
                                str(trial),
                                str(PARTITION_SIZE)]), 'wb') as f:
                pickle.dump(b0, f)
            with open('_'.join(['./violations/rmax', str(SEED),
                                str(trial),
                                str(PARTITION_SIZE)]), 'wb') as f:
                pickle.dump(r_max_raw, f)
            bad_cases += 1
            if bad_cases == 10:
                import sys
                sys.exit()
    
        trial += 1


import matplotlib.pyplot as plt
import numpy as np
import pickle
import multiprocessing
from scipy.special import comb
from functools import partial
from itertools import chain, islice, combinations

rng = np.random.RandomState(SEED)


a0 = np.array([0.3373581 , 0.23898464, 0.2959268 ])
b0 = np.array([0.59529115, 0.42148619, 0.5216534 ])

a0 = rng.uniform(low=1.0, high=10.0, size=int(3))
b0 = rng.uniform(low=1., high=10.0, size=int(3))

sortind = np.argsort(a0/b0)
a = a0[sortind]
b = b0[sortind]

part0 = [[0,1],[2]]
part1 = [[0,2],[1]]
gamma = 2.2
x = np.arange(0, 2*gamma, .0001)
y1 = np.array([np.sum([np.sum(a[p])**x0/np.sum(b[p]) for p in part0]) for x0 in x])
y2 = np.array([np.sum([np.sum(a[p])**x0/np.sum(b[p]) for p in part1]) for x0 in x])

plt.plot(x, y2-y1)
plt.pause(1e-3)
