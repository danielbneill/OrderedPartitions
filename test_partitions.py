import os
import sys
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
if __name__ == '__NOT_MAIN__':
    NUM_POINTS = 8
    POWER = 3.6
    NUM_WORKERS = multiprocessing.cpu_count() - 1
    INT_LIST= range(0, NUM_POINTS)
    
    partitions = subsets(NUM_POINTS)

    slices = slice_partitions(partitions)

    trial = 0
    while True:
        a0 = rng.uniform(low=1.0, high=10.0, size=int(NUM_POINTS))
        b0 = rng.uniform(low=1., high=10.0, size=int(NUM_POINTS))        
        
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

# Maximal ordered partition demonstration
if __name__ == '__main__':
    NUM_POINTS =        int(sys.argv[1]) or 3   # N
    PARTITION_SIZE =    int(sys.argv[2]) or 2   # T
    POWER =             float(sys.argv[3]) or 2.2      # gamma

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

        r_max_raw = optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS)
        
        if True:
            print('TRIAL: {} : max_raw: {:4.6f} pttn: {!r}'.format(trial, *r_max_raw))
            
        try:
            assert all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1)
        except AssertionError as e:
            # if any([len(x)==1 for x in r_max_abs[1]]):
            #     continue

            # Stop if exception found
            import pdb
            pdb.set_trace()
            if not os.path.exists('./violations'):
                os.mkdir('./violations')
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


