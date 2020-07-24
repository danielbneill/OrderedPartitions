import os
import sys
import numpy as np
import pickle
import multiprocessing
from scipy.special import comb
from functools import partial
from itertools import chain, islice, combinations

SEED = 349
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
            print('PARTITION: {!r}'.format(part))
            for part_ind, p in enumerate(part):
                part_sum = np.sum(a[p])**power/np.sum(b[p])
                part_val[part_ind] = part_sum
                val += part_sum
                print('    SUBSET: {!r} SUBSET SCORE: {:4.4f}'.format(p, part_sum))
            if self.cond(val, max_sum) == val:
                max_sum = val
                arg_max = part
            print('    PARTITION SCORE: {:4.4f}'.format(val))
        print('MAX PARTITION SCORE: {:4.4f}, MAX_PARTITION: {!r}'.format(max_sum, arg_max))
        # print()
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
        a0 = rng.uniform(low=-10.0, high=10.0, size=int(NUM_POINTS))
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

def plot_polytope(a0, b0):
    import matplotlib.pyplot as plot
    from scipy.spatial import ConvexHull, Delaunay

    ind = np.argsort(a0**PRIORITY_POWER/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))

    pi = subsets(len(a0))
    X = list()
    Y = list()
    txt = list()

    for subset in pi:
        s = subset[0]
        X.append(np.sum(a[s]))
        Y.append(np.sum(b[s]))
        txt.append(str(s))

    Xm, XM = np.min(X), np.max(X)
    Ym, YM = np.min(Y), np.max(Y)
    Cx, Cy = np.sum(a), np.sum(b)

    points = np.stack([X,Y]).transpose()
    hull = ConvexHull(points)
    vertices = [points[v] for v in hull.vertices]
    dhull = Delaunay(vertices)

    fig, ax = plot.subplots(1,1)
    
    PLOT_LEVEL_SETS = True
    
    if (PLOT_LEVEL_SETS):
        def in_hull(dhull, x, y):
            return dhull.find_simplex((x,y)) >= 0
    
        def F(x,y, gamma, dhull):
            return x**gamma/y + (Cx-x)**gamma/(Cy-y)
    
        xaxis = np.linspace(Xm, XM, 201)
        yaxis = np.linspace(Ym, YM, 201)
        xaxis, yaxis = xaxis[:-1], yaxis[:-1]
        Xgrid,Ygrid = np.meshgrid(xaxis, yaxis)
        Zgrid = F(Xgrid, Ygrid, POWER, dhull)
        
        for xi,xv in enumerate(xaxis):
            for yi,yv in enumerate(yaxis):
                if in_hull(dhull, xv, yv):
                    continue
                else:
                    Zgrid[yi,xi] = 0.

        cp = ax.contourf(Xgrid, Ygrid, Zgrid)
        fig.colorbar(cp)
        plot.pause(1e-3)

    ax.scatter(X, Y)
    for i,t in enumerate(txt):
        if i in hull.vertices:
            t = t.replace('[','<').replace(']','>')
        else:
            t = t.replace('[','').replace(']','')
        ax.annotate(t, (X[i], Y[i]))

    for simplex in hull.simplices:
        ax.plot(points[simplex,0], points[simplex,1], 'k-')

    vertices_txt = [txt[v] for v in hull.vertices]
    
    plot.pause(1e-3)

    import pdb
    pdb.set_trace()
    
    plot.close()

def optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER, cond=max):
    ind = np.argsort(a0**PRIORITY_POWER/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))

    # XXX
    # if num_mon_partitions > 100:
    if False:
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

    # import pdb
    # pdb.set_trace()

    # summands = [np.sum(a[p])**2/np.sum(b[p]) for p in r_max[1]]
    # parts = [ind[el] for el in [p for p in r_max[1]]]
    
    return r_max

# Maximal ordered partition demonstration
if __name__ == '__main__':
    NUM_POINTS =        int(sys.argv[1]) or 3   # N
    PARTITION_SIZE =    int(sys.argv[2]) or 2   # T
    POWER =             float(sys.argv[3]) or 2.2      # gamma
    PRIORITY_POWER =    float(sys.argv[4]) or 1.0

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

        a0 = rng.uniform(low=-10.0, high=10.0, size=int(NUM_POINTS))
        b0 = rng.uniform(low=1., high=10.0, size=int(NUM_POINTS))

        a0 = np.round(a0, 1)
        b0 = np.round(b0, 1)

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]    

        # a0 = np.array([8, 2, 9])
        # b0 = np.array([8, 1, 3])
        
        # gamma == 2.0, lambda > 1.0
        # x = 1e10
        # delta = 10
        # a0 = np.array([x-delta, delta, x+delta])
        # b0 = np.array([x, delta, x])

        # delta = 1
        # a0 = np.array([delta, 2*delta, 3*delta])
        # b0 = np.array([1, 1, 1])

        # gamma < 2.0
        # q = 1
        # epsilon = 1e-3
        # x = 1
        # b0 = np.array([1./(q*x+epsilon), 1./(q*epsilon), 1./(q*x-epsilon)])
        # a0 = np.array([1./x, 1./epsilon, 1./x])

        # a0 = np.array([-5.64313, -5.11986,  9.99038,  1.93718])
        # b0 = np.array([0.0772588, 1.22881  , 3.35838  , 0.0288292])

        # a0 = np.array([-5.64, -5.12,  10.0,  1.94])
        # b0 = np.array([0.077, 1.23, 3.36, 0.029])

        # a0 = np.array([ 0.37, 0.17, 0.50 ])
        # b0 = np.array([ 0.87, 0.37, 0.39 ])

        # a0 = np.array( [ 0.267532, 0.179856, 0.068246, 0.4343, 0.92863 ])
        # b0 = np.array( [ 0.6126, 0.312329, 0.090831, 0.566307, 0.566294 ] )
        # a0 = np.array([0.992819, 0.04904, 0.622353, 0.464107, 0.608956, 0.984192])
        # b0 = np.array([0.935323, 0.02541, 0.279373, 0.205452, 0.24599, 0.315633])
        # a0 = np.array([ -0.937353, -0.09833699999999999, -0.668365, 0.731261 ])
        # b0 = np.array([ 0.267252, 0.09030100000000001, 0.811923, 0.91252 ])
        # a0 = np.array([ 0.445962, 0.105416, 0.905763, 0.919112 ])
        # b0 = np.array([ 0.616067, 0.092109, 0.702833, 0.642663 ])
        # a0 = np.array([ 0.96563, 0.530856, 0.446896, 0.748362, 0.438265 ])
        # b0 = np.array([ 0.896815, 0.473519, 0.374769, 0.142674, 0.039357 ])
                      
        # a0 = np.array([.75, .25, 1.25])
        # b0 = np.array([1, .25, 1])

        sortind = np.argsort(a0/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        r_max_raw = optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER)
        # a0 = -1 * a0
        # r_max_neg = optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER)
        
        if True:
            print('TRIAL: {} : max_raw: {:4.6f} pttn: {!r}'.format(trial, *r_max_raw))

        try:
            assert all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1)
        except AssertionError as e:
            # if any([len(x)==1 for x in r_max_abs[1]]):
            #     continue
            
            # r_max_raw_minus_1 = optimize(a0, b0, PARTITION_SIZE-1, POWER, NUM_WORKERS, PRIORITY_POWER)

            # Stop if exception found

            plot_polytope(a0, b0)

            r_max_raw_less = optimize(a0, b0, PARTITION_SIZE-1, POWER, NUM_WORKERS, PRIORITY_POWER)

            if (r_max_raw[0] > r_max_raw_less[0]) or (not all(np.diff(list(chain.from_iterable(r_max_raw_less[1]))))):
                import pdb
                pdb.set_trace()
                
            # import pdb
            # pdb.set_trace()

            if False:
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

if (False):
#### Reconstruct ####
    import numpy as np
    def F(x,y):
        return np.sum(x)**2/np.sum(y)
    X1 = -3.70595
    Y1 = 0.106088
    X2 = 4.87052
    Y2 = 4.58719
    beta = 0.0772588
    alpha = -5.64313
    b = 3.35838
    a = 9.99038

    X1_val = X1
    Y1_val = Y1
    X2_val = X2
    Y2_val = Y2
    X1 = np.array([alpha, X1_val-alpha])
    Y1 = np.array([beta, Y1_val-beta])
    X2 = np.array([X2_val-a, a])
    Y2 = np.array([Y2_val-b, b])
    
    X1_m_alpha = X1[1:]
    Y1_m_beta  = Y1[1:]
    X1_p_a     = np.concatenate([X1, np.array([a])])
    Y1_p_b     = np.concatenate([Y1, np.array([b])])
    X2_p_alpha = np.concatenate([X2, np.array([alpha])])
    Y2_p_beta  = np.concatenate([Y2, np.array([beta])])
    X2_m_a     = X2[:-1]
    Y2_m_b     = Y2[:-1]
    X1_m_alpha_p_a = np.concatenate([X1_m_alpha, np.array([a])])
    Y1_m_beta_p_b = np.concatenate([Y1_m_beta, np.array([b])])
    X2_p_alpha_m_a = np.concatenate([X2[:-1], np.array([alpha])])
    Y2_p_beta_m_b = np.concatenate([Y2[:-1], np.array([beta])])
    
    assert (F(X1_m_alpha,Y1_m_beta)-F(X1,Y1)) > 0
    assert (F(X1_p_a,Y1_p_b)-F(X1,Y1)) < 0
    assert (F(X2_p_alpha,Y2_p_beta)-F(X2,Y2)) < 0
    assert (F(X2_m_a,Y2_m_b)-F(X2,Y2)) > 0
    
    top_row = F(X1_m_alpha,Y1_m_beta)+F(X2_p_alpha,Y2_p_beta)-F(X1,Y1)-F(X2,Y2)
    bot_row = F(X1_p_a,Y1_p_b)+F(X2_m_a,Y2_m_b)-F(X1,Y1)-F(X2,Y2)
    plus_minus = F(X1_m_alpha_p_a,Y1_m_beta_p_b)+F(X2_p_alpha_m_a,Y2_p_beta_m_b)-F(X1,Y1)-F(X2,Y2)

    a0 = np.array([X1[0], X2[0], X2[1], X1[1]])
    b0 = np.array([Y1[0], Y2[0], Y2[1], Y1[1]])

if (False):
    import numpy as np

    # gamma == 2.0, lambda > 1.0
    x = 1e6
    delta = 10.01
    theta = 2.0
    gamma = 2.0
    a0 = np.array([x-delta, delta, x+delta])
    b0 = np.array([x, delta, x])

    sortind = np.argsort(a0**theta/b0)
    a = a0[sortind]
    b = b0[sortind]

    part0 = [[0],[1,2]]
    part1 = [[0,1],[2]]
    part2 = [[0,2],[1]]

    sum([np.sum(a[part])**gamma/np.sum(b[part]) for part in part0])

if (False):
    import numpy as np
    import matplotlib.pyplot as plot
    
    gamma = 2.0
    delta = 1e-1
    C1 = 2+delta
    C2 = 2+delta
    epsilon = delta/2

    def F(x,y,gamma):
        return x**gamma/y + (C1-x)*gamma/(C2-y)

    xaxis = np.linspace(0.01, 2+delta-epsilon, 1000)
    yaxis = np.linspace(0.25, 2+delta-epsilon, 1000)
    X,Y = np.meshgrid(xaxis, yaxis)
    Z = F(X,Y,gamma)

    fig,ax = plot.subplots(1,1)
    cp = ax.contourf(X, Y, Z)
    fig.colorbar(cp)
    plot.show()
