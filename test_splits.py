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

SEED = 3369
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

def power_score_fn(a,b,gamma,p):
    return np.sum(a[p])**gamma/np.sum(b[p])

def double_power_score_fn(a,b,gamma,p):
    # XXX
    # return (np.sum(a[p])**gamma)*(np.sum(b[p])**-2.0)
    return np.sum(a[p])**gamma

def log_score_fn(a,b,gamma,p):
    return -1.*np.log(1. + np.sum(a[p]))

class Task(object):
    def __init__(self, a, b, partition, power=2, cond=max, score_fn=power_score_fn, fnArgs=()):
        self.partition = partition
        self.cond = cond
        self.score_fn = score_fn
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
                fnArgs = (a,b,power,p)
                part_sum = self.score_fn(*fnArgs)
                part_val[part_ind] = part_sum
                val += part_sum
                print('    SUBSET: {!r} SUBSET SCORE: {:4.4f}'.format(p, part_sum))
            if self.cond(val, max_sum) == val:
                max_sum = val
                arg_max = part
            print('    PARTITION SCORE: {:4.4f}'.format(val))
        print('MAX PARTITION SCORE: {:4.4f}, MAX_PARTITION: {!r}'.format(max_sum, arg_max))
        print()
        return (max_sum, arg_max)

class EndTask(object):
    pass

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

def plot_convex_hull(a0,
                     b0,
                     score_fn=power_score_fn,
                     plot_extended=False,
                     plot_symmetric=False,
                     show_plot=True,
                     show_contours=True):
    NUM_AXIS_POINTS = 201


    fig1, ax1 = None, None

    def in_hull(dhull, x, y):
        return dhull.find_simplex((x,y)) >= 0

    def F_symmetric(x,y,gamma):
        import warnings
        warnings.filterwarnings('ignore')
        # ret = x**gamma/y + (Cx-x)**gamma/(Cy-y)
        # ret = -1.*(np.log(1. + x) + np.log(1. + (Cx-x)))
        # ret = (x**gamma)/(y**2.0) + ((Cx-x)**gamma)/((Cy-y)**2.0)
        ret = (x**gamma) + ((Cx-x)**gamma)
        warnings.resetwarnings()
        return ret

    def F_orig(x,y,gamma):
        import warnings
        warnings.filterwarnings('ignore')
        # ret = x**gamma/y
        # ret = -1.*np.log(1. + x)
        # ret = (x**gamma)/(y**2.0)
        ret = x**gamma
        warnings.resetwarnings()
        return ret

    ind = np.argsort(a0**PRIORITY_POWER/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))

    pi = subsets(len(a))
    if not plot_extended:
        mp = [p[0] for p in pi if len(p[0]) == len(a0)]
        pi.remove(mp)

    if plot_symmetric:
        F = F_symmetric
        title = 'F Symmetric, '
    else:
        F = F_orig
        title = 'F Non-Symmetric, '

    if plot_extended:
        title += 'Full Hull'
    else:
        title += 'Constrained Hull'

    title += '  Case: (n, T) = : ( ' + str(len(a0)) + ', ' + str(PARTITION_SIZE) + ' )'
        
    X = list()
    Y = list()
    txt = list()

    for subset in pi:
        s = subset[0]
        X.append(np.sum(a[s]))
        Y.append(np.sum(b[s]))
        txt.append(str(s))

    if plot_extended:
        X = [0.] + X
        Y = [0.] + Y
        txt = ['-0-'] + txt

    points = np.stack([X,Y]).transpose()

    Xm, XM = np.min(X), np.max(X)
    Ym, YM = np.min(Y), np.max(Y)
    Cx, Cy = np.sum(a), np.sum(b)        

    hull = ConvexHull(points)
    vertices = [points[v] for v in hull.vertices]
    dhull = Delaunay(vertices)

    if show_plot:
        cmap = plot.cm.RdYlBu        
        fig1, ax1 = plot.subplots(1,1)

        xaxis = np.linspace(Xm, XM, NUM_AXIS_POINTS)
        yaxis = np.linspace(Ym, YM, NUM_AXIS_POINTS)
        xaxis, yaxis = xaxis[:-1], yaxis[:-1]
        Xgrid,Ygrid = np.meshgrid(xaxis, yaxis)
        Zgrid = F(Xgrid, Ygrid, POWER)

        for xi,xv in enumerate(xaxis):
            for yi,yv in enumerate(yaxis):
                if in_hull(dhull, xv, yv):
                    continue
                else:
                    Zgrid[yi,xi] = 0.

        if show_contours:
            cp = ax1.contourf(Xgrid, Ygrid, Zgrid, cmap=cmap)            
            cp.changed()
            fig1.colorbar(cp)
            

        ax1.scatter(X, Y)
        for i,t in enumerate(txt):
            if i in hull.vertices:
                t = t.replace('[','<').replace(']','>')
            else:
                t = t.replace('[','').replace(']','')
            t = t.replace(', ', ',')
            ax1.annotate(t, (X[i], Y[i]))

        for simplex in hull.simplices:
            ax1.plot(points[simplex,0], points[simplex,1], 'k-')

        plot.title(title)    

    vertices_txt = [txt[v] for v in hull.vertices]
    header = 'FULL_' if plot_extended else 'CONST_'
    header += 'SYM' if plot_symmetric else 'NONSYM'
    print('{:12} : {!r}'.format(header,vertices_txt))

    return fig1, ax1, vertices_txt
    

def plot_polytope(a0, b0, plot_constrained=True, score_fn=power_score_fn, show_plot=True, save_plot=False):

    fig1, ax1,vert_const_asym = plot_convex_hull(a0,
                                                 b0,
                                                 score_fn=power_score_fn,
                                                 plot_extended=False,
                                                 plot_symmetric=False,
                                                 show_plot=show_plot)
    if save_plot:
        plot.savefig('plot1.pdf')
    else:
        plot.pause(1e-3)
        
    fig2, ax2,vert_const_sym = plot_convex_hull(a0,
                                                b0,
                                                score_fn=power_score_fn,
                                                plot_extended=False,
                                                plot_symmetric=True,
                                                show_plot=show_plot)
    if save_plot:
        plot.savefig('plot2.pdf')
    else:
        plot.pause(1e-3)
        
    fig3, ax3,vert_ext_asym = plot_convex_hull(a0,
                                               b0,
                                               score_fn=power_score_fn,
                                               plot_extended=True,
                                               plot_symmetric=False,
                                               show_plot=show_plot)
    if save_plot:
        plot.savefig('plot3.pdf')
    else:
        plot.pause(1e-3)
        
    fig4, ax4,vert_ext_sym = plot_convex_hull(a0,
                                              b0,
                                              score_fn=power_score_fn,
                                              plot_extended=True,
                                              plot_symmetric=True,
                                              show_plot=show_plot)
    if save_plot:
        plot.savefig('plot4.pdf')
    else:
        plot.pause(1e-3)    

    if show_plot:
        plot.close()
        plot.close()
        plot.close()
        plot.close()

    return vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym

def optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER, cond=max):
    ind = np.argsort(a0**PRIORITY_POWER/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))

    # XXX
    # if num_mon_partitions > 100:
    # Doesn't work in certain cases?
    if False:
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(tasks, results) for i in range(NUM_WORKERS)]
        num_slices = len(slices)
        
        for worker in workers:
            worker.start()

        for i,slice in enumerate(slices):
            tasks.put(Task(a, b, slice, power=POWER, cond=cond, score_fn=SCORE_FN))

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
        allResults = [Task(a, b, partitions, power=POWER, cond=cond, score_fn=SCORE_FN)()]
    
            
    r_max = reduce(allResults, cond)

    return r_max

# Maximal ordered partition demonstration
if __name__ == '__main__':
    NUM_POINTS =        int(sys.argv[1]) or 3          # N
    PARTITION_SIZE =    int(sys.argv[2]) or 2          # T
    POWER =             float(sys.argv[3]) or 2.2      # gamma
    PRIORITY_POWER =    float(sys.argv[4]) or 1.0      # tau
    FORCE_MIXED =       float(sys.argv[5]) or False

    NUM_WORKERS = min(NUM_POINTS, multiprocessing.cpu_count() - 1)

    # SCORE_FN = power_score_fn
    # SCORE_FN = log_score_fn
    SCORE_FN = double_power_score_fn
    
    num_partitions = Bell_n_k(NUM_POINTS, PARTITION_SIZE)
    num_mon_partitions = _Mon_n_k(NUM_POINTS, PARTITION_SIZE)
    partitions = knuth_partition(range(0, NUM_POINTS), PARTITION_SIZE)
    
    slices = slice_partitions(partitions)

    trial = 0
    bad_cases = 0
    while True:
        # a0 = rng.choice(range(1,11), NUM_POINTS, True)
        # b0 = rng.choice(range(1,11), NUM_POINTS, True)

        # XXX
        a0 = rng.uniform(low=-0., high=10.0, size=int(NUM_POINTS))
        b0 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))

        if FORCE_MIXED:
            while all(a0>0) or all(a0<0):
                a0 = rng.uniform(low=-10., high=10.0, size=int(NUM_POINTS))
                b0 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))                

        # XXX
        # a0 = np.round(a0, 2)
        # b0 = np.round(b0, 2)
        
        a0 = np.round(a0, 8)
        b0 = np.round(b0, 8)

        # a0 = np.array([8, 2, 9])
        # b0 = np.array([8, 1, 3])
        
        # gamma == 2.0, lambda > 1.0
        # x = 2.
        # delta = 1.
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
        # (4,3) case
        # a0 = np.array([ 0.772736, 0.523752, 0.858158, 0.120438 ])
        # b0 = np.array([ 0.378871, 0.210641, 0.256059, 0.032089 ])
        # (4,3) mixed-sign case
        # a0 = np.array([ -0.649362, -0.546916, -0.661853, 0.863409 ])
        # b0 = np.array([ 0.14663, 0.445009, 0.605386, 0.441687 ])
        # a0 = np.array([-9.9525, -9.5814, -0.42452918,  4.4462,  9.2472])
        # b0 = np.array([5.8318, 5.7728, 0.5, 8.1294, 9.2019])
        # (4, 3, 4) case
        # a0 = np.array([-5.554 , -2.501 , -4.9577, -6.0844])
        # b0 = np.array([1.3083, 1.6324, 3.5354, 6.7042])
                      
        # a0 = np.array([.75, .25, 1.25])
        # b0 = np.array([1, .25, 1])

        sortind = np.argsort(a0**PRIORITY_POWER/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        r_max_raw = optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER)
        # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=False)
        print('=====')

        if False:
            print('TRIAL: {} : max_raw: {:4.6f} pttn: {!r}'.format(trial, *r_max_raw))

        try:
            assert all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1)
        except AssertionError as e:

            # Stop if exception found

            vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)
            
            def F_orig(x,y,gamma):
                return SCORE_FN(x,y,gamma,range(len(x)))
            def F_symmetric(x,y,Cx,Cy,gamma):
                if np.sum(x) == Cx:
                    return F_orig(Cx,Cy,gamma)
                else:
                    return F_orig(x,y,gamma,range(len(x))) + F_orig(Cx-x,Cy-y,gamma,range(len(x)))
                
            all_scores = [(i,F_orig(a0[:i], b0[:i], POWER)) for i in range(1,len(a0))] + \
                         [(i,F_orig(a0[i:], b0[i:], POWER)) for i in range(1,len(a0))] + \
                         [(len(a0),F_orig(a0, b0, POWER))]
            all_sym_scores = [(i, F_orig(a0[:i], b0[:i], POWER) + F_orig(a0[i:], b0[i:], POWER))
                              for i in range(1,len(a0))] + \
                              [(len(a0), F_orig(a0, b0, POWER))]            
            optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER) for i in range(1, 1+len(a0))]

            import pdb
            pdb.set_trace()

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
