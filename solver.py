import functools
from functools import partial
import numpy as np
import multiprocessing
import networkx as nx
import networkx.algorithms.shortest_paths
from utils import knuth_partitions, monotonic_partitions


class PartitionOptimizerTask(object):
    def __init__(self,
                 a,
                 b,
                 c,
                 solver_type,
                 partition,
                 partition_type='full'):
        
        self.partition = partition
        self.solver_type = solver_type
        self.task = partial(self._task, a, b, c)
        self.partition_type = partition_type # {'full', 'endpoints'}

    def __call__(self):
        return self._task(self.partition)

    def _task(self,
              a,
              b,
              c,
              partitions,
              report_each=1000):
        
        max_sum, arg_max = float('-inf'), -1
        
        for ind,part in enumerate(partitions):
            val, part_vertex = 0, [0] * len(part)

            for part_ind, p in enumerate(part):
                inds = range(p[0], p[1]) if self.partition_type == 'endpoints' else p
                if self.solver_type == 'linear_hessian':                    
                    part_sum = sum(a[inds])**2/sum(b[inds]) + sum(c[inds])
                # Is this correct?
                elif self.solver_type == 'quadratic':
                    part_sum = sum(a[inds])**2/sum(b[inds]) + sum(c[inds])
                elif self.solver_type == 'linear_constant':
                    part_sum = sum(a[inds])/sum(c[inds])
                else:
                    raise RuntimeError('incorrect solver_type specification')
                part_vertex[part_ind] = part_sum
                val += part_sum
                
            if val > max_sum:
                max_sum, arg_max, max_part_vertex = val, part, part_vertex

        return (max_sum, arg_max, max_part_vertex)

class PartitionOptimizerWorker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

class PartitionOptimizer(object):
    ''' Graph solver, brute force solver distributed optimizer, Python only.
        Creates and checks all partitions in brute-force mode, or checks only
        consecutive in graph mode. In order to distribute over partitions, the
        initial partition universe is consumed, then split.

        shortest_path_solver = True  => use graph-based solver
        shortest_path_solver = False => use brute-force solver
    '''
    
    def __init__(self,
                 a,
                 b,
                 c=None,
                 num_workers=None,
                 solver_type='linear_hessian',
                 use_monotonic_partitions=True,
                 shortest_path_solver=False):

        # Use abs(a) if c is defined
        self.n = len(a)
        self.a = a
        self.b = b
        self.c = c if c is not None else np.zeros(a.shape)
        self.num_workers = num_workers or multiprocessing.cpu_count() - 1
        self.solver_type = solver_type
        self.use_monotonic_partitions = use_monotonic_partitions
        self.shortest_path_solver = shortest_path_solver

        self.partition_type = 'endpoints' if self.use_monotonic_partitions else 'full'        
        self.INT_LIST = range(0, self.n)

        if self.solver_type == 'linear_hessian':
            self.sortind = np.argsort(self.a / self.b + self.c)
        elif self.solver_type == 'quadratic':
            self.sortind = np.argsort(self.a / self.b + self.c)
        elif self.solver_type == 'linear_constant':
            self.sortind = np.argsort(self.a / self.c)
        else:
            raise RuntimeError('incorrect solver_type specification')
        
        (self.a,self.b,self.c) = (seq[self.sortind] for seq in (self.a,self.b,self.c))

    def run(self, num_partitions):
        self.run_shortest_path_solver(num_partitions) if self.shortest_path_solver \
                                                      else self.run_brute_force(num_partitions)
                                                      
    @functools.lru_cache(4096)
    def _calc_weight(self, i, j):
        return np.sum(self.a[i:j])**2/np.sum(self.b[i:j])

    def run_shortest_path_solver(self, num_partitions):
        ''' Graph setup for shortest-path-based graph solver,
            assumes solution can be found among consecutive partitions.
        '''
        
        G = nx.DiGraph()
        
        # source
        G.add_node((0,0))
        # sink
        G.add_node((self.n, num_partitions))

        def connect_nodes(G, j, k):
            for i in range(j):
                if G.has_node((i,k-1)):
                    weight = -1 * self._calc_weight(i, j)
                    G.add_edge((i,k-1), (j,k), weight=weight)

        for k in range(1, num_partitions):
            for j in range(k, self.n):
                if j <= self.n-(num_partitions-k):
                    G.add_node((j, k))
                    connect_nodes(G, j, k)
                    if not j % 1000:
                        print('added edges: ({}, {})'.format(j,k))
            print('completed layer {}'.format(k))
                    
        # connect sink to previous layer
        for i in range(self.n):
            if G.has_node((i, num_partitions - 1)):
                weight = -1 * self._calc_weight(i, self.n)
                G.add_edge((i, num_partitions-1), (self.n, num_partitions), weight=weight)

        # Bellman-Ford shortest path
        path = networkx.algorithms.shortest_paths.weighted.bellman_ford_path(G,
                                                                             (0,0),
                                                                             (self.n, num_partitions),
                                                                             weight='weight')

        subsets = [range(p[0], q[0]) for p,q in zip(path[:-1], path[1:])]
        summands = [np.sum(self.a[subset])**2/np.sum(self.b[subset]) for subset in subsets]

        self.maximal_val = sum(summands)
        self.maximal_sorted_part = subsets
        self.maximal_part = [list(self.sortind[subset]) for subset in subsets]
        self.leaf_values = [np.sum(self.a[part])/np.sum(self.b[part]) for part in subsets]
        self.summands = summands

    def run_brute_force(self, num_partitions):
        self.slice_partitions(num_partitions)

        num_slices = len(self.slices) # should be the same as num_workers
        if num_slices > 1:            
            tasks = multiprocessing.JoinableQueue()
            results = multiprocessing.Queue()
            workers = [PartitionOptimizerWorker(tasks, results) for i in range(self.num_workers)]

            for worker in workers:
                worker.start()

            for i,slice in enumerate(self.slices):
                tasks.put(PartitionOptimizerTask(self.a,
                                                 self.b,
                                                 self.c,
                                                 self.solver_type,
                                                 slice,
                                                 partition_type=self.partition_type))

            for i in range(self.num_workers):
                tasks.put(None)

            tasks.join()

            allResults = list()
            while not results.empty():
                result = results.get()
                allResults.append(result)            
        else:
            task = PartitionOptimizerTask(self.a,
                                          self.b,
                                          self.c,
                                                        self.solver_type,
                                          self.slices[0],
                                          partition_type=self.partition_type)
            allResults = [task()]
            
        def reduce(allResults, fn):
            return fn(allResults, key=lambda x: x[0])

        try:
            val,subsets,summands = reduce(allResults, max)
        except ValueError:
            raise RuntimeError('optimization failed for some reason')

        if self.partition_type == 'endpoints':
            subsets = [range(s[0],s[1]) for s in subsets]

        self.allResults = allResults
        self.maximal_val = val            
        self.maximal_sorted_part = subsets
        self.maximal_part = [list(self.sortind[subset]) for subset in subsets]
        self.leaf_values = [np.sum(self.a[part])/np.sum(self.b[part]) for part in subsets]
        self.summands = summands

    def slice_partitions(self, num_partitions):
        if self.use_monotonic_partitions == True:
            partitions = monotonic_partitions(self.n, num_partitions)
        else:
            partitions = knuth_partitions(self.INT_LIST, num_partitions)
        
        # Have to consume; can't split work on generator
        partitions = list(partitions)
        num_partitions = len(partitions)

        stride = max(int(num_partitions/self.num_workers), 1)
        bin_ends = list(range(0, num_partitions, stride))
        bin_ends = bin_ends + [num_partitions] if num_partitions/self.num_workers else bin_ends
        islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))

        # Shuffle partitions, then form slices
        rng.shuffle(partitions)
        slices = [list(islice(partitions, *ind)) for ind in islice_on]
        self.slices = slices

