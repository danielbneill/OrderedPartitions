import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from functools import partial
from itertools import chain, islice

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

X,y = make_classification(random_state=55, n_samples=15)
X_train, X_test, y_train, y_test = train_test_split(X, y)

SEED = 55
rng = np.random.RandomState(SEED)

class GradientBoostingPartitionClassifier(object):
    def __init__(self,
                 X,
                 y,
                 min_partition_size,
                 max_partition_size,
                 gamma,
                 eta,
                 num_classifiers=100,
                 ):

        # Inputs
        self.X = theano.shared(value=X, name='X', borrow=True)
        self.y = theano.shared(value=y, name='y', borrow=True)
        initial_X = self.X.get_value()
        self.N, self.num_features = initial_X.shape
        
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.gamma = gamma
        self.eta = eta
        self.num_classifiers = num_classifiers
        self.curr_classifier = 0

        # Derived
        # optimal partition at each step, not part of any
        # gradient calculation so not a tensor
        self.partitions = list()
        # distinct leaf values at each step, also not a tesnor
        self.distinct_leaf_values = np.zeros((self.num_classifiers,
                                             self.N))
        # regularization penalty at each step
        self.regularization = theano.shared(name='regularization',
                                            value=np.zeros((self.num_classifiers,
                                                            1)).astype(theano.config.floatX))
        # optimal learner at each step
        self.leaf_values = theano.shared(name='leaf_values',
                                         value=np.zeros((self.num_classifiers,
                                                         self.N,
                                                         1)).astype(theano.config.floatX))
        # optimal approximate tree at each step
        self.approx_partition_trees = theano.shared(name='approx_partition_trees',
                                                    value=np.zeros((self.num_classifiers,
                                                                    self.num_features + 1,
                                                                    1)).astype(theano.config.floatX))

        # set initial classifier to the output mean
        y_bar = theano.function([], T.mean(self.y))()
        leaf_value = y_bar * np.ones((self.N, 1)).astype(theano.config.floatX)
        self.set_next_leaf_value(leaf_value)

        # Set initial partition to be the size 1 partition (all leaf values the same)
        self.partitions.append(list(range(self.N)))

        # Set initial classifier
        approx_partition_tree = self.imply_tree(leaf_value)
        self.set_next_classifier(approx_partition_tree)        
        
        # For testing
        self.srng = T.shared_randomstreams.RandomStreams(seed=SEED)                

    def set_next_classifier(self, classifier):
        i = self.curr_classifier
        c = T.dmatrix()
        update = (self.approx_partition_trees,
                  T.set_subtensor(self.approx_partition_trees[i, :, :], c))
        f = theano.function([c], updates=[update])
        f(classifier)

    def set_next_leaf_value(self, leaf_value):
        i = self.curr_classifier
        c = T.dmatrix()
        update = (self.leaf_values,
                  T.set_subtensor(self.leaf_values[i, :, :], c))
        f = theano.function([c], updates=[update])
        f(leaf_value)

    def imply_tree(self, leaf_values):
        from sklearn.linear_model import LinearRegression
        X0 = self.X.get_value()
        y0 = leaf_values
        reg = LinearRegression(fit_intercept=True).fit(X0, y0)
        implied_tree = np.concatenate([reg.coef_.squeeze(), reg.intercept_]).reshape(-1,1)
        return implied_tree

    def weak_learner_predict(self, approx_partition_tree):
        X = T.concatenate([self.X, T.as_tensor(np.ones((self.N, 1)).astype(theano.config.floatX))], axis=1)
        y_hat = T.dot(X, approx_partition_tree)
        return y_hat

    def predict(self):
        def iter_step(learner):
            y_step = self.weak_learner_predict(learner)
            return y_step
        
        y,inner_updates = theano.scan(
            fn=iter_step,
            sequences=[self.approx_partition_trees],
            outputs_info=[None]
            )

        return T.sum(y, axis=0)
    
    def loss(self, y_hat):
        return self._mse(y_hat) + T.sum(self.regularization)

    def loss_without_regularization(self, y_hat):
        return self._mse(y_hat)

    def find_optimal_partition(self, num_partitions):
        g,h = self.generate_coefficients()

        optimizer = PartitionTreeOptimizer(g, h)
        optimizer.run(num_partitions)

        # Set next partition to be optimal
        self.partitions.append(optimizer.maximal_part)

        assert np.isclose(optimizer.maximal_val, np.sum(optimizer.summands)), 'optimal value mismatch'
        for part,val in zip(optimizer.maximal_part, optimizer.summands):
            assert np.isclose(np.sum(np.abs(g)[part])**2/np.sum(h[part]), val), 'optimal partitions mismatch'

        # Find optimal tree cuts
        leaf_values = np.zeros((self.N, 1))
        for part in optimizer.maximal_part:
            min_val = -1 * np.sum(g[part])/np.sum(h[part])
            leaf_values[part] = min_val
            
        # Find implied tree
        implied_tree = self.imply_tree(leaf_values)
            
        
        return 10
        
    def generate_coefficients(self):
        ''' Generate gradient, hessian sequences for offgraph
            optimizaition, return types are np.arrays
        '''
        x = T.dvector('x')
        grads = T.grad(self.loss_without_regularization(x), x)
        hess = T.hessian(self.loss_without_regularization(x), x)

        G = theano.function([x], grads)
        H = theano.function([x], hess)

        # Test
        y_hat0 = rng.uniform(low=0., high=1., size=(self.N,))
        f_t = rng.uniform(low=0., high=.01, size=(self.N,))
        loss0 = theano.function([x], self.loss_without_regularization(x))(y_hat0 + f_t)
        
        loss0_approx = theano.function([x], self.loss_without_regularization(x))(y_hat0) + \
                      np.dot(G(y_hat0), f_t) + \
                      0.5 * np.dot(y_hat0.T, np.dot(H(y_hat0), f_t))
        assert np.isclose(loss0, loss0_approx, rtol=0.01)

        y_hat0 = theano.function([], self.predict())().squeeze()
        g = G(y_hat0) + self.eta
        h = np.zeros(y_hat0.shape)
        for i in range(g.shape[0]):
            h[i] = np.sum(np.dot(H(y_hat0)[i,:], y_hat0))

        # Test
        quadratic_term = 0.5 * np.dot(y_hat0.T, np.dot(H(y_hat0), f_t))
        quadratic_term_coeff = 0.5 * np.dot(h, f_t)
        assert quadratic_term == quadratic_term_coeff

        return (g, h)        

    def _mse(self, y_hat):
        return T.sqrt(T.sum((T.shape_padaxis(self.y, 1) - y_hat)**2))

    def _regularizer(self, leaf_values):
        size_reg = self.gamma * partition_size
        coeff_reg = 0.5 * self.eta * np.sum(leaf_values**2)
        return size_req + coeff_reg

import multiprocessing

class Task(object):
    def __init__(self, a, b, partition):
        self.partition = partition
        self.task = partial(Task._task, a, b)

    def __call__(self):
        return self.task(self.partition)

    @staticmethod
    def _task(a, b, partitions, report_each=1000):
        max_sum = float('-inf')
        arg_max = -1
        for ind,part in enumerate(partitions):
            val = 0
            part_val = [0] * len(part)
            part_vertex = [0] * len(part)
            for part_ind, p in enumerate(part):
                part_sum = sum(a[p])**2/sum(b[p])
                part_vertex[part_ind] = part_sum
                part_val[part_ind] = part_sum
                val += part_sum
            if val > max_sum:
                max_sum = val
                arg_max = part
                max_part_vertex = part_vertex
            # if not ind%report_each:
            #     print('Percent complete: {:.{prec}f}'.
            #           format(100*len(slices)*ind/num_partitions, prec=2))
        return (max_sum, arg_max, max_part_vertex)

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
            if task is None:
                # print('Exiting: {}'.format(proc_name))
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

class PartitionTreeOptimizer(object):
    def __init__(self, a, b, num_workers=None):
        self.a = abs(a)
        self.b = b
        self.n = len(a)
        self.num_workers = num_workers or multiprocessing.cpu_count() - 1
        
        self.INT_LIST = range(0, self.n)

        assert (b > 0).all(), 'b must be a positive sequence'

        sortind = np.argsort(self.a / self.b)
        self.sortind = sortind
        (self.a,self.b) = (seq[sortind] for seq in (self.a,self.b))

    def run(self, num_partitions):
        self.slice_partitions(num_partitions)

        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(tasks, results) for i in range(self.num_workers)]
        num_slices = len(self.slices) # should be the same as num_workers

        for worker in workers:
            worker.start()

        for i,slice in enumerate(self.slices):
            tasks.put(Task(self.a, self.b, slice))

        for i in range(self.num_workers):
            tasks.put(None)

        tasks.join()

        def reduce(allResults, fn):
            return fn(allResults, key=lambda x: x[0])

        allResults = list()
        while not results.empty():
            result = results.get()
            allResults.append(result)


        val,subsets,summands = reduce(allResults, max)
        self.maximal_val = val
        self.maximal_sorted_part = subsets
        self.maximal_part = [list(self.sortind[subset]) for subset in subsets]
        self.summands = summands

    def slice_partitions(self, num_partitions):
        partitions = PartitionTreeOptimizer._knuth_partitions(self.INT_LIST, num_partitions)
        
        # Have to consume it; can't split work on generator
        partitions = list(partitions)
        num_partitions = len(partitions)
        
        bin_ends = list(range(0,num_partitions,int(num_partitions/self.num_workers)))
        bin_ends = bin_ends + [num_partitions] if num_partitions/self.num_workers else bin_ends
        islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))
        
        rng.shuffle(partitions)
        slices = [list(islice(partitions, *ind)) for ind in islice_on]
        self.slices = slices
        
    @staticmethod
    def _knuth_partitions(ns, m):
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

    @staticmethod
    def _Bell_n_k(n, k):
        ''' Number of partitions of {1,...,n} into
            k subsets, a restricted Bell number
        '''
        if (n == 0 or k == 0 or k > n): 
            return 0
        if (k == 1 or k == n): 
            return 1

        return (k * PartitionTreeOptimizer._Bell_n_k(n - 1, k) + 
                    PartitionTreeOptimizer._Bell_n_k(n - 1, k - 1))
    
