import logging
import multiprocessing
import numpy as np
import pandas as pd
from itertools import combinations
from functools import partial
from scipy.special import comb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from functools import partial, lru_cache
from itertools import chain, islice
import functools
import multiprocessing

import networkx as nx
import networkx.algorithms.shortest_paths    

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import proto

import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_confusion(confusion_matrix, class_names, figsize=(10,7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
  
X,y = make_classification(random_state=551, n_samples=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

SEED = 144
rng = np.random.RandomState(SEED)

class GradientBoostingPartitionClassifier(object):
    def __init__(self,
                 X,
                 y,
                 min_partition_size,
                 max_partition_size,
                 gamma=0.,
                 eta=0.,
                 num_classifiers=100,
                 use_constant_term=False,
                 solver_type='linear_hessian',
                 learning_rate=0.1,
                 distill_method='OLS',
                 use_monotonic_partitions=True,
                 shortest_path_solver=False,
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

        # algorithm directives
        # solver_type is one of
        # ('quadratic, 'linear_hessian', 'linear_constant')
        self.use_constant_term = use_constant_term
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        self.distill_method = distill_method
        self.use_monotonic_partitions = use_monotonic_partitions
        self.shortest_path_solver = shortest_path_solver

        # Derived
        # optimal partition at each step, not part of any
        # gradient calculation so not a tensor
        self.partitions = list()
        # distinct leaf values at each step, also not a tesnor
        self.distinct_leaf_values = np.zeros((self.num_classifiers + 1,
                                             self.N))
        # regularization penalty at each step
        self.regularization = theano.shared(name='regularization',
                                            value=np.zeros((self.num_classifiers + 1,
                                                            1)).astype(theano.config.floatX))
        # optimal learner at each step
        self.leaf_values = theano.shared(name='leaf_values',
                                         value=np.zeros((self.num_classifiers + 1,
                                                         self.N,
                                                         1)).astype(theano.config.floatX))
        # classifier at each step
        self.implied_trees = [DirectImpliedTree(self.X.get_value(), 0.5 * np.ones((self.N, 1)))] * \
                             (self.num_classifiers + 1)
        
        # set initial random leaf values
        # leaf_value = np.asarray(rng.choice((0, 1),
        #                               self.N)).reshape(self.N, 1).astype(theano.config.floatX)
        # noise = np.asarray(rng.choice((-1e-1, 1e-1),
        #                               self.N)).reshape(self.N, 1).astype(theano.config.floatX)
        # leaf_value = self.y.get_value().reshape((self.N, 1)) + noise
        if self.distill_method in ('LDA', 'DecisionTree'):
            # Cannot have number of unique classes == number of samples, so
            # we must restrict sampling to create fewer classes
            leaf_value = rng.choice(rng.uniform(low=0.0, high=1.0, size=(self.min_partition_size,)),
                                    size=(self.N, 1)).astype(theano.config.floatX)
        else:
            leaf_value = np.asarray(rng.uniform(low=0.0, high=1.0, size=(self.N, 1))).astype(theano.config.floatX)
        self.set_next_leaf_value(leaf_value)

        # Set initial partition to be the size 1 partition (all leaf values the same)
        self.partitions.append(list(range(self.N)))

        # Set initial classifier
        implied_tree = self.imply_tree(leaf_value)
        
        self.set_next_classifier(implied_tree)        
        
        # For testing
        self.srng = T.shared_randomstreams.RandomStreams(seed=SEED)

        self.curr_classifier += 1

    def set_next_classifier(self, classifier):
        i = self.curr_classifier
        self.implied_trees[i] = classifier

    def set_next_leaf_value(self, leaf_value):
        i = self.curr_classifier
        c = T.dmatrix()
        update = (self.leaf_values,
                  T.set_subtensor(self.leaf_values[i, :, :], c))
        f = theano.function([c], updates=[update])
        f(leaf_value)

    def imply_tree(self, leaf_values):
        X0 = self.X.get_value()
        y0 = leaf_values
            
        if self.distill_method == 'OLS':
            implied_tree = OLSImpliedTree(X0, y0)
        elif self.distill_method == 'LDA':
            implied_tree = LDAImpliedTree(X0, y0)
        elif self.distill_method == 'SVM':
            implied_tree = SVMImpliedTree(X0, y0)
        elif self.distill_method == 'direct':
            implied_tree = DirectImpliedTree(X0, y0)
        elif self.distill_method == 'DecisionTree':
            implied_tree = DecisionTreeImpliedTree(X0, y0)

        return implied_tree

    def weak_learner_predict(self, classifier_ind):
        classifier = self.implied_trees[classifier_ind]
        return classifier.predict(self.X)

    def predict_from_input(self, X0):
        X = theano.shared(value=X0.astype(theano.config.floatX))
        y_hat = theano.shared(name='y_hat', value=np.zeros((X0.shape[0], 1)))
        for classifier_ind in range(self.curr_classifier):
            y_hat += self.implied_trees[classifier_ind].predict(X)
        return y_hat
        
    def predict(self):        
        y_hat = theano.shared(name='y_hat', value=np.zeros((self.N, 1)))
        for classifier_ind in range(self.curr_classifier):
            y_hat += self.implied_trees[classifier_ind].predict(self.X)
        return y_hat

    def predict_old(self):
        def iter_step(classifier_ind):
            y_step = self.weak_learner_predict(classifier_ind)
            return y_step

        # scan is short-circuited by length of T.arange(self.curr_classifier)
        y,inner_updates = theano.scan(
            fn=iter_step,
            sequences=[T.arange(self.curr_classifier)],
            outputs_info=[None],
            )

        return T.sum(y, axis=0)
    
    def fit(self, num_steps=None):
        num_steps = num_steps or self.num_classifiers
        print('STEP {}: LOSS: {:4.6f}'.format(0,
                                              theano.function([],
                                                              clf.loss_without_regularization(
                                                                  clf.predict()))()))        
        for i in range(num_steps):
            self.fit_step()
            print('STEP {}: LOSS: {:4.6f}'.format(i,
                                                  theano.function([],
                                                                  self.loss_without_regularization(
                                                                      clf.predict()))()))
        print('Training finished')

    def fit_step(self):
        g, h, c = self.generate_coefficients(constantTerm=self.use_constant_term)

        # Pure Python optimizer
        # optimizer = PartitionTreeOptimizer(g,
        #                                    h,
        #                                    c,
        #                                    solver_type=self.solver_type,
        #                                    use_monotonic_partitions=self.use_monotonic_partitions,
        #                                    shortest_path_solver=self.shortest_path_solver)

        

        # num_partitions = rng.choice(range(self.min_partition_size, self.max_partition_size))
        # print('num_partitions: {}'.format(num_partitions))
        
        # optimizer.run(num_partitions)

        # Set next partition to be optimal
        # self.partitions.append(optimizer.maximal_part)

        # Assert optimization correct
        # assert np.isclose(optimizer.maximal_val, np.sum(optimizer.summands)), \
        #        'optimal value mismatch'
        # for part,val in zip(optimizer.maximal_part, optimizer.summands):
        #     if self.solver_type == 'linear_hessian':
        #         assert np.isclose(np.sum(abs(g[part]))**2/np.sum(h[part]), val), \
        #                'optimal partitions mismatch'

        # print('LENGTH OF OPTIMAL PARTITIONS: {!r}'.format(
        #     [len(part) for part in optimizer.maximal_part]))
        # print('OPTIMAL SORTED PARTITION ENDPOITS: {!r}'.format(
        #     [(p[0],p[-1]) for p in optimizer.maximal_sorted_part]))
        # print('OPTIMAL SUMMANDS: {!r}'.format(
        #     optimizer.summands))
        # print('OPTIMAL LEAF VALUES: {!r}'.format(
        #     optimizer.leaf_values))
                            
        # Calculate optimal leaf_values
        # leaf_value = np.zeros((self.N, 1))        
        # for part in optimizer.maximal_part:
        #     if self.solver_type == 'quadratic':
        #         r1, r2 = self.quadratic_solution_scalar(np.sum(g[part]),
        #                                                 np.sum(h[part]),
        #                                                 np.sum(c[part]))
        #         leaf_value[part] = self.learning_rate * r1
        #     elif self.solver_type == 'linear_hessian':
        #         min_val = -1 * np.sum(g[part])/np.sum(h[part])
        #         leaf_value[part] = self.learning_rate * min_val
        #     elif self.solver_type == 'linear_constant':
        #         min_val = -1 * np.sum(g[part])/np.sum(c[part])
        #         leaf_value[part] = self.learning_rate * min_val
        #     else:
        #         raise RuntimeError('Incorrect solver_type')

        # SWIG optimizer
        # optimize over all smaller partition sizes
        num_partitions = int(rng.choice(range(self.min_partition_size, self.max_partition_size)))
        num_workers = multiprocessing.cpu_count() - 1
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(tasks, results) for i in range(num_workers)]
                
        for worker in workers:
            worker.start()

        for splits in range(num_partitions, 2, -1):
            tasks.put(OptimizerTask(self.N, splits, g, h))

        for i in range(num_workers):
            tasks.put(EndTask())

        tasks.join()
                      
        allResults = list()
        while not results.empty():
            result = results.get(block=True)
            allResults.append(result)
            
        x = T.dmatrix('x')
        loss = theano.function([x], self.loss_without_regularization(x))
        min_loss = float('inf')
        
        for s,w in allResults:
            leaf_value = np.zeros((self.N, 1))
            for subset in s:
                part = list(subset)
                min_val = -1 * np.sum(g[part])/np.sum(h[part])
                leaf_value[part] = self.learning_rate * min_val
            implied_tree = self.imply_tree(leaf_value)
            loss_new = loss(theano.function([], self.predict())() + theano.function([], implied_tree.predict(self.X))())
            if loss_new < min_loss:
                min_loss = loss_new
                subsets = s

        self.partitions.append(subsets)

        leaf_value = np.zeros((self.N, 1))
        for subset in subsets:
            part = list(subset)
            min_val = -1 * np.sum(g[part])/np.sum(h[part])
            leaf_value[part] = self.learning_rate * min_val

        print('{!r}'.format([(round(val,4), np.sum(leaf_value==val)) for val in np.unique(leaf_value)]))

        # Set leaf_value
        self.set_next_leaf_value(leaf_value)

        # Calculate implied_tree
        implied_tree = self.imply_tree(leaf_value)

        # Set implied_tree
        self.set_next_classifier(implied_tree)

        self.curr_classifier += 1

    def generate_coefficients(self, constantTerm=False):
        x = T.dvector('x')
        loss = self.loss_without_regularization(T.shape_padaxis(x, 1))

        grads = T.grad(loss, x)
        hess = T.hessian(loss, x)

        G = theano.function([x], grads)
        H = theano.function([x], hess)

        y_hat0 = theano.function([], self.predict())().squeeze()
        g = G(y_hat0)
        h = np.diag(H(y_hat0))

        c = None
        if constantTerm and not self.solver_type == 'linear_hessian':
            c = theano.function([], self._mse_coordinatewise(self.predict()))().squeeze()
            return (g, h, c)

        return (g, h, c)        
        
    def generate_coefficients_old(self, constantTerm=False):
        ''' Generate gradient, hessian sequences for offgraph
            optimizaition, return types are np.arrays
        '''
        x = T.dvector('x')
        loss = self.loss_without_regularization(T.shape_padaxis(x, 1))


        grads = T.grad(loss, x)
        hess = T.hessian(loss, x)

        G = theano.function([x], grads)
        H = theano.function([x], hess)

        # Test - random y_hat, random increment f_T
        y_hat0 = rng.uniform(low=0., high=1., size=(self.N,))
        f_t = rng.uniform(low=0., high=.01, size=(self.N,))
        loss0 = theano.function([x], loss)(y_hat0 + f_t)        
        loss0_approx0 = theano.function([x], loss)(y_hat0) + \
                        np.dot(G(y_hat0), f_t) + \
                        0.5 * np.dot(f_t.T, np.dot(H(y_hat0), f_t))
        loss0_approx1 = theano.function([x], loss)(y_hat0) + \
                        np.dot(G(y_hat0), f_t) + \
                        0.5 * np.dot(np.dot(f_t.T, H(y_hat0)), f_t)
        assert np.isclose(loss0, loss0_approx0, rtol=0.01)
        assert np.isclose(loss0, loss0_approx1, rtol=0.01)

        # Test
        y_hat0 = rng.uniform(low=0., high=1., size=(self.N,))
        y0 = self.y.get_value()
        f_t0 = y0 - y_hat0
        g = G(y_hat0)
        h = np.dot(f_t0.T, H(y_hat0))
        y_tilde = y_hat0 - (0.5 * -g*g/h)
        assert np.isclose(y_tilde, y0).all()

        # Test
        quadratic_term0 = 0.5 * np.dot(f_t.T, np.dot(H(y_hat0), f_t))
        quadratic_term1 = 0.5 * np.dot(np.dot(f_t.T, H(y_hat0)), f_t)
        assert np.isclose(quadratic_term0, quadratic_term1)

        # Test - y_hat = prediction, f_t = y - y_hat
        y_hat0 = theano.function([], self.predict())().squeeze()
        f_t = self.y.get_value() - y_hat0
        loss0 = theano.function([x], loss)(y_hat0 + f_t)        
        loss0_approx = theano.function([x], loss)(y_hat0) + \
                      np.dot(G(y_hat0), f_t) + \
                      0.5 * np.dot(np.dot(f_t.T, H(y_hat0)), f_t)
        assert np.isclose(loss0, loss0_approx, rtol=0.01)

        y_hat = theano.function([], self.predict())().squeeze()
        f_t = self.y.get_value() - y_hat0
        g = G(y_hat0)
        h = np.dot(f_t.T, H(y_hat0))                

        if constantTerm:
            c = theano.function([], self._mse_coordinatewise(self.predict()))().squeeze()
            return (g, h, c)

        return (g, h)        

    def loss(self, y_hat):
        return self._mse(y_hat) + T.sum(self.regularization)

    def loss_without_regularization(self, y_hat):
        return self._mse(y_hat)

    def _mse(self, y_hat):
        # XXX
        # return T.sum(self._mse_coordinatewise(y_hat))
        return T.sqrt(T.sum(self._mse_coordinatewise(y_hat)))

    def _mse_coordinatewise(self, y_hat):
        return (T.shape_padaxis(self.y, 1) - y_hat)**2

    def _regularizer(self, leaf_values):
        size_reg = self.gamma * partition_size
        coeff_reg = 0.5 * self.eta * np.sum(leaf_values**2)
        return size_req + coeff_reg

    def quadratic_solution_scalar(self, g, h, c):
        a,b = 0.5*h, g
        s1 = -b
        s2 = np.sqrt(b**2 - 4*a*c)
        r1 = (s1 + s2) / (2*a)
        r2 = (s1 - s2) / (2*a)

        return r1, r2
    
    def quadratic_solution( self, g, h, c):
        a,b = 0.5*h, g
        s1 = -b
        s2 = np.sqrt(b**2 - 4*a*c)
        r1 = (s1 + s2) / (2*a)
        r2 = (s1 - s2) / (2*a)

        return (r1.reshape(-1,1), r2.reshape(-1, 1))

class DirectImpliedTree(object):
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y

    def predict(self, X):
        return T.as_tensor(self.y.astype(theano.config.floatX))

class OLSImpliedTree(LinearRegression):
    def __init__(self, X=None, y=None):
        super(OLSImpliedTree, self).__init__(fit_intercept=True)
        self.X = X
        self.y = y
        self.fit(X, y)
        
    def predict(self, X):
        y_hat0 = super(OLSImpliedTree, self).predict(X.get_value())
        y_hat = T.as_tensor(y_hat0.astype(theano.config.floatX))
        return y_hat

class ClassifierImpliedTree(object):
    _classifierClz = None

    def __init__(self, X=None, y=None):
        self._classifier = _classifierClz()
        self.X = X
        unique_vals = np.sort(np.unique(y))
        self.val_to_class = dict(zip(unique_vals, range(len(unique_vals))))
        self.class_to_val = {v:k for k,v in self.val_to_class.items()}
        self.y = np.array([self.val_to_class[x[0]] for x in y])
        self._classifier.fit(self.X, self.y)

class LDAImpliedTree(LinearDiscriminantAnalysis):
    def __init__(self, X=None, y=None):
        super(LDAImpliedTree, self).__init__()
        self.X = X
        unique_vals = np.sort(np.unique(y))
        self.val_to_class = dict(zip(unique_vals, range(len(unique_vals))))
        self.class_to_val = {v:k for k,v in self.val_to_class.items()}
        self.y = np.array([self.val_to_class[x[0]] for x in y])
        super(LDAImpliedTree, self).fit(self.X, self.y)

    def predict(self, X):
        y_hat0 = super(LDAImpliedTree, self).predict(X.get_value())
        y_hat = T.as_tensor(np.array([self.class_to_val[x] for x in y_hat0]).reshape(-1, 1).astype(theano.config.floatX))
        return y_hat

class SVMImpliedTree(SVC):
    def __init__(self, X=None, y=None):
        super(SVMImpliedTree, self).__init__()
        self.X = X
        unique_vals = np.sort(np.unique(y))
        self.val_to_class = dict(zip(unique_vals, range(len(unique_vals))))
        self.class_to_val = {v:k for k,v in self.val_to_class.items()}
        self.y = np.array([self.val_to_class[x[0]] for x in y])
        super(SVMImpliedTree, self).fit(self.X, self.y)

    def predict(self, X):
        y_hat0 = super(SVMImpliedTree, self).predict(X.get_value())
        y_hat = T.as_tensor(np.array([self.class_to_val[x] for x in y_hat0]).reshape(-1, 1).astype(theano.config.floatX))
        return y_hat
        

class DecisionTreeImpliedTree(DecisionTreeClassifier):
    def __init__(self, X=None, y=None):
        super(DecisionTreeImpliedTree, self).__init__(max_depth=2)
        self.X = X
        unique_vals = np.sort(np.unique(y))
        self.val_to_class = dict(zip(unique_vals, range(len(unique_vals))))
        self.class_to_val = {v:k for k,v in self.val_to_class.items()}
        self.y = np.array([self.val_to_class[x[0]] for x in y])
        super(DecisionTreeClassifier, self).fit(self.X, self.y)

    def predict(self, X):
        y_hat0 = super(DecisionTreeImpliedTree, self).predict(X.get_value())
        y_hat = T.as_tensor(np.array([self.class_to_val[x] for x in y_hat0]).reshape(-1, 1).astype(theano.config.floatX))
        return y_hat

class Task(object):
    def __init__(self, a, b, c, solver_type, partition, partition_type='full'):
        self.partition = partition
        self.solver_type = solver_type
        self.task = partial(self._task, a, b, c)
        self.partition_type = partition_type # {'full', 'endpoints'}

    def __call__(self):
        return self.task(self.partition)

    def _task(self, a, b, c, partitions, report_each=1000):
        max_sum = float('-inf')
        arg_max = -1
        for ind,part in enumerate(partitions):
            val = 0
            part_vertex = [0] * len(part)
            for part_ind, p in enumerate(part):
                inds = p
                if self.partition_type == 'endpoints':
                    inds = range(p[0], p[1])
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
    def __init__(self,
                 a,
                 b,
                 c=None,
                 num_workers=None,
                 solver_type='linear_hessian',
                 use_monotonic_partitions=True,
                 shortest_path_solver=False):


        self.a = a if c is not None else abs(a)
        self.b = b
        self.c = c if c is not None else np.zeros(a.shape)
        self.n = len(a)
        self.num_workers = num_workers or multiprocessing.cpu_count() - 1
        self.solver_type = solver_type
        self.use_monotonic_partitions = use_monotonic_partitions
        self.partition_type = 'endpoints' if self.use_monotonic_partitions else 'full'
        self.shortest_path_solver = shortest_path_solver
        
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
        if self.shortest_path_solver:
            self.run_shortest_path_solver(num_partitions)
        else:
            self.run_brute_force(num_partitions)

    @functools.lru_cache(4096)
    def _calc_weight(self, i, j):
        return np.sum(self.a[i:j])**2/np.sum(self.b[i:j])

    def run_shortest_path_solver(self, num_partitions):
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
            workers = [Worker(tasks, results) for i in range(self.num_workers)]

            for worker in workers:
                worker.start()

            for i,slice in enumerate(self.slices):
                tasks.put(Task(self.a,
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
            task = Task(self.a,
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
            partitions = PartitionTreeOptimizer._monotonic_partitions(self.n, num_partitions)
        else:
            partitions = PartitionTreeOptimizer._knuth_partitions(self.INT_LIST, num_partitions)
        
        # Have to consume; can't split work on generator
        partitions = list(partitions)
        num_partitions = len(partitions)

        stride = max(int(num_partitions/self.num_workers), 1)
        bin_ends = list(range(0, num_partitions, stride))
        bin_ends = bin_ends + [num_partitions] if num_partitions/self.num_workers else bin_ends
        islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))
        
        rng.shuffle(partitions)
        slices = [list(islice(partitions, *ind)) for ind in islice_on]
        self.slices = slices

    @staticmethod
    def _monotonic_partitions(n, m):
        ''' Returns endpoints of all monotonic
            partitions
        '''
        combs = combinations(range(n-1), m-1)
        parts = list()
        for comb in combs:
            yield [(l+1, r+1) for l,r in zip((-1,)+comb, comb+(n-1,))]
    
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

    @staticmethod
    def _Mon_n_k(n, k):
        return comb(n-1, k-1, exact=True)

class EndTask(object):
    pass

class OptimizerTask(object):
    def __init__(self, N, num_partitions, g, h):
        g_c = proto.FArray()
        h_c = proto.FArray()        
        g_c = g
        h_c = h
        self.task = partial(self._task, N, num_partitions, g_c, h_c)

    def __call__(self):
        return self.task()
        
    @staticmethod
    def _task(N, num_partitions, g, h):
        s, w = proto.optimize_one(N, num_partitions, g, h)
        return s, w

class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            task = self.task_queue.get()
            if isinstance(task, EndTask):
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

    

# Test
if __name__ == '__main__':
    num_steps = 20
    num_classifiers = num_steps
    min_partitions = 50
    max_partitions = 51

    clf = GradientBoostingPartitionClassifier(X_train,
                                              y_train,
                                              min_partition_size=min_partitions,
                                              max_partition_size=max_partitions,
                                              gamma=0.,
                                              eta=0.,
                                              num_classifiers=num_classifiers,
                                              use_constant_term=False,
                                              solver_type='linear_hessian',
                                              learning_rate=.75,
                                              distill_method='DecisionTree',
                                              use_monotonic_partitions=False,
                                              shortest_path_solver=True
                                              )

    clf.fit(num_steps)

    # Vanilla regression model
    from sklearn.linear_model import LinearRegression
    from catboost import CatBoostClassifier, Pool
    X0 = clf.X.get_value()
    y0 = clf.y.get_value()
    reg = LinearRegression(fit_intercept=True).fit(X0, y0)
    logreg = LogisticRegression().fit(X0, y0)
    clf_cb = CatBoostClassifier(iterations=100,
                                depth=2,
                                learning_rate=1,
                                loss_function='CrossEntropy',
                                verbose=False)
    
    clf_cb.fit(X0, y0)
    
    x = T.dmatrix('x')
    _loss = theano.function([x], clf.loss_without_regularization(x))
    
    y_hat_clf = theano.function([], clf.predict())()
    y_hat_ols = reg.predict(X0).reshape(-1,1)
    y_hat_lr = logreg.predict(X0).reshape(-1,1)
    y_hat_cb = clf_cb.predict(X0).reshape(-1,1)
    
    y_hat_clf = (y_hat_clf > .5).astype(int)
    y_hat_ols = (y_hat_ols > .5).astype(int)
    
    print('IS _loss_clf: {:4.6f}'.format(_loss(y_hat_clf)))
    print('IS _loss_ols: {:4.6f}'.format(_loss(y_hat_ols)))
    print('IS _loss_lr:  {:4.6f}'.format(_loss(y_hat_lr)))
    print('IS _loss_cb:  {:4.6f}'.format(_loss(y_hat_cb)))
    print()
    
    # Out-of-sample predictions
    X0 = X_test
    y0 = y_test.reshape(-1,1)
    
    y_hat_clf = theano.function([], clf.predict_from_input(X0))()
    y_hat_ols = reg.predict(X0).reshape(-1,1)
    y_hat_lr = logreg.predict(X0).reshape(-1,1)
    y_hat_cb = clf_cb.predict(X0).reshape(-1,1)

    y_hat_clf = (y_hat_clf > .5).astype(int)
    y_hat_ols = (y_hat_ols > .5).astype(int)
                        
    def _loss(y_hat):
        return np.sum((y0 - y_hat)**2)
    
    print('OOS _loss_clf: {:4.6f}'.format(_loss(y_hat_clf)))
    print('OOS _loss_ols: {:4.6f}'.format(_loss(y_hat_ols)))
    print('OOS _loss_lr: {:4.6f}'.format(_loss(y_hat_lr)))
    print('OOS _loss_cb: {:4.6f}'.format(_loss(y_hat_cb)))
    print()
    
    print('OOS _accuracy_clf: {:1.4f}'.format(metrics.accuracy_score(y_hat_clf, y0)))
    print('OOS _accuracy_ols: {:1.4f}'.format(metrics.accuracy_score(y_hat_ols, y0)))
    print('OOS _accuracy_lr: {:1.4f}'.format(metrics.accuracy_score(y_hat_lr, y0)))
    print('OOS _accuracy_cb: {:1.4f}'.format(metrics.accuracy_score(y_hat_cb, y0)))
    
    target_names = ['0', '1']
    conf = plot_confusion(metrics.confusion_matrix(y_hat_clf, y0), target_names)
    plt.show()
