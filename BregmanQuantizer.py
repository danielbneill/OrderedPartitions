import heapq
import numpy as np
import theano
import theano.tensor as T
import sklearn.base
import sklearn.tree
import multiprocessing

import solverSWIG_PG
import solverSWIG_DP


SEED = 515
rng = np.random.RandomState(SEED)

class BregmanDivergenceQuantizer(object):
    def __init__(self,
                 X,
                 min_partition_size,
                 max_partition_size,
                 gamma=0.1,
                 eta=0.1,
                 use_constant_term=False,
                 solver_type='linear_hessian',
                 learning_rate=0.1,
                 ):

        ############
        ## Inputs ##
        ############
        self.X = theano.shared(value=X.flatten(), name='X', borrow=True)
        self.N, self.num_features = X.shape
        self.L = self.N * self.num_features
        
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.gamma = gamma
        self.eta = eta
        self.num_classifiers = 1

        # solver_type is one of
        # ('quadratic, 'linear_hessian', 'linear_constant')
        self.use_constant_term = use_constant_term
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        ################
        ## END Inputs ##
        ################

        # leaf values at each step - override as this is unsupervised
        # self.leaf_values = theano.shared(name='leaf_values',
        #                                  value=np.zeros((self.L,
        #                                                  1)).astype(theano.config.floatX))
        self.leaf_values = theano.shared(name='leaf_values',
                                         value=np.mean(X)*np.ones((self.L,1)).astype(theano.config.floatX))
        # current number of distinct leaf_values
        self.num_partitions = 1 

        # optimal partition at each step, not part of any gradient, not a tensor
        self.partitions = list()
        # distinct leaf values at each step, not a tesnor
        self.distinct_leaf_values = np.zeros((self.num_classifiers + 1,
                                             self.L))
        # regularization penalty at each step
        self.regularization = theano.shared(name='regularization',
                                            value=np.zeros((self.num_classifiers + 1,
                                                            1)).astype(theano.config.floatX))

        # Set initial partition to be the size 1 partition (all leaf values the same)
        self.partitions.append(list(range(self.L)))

        # For testing
        self.srng = T.shared_randomstreams.RandomStreams(seed=SEED)
        self.curr_classifier = 1

        self.fit()

    def fit(self, num_steps=None):
        num_steps = num_steps or self.num_classifiers

        # Initial leaf_values for initial loss calculation
        leaf_values = theano.shared(name='leaf_values', value=np.zeros((self.L,
                                                                        1)).astype(theano.config.floatX))        

        print('STEP {}: LOSS: {:4.6f}'.format(0,
                                              theano.function([],
                                                              self.loss(
                                                                  self.predict()))()))
        self.fit_step()
        leaf_values = self.leaf_values.get_value()
        print('STEP {}: LOSS: {:4.6f}'.format(1,
                                              theano.function([],
                                                              self.loss(
                                                                  self.predict()))()))
        # Summary statistics mid-training
        print('Training finished')

    def fit_step(self):
        self.num_partitions = int(rng.choice(range(self.min_partition_size, self.max_partition_size)))

        g, h, c = self.generate_coefficients(constantTerm=False)        

        # Find best optimal split
        opt_leaf_values = self.find_best_optimal_split(g, h, self.num_partitions)

        self.inc_leaf_values(opt_leaf_values)

    def generate_coefficients(self, constantTerm=False):

        x = T.dvector('x')
        leaf_values = self.leaf_values.get_value().squeeze()
        loss = self.loss(x)

        grads = T.grad(loss, x)
        hess = T.hessian(loss, x)

        G = theano.function([x], grads)
        H = theano.function([x], hess)

        g = G(leaf_values)
        h = np.diag(H(leaf_values))
        # XXX
        # h = h + np.array([self.gamma]*h.shape[0])
        c = None
        
        return (g, h, c)        

    def set_next_classifier(self, classifier):
        i = self.curr_classifier
        self.implied_trees[i] = classifier

    def inc_leaf_values(self, inc_leaf_values):
        c = T.dmatrix('c')
        update=(self.leaf_values, self.leaf_values + c.astype(theano.config.floatX))
        f = theano.function([c], updates=[update])
        f(inc_leaf_values)

    def predict(self):
        return self.leaf_values.get_value().squeeze()

    def find_best_optimal_split(self, g, h, num_partitions):
        ''' Method: results contains the optimal partitions for all partition sizes in
            [1, num_partitions]. We take each, from the optimal_split_tree from an
            inductive fitting of the classifier, then look at the loss of the new
            predictor (current predictor + optimal_split_tree predictor). The minimal
            loss wins.
        '''

        results = (solverSWIG_DP.OptimizerSWIG(num_partitions, g, h)(),)

        for rind, result in enumerate(results):
            leaf_values = np.zeros((self.L, 1))
            subsets = result[0]
            for subset in subsets:
                s = list(subset)
                min_val = -1 * np.sum(g[s])/np.sum(h[s])
                # XXX
                # impliedSolverKwargs = dict(max_depth=max([int(len(s)/2), 2]))
                # impliedSolverKwargs = dict(max_depth=int(np.log2(num_partitions)))
                impliedSolverKwargs = dict(max_depth=None)
                leaf_values[s] = self.learning_rate * min_val

        print('leaf_values:    {!r}'.format([(round(val,8), np.sum(leaf_values==val))
                                            for val in np.unique(leaf_values)]))
        return leaf_values

    def loss(self, X_q):
        # return self.loss_without_regularization(X_q) + self.regularization_loss(X_q)
        return self.loss_without_regularization(X_q)

    def loss_without_regularization(self, X_q):
        ''' Dependent on loss function '''
        return self.mse_loss_without_regularization(X_q)

    def regularization_loss(self, X_q):
        ''' Independent of loss function '''
        size_reg = self.gamma * self.num_partitions
        # XXX
        # Problem taking grad of Unique
        # coeff_reg = 0.5 * self.eta * T.sum(T.extra_ops.Unique(False,False,False)(X_q)**2)
        coeff_reg = 0.0
        return size_reg + coeff_reg
    
    def mse_loss_without_regularization(self, X_q):
        return self._mse(X_q)

    def exp_loss_without_regularization(self, X_q):
        return T.sum(T.exp(-X_q * T.shape_padaxis(self.X, 1)))

    def cosh_loss_without_regularization(self, X_q):
        return T.sum(T.log(T.cosh(X_q - T.shape_padaxis(self.X, 1))))

    def hinge_loss_without_regularization(self, X_q):
        return T.sum(T.abs_(X_q - T.shape_padaxis(self.X, 1)))

    def logistic_loss_without_regularization(self, X_q):
        return T.sum(T.log(1 + T.exp(-X_q * T.shape_padaxis(self.X, 1))))

    def cross_entropy(self, X_q):
        y0 = T.shape_padaxis(self.X, 1)
        # return T.sum(-(y0 * T.log(X_q) + (1-y0)*T.log(1-X_q)))
        return T.sum(T.nnet.binary_crossentropy(X_q, y0))
    
    def _mse(self, X_q):
        return T.sum(self._mse_coordinatewise(X_q))

    def _mse_coordinatewise(self, X_q):
        return (T.shape_padaxis(self.X, 1) - T.shape_padaxis(X_q,1))**2

