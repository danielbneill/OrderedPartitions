import heapq
import numpy as np
import sklearn.base
import sklearn.tree
import theano
import theano.tensor as T

import classifier
import solverSWIG

SEED = 515
rng = np.random.RandomState(SEED)

class OptimalSplitGradientBoostingClassifier(object):
    def __init__(self,
                 X,
                 y,
                 min_partition_size,
                 max_partition_size,
                 gamma=0.1,
                 eta=0.1,
                 num_classifiers=100,
                 use_constant_term=False,
                 solver_type='linear_hessian',
                 learning_rate=0.1,
                 distiller=classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier),
                 use_monotonic_partitions=True,
                 shortest_path_solver=False,
                 ):
        ############
        ## Inputs ##
        ############
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

        # solver_type is one of
        # ('quadratic, 'linear_hessian', 'linear_constant')
        self.use_constant_term = use_constant_term
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        self.distiller = distiller
        self.use_monotonic_partitions = use_monotonic_partitions
        self.shortest_path_solver = shortest_path_solver
        ################
        ## END Inputs ##
        ################

        # optimal partition at each step, not part of any gradient, not a tensor
        self.partitions = list()
        # distinct leaf values at each step, not a tesnor
        self.distinct_leaf_values = np.zeros((self.num_classifiers + 1,
                                             self.N))
        # regularization penalty at each step
        self.regularization = theano.shared(name='regularization',
                                            value=np.zeros((self.num_classifiers + 1,
                                                            1)).astype(theano.config.floatX))
        # leaf values at each step
        self.leaf_values = theano.shared(name='leaf_values',
                                         value=np.zeros((self.num_classifiers + 1,
                                                         self.N,
                                                         1)).astype(theano.config.floatX))
        # classifier at each step
        self.implied_trees = [classifier.LeafOnlyTree(self.X.get_value(), 0.5 * np.ones((self.N, 1)))] * \
                             (self.num_classifiers + 1)

        # Set initial set of leaf_values to be random
        if isinstance(self.distiller.args[0], sklearn.base.ClassifierMixin):
            # Cannot have number of unique classes == number of samples, so
            # we must restrict sampling to create fewer classes
            leaf_value = rng.choice(rng.uniform(low=0.0, high=1.0, size=(self.min_partition_size,)),
                                    size=(self.N, 1)).astype(theano.config.floatX)
        elif isinstance(self.distiller.args[0], sklearn.base.RegressorMixin):
            leaf_value = np.asarray(rng.uniform(low=0.0, high=1.0, size=(self.N, 1))).astype(theano.config.floatX)
        else:
            raise RuntimeError('Cannot determine whether distiller is of classifier or regressor type.')
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

    def imply_tree(self, leaf_values, **impliedSolverKwargs):
        X0 = self.X.get_value()
        y0 = leaf_values
        return self.distiller(X0, y0, **impliedSolverKwargs)
            
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

        # Initial leaf_values for initial loss calculation
        leaf_values = self.leaf_values.get_value()[0,:]
        print('STEP {}: LOSS: {:4.6f}'.format(0,
                                              theano.function([],
                                                              self.loss(
                                                                  self.predict(),
                                                                  len(np.unique(leaf_values)),
                                                                  leaf_values))()))
        # Iterative boosting
        for i in range(1,num_steps):
            self.fit_step()
            leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
            if (i==24):
                import pdb
                pdb.set_trace()
            print('STEP {}: LOSS: {:4.6f}'.format(i,
                                                  theano.function([],
                                                                  self.loss(
                                                                      self.predict(),
                                                                      len(np.unique(leaf_values)),
                                                                      leaf_values))()))
            # Summary statistics mid-training
        print('Training finished')

    def find_best_optimal_split(self, g, h, num_partitions):
        ''' Method: results contains the optimal partitions for all partition sizes in
            [1, num_partitions]. We take each, form the optimal_split_tree from an
            inductive fitting of the classifier, then look at the loss of the new
            predictor (current predictor + optimal_split_tree predictor). The minimal
            loss wins.
        '''

        results = solverSWIG.OptimizerSWIG(num_partitions, g, h)()

        npart = T.scalar('npart')
        lv = T.dmatrix('lv')
        x = T.dmatrix('x')
        loss = theano.function([x,npart,lv], self.loss(x, npart, lv))
        
        leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
        loss_heap = []

        results = (results,)

        for rind, result in enumerate(results):
            leaf_values = np.zeros((self.N, 1))
            subsets = result[0]
            for subset in subsets:
                s = list(subset)
                min_val = -1 * np.sum(g[s])/np.sum(h[s])
                # XXX
                # impliedSolverKwargs = dict(max_depth=max([int(len(s)/2), 2]))
                # impliedSolverKwargs = dict(max_depth=int(np.log2(num_partitions)))
                impliedSolverKwargs = dict(max_depth=None)
                leaf_values[s] = self.learning_rate * min_val
            optimal_split_tree = self.imply_tree(leaf_values, **impliedSolverKwargs)
            loss_new = loss(theano.function([], self.predict())() +
                            theano.function([], optimal_split_tree.predict(self.X))(),
                            len(subsets),
                            leaf_values)
            heapq.heappush(loss_heap, (loss_new.item(0), rind, leaf_values))

        best_loss, best_rind, best_leaf_values = heapq.heappop(loss_heap)

        # XXX
        # solverKwargs = dict(max_depth=int(len(np.unique(best_leaf_values))))
        solverKwargs = dict(max_depth=int(np.log2(num_partitions)))
        solverKwargs = dict(max_depth=None)        
        optimal_split_tree = self.imply_tree(best_leaf_values, **solverKwargs)

        # ===============================
        # == If DecisionTreeClassifier ==
        # ===============================
        # from sklearn import tree
        # import graphviz
        # dot_data = tree.export_graphviz(tr, out_file=None)
        # graph = graphviz.Source(dot_data)
        # graph.render('Boosting')

        self.partitions.append(results[best_rind][0])

        implied_values = theano.function([], optimal_split_tree.predict(self.X))()

        print('leaf_values:    {!r}'.format([(round(val,4), np.sum(best_leaf_values==val))
                                            for val in np.unique(best_leaf_values)]))
        print('implied_values: {!r}'.format([(round(val,4), np.sum(implied_values==val))
                                            for val in np.unique(implied_values)]))

        return best_leaf_values
        
    def fit_step(self):
        g, h, c = self.generate_coefficients(constantTerm=self.use_constant_term)

        # SWIG optimizer, task-based C++ distribution
        num_partitions = int(rng.choice(range(self.min_partition_size, self.max_partition_size)))

        # Find best optimal split
        best_leaf_values = self.find_best_optimal_split(g, h, num_partitions)

        # Set leaf_value, return leaf values used to generate
        self.set_next_leaf_value(best_leaf_values)

        # Calculate optimal_split_tree
        # XXX
        # impliedSolverKwargs = dict(max_depth=len(np.unique(best_leaf_values))/2)
        impliedSolverKwargs = dict(max_depth=int(np.log2(num_partitions)))
        impliedSolverKwargs = dict(max_depth=None)        
        optimal_split_tree = self.imply_tree(best_leaf_values, **impliedSolverKwargs)

        # Set implied_tree
        self.set_next_classifier(optimal_split_tree)

        self.curr_classifier += 1

    def generate_coefficients(self, constantTerm=False):

        x = T.dvector('x')
        leaf_values = self.leaf_values.get_value()[-1+self.curr_classifier,:]
        loss = self.loss(T.shape_padaxis(x, 1), len(np.unique(leaf_values)), leaf_values)

        grads = T.grad(loss, x)
        hess = T.hessian(loss, x)

        G = theano.function([x], grads)
        H = theano.function([x], hess)

        y_hat0 = theano.function([], self.predict())().squeeze()
        g = G(y_hat0)
        h = np.diag(H(y_hat0))
        h = h + np.array([self.gamma]*h.shape[0])

        c = None
        if constantTerm and not self.solver_type == 'linear_hessian':
            c = theano.function([], self._mse_coordinatewise(self.predict()))().squeeze()
            return (g, h, c)

        return (g, h, c)        
        
    def loss(self, y_hat, num_partitions, leaf_values):
        return self.loss_without_regularization(y_hat) + self.regularization_loss(num_partitions, leaf_values)

    def loss_without_regularization(self, y_hat):
        ''' Dependent on loss function '''
        return self.mse_loss_without_regularization(y_hat)

    def regularization_loss(self, num_partitions, leaf_values):
        ''' Independent of loss function '''
        size_reg = self.gamma * num_partitions
        coeff_reg = 0.5 * self.eta * T.sum(T.extra_ops.Unique(False,False,False)(leaf_values)**2)
        return size_reg + coeff_reg
    
    def mse_loss_without_regularization(self, y_hat):
        return self._mse(y_hat)

    def exp_loss_without_regularization(self, y_hat):
        return T.sum(T.exp(-y_hat * T.shape_padaxis(self.y, 1)))

    def cosh_loss_without_regularization(self, y_hat):
        return T.sum(T.log(T.cosh(y_hat - T.shape_padaxis(self.y, 1))))

    def hinge_loss_without_regularization(self, y_hat):
        return T.sum(T.abs_(y_hat - T.shape_padaxis(self.y, 1)))

    def logistic_loss_without_regularization(self, y_hat):
        return T.sum(T.log(1 + T.exp(-y_hat * T.shape_padaxis(self.y, 1))))

    def cross_entropy(self, y_hat):
        y0 = T.shape_padaxis(self.y, 1)
        # return T.sum(-(y0 * T.log(y_hat) + (1-y0)*T.log(1-y_hat)))
        return T.sum(T.nnet.binary_crossentropy(y_hat, y0))
    
    def _mse(self, y_hat):
        return T.sqrt(T.sum(self._mse_coordinatewise(y_hat)))

    def _mse_coordinatewise(self, y_hat):
        return (T.shape_padaxis(self.y, 1) - y_hat)**2

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
