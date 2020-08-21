class OptimalSplitGradientBoostingClassifier(object):
    def __init__(self,
                 X,
                 y,
                 min_partition_size,
                 max_partition_size,
                 num_classifiers,
                 gamma=0.,
                 eta=0.,
                 learning_rate=0.1,
                 solver_type='linear_hessian',
                 distillation_method='OLS',
                 use_consecutive_paritions=True,
                 use_shortest_path_solver=False
                 ):
        ##################
        ## BEGIN Inputs ##
        ##################
        self.x = theano.shared(value=X, name='X', borrow=True)
        self.y = theano.shared(value=y, name='y', borrow=True)
        initial_X = self.X.get_value()
        self.N, self.num_features = initial_X.shape
        self.min_partition_size = min_partition_size
        self.max_partition_size = max_partition_size
        self.gamma = gamma
        self.eta = eta
        self.num_classifiers = num_classifiers
        self.curr_classifier = 0
        
        # Solver details
        # solver_type is one of
        # ('quadratic', 'linear_hessian', 'linear_constant')
        self.use_constant_term = False
        self.solver_type = solver_type
        self.learning_rate = learning_rate
        self.distill_method = distill_method
        self.use_monotonic_partitions = use_monotonic_partitions
        self.shortest_path_solver = shortest_path_solver
        ################
        ## END Inputs ##
        ################

        
        

        
                 
        
