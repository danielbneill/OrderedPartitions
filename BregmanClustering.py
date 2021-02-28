from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, load_breast_cancer

import numpy as np
import classifier
import utils
from BregmanQuantizer import BregmanDivergenceQuantizer

USE_SIMULATED_DATA = True # True
USE_01_LOSS = False # False
# XXX
# TEST_SIZE = 0.10 # .10
TEST_SIZE = 0.5

##########################
## Generate Random Data ##
##########################
if (USE_SIMULATED_DATA):
    SEED = 254 # 254
    NUM_SAMPLES = 100 # 1000
    NUM_FEATURES = 20 # 20
    rng = np.random.RandomState(SEED)
    
    X,y = make_classification(random_state=SEED, n_samples=NUM_SAMPLES, n_features=NUM_FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
##############################
## END Generate Random Data ##
##############################
else:
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=TEST_SIZE)
if USE_01_LOSS:
    X_train = 2*X_train - 1
    X_test = 2*X_test - 1
    y_train = 2*y_train - 1
    y_test = 2*y_test - 1
#############################
## Generate Empirical Data ##
#############################
if __name__ == '__main__':
    min_partitions = 1 # 1
    max_partitions = 11 # 21


    import sklearn.tree
    clstr = BregmanDivergenceQuantizer(X_train,
                                       min_partitions,
                                       max_partitions,
                                       gamma=0.25,
                                       eta=0.5,
                                       use_constant_term=False,
                                       solver_type='linear_hessian',
                                       learning_rate=1.0,
                                       )
