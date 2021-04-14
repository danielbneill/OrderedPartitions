import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
import sklearn.tree
import sklearn.svm
import sklearn.discriminant_analysis

import classifier
import utils
from optimalsplitboost import OptimalSplitGradientBoostingClassifier

USE_SIMULATED_DATA = True # True
USE_01_LOSS = False # False
TEST_SIZE = 0.20 # .10

##########################
## Generate Random Data ##
##########################
if (USE_SIMULATED_DATA):
    SEED = 254 # 254
    NUM_SAMPLES = 1000 # 1000, 10000
    NUM_FEATURES = 100 # 20, 500
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

    num_steps = 50
    
    distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier) # use classifier
    # print('USING LDA')
    # distiller = classifier.classifierFactory(sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
    # distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeRegressor)

    clfKwargs = { 'min_partition_size': 1,
                  'max_partition_size': 10, # 50
                  'row_sample_ratio':   0.75,
                  'col_sample_ratio':   1.00,
                  'gamma':              0.0025,
                  'eta':                0.05,
                  'num_classifiers':    num_steps,
                  'use_constant_term':  False,
                  'solver_type':        'linear_hessian',
                  'learning_rate':      0.65,
                  'distiller':          distiller
                  }
                  
    clf = OptimalSplitGradientBoostingClassifier( X_train,
                                                  y_train,
                                                  **clfKwargs
                                                 )

    clf.fit(num_steps)

    utils.oos_summary(clf, X_test, y_test)

