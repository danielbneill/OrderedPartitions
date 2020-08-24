import logging
import multiprocessing
import numpy as np
import pandas as pd
import heapq
from itertools import combinations
from functools import partial
from scipy.special import comb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from functools import partial, lru_cache
from itertools import chain, islice
import functools

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import classifier
import solver
from optimalsplitboost import OptimalSplitGradientBoostingClassifier


import proto

import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

###################
## Generate Data ##
###################
SEED = 144
NUM_SAMPLES = 150
TEST_SIZE = 0.2
rng = np.random.RandomState(SEED)

X,y = make_classification(random_state=SEED, n_samples=NUM_SAMPLES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
#######################
## END Generate Data ##
#######################

def oos_summary(clf):
    # Vanilla regression model
    from sklearn.linear_model import LinearRegression
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
    print('OOS _loss_lr:  {:4.6f}'.format(_loss(y_hat_lr)))
    print('OOS _loss_cb:  {:4.6f}'.format(_loss(y_hat_cb)))
    print()
    
    print('OOS _accuracy_clf: {:1.4f}'.format(metrics.accuracy_score(y_hat_clf, y0)))
    print('OOS _accuracy_ols: {:1.4f}'.format(metrics.accuracy_score(y_hat_ols, y0)))
    print('OOS _accuracy_lr:  {:1.4f}'.format(metrics.accuracy_score(y_hat_lr, y0)))
    print('OOS _accuracy_cb:  {:1.4f}'.format(metrics.accuracy_score(y_hat_cb, y0)))
    
    # target_names = ['0', '1']
    # conf = plot_confusion(metrics.confusion_matrix(y_hat_clf, y0), target_names)
    # plt.show()

XXX

# Test
if __name__ == '__main__':
    num_steps = 50
    num_classifiers = num_steps
    min_partitions = 10
    max_partitions = 26

    import sklearn.tree
    distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier)

    clf = OptimalSplitGradientBoostingClassifier(X_train,
                                                 y_train,
                                                 min_partition_size=min_partitions,
                                                 max_partition_size=max_partitions,
                                                 gamma=0.,
                                                 eta=0.,
                                                 num_classifiers=num_classifiers,
                                                 use_constant_term=False,
                                                 solver_type='linear_hessian',
                                                 learning_rate=.75,
                                                 distiller=distiller,
                                                 use_monotonic_partitions=False,
                                                 shortest_path_solver=True
                                                 )

    clf.fit(num_steps)

def oos_summary(clf):
    # Vanilla regression model
    from sklearn.linear_model import LinearRegression
    X0 = clf.X.get_value()
    y0 = clf.y.get_value()
    reg = LinearRegression(fit_intercept=True).fit(X0, y0)
    logreg = LogisticRegression().fit(X0, y0)
    clf_cb = CatBoostClassifier(iterations=100,
                                depth=6,
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
    
    # target_names = ['0', '1']
    # conf = plot_confusion(metrics.confusion_matrix(y_hat_clf, y0), target_names)
    # plt.show()
