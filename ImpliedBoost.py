import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import classifier
import solver
import utils
from optimalsplitboost import OptimalSplitGradientBoostingClassifier

###################
## Generate Data ##
###################
SEED = 249
NUM_SAMPLES = 450
TEST_SIZE = 0.25
rng = np.random.RandomState(SEED)

X,y = make_classification(random_state=SEED, n_samples=NUM_SAMPLES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
#######################
## END Generate Data ##
#######################

if __name__ == '__main__':
    num_steps = 20
    num_classifiers = num_steps
    min_partitions = 100
    max_partitions = 101

    import sklearn.tree
    distiller = classifier.classifierFactory(sklearn.tree.DecisionTreeClassifier)

    clf = OptimalSplitGradientBoostingClassifier(X_train,
                                                 y_train,
                                                 min_partition_size=min_partitions,
                                                 max_partition_size=max_partitions,
                                                 gamma=0.01,
                                                 eta=0.01,
                                                 num_classifiers=num_classifiers,
                                                 use_constant_term=False,
                                                 solver_type='linear_hessian',
                                                 learning_rate=.80,
                                                 distiller=distiller,
                                                 use_monotonic_partitions=False,
                                                 shortest_path_solver=True
                                                 )

    clf.fit(num_steps)
    utils.oos_summary(clf, X_test, y_test)
