import numpy as np
import solverSWIG_DP
import solverSWIG_DP_Multi

g = np.array([0.0212651 , -0.20654906, -0.20654906, -0.20654906, -0.20654906,
      0.0212651 , -0.20654906,  0.0212651 , -0.20654906,  0.0212651])
h = np.array([0.22771114, 0.21809504, 0.21809504, 0.21809504, 0.21809504,
      0.22771114, 0.21809504, 0.22771114, 0.21809504, 0.22771114])
num_partitions = 4

results_DP = solverSWIG_DP.OptimizerSWIG(num_partitions, g, h)()
results_DP_Multi = solverSWIG_DP_Multi.OptimizerSWIG(num_partitions, g, h)()
