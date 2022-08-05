"""
The data structure like is the list of [Name-method, Model-size, Time-step, MSE]
"""
import numpy as np
from numpy import nan

from export_data import compare_auto_correlation_CNU, compare_auto_correlation_Household

# data, num_dataset = np.array(compare_auto_correlation_CNU(), dtype=object)

# data, num_dataset = compare_auto_correlation_Household()
data, num_dataset = compare_auto_correlation_CNU()
data = np.array(data, dtype=object)

# data = np.array([
#     ['VGG-16', 138, 31e3, 71],
#     ['MobNetV2-1', 3.4, 2 * 300, 71.7],
#     ['MobNetV2-1.4', 6.9, 2 * 585, 74.7],
# ], dtype=object)
