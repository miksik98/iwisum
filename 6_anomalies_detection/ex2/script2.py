import numpy as np
from ex2.solution2 import detect
from utils import report_results_2d

# load data
train_data = np.genfromtxt("ex2_train_data.csv")
test_data = np.genfromtxt("ex2_test_data.csv")
test_labels = np.genfromtxt("ex2_test_labels.csv")

# make predictions
predictions = detect(train_data,
                     test_data)

# show results
report_results_2d(test_data, test_labels, {
    "Covariance-Mahalanobis": predictions
})
