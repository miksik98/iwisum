import numpy as np
from ex2.solution2 import detect as detect2
from ex3.solution3 import detect as detect3
from utils import report_results_2d

# load data
train_data = np.genfromtxt("ex3_train_data.csv")
test_data = np.genfromtxt("ex3_test_data.csv")
test_labels = np.genfromtxt("ex3_test_labels.csv")

# make predictions
predictions_cov = detect2(train_data,
                          test_data)

predictions_ocsv = detect3(train_data,
                           test_data)

report_results_2d(test_data, test_labels, {
    "Covariance-Mahalanobis": predictions_cov,
    "One-Class SVM": predictions_ocsv
})
