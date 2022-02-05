import numpy as np
from ex4.solution4 import detect_cov, detect_ocsvm, detect_iforest, detect_lof
from utils import report_results_2d

# load data
train_data = np.genfromtxt("ex4_train_data.csv")
train_labels = np.genfromtxt("ex4_train_labels.csv")

outliers_fraction = np.count_nonzero(train_labels) / len(train_labels)

# make predictions
predictions_cov = detect_cov(train_data,
                             outliers_fraction)

predictions_ocsv = detect_ocsvm(train_data,
                                outliers_fraction)

predictions_iforest = detect_iforest(train_data,
                                     outliers_fraction)

predictions_lof = detect_lof(train_data,
                             outliers_fraction)

# show results
# on train data!
report_results_2d(train_data, train_labels, {
    "Covariance-Mahalanobis": predictions_cov,
    "One-Class SVM": predictions_ocsv,
    "Isolation Forest": predictions_iforest,
    "Local Outlier Factor": predictions_lof
})
