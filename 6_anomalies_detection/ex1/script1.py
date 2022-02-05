import numpy as np
from ex1.solution1 import detect
from utils import print_results

# load data
train_data = np.genfromtxt("ex1_train_data.csv")
test_data = np.genfromtxt("ex1_test_data.csv")
test_labels = np.genfromtxt("ex1_test_labels.csv")

predictions = detect(train_data,
                     test_data)

print_results(test_labels, predictions)
