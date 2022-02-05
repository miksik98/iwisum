import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from os.path import join
from sklearn.utils import shuffle

# ---------------------------------------------
# settings
# ---------------------------------------------

train_cnt = 100000
test_nominal_cnt = 9000
test_anomaly_cnt = 1000

nominal_mu = 10
nominal_sigma = 2.5

anomaly_mu = 20
anomaly_sigma = 2.5

seed = 1

# ---------------------------------------------
# generate
# ---------------------------------------------

np.random.seed(seed)

# train
# all nominal
train_data = nominal_sigma * np.random.randn(train_cnt, 1) + nominal_mu

# test
# nominal + anomaly
test_data = np.vstack((
    nominal_sigma * np.random.randn(test_nominal_cnt, 1) + nominal_mu,
    anomaly_sigma * np.random.randn(test_anomaly_cnt, 1) + anomaly_mu
))
test_labels = np.hstack((
    np.zeros(test_nominal_cnt),
    np.ones(test_anomaly_cnt)
))

test_data, test_labels = shuffle(test_data, test_labels,
                                 random_state=seed)

# ---------------------------------------------
# save
# ---------------------------------------------

np.savetxt("ex1_train_data.csv",
           train_data,
           fmt="%.4f")
np.savetxt("ex1_test_data.csv",
           test_data,
           fmt="%.4f")
np.savetxt("ex1_test_labels.csv",
           test_labels,
           fmt="%d")

# ---------------------------------------------
# plot
# ---------------------------------------------

# plt.plot(train_data)
plt.hist(train_data,
         bins=100,
         density=True,
         histtype="step",
         label="train data")
plt.hist(test_data,
         bins=100,
         density=True,
         histtype="step",
         label="test data")
plt.legend()
plt.show()

print(test_labels)
