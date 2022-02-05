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

nominal_cov = [[1, .5],
               [.5, 1]]

anomaly_cov = [[1, 0],
               [0, 7]]

nominal_mu = [0, 0]
anomaly_mu = [0, 0]

seed = 2

# ---------------------------------------------
# generate
# ---------------------------------------------

np.random.seed(seed)

# train
# all nominal
train_data = np.dot(np.random.randn(train_cnt, 2),
                    nominal_cov) \
             + nominal_mu

# test
# nominal + anomaly
test_data = np.vstack((
    np.dot(np.random.randn(test_nominal_cnt, 2),
           nominal_cov)
    + nominal_mu,
    np.dot(np.random.randn(test_anomaly_cnt, 2),
           anomaly_cov)
    + anomaly_mu
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

np.savetxt("ex2_train_data.csv",
           train_data,
           fmt="%.4f")
np.savetxt("ex2_test_data.csv",
           test_data,
           fmt="%.4f")
np.savetxt("ex2_test_labels.csv",
           test_labels,
           fmt="%d")

# ---------------------------------------------
# plot
# ---------------------------------------------


fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True,
                        tight_layout=True)
axs[0].hist2d(train_data[:, 0],
              train_data[:, 1],
              bins=100,
              norm=matplotlib.colors.LogNorm())
axs[0].set_title("Train data")

axs[1].hist2d(test_data[:, 0],
              test_data[:, 1],
              bins=100,
              norm=matplotlib.colors.LogNorm())
axs[1].set_title("Test data")
plt.show()
