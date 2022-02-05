import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.utils import shuffle

# ---------------------------------------------
# settings
# ---------------------------------------------

train_cnt = 100000
test_nominal_cnt = 9000
test_anomaly_cnt = 1000

mu = 0
sigma = 0.3
mu_shift = 2

seed = 3

# ---------------------------------------------
# generate
# ---------------------------------------------

np.random.seed(seed)

# train
# all nominal
train_data = sigma * np.random.randn(train_cnt // 2, 2) + mu
train_data = np.r_[train_data + mu_shift, train_data - mu_shift]

# test
# nominal + anomaly
test_data = sigma * np.random.randn(test_nominal_cnt // 2, 2) + mu
test_data = np.r_[test_data + mu_shift, test_data - mu_shift]
test_data = np.vstack((test_data,
                       np.random.uniform(low=-4, high=4, size=(test_anomaly_cnt, 2))))

test_labels = np.hstack((
    np.zeros(test_nominal_cnt),
    np.ones(test_anomaly_cnt)
))

test_data, test_labels = shuffle(test_data, test_labels,
                                 random_state=seed)

# ---------------------------------------------
# save
# ---------------------------------------------

np.savetxt("ex3_train_data.csv",
           train_data,
           fmt="%.4f")
np.savetxt("ex3_test_data.csv",
           test_data,
           fmt="%.4f")
np.savetxt("ex3_test_labels.csv",
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
