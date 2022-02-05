import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn.utils import shuffle

# ---------------------------------------------
# settings
# ---------------------------------------------

train_nominal_cnt = 9000
train_anomaly_cnt = 1000

mu = 0
sigma = 0.3
mu_shift = 1.5

seed = 3

# ---------------------------------------------
# generate
# ---------------------------------------------

np.random.seed(seed)

# train
# nominal + anomaly
#   nominal
X1 = sigma * np.random.randn(train_nominal_cnt // 2, 2) - mu_shift
X2 = sigma * np.random.randn(train_nominal_cnt // 2, 2) + mu_shift

#   add outliers
train_data = np.r_[np.r_[X1, X2],
                   np.random.uniform(low=-6, high=6, size=(train_anomaly_cnt, 2))]

# test
train_labels = np.hstack((
    np.zeros(train_nominal_cnt),
    np.ones(train_anomaly_cnt)
))

train_data, train_labels = shuffle(train_data, train_labels,
                                   random_state=seed)

# ---------------------------------------------
# save
# ---------------------------------------------

np.savetxt("ex4_train_data.csv",
           train_data,
           fmt="%.4f")
np.savetxt("ex4_train_labels.csv",
           train_labels,
           fmt="%d")

# ---------------------------------------------
# plot
# ---------------------------------------------
plt.hist2d(train_data[:, 0],
           train_data[:, 1],
           bins=100,
           norm=matplotlib.colors.LogNorm())
plt.title("Train data")
plt.show()
