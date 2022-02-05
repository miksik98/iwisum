import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    covs = MinCovDet().fit(train_data)
    mahalanobis = covs.mahalanobis(train_data)
    maximum = np.max(mahalanobis)

    result = []
    for test in covs.mahalanobis(test_data):
        if test <= maximum:
            result.append(0)
        else:
            result.append(1)
    return result


