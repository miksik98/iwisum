from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils import binary2neg_boolean
import numpy as np

SEED = 1


def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    cov = EllipticEnvelope(contamination=outliers_fraction).fit(data)
    result = cov.predict(data)
    return binary2neg_boolean(result)


def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    ocsvm = svm.OneClassSVM(nu=outliers_fraction).fit_predict(data)
    return binary2neg_boolean(ocsvm)


def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    iforest = IsolationForest(contamination=outliers_fraction).fit_predict(data)
    return binary2neg_boolean(iforest)


def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    lof = LocalOutlierFactor(contamination=outliers_fraction, n_neighbors=500).fit_predict(data)
    return binary2neg_boolean(lof)
