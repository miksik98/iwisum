import numpy as np


def reconstruction_errors(inputs: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Calculate reconstruction errors.

    :param inputs: Numpy array of input images
    :param reconstructions: Numpy array of reconstructions
    :return: Numpy array (1D) of reconstruction errors for each pair of input and its reconstruction
    """
    return (np.square(inputs - reconstructions)).mean()


def calc_threshold(reconstr_err_nominal: np.ndarray) -> float:
    """Calculate threshold for anomaly-detection

    :param reconstr_err_nominal: Numpy array of reconstruction errors for examples drawn from nominal class.
    :return: Anomaly-detection threshold
    """
    return np.percentile(reconstr_err_nominal, 0.95)


def detect(reconstr_err_all: np.ndarray, threshold: float) -> list:
    """Recognize anomalies using given reconstruction errors and threshold.

    :param reconstr_err_all: Numpy array of reconstruction errors.
    :param threshold: Anomaly-detection threshold
    :return: list of 0/1 values
    """
    result = []
    for r in reconstr_err_all:
        if r <= threshold:
            result.append(0)
        else:
            result.append(1)
    return result

