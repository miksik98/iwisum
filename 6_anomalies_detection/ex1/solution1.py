import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    std = np.std(train_data)
    mean = np.mean(train_data)
    result = []
    for test in test_data:
        if mean - 3 * std <= test <= mean + 3 * std:
            result.append(0)
        else:
            result.append(1)
    return result

