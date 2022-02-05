import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import math


def report_results_2d(data, labels, predictions_dict):
    plot_it = 0
    plot_cols_cnt = 2 if len(predictions_dict) > 1 else 1
    plot_rows_cnt = math.ceil(len(predictions_dict) / 2)

    for method in predictions_dict:
        plot_it += 1

        print("Results for: '{}'".format(method))
        predictions = predictions_dict[method]
        print_results(labels, predictions)

        subplot = plt.subplot(plot_rows_cnt, plot_cols_cnt,
                              plot_it)
        plt.title(method)
        plot_results_2D(data, labels, predictions, subplot)

    plt.subplots_adjust(hspace=0.3)
    plt.show()


def print_results(labels, predictions):
    # print results
    f1 = f1_score(labels, predictions)
    print("F1-score = {0:.2}".format(f1))

    print("Confusion matrix")
    print(confusion_matrix(labels, predictions))
    print("")


def plot_results_2D(data, labels, predictions, subplot):
    # plot
    idx_reference_anomaly = labels.astype(np.bool)
    idx_reference_nominal = np.logical_not(idx_reference_anomaly)

    idx_result_anomaly = np.array(predictions).astype(np.bool)
    idx_result_nominal = np.logical_not(idx_result_anomaly)

    idx_TN = np.logical_and(idx_result_nominal, idx_reference_nominal)
    idx_FN = np.logical_and(idx_result_nominal, idx_reference_anomaly)

    idx_TP = np.logical_and(idx_result_anomaly, idx_reference_anomaly)
    idx_FP = np.logical_and(idx_result_anomaly, idx_reference_nominal)

    subplot.plot(data[idx_TN, 0],
                 data[idx_TN, 1], 'k.',
                 label="TN")
    subplot.plot(data[idx_FN, 0],
                 data[idx_FN, 1], 'b.',
                 label="FN")
    subplot.plot(data[idx_TP, 0],
                 data[idx_TP, 1], 'r.',
                 label="TP")
    subplot.plot(data[idx_FP, 0],
                 data[idx_FP, 1], 'm.',
                 label="FP")

    subplot.legend(loc='lower right')


def binary2neg_boolean(x: np.ndarray) -> list:
    """Negate vector of binary classification (-1 and 1) and convert to list of (0 and 1)

    Finally every -1 on input becomes 1 on output and ever 1 becomes 0.

    :param x:
    :return:
    """
    x = x * -1
    x[x < 0] = 0

    return x.tolist()
