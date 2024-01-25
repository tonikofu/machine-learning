import numpy as np

def my_accuracy_score(y_true, y_pred):
    n_objects = y_true.shape[0]
    if n_objects != y_pred.shape[0]: return

    correct = np.count_nonzero((y_true - y_pred) == 0.0)

    return (correct / n_objects)

def my_confusion_matrix(y_true, y_pred):
    if y_true.shape[0] != y_pred.shape[0]: return

    yT = y_true + y_pred
    yF = y_true - y_pred

    TP = np.count_nonzero((yT) == 2.0)
    TN = np.count_nonzero((yT) == 0.0)
    FP = np.count_nonzero((yF) == -1.0)
    FN = np.count_nonzero((yF) == 1.0)

    return (np.array([[TN, FP], [FN, TP]]))

def my_classification_report(y_true, y_pred):
    if y_true.shape[0] != y_pred.shape[0]: return

    cm = my_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    N = TN + FP
    precision_N = TN / (TN + FN)
    recall_N = TN / N
    f1_N = 2 / ((1 / precision_N) + (1 / recall_N))

    P = TP + FN
    precision_P = TP / (TP + FP)
    recall_P = TP / P
    f1_P = 2 / ((1 / precision_P) + (1 / recall_P))

    report = '          precision          recall        f1-score  \n'
    report += "0.0" + "{:16.2f}".format(precision_N) + "{:16.2f}".format(recall_N) + "{:16.2f}".format(f1_N) + "\n"
    report += "1.0" + "{:16.2f}".format(precision_P) + "{:16.2f}".format(recall_P) + "{:16.2f}".format(f1_P) + "\n"
    return (report)
