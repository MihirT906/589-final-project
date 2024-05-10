import numpy as np


def multiclass_eval(y, y_pred):
    precision, recall, f1 = [], [], []
    for c in np.unique(y):
        tp = np.sum((y == c) & (y_pred == c))
        fp = np.sum((y != c) & (y_pred == c))
        fn = np.sum((y == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return np.mean(precision), np.mean(recall), np.mean(f1)


def binary_eval(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f
