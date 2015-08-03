import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
Model Evaluation Functions
"""
AUC_WEIGHTS = [(.2, 2.0), (.2, 1.5), (.2, 1.0), (.2, 0.5), (.2, 0.0)]

# given a TPR score, returns the adjusted score given weightings
def weight_normalizer(p):
    x = p.copy()
    score = 0.0
    for t, w in AUC_WEIGHTS:
        if x <= t:
            score += x * w
            break
        else:
            score += t * w
            x -= t
    return score


# given sorted, monotonically increasing arrays of x and y coordinates,
# approximates the area under the curve
def approx_area_rectangular(x, y):
    n = len(x)
    assert(len(y) == n)
    area = 0.0
    for i, xi in enumerate(x[:-1]):
        xa = x[i + 1] - x[i]
        ya = (y[i] + y[i + 1]) / 2.0
        area += xa * ya
    return area


def generate_roc_points(preds_and_labels, grain=1000):
    # split into positive and negative classes and sort for easy thresholding
    sorted_pos = sorted([p for p, l in preds_and_labels if l == 1], reverse=True)
    sorted_neg = sorted([p for p, l in preds_and_labels if l == 0], reverse=True)
    nt = grain + 1
    thresholds = np.linspace(1.0, 0.0, nt)
    pos_idx = 0
    neg_idx = 0
    n_pos = len(sorted_pos)
    n_neg = len(sorted_neg)
    max_pos_idx = n_pos - 1
    max_neg_idx = n_neg - 1
    tps = np.zeros(nt)
    fps = np.zeros(nt)

    for i, t in enumerate(thresholds):
        # move indices
        while pos_idx < max_pos_idx and sorted_pos[pos_idx] >= t:
            pos_idx += 1
        while neg_idx < max_neg_idx and sorted_neg[neg_idx] >= t:
            neg_idx += 1
        # count TPs and FPs
        tps[i] = pos_idx
        fps[i] = neg_idx

    return tps / n_pos, fps / n_neg


# competition evaluation function.
# See: https://www.kaggle.com/c/flavours-of-physics/details/evaluation
def weighted_auc(tprs, fprs, areafun=approx_area_rectangular):
    # normalize tprs
    wtprs = np.array([weight_normalizer(x) for x in tprs])
    return areafun(fprs, wtprs)


def regular_auc(tprs, fprs, areafun=approx_area_rectangular):
    return areafun(fprs, tprs)


def score_weighted(model, X, y):
    tpr, fpr = generate_roc_points(zip(model.predict_proba(X)[:, 1], y))
    return weighted_auc(tpr, fpr)