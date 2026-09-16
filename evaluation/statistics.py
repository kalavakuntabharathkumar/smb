"""Statistical analysis utilities for evaluation results."""
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix

def significance_test(group_a, group_b):
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    return {"t_statistic": float(t_stat), "p_value": float(p_value), "significant_at_0.05": bool(p_value < 0.05), "mean_a": float(np.mean(group_a)), "mean_b": float(np.mean(group_b))}

def confusion_matrix_report(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel() if matrix.size == 4 else (0, 0, 0, 0)
    return {"matrix": matrix.tolist(), "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)}
