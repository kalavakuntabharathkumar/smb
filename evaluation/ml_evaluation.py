"""Classic ML evaluation metrics."""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def classification_report_ml(y_true, y_pred, y_scores):
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    report["roc_auc"] = float(roc_auc_score(y_true, y_scores)) if len(set(y_true)) > 1 else None
    return report
