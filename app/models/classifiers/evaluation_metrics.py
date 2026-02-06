from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize
from scipy import stats
from collections import defaultdict


def binary_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    return float(accuracy_score(y_true, y_pred))


def binary_precision(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        precision_score(y_true, y_pred, average="binary", zero_division=zero_division)
    )


def binary_recall(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        recall_score(y_true, y_pred, average="binary", zero_division=zero_division)
    )


def binary_f1(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        f1_score(y_true, y_pred, average="binary", zero_division=zero_division)
    )


def macro_precision(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        precision_score(y_true, y_pred, average="macro", zero_division=zero_division)
    )


def macro_recall(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        recall_score(y_true, y_pred, average="macro", zero_division=zero_division)
    )


def macro_f1(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=zero_division)
    )


def weighted_precision(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        precision_score(y_true, y_pred, average="weighted", zero_division=zero_division)
    )


def weighted_recall(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        recall_score(y_true, y_pred, average="weighted", zero_division=zero_division)
    )


def weighted_f1(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        f1_score(y_true, y_pred, average="weighted", zero_division=zero_division)
    )


def micro_f1(y_true: NDArray, y_pred: NDArray, zero_division: float = 0.0) -> float:
    return float(
        f1_score(y_true, y_pred, average="micro", zero_division=zero_division)
    )


def roc_auc_binary(y_true: NDArray, y_prob: NDArray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_prob))


def roc_auc_multiclass_ovr(
    y_true: NDArray, y_prob: NDArray, n_classes: Optional[int] = None
) -> float:
    classes = np.unique(y_true)
    if n_classes is None:
        n_classes = len(classes)
    if n_classes < 2:
        return 0.0
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if y_prob.shape[1] != n_classes:
        return 0.0
    return float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))


def average_precision_binary(y_true: NDArray, y_prob: NDArray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_prob))


def log_loss_score(y_true: NDArray, y_prob: NDArray, eps: float = 1e-15) -> float:
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(log_loss(y_true, y_prob))


def brier_score(y_true: NDArray, y_prob: NDArray) -> float:
    if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    return float(brier_score_loss(y_true, y_prob))


def matthews_correlation(y_true: NDArray, y_pred: NDArray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def cohen_kappa(y_true: NDArray, y_pred: NDArray, weights: Optional[str] = None) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights=weights))


def confusion_matrix_dict(y_true: NDArray, y_pred: NDArray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "matrix": cm.tolist(),
        "true_positives": int(np.diag(cm).sum()),
        "total": int(len(y_true)),
        "per_class_tp": [int(cm[i, i]) for i in range(cm.shape[0])],
        "per_class_support": [int(cm[i, :].sum()) for i in range(cm.shape[0])],
    }


def true_positive_rate(y_true: NDArray, y_pred: NDArray, positive_class: int = 1) -> float:
    mask = y_true == positive_class
    if mask.sum() == 0:
        return 0.0
    return float(np.logical_and(mask, y_pred == positive_class).sum() / mask.sum())


def false_positive_rate(y_true: NDArray, y_pred: NDArray, positive_class: int = 1) -> float:
    mask = y_true != positive_class
    if mask.sum() == 0:
        return 0.0
    return float(np.logical_and(mask, y_pred == positive_class).sum() / mask.sum())


def specificity(y_true: NDArray, y_pred: NDArray, positive_class: int = 1) -> float:
    return 1.0 - false_positive_rate(y_true, y_pred, positive_class)


def negative_predictive_value(
    y_true: NDArray, y_pred: NDArray, positive_class: int = 1
) -> float:
    mask = y_pred != positive_class
    if mask.sum() == 0:
        return 0.0
    return float(np.logical_and(mask, y_true != positive_class).sum() / mask.sum())


def balanced_accuracy(y_true: NDArray, y_pred: NDArray) -> float:
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(np.logical_and(mask, y_pred == c).sum() / mask.sum())
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def g_mean_score(y_true: NDArray, y_pred: NDArray, positive_class: int = 1) -> float:
    tpr = true_positive_rate(y_true, y_pred, positive_class)
    spec = specificity(y_true, y_pred, positive_class)
    return float(np.sqrt(tpr * spec))


def youden_index(y_true: NDArray, y_pred: NDArray, positive_class: int = 1) -> float:
    tpr = true_positive_rate(y_true, y_pred, positive_class)
    fpr = false_positive_rate(y_true, y_pred, positive_class)
    return float(tpr - fpr)


def classification_metrics_full(
    y_true: NDArray,
    y_pred: NDArray,
    y_prob: Optional[NDArray] = None,
    positive_class: int = 1,
) -> Dict[str, float]:
    out = {
        "accuracy": binary_accuracy(y_true, y_pred),
        "precision_macro": macro_precision(y_true, y_pred),
        "recall_macro": macro_recall(y_true, y_pred),
        "f1_macro": macro_f1(y_true, y_pred),
        "precision_weighted": weighted_precision(y_true, y_pred),
        "recall_weighted": weighted_recall(y_true, y_pred),
        "f1_weighted": weighted_f1(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "matthews_corrcoef": matthews_correlation(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        "true_positive_rate": true_positive_rate(y_true, y_pred, positive_class),
        "false_positive_rate": false_positive_rate(y_true, y_pred, positive_class),
        "specificity": specificity(y_true, y_pred, positive_class),
        "g_mean": g_mean_score(y_true, y_pred, positive_class),
        "youden_index": youden_index(y_true, y_pred, positive_class),
    }
    if y_prob is not None:
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            out["roc_auc"] = roc_auc_binary(y_true, prob_pos)
            out["average_precision"] = average_precision_binary(y_true, prob_pos)
            out["brier_score"] = brier_score(y_true, prob_pos)
            try:
                out["log_loss"] = log_loss_score(y_true, prob_pos)
            except Exception:
                pass
        else:
            out["roc_auc"] = roc_auc_multiclass_ovr(y_true, y_prob)
    return out


def bootstrap_confidence_interval(
    y_true: NDArray,
    y_pred: NDArray,
    metric_fn: Any,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = 42,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s = metric_fn(y_true[idx], y_pred[idx])
        scores.append(s)
    scores = np.array(scores)
    alpha = 1.0 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    return float(np.mean(scores)), float(lower), float(upper)


def per_class_metrics(
    y_true: NDArray, y_pred: NDArray
) -> Dict[int, Dict[str, float]]:
    classes = np.unique(y_true)
    result = {}
    for c in classes:
        y_true_bin = (y_true == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)
        result[int(c)] = {
            "precision": binary_precision(y_true_bin, y_pred_bin),
            "recall": binary_recall(y_true_bin, y_pred_bin),
            "f1": binary_f1(y_true_bin, y_pred_bin),
            "support": int((y_true == c).sum()),
        }
    return result


def calibration_error(
    y_true: NDArray, y_prob: NDArray, n_bins: int = 10
) -> float:
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    total_error = 0.0
    count = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            total_error += np.abs(avg_pred - avg_true) * mask.sum()
            count += mask.sum()
    return float(total_error / count) if count > 0 else 0.0


def expected_calibration_error(
    y_true: NDArray, y_prob: NDArray, n_bins: int = 10
) -> float:
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            ece += (mask.sum() / n) * np.abs(avg_pred - avg_true)
    return float(ece)


def maximum_calibration_error(
    y_true: NDArray, y_prob: NDArray, n_bins: int = 10
) -> float:
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    bins = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            avg_pred = y_prob[mask].mean()
            avg_true = y_true[mask].mean()
            mce = max(mce, np.abs(avg_pred - avg_true))
    return float(mce)


def multiclass_roc_auc_ovo(
    y_true: NDArray, y_prob: NDArray, n_classes: Optional[int] = None
) -> float:
    classes = np.unique(y_true)
    if n_classes is None:
        n_classes = len(classes)
    if n_classes < 2:
        return 0.0
    return float(
        roc_auc_score(
            y_true, y_prob, average="macro", multi_class="ovo"
        )
    )


def multiclass_roc_auc_ovr(
    y_true: NDArray, y_prob: NDArray, n_classes: Optional[int] = None
) -> float:
    return roc_auc_multiclass_ovr(y_true, y_prob, n_classes)


def top_k_accuracy(y_true: NDArray, y_prob: NDArray, k: int = 5) -> float:
    if y_prob.ndim == 1:
        return float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k_pred[i]:
            correct += 1
    return float(correct / len(y_true))
