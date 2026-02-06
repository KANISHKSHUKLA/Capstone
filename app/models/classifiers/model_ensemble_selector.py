from typing import List, Dict, Optional, Tuple, Any, Callable
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import warnings


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimators: List[BaseEstimator],
        meta_estimator: Optional[BaseEstimator] = None,
        cv: int = 5,
        use_proba: bool = True,
        random_state: Optional[int] = 42,
    ):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator or LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        self.cv = cv
        self.use_proba = use_proba
        self.random_state = random_state
        self.fitted_base_: List[BaseEstimator] = []
        self.fitted_meta_: Optional[BaseEstimator] = None
        self.classes_: Optional[NDArray] = None
        self.label_binarizer_: Optional[LabelBinarizer] = None

    def _get_base_predictions(self, estimators: List[BaseEstimator], X: NDArray) -> NDArray:
        preds = []
        for est in estimators:
            if self.use_proba and hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                if proba.shape[1] == 2:
                    preds.append(proba[:, 1])
                else:
                    preds.append(proba)
            else:
                preds.append(est.predict(X))
        if preds and isinstance(preds[0], np.ndarray) and preds[0].ndim == 1:
            return np.column_stack(preds)
        return np.concatenate([p.reshape(-1, 1) if p.ndim == 1 else p for p in preds], axis=1)

    def fit(self, X: NDArray, y: NDArray) -> "StackingEnsemble":
        self.classes_ = np.unique(y)
        self.label_binarizer_ = LabelBinarizer()
        self.label_binarizer_.fit(y)
        self.fitted_base_ = [clone(est) for est in self.base_estimators]
        for est in self.fitted_base_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X, y)
        meta_features = self._get_base_predictions(self.fitted_base_, X)
        self.fitted_meta_ = clone(self.meta_estimator)
        self.fitted_meta_.fit(meta_features, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        meta_features = self._get_base_predictions(self.fitted_base_, X)
        return self.fitted_meta_.predict(meta_features)

    def predict_proba(self, X: NDArray) -> NDArray:
        meta_features = self._get_base_predictions(self.fitted_base_, X)
        if hasattr(self.fitted_meta_, "predict_proba"):
            return self.fitted_meta_.predict_proba(meta_features)
        pred = self.fitted_meta_.predict(meta_features)
        return self.label_binarizer_.transform(pred)


class BlendingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimators: List[BaseEstimator],
        weights: Optional[List[float]] = None,
        use_proba: bool = True,
    ):
        self.estimators = estimators
        self.weights = weights
        self.use_proba = use_proba
        self.fitted_estimators_: List[BaseEstimator] = []
        self.classes_: Optional[NDArray] = None

    def fit(self, X: NDArray, y: NDArray) -> "BlendingEnsemble":
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = [clone(est) for est in self.estimators]
        n = len(self.fitted_estimators_)
        if self.weights is None:
            self.weights = [1.0 / n] * n
        else:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        for est in self.fitted_estimators_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X, y)
        return self

    def _predict_proba_aggregate(self, X: NDArray) -> NDArray:
        all_proba = []
        for est in self.fitted_estimators_:
            if hasattr(est, "predict_proba"):
                all_proba.append(est.predict_proba(X))
            else:
                pred = est.predict(X)
                n_classes = len(self.classes_)
                proba = np.zeros((len(pred), n_classes))
                for i, c in enumerate(self.classes_):
                    proba[pred == c, i] = 1.0
                all_proba.append(proba)
        stacked = np.stack(all_proba, axis=0)
        w = np.array(self.weights).reshape(-1, 1, 1)
        return np.sum(stacked * w, axis=0)

    def predict(self, X: NDArray) -> NDArray:
        proba = self._predict_proba_aggregate(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
        return self._predict_proba_aggregate(X)


class VotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimators: List[BaseEstimator],
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.fitted_estimators_: List[BaseEstimator] = []
        self.classes_: Optional[NDArray] = None

    def fit(self, X: NDArray, y: NDArray) -> "VotingEnsemble":
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = [clone(est) for est in self.estimators]
        n = len(self.fitted_estimators_)
        if self.weights is None:
            self.weights = [1.0] * n
        for est in self.fitted_estimators_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X, y)
        return self

    def predict(self, X: NDArray) -> NDArray:
        if self.voting == "soft":
            proba = np.zeros((len(X), len(self.classes_)))
            for i, est in enumerate(self.fitted_estimators_):
                w = self.weights[i]
                if hasattr(est, "predict_proba"):
                    proba += w * est.predict_proba(X)
                else:
                    pred = est.predict(X)
                    for j, c in enumerate(self.classes_):
                        proba[pred == c, j] += w
            total_w = sum(self.weights)
            proba /= total_w
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            preds = np.array([est.predict(X) for est in self.fitted_estimators_])
            from scipy.stats import mode
            mode_result = mode(preds, axis=0, keepdims=True)
            return mode_result.mode.squeeze().astype(self.classes_.dtype)

    def predict_proba(self, X: NDArray) -> NDArray:
        proba = np.zeros((len(X), len(self.classes_)))
        for i, est in enumerate(self.fitted_estimators_):
            w = self.weights[i]
            if hasattr(est, "predict_proba"):
                proba += w * est.predict_proba(X)
            else:
                pred = est.predict(X)
                for j, c in enumerate(self.classes_):
                    proba[pred == c, j] += w
        proba /= sum(self.weights)
        return proba


class ModelSelector:
    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        X: NDArray,
        y: NDArray,
        metric: str = "f1_weighted",
        cv: int = 5,
        random_state: Optional[int] = 42,
    ):
        self.estimators = estimators
        self.X = X
        self.y = y
        self.metric = metric
        self.cv = cv
        self.random_state = random_state
        self.scores_: Dict[str, float] = {}
        self.best_name_: Optional[str] = None
        self.best_estimator_: Optional[BaseEstimator] = None

    def _score_estimator(self, name: str, estimator: BaseEstimator) -> float:
        skf = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )
        preds = cross_val_predict(estimator, self.X, self.y, cv=skf)
        if self.metric == "f1_weighted":
            return float(f1_score(self.y, preds, average="weighted", zero_division=0))
        if self.metric == "accuracy":
            return float(accuracy_score(self.y, preds))
        return float(f1_score(self.y, preds, average="weighted", zero_division=0))

    def fit(self) -> "ModelSelector":
        for name, est in self.estimators:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = self._score_estimator(name, est)
            self.scores_[name] = score
        if self.scores_:
            self.best_name_ = max(self.scores_, key=self.scores_.get)
            best_est = dict(self.estimators).get(self.best_name_)
            if best_est is not None:
                self.best_estimator_ = clone(best_est)
                self.best_estimator_.fit(self.X, self.y)
        return self

    def get_best(self) -> Tuple[Optional[str], Optional[BaseEstimator]]:
        return self.best_name_, self.best_estimator_

    def get_ranked(self) -> List[Tuple[str, float]]:
        return sorted(
            self.scores_.items(), key=lambda x: x[1], reverse=True
        )


class ThresholdOptimizer:
    def __init__(
        self,
        y_true: NDArray,
        y_prob: NDArray,
        metric: str = "f1",
    ):
        self.y_true = y_true
        self.y_prob = y_prob
        self.metric = metric
        self.best_threshold_: float = 0.5
        self.best_score_: float = 0.0

    def _score_at_threshold(self, t: float) -> float:
        y_pred = (self.y_prob >= t).astype(int)
        return float(
            f1_score(self.y_true, y_pred, average="weighted", zero_division=0)
        )

    def fit(self, n_steps: int = 100) -> "ThresholdOptimizer":
        thresholds = np.linspace(0.0, 1.0, n_steps)
        best_s = -1.0
        best_t = 0.5
        for t in thresholds:
            s = self._score_at_threshold(t)
            if s > best_s:
                best_s = s
                best_t = t
        self.best_threshold_ = best_t
        self.best_score_ = best_s
        return self

    def predict(self, y_prob: NDArray) -> NDArray:
        return (y_prob >= self.best_threshold_).astype(int)


class DynamicWeightedEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        estimators: List[BaseEstimator],
        validation_fraction: float = 0.2,
        random_state: Optional[int] = 42,
    ):
        self.estimators = estimators
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.fitted_estimators_: List[BaseEstimator] = []
        self.weights_: Optional[NDArray] = None
        self.classes_: Optional[NDArray] = None

    def fit(self, X: NDArray, y: NDArray) -> "DynamicWeightedEnsemble":
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        idx = rng.permutation(n)
        n_val = int(n * self.validation_fraction)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        self.fitted_estimators_ = [clone(est) for est in self.estimators]
        for est in self.fitted_estimators_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X_train, y_train)
        preds = np.array([est.predict(X_val) for est in self.fitted_estimators_])
        accs = np.array(
            [accuracy_score(y_val, preds[i]) for i in range(len(self.fitted_estimators_))]
        )
        accs = np.maximum(accs, 1e-9)
        self.weights_ = accs / accs.sum()
        return self

    def predict(self, X: NDArray) -> NDArray:
        preds = np.array([est.predict(X) for est in self.fitted_estimators_])
        proba = np.zeros((len(X), len(self.classes_)))
        for i, w in enumerate(self.weights_):
            for j, c in enumerate(self.classes_):
                proba[preds[i] == c, j] += w
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: NDArray) -> NDArray:
        all_proba = []
        for est in self.fitted_estimators_:
            if hasattr(est, "predict_proba"):
                all_proba.append(est.predict_proba(X))
            else:
                pred = est.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for j, c in enumerate(self.classes_):
                    proba[pred == c, j] = 1.0
                all_proba.append(proba)
        stacked = np.stack(all_proba, axis=0)
        w = self.weights_.reshape(-1, 1, 1)
        return np.sum(stacked * w, axis=0)


def rank_models_by_cv(
    estimators: List[Tuple[str, BaseEstimator]],
    X: NDArray,
    y: NDArray,
    cv: int = 5,
    metric: str = "f1_weighted",
    random_state: Optional[int] = 42,
) -> List[Tuple[str, float]]:
    selector = ModelSelector(
        estimators=estimators,
        X=X,
        y=y,
        metric=metric,
        cv=cv,
        random_state=random_state,
    )
    selector.fit()
    return selector.get_ranked()


def blend_predictions(
    prediction_list: List[NDArray],
    weights: Optional[NDArray] = None,
) -> NDArray:
    preds = np.array(prediction_list)
    if preds.dtype in (np.int32, np.int64):
        from scipy.stats import mode
        if weights is not None:
            unique_classes = np.unique(preds)
            n_classes = len(unique_classes)
            n_samples = preds.shape[1]
            votes = np.zeros((n_samples, n_classes))
            for i in range(preds.shape[0]):
                w = weights[i] if i < len(weights) else 1.0
                for j, c in enumerate(unique_classes):
                    votes[preds[i] == c, j] += w
            return unique_classes[np.argmax(votes, axis=1)]
        return mode(preds, axis=0, keepdims=True).mode.squeeze()
    if weights is None:
        weights = np.ones(preds.shape[0]) / preds.shape[0]
    weights = weights.reshape(-1, 1, 1)
    proba = np.sum(preds * weights, axis=0)
    return (proba >= 0.5).astype(int)


def stack_oof_predictions(
    estimators: List[BaseEstimator],
    X: NDArray,
    y: NDArray,
    cv: int = 5,
    use_proba: bool = True,
    random_state: Optional[int] = 42,
) -> NDArray:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    n_samples = len(X)
    n_estimators = len(estimators)
    if use_proba:
        n_classes = len(np.unique(y))
        oof = np.zeros((n_samples, n_estimators * n_classes))
    else:
        oof = np.zeros((n_samples, n_estimators))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        for i, est in enumerate(estimators):
            clone_est = clone(est)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clone_est.fit(X_train, y[train_idx])
            if use_proba and hasattr(clone_est, "predict_proba"):
                proba = clone_est.predict_proba(X_val)
                oof[val_idx, i * proba.shape[1] : (i + 1) * proba.shape[1]] = proba
            else:
                oof[val_idx, i] = clone_est.predict(X_val)
    return oof
