import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
import numpy as np
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    StratifiedKFold,
    KFold,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.base import BaseEstimator, clone
import warnings


@dataclass
class ExperimentResult:
    experiment_id: str
    model_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    cv_scores: Optional[Dict[str, List[float]]] = None
    fit_time: float = 0.0
    predict_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_experiment_id(config: Dict[str, Any], prefix: str = "exp") -> str:
    config_str = json.dumps(config, sort_keys=True)
    h = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


class CrossValidationRunner:
    def __init__(
        self,
        n_splits: int = 5,
        stratified: bool = True,
        random_state: Optional[int] = 42,
        scoring: Optional[List[str]] = None,
    ):
        self.n_splits = n_splits
        self.stratified = stratified
        self.random_state = random_state
        self.scoring = scoring or ["accuracy", "f1_weighted", "roc_auc"]
        self.cv = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            if stratified
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )

    def run(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, List[float]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cross_validate(
                model,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                return_train_score=True,
            )
        output = {}
        for key in results:
            if key.startswith("test_"):
                output[key.replace("test_", "")] = list(results[key])
            if key.startswith("train_"):
                output["train_" + key.replace("train_", "")] = list(results[key])
        output["fit_time"] = list(results.get("fit_time", []))
        output["score_time"] = list(results.get("score_time", []))
        return output

    def run_single_metric(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "accuracy",
    ) -> List[float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring)
        return list(scores)


class GridSearchRunner:
    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        cv: int = 3,
        scoring: str = "f1_weighted",
        n_jobs: Optional[int] = None,
        refit: bool = True,
    ):
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.best_estimator_: Optional[BaseEstimator] = None
        self.best_params_: Optional[Dict] = None
        self.cv_results_: Optional[Dict] = None

    def run(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        gs = GridSearchCV(
            model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X, y)
        self.best_estimator_ = gs.best_estimator_
        self.best_params_ = gs.best_params_
        self.cv_results_ = gs.cv_results_
        return {
            "best_params": gs.best_params_,
            "best_score": float(gs.best_score_),
            "cv_results": {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in gs.cv_results_.items()
            },
        }


class RandomizedSearchRunner:
    def __init__(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 20,
        cv: int = 3,
        scoring: str = "f1_weighted",
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = None,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_estimator_: Optional[BaseEstimator] = None
        self.best_params_: Optional[Dict] = None

    def run(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        rs = RandomizedSearchCV(
            model,
            self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rs.fit(X, y)
        self.best_estimator_ = rs.best_estimator_
        self.best_params_ = rs.best_params_
        return {
            "best_params": rs.best_params_,
            "best_score": float(rs.best_score_),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(
                average_precision_score(y_true, y_prob)
            )
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except Exception:
            pass
    return metrics


def compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "confusion_matrix": cm.tolist(),
        "true_positives": int(np.diag(cm).sum()),
        "total": int(len(y_true)),
    }


class ExperimentRunner:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        save_predictions: bool = False,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_predictions = save_predictions
        self.verbose = verbose
        self.results: List[ExperimentResult] = []
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_single(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        config: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        config = config or {}
        experiment_id = generate_experiment_id(config, prefix=model_name)
        model_clone = clone(model)
        t0 = time.perf_counter()
        model_clone.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = model_clone.predict(X_test)
        predict_time = time.perf_counter() - t0
        y_prob = None
        if hasattr(model_clone, "predict_proba"):
            y_prob = model_clone.predict_proba(X_test)[:, 1]
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        metrics["fit_time"] = fit_time
        metrics["predict_time"] = predict_time
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            config=config,
            metrics=metrics,
            fit_time=fit_time,
            predict_time=predict_time,
            metadata={"n_train": len(y_train), "n_test": len(y_test)},
        )
        self.results.append(result)
        if self.save_predictions and self.output_dir:
            pred_path = self.output_dir / f"{experiment_id}_predictions.npy"
            np.save(pred_path, y_pred)
        if self.verbose:
            print(f"{model_name} accuracy={metrics.get('accuracy', 0):.4f} f1={metrics.get('f1_weighted', 0):.4f}")
        return result

    def run_cv(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "model",
        config: Optional[Dict[str, Any]] = None,
        n_splits: int = 5,
    ) -> ExperimentResult:
        config = config or {}
        experiment_id = generate_experiment_id(config, prefix=model_name)
        cv_runner = CrossValidationRunner(n_splits=n_splits)
        cv_scores = cv_runner.run(model, X, y)
        mean_metrics = {
            k: float(np.mean(v)) for k, v in cv_scores.items() if isinstance(v, list)
        }
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            config=config,
            metrics=mean_metrics,
            cv_scores=cv_scores,
            metadata={"n_samples": len(y), "n_splits": n_splits},
        )
        self.results.append(result)
        return result

    def run_grid(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        model_name: str = "model",
    ) -> ExperimentResult:
        gs_runner = GridSearchRunner(param_grid=param_grid, cv=3)
        gs_result = gs_runner.run(model, X, y)
        experiment_id = generate_experiment_id(gs_result, prefix=model_name)
        result = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            config=gs_result["best_params"],
            metrics={"best_cv_score": gs_result["best_score"]},
            metadata={"grid_search": True, "cv_results": gs_result.get("cv_results")},
        )
        self.results.append(result)
        return result

    def save_results(self, filename: str = "experiment_results.json") -> Optional[Path]:
        if not self.output_dir:
            return None
        path = self.output_dir / filename
        data = [r.to_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def get_best_by_metric(self, metric: str = "f1_weighted") -> Optional[ExperimentResult]:
        if not self.results:
            return None
        valid = [r for r in self.results if metric in r.metrics]
        if not valid:
            return None
        return max(valid, key=lambda r: r.metrics[metric])


def run_ablation(
    model_factory: Callable[[Dict], BaseEstimator],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ablation_configs: List[Dict[str, Any]],
    metric: str = "f1_weighted",
) -> List[Tuple[Dict[str, Any], float]]:
    outcomes = []
    for config in ablation_configs:
        model = model_factory(config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        outcomes.append((config, score))
    return outcomes


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 200,
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
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
