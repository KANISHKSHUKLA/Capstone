

import random
import math
import json
from typing import Dict, List, Tuple

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score

from ml_pipeline.academic_feature_engineering import AcademicFeatureEngineer
from ml_pipeline.deep_bert_embedding_pipeline import DeepBERTEmbeddingPipeline

from ml_pipeline.classifiers.logistic_regression_baseline import LogisticRelevanceClassifier
from ml_pipeline.classifiers.naive_bayes_text_model import NaiveBayesTextClassifier
from ml_pipeline.classifiers.random_forest_classifier import RandomForestRelevanceModel
from ml_pipeline.classifiers.gradient_boosting_classifier import GradientBoostingRelevanceModel
from ml_pipeline.classifiers.lightgbm_classifier import LightGBMRelevanceModel
from ml_pipeline.classifiers.knn_similarity_classifier import KNNSimilarityClassifier
from ml_pipeline.svm_relevance_classifier import SVMRelevanceClassifier


__all__ = [
    "ResearchPaperMLSystem",
    "run_full_pipeline"
]


class ResearchPaperMLSystem:
    """
    Central ML system coordinating all classifiers and representations.
    """

    def __init__(self, paper_text: str):
        self.paper_text = paper_text
        self.interpretable_vector = None
        self.embedding_vector = None

        self.scaler = StandardScaler()
        self.minmax = MinMaxScaler()

        self.models = {}
        self.probabilities = {}
        self.predictions = {}

        self._initialize_models()

    def _initialize_models(self):
        self.models["logistic"] = LogisticRelevanceClassifier()
        self.models["naive_bayes"] = NaiveBayesTextClassifier()
        self.models["svm"] = SVMRelevanceClassifier()
        self.models["random_forest"] = RandomForestRelevanceModel()
        self.models["gradient_boosting"] = GradientBoostingRelevanceModel()
        self.models["lightgbm"] = LightGBMRelevanceModel()
        self.models["knn"] = KNNSimilarityClassifier()

    def _extract_interpretable_features(self):
        engineer = AcademicFeatureEngineer(self.paper_text)
        features = engineer.extract_all_features()
        ordered_keys = sorted(features.keys())
        vector = [features[k] for k in ordered_keys]
        self.interpretable_vector = np.array(vector, dtype=np.float32)

    def _generate_embeddings(self):
        bert = DeepBERTEmbeddingPipeline()
        self.embedding_vector = bert.generate_embedding(
            self.paper_text,
            strategy="layer_mean"
        )

    def _assemble_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        combined = np.concatenate(
            [self.interpretable_vector, self.embedding_vector],
            axis=0
        )

        X = np.vstack([combined for _ in range(150)])
        y = np.array([random.randint(0, 1) for _ in range(len(X))])

        X = self.scaler.fit_transform(X)
        X = self.minmax.fit_transform(X)

        return X, y

    def _split_dataset(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

    def _store_results(self, model_name: str, preds, probs):
        self.predictions[model_name] = preds
        self.probabilities[model_name] = probs

    def _run_logistic(self, X_train, X_test, y_train):
        model = self.models["logistic"]
        model.train(X_train, y_train)
        preds = model.pipeline.predict(X_test)
        probs = model.pipeline.predict_proba(X_test)[:, 1]
        self._store_results("logistic", preds, probs)

    def _run_naive_bayes(self, X_train, X_test, y_train):
        model = self.models["naive_bayes"]
        model.train(np.abs(X_train), y_train)
        preds = model.model.predict(np.abs(X_test))
        probs = model.model.predict_proba(np.abs(X_test))[:, 1]
        self._store_results("naive_bayes", preds, probs)

    def _run_svm(self, X_train, X_test, y_train):
        model = self.models["svm"]
        model.fit_pipeline(X_train, y_train)
        preds = model.model.predict(X_test)
        probs = model.model.predict_proba(X_test)[:, 1]
        self._store_results("svm", preds, probs)

    def _run_random_forest(self, X_train, X_test, y_train):
        model = self.models["random_forest"]
        model.train(X_train, y_train)
        preds = model.model.predict(X_test)
        probs = model.model.predict_proba(X_test)[:, 1]
        self._store_results("random_forest", preds, probs)

    def _run_gradient_boosting(self, X_train, X_test, y_train):
        model = self.models["gradient_boosting"]
        model.train(X_train, y_train)
        preds = model.model.predict(X_test)
        probs = model.model.predict_proba(X_test)[:, 1]
        self._store_results("gradient_boosting", preds, probs)

    def _run_lightgbm(self, X_train, X_test, y_train):
        model = self.models["lightgbm"]
        model.train(X_train, y_train)
        preds = model.model.predict(X_test)
        probs = model.model.predict_proba(X_test)[:, 1]
        self._store_results("lightgbm", preds, probs)

    def _run_knn(self, X_train, X_test, y_train):
        model = self.models["knn"]
        model.train(X_train, y_train)
        preds = model.model.predict(X_test)
        probs = model.model.predict_proba(X_test)[:, 1]
        self._store_results("knn", preds, probs)

    def _aggregate_outputs(self):
        stacked = np.column_stack(list(self.probabilities.values()))
        mean_score = np.mean(stacked, axis=1)
        median_score = np.median(stacked, axis=1)
        weighted_score = np.average(
            stacked,
            axis=1,
            weights=[1.0 + i * 0.1 for i in range(stacked.shape[1])]
        )
        return mean_score, median_score, weighted_score

    def execute(self) -> Dict[str, float]:
        self._extract_interpretable_features()
        self._generate_embeddings()

        X, y = self._assemble_feature_matrix()
        X_train, X_test, y_train, y_test = self._split_dataset(X, y)

        self._run_logistic(X_train, X_test, y_train)
        self._run_naive_bayes(X_train, X_test, y_train)
        self._run_svm(X_train, X_test, y_train)
        self._run_random_forest(X_train, X_test, y_train)
        self._run_gradient_boosting(X_train, X_test, y_train)
        self._run_lightgbm(X_train, X_test, y_train)
        self._run_knn(X_train, X_test, y_train)

        mean_score, median_score, weighted_score = self._aggregate_outputs()

        return {
            "auc_mean": roc_auc_score(y_test, mean_score),
            "auc_median": roc_auc_score(y_test, median_score),
            "auc_weighted": roc_auc_score(y_test, weighted_score),
            "models_used": list(self.models.keys())
        }


def run_full_pipeline(paper_text: str) -> Dict[str, float]:
    system = ResearchPaperMLSystem(paper_text)
    return system.execute()
