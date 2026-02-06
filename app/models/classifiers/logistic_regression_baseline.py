

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


class LogisticRelevanceClassifier:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=1000
        )

        self.pipeline = Pipeline([
            ("scaling", self.scaler),
            ("classifier", self.model)
        ])

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.pipeline.predict(X_test)
        probabilities = self.pipeline.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, predictions))
        print("ROC-AUC:", roc_auc_score(y_test, probabilities))

    def get_feature_importance(self, feature_names):
        coefs = self.model.coef_[0]
        return dict(zip(feature_names, coefs))
