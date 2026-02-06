

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


class GradientBoostingRelevanceModel:

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))
