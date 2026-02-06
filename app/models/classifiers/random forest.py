

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


class RandomForestRelevanceModel:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))

    def feature_importance(self, feature_names):
        return dict(
            zip(feature_names, self.model.feature_importances_)
        )
