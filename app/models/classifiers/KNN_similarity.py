

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


class KNNSimilarityClassifier:

    def __init__(self):
        self.model = KNeighborsClassifier(
            n_neighbors=7,
            metric="cosine",
            weights="distance"
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))
