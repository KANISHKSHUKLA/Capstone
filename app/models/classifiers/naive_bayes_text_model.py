

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class NaiveBayesTextClassifier:

    def __init__(self):
        self.model = MultinomialNB(alpha=0.5)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))
