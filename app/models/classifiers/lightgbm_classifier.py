

import lightgbm as lgb
from sklearn.metrics import classification_report


class LightGBMRelevanceModel:

    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=400,
            num_leaves=64,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))
