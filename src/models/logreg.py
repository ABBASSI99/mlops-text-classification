from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class LogisticRegressionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        X_vect = self.vectorizer.fit_transform(X)
        self.model.fit(X_vect, y)

    def predict(self, X):
        X_vect = self.vectorizer.transform(X)
        return self.model.predict(X_vect).tolist()

    def evaluate(self, X, y):
        X_vect = self.vectorizer.transform(X)
        return self.model.score(X_vect, y)