from model import FactorModel, BaseModel
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet
import numpy as np
from constants.data import ALL_CHARS


class Factor(FactorModel):
    def __init__(self, factors='all', **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(fit_intercept=True)
        self.name = factors
        idx_maps = {'all': [0, 1, 2, 3],
                    'capm': [0],
                    'ff': [0, 1, 2]}
        self.factors = idx_maps[factors]

    def _model(self):
        self.model = LinearRegression(fit_intercept=True)

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: np.array):
        return self.model.predict(X)

    def validate(self):
        train_score = np.empty(len(self.X_test))
        test_score = np.empty(len(self.X_test))
        for i in range(len(self.X_train)):
            self._model()
            self.fit(self.X_train[i][:, self.factors], self.y_train[i])
            train_score[i] = self.score(self.X_train[i][:, self.factors], self.y_train[i])
            test_score[i] = self.score(self.X_test[i][:, self.factors], self.y_test[i])

        avg_train_score = np.mean(train_score)
        avg_test_score = np.mean(test_score)

        return avg_train_score, avg_test_score


class ReducedOLS(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(fit_intercept=True)
        reduced_chars = ['bm', 'mom12m', 'mv']
        idxs = [np.where(np.array(ALL_CHARS) == c)[0][0] for c in reduced_chars]
        self.X = self.X[:, idxs]
        self.X_train = self.X_train[:, idxs]
        self.X_test = self.X_test[:, idxs]
        self.X_val = self.X_val[:, idxs]

    def fit(self, X: np.array, y: np.array):
        self.model.fit(X, y)

    def predict(self, X: np.array):
        return self.model.predict(X)

    def validate(self):
        X_train = np.concatenate((self.X_train, self.X_val), axis=0)
        y_train = np.concatenate((self.y_train, self.y_val), axis=0)

        X_test, y_test = self.X_test, self.y_test

        self.model.fit(X_train, y_train)
        train_score = self.score(X_train, y_train)
        test_score = self.score(X_test, y_test)

        return train_score, test_score


