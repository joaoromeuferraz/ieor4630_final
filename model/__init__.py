from data import Data, load_features
from constants.model import *
from constants.data import OUTPUT_DATA_DIR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from data.helpers import loc_nearest


class BaseModel:
    def __init__(self, val_split=0.1, test_split=0.1, label_col='20d_ret', features_dir=None):
        features_dir = features_dir or FEATURES_DIR
        X, y, dates = load_features(features_dir, label_col=label_col)
        self.X = X
        self.y = y
        self.dates = pd.to_datetime(dates)

        self.val_split, self.test_split = val_split, test_split

        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

        print("Processing data...")
        self._process_data()
        print("Done")

    def _process_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_split, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_split, shuffle=False)

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def fit(self, X: np.array, y: np.array):
        """

        :param X: input of shape (batch_size, num_features)
        :param y: target of shape (batch_size, 1)
        :return: self
        """
        raise NotImplementedError("Must implement fit")

    def predict(self, X: np.array):
        """

        :param X: input of shape (batch_size, num_features)
        :return: output of shape (batch_size, 1)
        """
        raise NotImplementedError("Must implement predict")

    def score(self, X: np.array, y: np.array):
        """
        Calculated out-of-sample R2

        :param X: input of shape (batch_size, num_features)
        :param y: target of shape (batch_size, 1)
        :return:R2
        """
        y_pred = self.predict(X)
        r2 = np.sum((y - y_pred)**2)/np.sum(y**2)
        r2 = 1 - r2
        return r2


class FactorModel:
    def __init__(self, path=None, min_len=100, test_size=20, label_col='20d_ret'):
        path = path or OUTPUT_DATA_DIR
        self.path = path
        self.min_len = min_len
        self.test_size = test_size
        self.label_col = label_col

        fnames = os.listdir(path)
        dates = [f[:-4] for f in fnames]
        dates = pd.to_datetime(dates)
        dates = dates.sort_values()
        self.dates = dates
        self.fnames = fnames

        self.d = Data()
        self.factors = self._get_factors()
        self.cum_factors = (1 + self.factors).cumprod()
        self.labels = self._load_labels()

        self.X, self.y = None, None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None

        print("Processing data...")
        self._get_training_data()  # list
        self._train_test_split()
        print("Done")

    def _get_factors(self):
        start_date = self.dates.min().strftime('%Y-%m-%d')
        factor_start = loc_nearest(start_date, self.d.dates_f, method='forward')

        end_date = self.dates.max().strftime('%Y-%m-%d')
        factor_end = loc_nearest(end_date, self.d.dates_f, method='backward')

        factors = self.d._get_factors(factor_start, factor_end)
        return factors

    def _load_labels(self):
        """

        :param path: directory that contains csv files
        :param label_col: name of column containing label
        :return:
        """
        all_labels = None
        print(f"Loading labels...")
        for f in self.fnames:
            df = pd.read_csv(os.path.join(self.path, f))
            df = df.set_index(df.columns[0])
            df.index.name = 'permno'
            labels = df.loc[:, [self.label_col]]
            labels = labels.rename(columns={self.label_col: f[:-4]})
            if all_labels is None:
                all_labels = labels
            else:
                all_labels = pd.concat((all_labels, labels), axis=1, join='outer')

        print(f"Done")

        return all_labels

    def _get_training_data(self):
        all_X, all_y = [], []
        for p in self.labels.index:
            label_series = self.labels.loc[p].dropna()
            label_series.index = pd.to_datetime(label_series.index)
            if len(label_series) >= self.min_len:
                X = self.cum_factors.reindex(label_series.index, method='ffill')
                X = X.pct_change(periods=1).values
                y = label_series.values
                all_X.append(X[1:])
                all_y.append(y[1:])
        self.X = all_X
        self.y = all_y

    def _train_test_split(self):
        X_train, X_test, y_train, y_test = [], [], [], []
        ts = self.test_size
        for i in range(len(self.X)):
            X_train.append(self.X[i][:-ts])
            X_test.append(self.X[i][-ts:])
            y_train.append(self.y[i][:-ts])
            y_test.append(self.y[i][-ts:])

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def fit(self, X: np.array, y: np.array):
        """

        :param X: input of shape (batch_size, num_features)
        :param y: target of shape (batch_size, 1)
        :return: self
        """
        raise NotImplementedError("Must implement fit")

    def predict(self, X: np.array):
        """

        :param X: input of shape (batch_size, num_features)
        :return: output of shape (batch_size, 1)
        """
        raise NotImplementedError("Must implement predict")

    def score(self, X: np.array, y: np.array):
        """
        Calculated out-of-sample R2

        :param X: input of shape (batch_size, num_features)
        :param y: target of shape (batch_size, 1)
        :return:R2
        """
        y_pred = self.predict(X)
        r2 = np.sum((y - y_pred)**2)/np.sum(y**2)
        r2 = 1 - r2
        return r2







