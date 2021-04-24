import tensorflow as tf
from tensorflow import keras
import numpy as np
from model import BaseModel
from constants.model import *
import os


class FFN(BaseModel):
    def __init__(self, batch_size=32, lr=1e-06, epochs=100,
                 loss='mean_squared_error', dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss = loss
        self.model = None
        self.dropout = dropout

    def _model(self, neurons, activation='relu'):
        keras.backend.clear_session()
        structure = []
        for i, num in enumerate(neurons):
            structure += [
                keras.layers.Dense(num, activation=activation, name=f"layer{i + 1}"),
                keras.layers.BatchNormalization()
            ]
        structure.append(keras.layers.Dropout(self.dropout))
        structure.append(keras.layers.Dense(1, name='output'))
        self.model = keras.Sequential(structure)
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr=self.lr),
            loss=self.loss
        )
        self.model.build(input_shape=(None, self.X.shape[1]))
        self.model.summary()

    def fit(self, X: np.array, y: np.array):
        self.model.fit(
            x=X,
            y=y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.val_split,
            verbose=1
        )

    def score(self, X: np.array, y: np.array, batch_size=1000):
        N = X.shape[0] // batch_size
        batches_X, batches_y = np.array_split(X, N), np.array_split(y, N)
        r2 = 0
        for i in range(N):
            y_pred = self.predict(batches_X[i])
            res = np.sum((batches_y[i] - y_pred) ** 2) / np.sum((batches_y[i]) ** 2)
            res = 1 - res
            r2 += res / N
        return r2

    def predict(self, X: np.array):
        return self.model.predict(X)

    def validate(self, neurons, activation='relu'):
        self._model(neurons, activation=activation)
        self.fit(self.X_train, self.y_train)
        train_score = self.score(self.X_train, self.y_train)
        test_score = self.score(self.X_test, self.y_test)
        return train_score, test_score

    def save_model(self, checkpoint_dir=None, model_name=None):
        checkpoint_dir = checkpoint_dir or CHECKPOINT_PATH
        if model_name is None:
            model_name = 'nn/'
        else:
            model_name += "/"
        path = os.path.join(checkpoint_dir, model_name)
        checkpoint = tf.train.Checkpoint(model=self.model)
        save_path = checkpoint.save(path)
        return save_path

    def load_model(self, checkpoint_dir=None, model_name=None):
        checkpoint_dir = checkpoint_dir or CHECKPOINT_PATH
        if model_name is None:
            model_name = 'nn/'
        else:
            model_name += "/"
        path = os.path.join(checkpoint_dir, model_name)
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.restore(path)





