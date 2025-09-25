from mpi4py import MPI
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleParallelScaler:
    def __init__(self, comm, with_mean=True, with_std=True, eps=1e-12):
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self.comm = comm

    def fit(self, X_local):
        X_local = np.asarray(X_local, dtype=np.float64)
        n_local, n_features = X_local.shape

        # local stats
        sum_local = X_local.sum(axis=0)
        sumsq_local = (X_local**2).sum(axis=0)

        # global stats
        n_total = self.comm.allreduce(n_local, op=MPI.SUM)
        sum_total = np.zeros(n_features)
        sumsq_total = np.zeros(n_features)
        self.comm.Allreduce(sum_local, sum_total, op=MPI.SUM)
        self.comm.Allreduce(sumsq_local, sumsq_total, op=MPI.SUM)

        mean = sum_total / n_total
        var = sumsq_total / n_total - mean**2
        std = np.sqrt(np.maximum(var, 0.0))
        std[std < self.eps] = 1.0

        self.mean_ = mean if self.with_mean else np.zeros_like(mean)
        self.std_ = std if self.with_std else np.ones_like(std)

    def transform(self, X_local):
        X_local = np.asarray(X_local, dtype=np.float64)
        X_local = X_local.copy()
        if self.with_mean:
            X_local -= self.mean_
        if self.with_std:
            X_local /= self.std_
        return X_local

    def fit_transform(self, X_local):
        self.fit(X_local)
        return self.transform(X_local)


class MPICyclicEncoder(BaseEstimator, TransformerMixin):
    """
    MPI‑friendly cyclic encoder transformer.
    Fits by finding global max values for each cyclic column.
    Transforms by adding sin/cos features for each cyclic column.

    Parameters
    ----------
    drop_original : bool, default=True
        Whether to drop the original cyclic columns after encoding.
    """

    def __init__(self, comm, drop_original=True):
        self.drop_original = drop_original
        self.global_max_ = {}
        self.comm = comm

    def fit(self, X, y=None):
        """
        Fit the encoder by finding global max value for each cyclic column.
        X : pandas.DataFrame
            Local chunk of training data.
        """
        for feature in X.columns:
            local_max = X[feature].max()
            global_max = self.comm.allreduce(local_max, op=MPI.MAX)
            self.global_max_[feature] = global_max
        return self

    def transform(self, X):
        result = []
        for feature in X.columns:
            max_value = self.global_max_[feature]
            angles = 2 * np.pi * X[feature].values / max_value
            result.append(np.sin(angles))
            result.append(np.cos(angles))
        return np.column_stack(result)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
