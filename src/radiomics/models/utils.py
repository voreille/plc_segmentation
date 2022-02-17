import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class RemoveHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.85, metric='pearson'):
        self.indices_to_drop = None
        self.metric = metric  # pearson’, ‘kendall’, ‘spearman’
        self.threshold = threshold

    def fit(self, X, y=None):
        corr_matrix = pd.DataFrame(X).corr(
            method=self.metric).abs()  # correlation matrix
        # Find index of feature columns with correlation greater than
        # a defined score and then drop these features
        upper = np.triu(corr_matrix.values, k=1)
        self.indices_to_drop = [
            column for column in range(corr_matrix.shape[1])
            if any(upper[column] > self.threshold)
        ]
        return self

    def transform(self, X, y='deprecated', copy=None):
        return np.delete(X, self.indices_to_drop, axis=1)