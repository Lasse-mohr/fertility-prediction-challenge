import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator
import numpy as np


class TextTransform(BaseEstimator):
    """
    Class to transform multilingual sentences into vectors
    """

    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v1") -> None:
        assert model_name in ["distiluse-base-multilingual-cased-v1", "LaBSE",
                              "average_word_embeddings_glove.6B.300d"], "Wrong model name!"
        self.model = SentenceTransformer(model_name)

    def _vectorize_column(self, column: pd.Series):
        column = column.values
        non_empty_rows = np.where((column != None) & (column != ""))
        embeddings = self.model.encode(column[non_empty_rows])
        output = np.zeros(shape=(column.shape[0], embeddings.shape[1]))
        output[non_empty_rows] = embeddings
        return output

    def fit(self, column: pd.Series):
        """
        Return: np.array of size [n_rows x 512]
                empty and Null rows are assigned vector of 0s
        """
        return self._vectorize_column(column)

    def transform(self, column: pd.Series):
        return self._vectorize_column(column)
